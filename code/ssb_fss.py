"""SC: finite-size scaling collapse of the Sym_k breaking (RESEARCH_PROPOSAL_v2, SC1-SC3).
Independent samples at z = z_c + lambda N^{-1/3}; each sample is ssb.run_sample (order parameter, residual group,
one-edge perturbation).

  python ssb_fss.py --out data/ssb/fss.parquet [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spec_card import check_spec, write_with_header  # noqa: E402
from ssb import run_sample  # noqa: E402

ZC = {0.1: 1.0, 0.25: 1.1140}
LAMBDAS = np.round(np.arange(-3.0, 4.01, 0.5), 2)
SAMPLES = {1000: 4000, 4000: 2000, 16000: 1000, 64000: 500, 256000: 200}


def z_of(phi, lam, N):
    return float(ZC[phi] + lam * N ** (-1 / 3))


def tasks(smoke=False):
    samples = {1000: 20, 4000: 10} if smoke else SAMPLES
    lams = LAMBDAS[::4] if smoke else LAMBDAS
    out = []
    i = 0
    for N, R in samples.items():
        for phi in ZC:
            for lam in lams:
                z = z_of(phi, lam, N)
                for r in range(R):
                    out.append((N, 3, phi, z, 8 * 10 ** 12 + 1_000_003 * i + r, float(lam)))
                i += 1
    out.sort(key=lambda t: -t[0])
    return out


def one(t):
    N, k, phi, z, seed, lam = t
    rec = run_sample((N, k, phi, z, seed))[0]
    rec["lam"] = lam
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ts = tasks(a.smoke)
    check_spec(mode="fss", phis=sorted({t[2] for t in ts}), graph="er")
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    res = []
    with Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(one, ts, chunksize=4)):
            res.append(r)
            if (i + 1) % max(1, len(ts) // 20) == 0:
                print(f"{i + 1}/{len(ts)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_with_header(pd.DataFrame(res), a.out, mode="fss", rows=len(res), graph="er")
    print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
