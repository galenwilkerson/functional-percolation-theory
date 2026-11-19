"""OR vs non-OR response when the number of inputs grows with N.

m input groups of g seeds each (k = m g seeds in all; input density h = k/N).  All seeds of a group
are clamped together, so every node computes a monotone Boolean function of m bits; all 2^m - 1
non-zero patterns are run.  g = 1 recovers individual inputs (m = k).

A node's function f is an OR iff f(x) = OR_{i in R} x_i with R = {i : f(e_i) = 1}; otherwise it is
non-OR (some minimal true pattern has two or more groups: synergy).

Stages (cells = N, z, phi, m, g, M graphs):
  FIXH   fixed density h = k/N in {3e-2, 1e-2, 3e-3, 1e-3}, m = 2: compare with nonor_theory.py
  GROW   m = 2, g = round(N^a / 2), a in {0, 1/3, 1/2, 2/3}, N = 1e3..1e6, z near z_c
  M3     m = 3 at fixed h, N = 1e5: which non-OR functions (majority, AND, x1 or (x2 and x3), ...)

  python grouped_inputs.py --stage FIXH --out data/grouped
Writes <out>/<stage>.parquet (one row per graph); resumable per cell (<out>/cells/<id>.parquet).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from er_checks import sample_gnp  # noqa: E402

GRAPHS_NM = 2e6     # graphs per cell ~ GRAPHS_NM / N (at least 3)


def zgrid_onset():
    return [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0]


def cells(stage):
    out = []
    if stage == "FIXH":
        zs = zgrid_onset() + [2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 5.0]
        for N in (10**4, 10**5, 10**6):
            for h in (3e-2, 1e-2, 3e-3, 1e-3):
                g = max(1, round(h * N / 2))
                for z in zs:
                    out.append(dict(N=N, z=z, phi="0.25", m=2, g=g, h=h, scheme=f"h{h:g}"))
    elif stage == "GROW":
        for N in (10**3, 10**4, 10**5, 10**6):
            for a in (0.0, 1 / 3, 0.5, 2 / 3):
                g = 2 if a == 0 else max(1, round(N ** a / 2))
                for z in zgrid_onset():
                    out.append(dict(N=N, z=z, phi="0.25", m=2, g=g, h=2 * g / N,
                                    scheme=f"a{a:.3f}"))
    elif stage == "GROW2":   # fine z near z_c = 1.1141 (phi = 0.25), 5x more graphs
        for N in (10**3, 10**4, 10**5, 10**6):
            for a in (0.0, 1 / 3, 0.5, 2 / 3):
                g = 2 if a == 0 else max(1, round(N ** a / 2))
                for z in (1.06, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.15):
                    out.append(dict(N=N, z=z, phi="0.25", m=2, g=g, h=2 * g / N,
                                    scheme=f"a{a:.3f}", nm=1e7))
    elif stage == "M3":
        for h in (3e-2, 3e-3):
            for z in zgrid_onset() + [3.0, 4.0, 4.5]:
                out.append(dict(N=10**5, z=z, phi="0.25", m=3, g=max(1, round(h * 1e5 / 3)), h=h,
                                scheme=f"h{h:g}"))
    for c in out:
        c["M"] = max(3, int(round(c.pop("nm", GRAPHS_NM) / c["N"])))
        c["stage"] = stage
        key = {k: c[k] for k in ("N", "z", "phi", "m", "g", "M")}
        if stage not in ("FIXH", "GROW", "M3"):   # (old stages keep their ids; GROW shares FIXH files)
            key["stage"] = stage
        c["id"] = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return out


def node_phi(N, lab, rng):
    if lab.startswith("N"):
        mu, sd = (float(x) for x in lab[1:].split("_"))
        return np.clip(mu + sd * rng.standard_normal(N), 1e-6, 1.0)
    return np.full(N, float(lab))


def run_graph(N, z, lab, m, g, rng):
    src, dst = sample_gnp(N, z, rng)
    A = csr_matrix((np.ones(2 * src.size, np.float32), (np.concatenate([src, dst]),
                                                         np.concatenate([dst, src]))), shape=(N, N))
    deg = np.asarray(A.sum(1)).ravel()
    tau = np.ceil(node_phi(N, lab, rng) * deg - 1e-9)
    tau[deg == 0] = np.inf
    seeds = rng.choice(N, m * g, replace=False)
    grp = np.full(N, -1)
    grp[seeds] = np.repeat(np.arange(m), g)
    P = (1 << m) - 1
    pats = np.arange(1, P + 1)
    on = np.zeros((N, P), bool)                    # clamped ON seeds per pattern
    isseed = grp >= 0
    on[isseed] = ((pats[None, :] >> grp[isseed][:, None]) & 1).astype(bool)
    X = on.astype(np.float32)
    free = ~isseed
    rounds = 0
    while True:
        rounds += 1
        cnt = A @ X
        new = on | (free[:, None] & (cnt >= tau[:, None]))
        Xn = new.astype(np.float32)
        if np.array_equal(Xn, X):
            break
        X = Xn
    T = X[free].astype(bool)                       # truth tables on patterns 1..P (00 -> 0)
    code = (T * (1 << np.arange(P))).sum(1)        # bit (p-1) = output on pattern p
    resp = code > 0
    single = np.zeros_like(code)                   # R as a mask over groups
    for i in range(m):
        single |= (T[:, (1 << i) - 1].astype(np.int64) << i)
    orpred = ((pats[None, :] & single[:, None]) != 0)
    is_or = (T == orpred).all(1)
    nonor = resp & ~is_or
    u, c = np.unique(code[nonor], return_counts=True)
    return dict(n_free=int(free.sum()), n_resp=int(resp.sum()), n_nonor=int(nonor.sum()),
                n_all=int((code == (1 << P) - 1).sum()), rounds=rounds,
                nonor_codes=json.dumps({int(a): int(b) for a, b in zip(u, c)}))


def run_cell(args):
    c, outdir, seed = args
    f = Path(outdir) / "cells" / f"{c['id']}.parquet"
    if f.exists():
        return str(f)
    rng = np.random.default_rng(int(c["id"], 16) ^ seed)
    rows = []
    for r in range(c["M"]):
        d = run_graph(c["N"], c["z"], c["phi"], c["m"], c["g"], rng)
        rows.append(dict(c, rep=r, **d))
    tmp = f.with_name(f.name + ".part")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, f)
    return str(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["FIXH", "GROW", "GROW2", "M3"])
    ap.add_argument("--out", default="data/grouped")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--max-N", type=int)
    a = ap.parse_args()
    out = HERE / a.out
    (out / "cells").mkdir(parents=True, exist_ok=True)
    cs = [c for c in cells(a.stage) if not a.max_N or c["N"] <= a.max_N]
    cs.sort(key=lambda c: -c["N"] * c["M"] * c["z"])     # big ones first
    print(f"{a.stage}: {len(cs)} cells", flush=True)
    with Pool(int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))) as pool:
        files = pool.map(run_cell, [(c, str(out), a.seed) for c in cs], chunksize=1)
    df = pd.concat([pd.read_parquet(f) for f in files])
    df.to_parquet(out / f"{a.stage}.parquet", index=False)
    s = df.groupby(["scheme", "N", "z"])[["n_resp", "n_nonor"]].sum()
    s["share"] = s.n_nonor / s.n_resp
    pd.set_option("display.width", 200)
    print(s["share"].unstack("N").round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
