"""Finite input density: the field rounds activity but not complexity.

Simple-contagion limit on G(N, z/N), k = hN inputs (exact classes). Inputs play two roles:
  * under the all-ON pattern they are sources (ghost field): responsive fraction
      Psi_h = 1 - e^{-z P},  P = 1 - (1-h) e^{-z P}          (smooth in z for h > 0)
  * for relevance, the other inputs are holes (clamped-off inputs block): input i is relevant to u
    iff u lies in a component of G \\ I adjacent to i. G \\ I is ER with z' = z(1-h), so
      C_N/k = q^2,  q = 1 - e^{-z P'},  P' = (1-h)(1 - e^{-z P'})   (sharp at z(1-h) = 1)

  python finite_h.py --out data/finite_h
  python finite_h.py --out data/finite_h --analyze
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from er_checks import sample_gnp  # noqa: E402

HS = [0.05, 0.1, 0.2]
GRAPHS = {10000: 200, 100000: 40, 1000000: 8}


def zgrid(h):
    zc = 1 / (1 - h)
    return sorted(set(np.round(np.concatenate([np.linspace(0.6, 2.0, 15), zc + np.array([-0.08, -0.04, -0.02, 0, 0.02, 0.04, 0.08])]), 4)))


def one(task):
    N, h, z, seed = task
    rng = np.random.default_rng(seed)
    k = int(round(h * N))
    src, dst = sample_gnp(N, z, rng)
    inputs = rng.choice(N, k, replace=False)
    inp = np.zeros(N, bool)
    inp[inputs] = True
    keep = ~inp[src] & ~inp[dst]
    A = coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    size = np.bincount(lab[~inp], minlength=N)
    e1 = inp[src] & ~inp[dst]
    e2 = inp[dst] & ~inp[src]
    comp = np.concatenate([lab[dst[e1]], lab[src[e2]]])
    who = np.concatenate([src[e1], dst[e2]])
    pair = np.unique(comp.astype(np.int64) * N + who)
    pc, pw = pair // N, pair % N
    r = np.bincount(pw, weights=size[pc], minlength=N)[inputs]
    nn = N - k
    psi = size[np.unique(pc)].sum() / nn
    return dict(N=N, h=h, z=z, psi=psi, Ck=r.sum() / (k * nn))


def main(out):
    rng = np.random.default_rng(77)
    T = [(N, h, z, int(rng.integers(2**62))) for N in sorted(GRAPHS, reverse=True) for h in HS
         for z in zgrid(h) for _ in range(GRAPHS[N])]
    with Pool(int(os.environ.get("GQ_CPUS", "4"))) as pool:
        rows = list(pool.imap_unordered(one, T, chunksize=2))
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
    print("done", len(rows), flush=True)


def fixed(f, x0=0.9):
    x = x0
    for _ in range(100000):
        xn = f(x)
        if abs(xn - x) < 1e-14:
            break
        x = xn
    return x


def theory(z, h):
    P = fixed(lambda P: 1 - (1 - h) * math.exp(-z * P))
    Pp = fixed(lambda P: (1 - h) * (1 - math.exp(-z * P)))
    Pp = Pp if Pp > 1e-9 else 0.0
    q = 1 - math.exp(-z * Pp)
    return 1 - math.exp(-z * P), q * q


def analyze(out):
    d = pd.read_csv(out / "samples.csv")
    g = d.groupby(["N", "h", "z"])
    s = g.mean().reset_index()
    s["Ck_se"] = (g.Ck.std() / np.sqrt(g.size())).values
    th = [theory(z, h) for z, h in zip(s.z, s.h)]
    s["psi_th"] = [t[0] for t in th]
    s["Ck_th"] = [t[1] for t in th]
    s.to_csv(out / "summary.csv", index=False)
    big = s[(s.N == s.N.max())]
    above = big[big.z * (1 - big.h) > 1.15]
    print("C/k / theory (N max, z(1-h)>1.15): median", round((above.Ck / above.Ck_th).median(), 4),
          "range", round((above.Ck / above.Ck_th).min(), 3), round((above.Ck / above.Ck_th).max(), 3))
    print("Psi / theory (N max): median", round((big.psi / big.psi_th).median(), 4))
    below = s[s.z * (1 - s.h) < 0.9]
    print("C/k below z(1-h)=0.9 by N (max):", below.groupby("N").Ck.max().to_dict())
    print("Psi below z(1-h)=0.9 by N (min):", below.groupby("N").psi.min().round(4).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    analyze(o) if a.analyze else main(o)
