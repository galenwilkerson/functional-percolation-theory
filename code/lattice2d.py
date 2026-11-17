"""Universality beyond mean field: functional percolation on the 2D square lattice.

Bond-diluted square lattice (L x L, periodic), bond probability p. Degree <= 4, so for
phi <= 1/4 every node fires on one active neighbour: the simple-contagion limit is exact and
f_u = OR of the inputs reachable from u (avoiding the other inputs). k = 5 uniform inputs.

Prediction (2D bond percolation: p_c = 1/2, beta = 5/36, nu = 4/3):
  C_N/k = Pr[node and input both in the infinite cluster] -> P_inf^2 ~ (p - p_c)^{2 beta}, 2 beta = 5/18
  FSS:  (C_N/k) L^{2 beta/nu} = F((p - p_c) L^{1/nu}),   2 beta/nu = 5/24,  1/nu = 3/4
  G_2 = Pr[f_u = f_v != 0] -> P_inf^2 [1 - (1 - P_inf)^k]

  python lattice2d.py --out data/lattice2d
  python lattice2d.py --out data/lattice2d --analyze
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

K = 5
PS = [0.40, 0.44, 0.46, 0.48, 0.49, 0.495, 0.50, 0.505, 0.51, 0.52, 0.54, 0.56, 0.60, 0.65, 0.70]
SAMPLES = {128: 1000, 256: 500, 512: 250, 1024: 100, 2048: 40}


def lattice_edges(L, p, rng):
    idx = np.arange(L * L).reshape(L, L)
    right = np.stack([idx.ravel(), np.roll(idx, -1, axis=1).ravel()], 1)
    down = np.stack([idx.ravel(), np.roll(idx, -1, axis=0).ravel()], 1)
    e = np.concatenate([right, down])
    keep = rng.random(len(e)) < p
    return e[keep, 0], e[keep, 1]


def one(task):
    L, p, seed = task
    rng = np.random.default_rng(seed)
    N = L * L
    src, dst = lattice_edges(L, p, rng)
    A0 = coo_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    _, lab0 = connected_components(A0, directed=False)
    S = np.bincount(lab0).max() / N
    inputs = rng.choice(N, K, replace=False)
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
    nn = N - K
    r = np.bincount(pw, weights=size[pc], minlength=N)[inputs]
    pos = np.zeros(N, np.int64)
    pos[inputs] = np.arange(K)
    mask = np.zeros(N, np.int64)
    np.bitwise_or.at(mask, pc, 1 << pos[pw])
    comps = np.unique(pc)
    fm = pd.Series(size[comps]).groupby(mask[comps]).sum().values / nn
    return dict(L=L, p=p, S=S, psi=fm.sum(), Ck=r.sum() / (K * nn), G2=float((fm ** 2).sum()))


def main(out):
    rng = np.random.default_rng(2036)
    T = [(L, p, int(rng.integers(2**62))) for L in sorted(SAMPLES, reverse=True)
         for p in PS for _ in range(SAMPLES[L])]
    with Pool(int(os.environ.get("GQ_CPUS", "4"))) as pool:
        rows = list(pool.imap_unordered(one, T, chunksize=1))
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
    print("done", len(rows), flush=True)


def analyze(out):
    d = pd.read_csv(out / "samples.csv")
    g = d.groupby(["L", "p"])
    s = g.mean().reset_index()
    se = g.std().reset_index()
    n = g.size().values
    for c in ("S", "Ck", "G2", "psi"):
        s[c + "_se"] = se[c] / np.sqrt(n)
    s["P1"] = 1 - (1 - s.S) ** K
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[["L", "p", "S", "Ck", "Ck_se", "G2"]].round(5).to_string(index=False))
    # ratios to P_inf^2 above p_c at the largest L
    big = s[(s.L == s.L.max()) & (s.p >= 0.54)]
    print("\nC/k / S^2 (L max, p>=0.54):", (big.Ck / big.S ** 2).round(3).tolist())
    print("G2 / (S^2 P1):", (big.G2 / (big.S ** 2 * big.P1)).round(3).tolist())
    # exponent of C/k vs p - p_c at largest L, above the window
    for L in sorted(s.L.unique()):
        t = s[(s.L == L) & (s.p > 0.5)].copy()
        t = t[(t.p - 0.5) * L ** 0.75 > 8]
        if len(t) >= 3:
            sl = np.polyfit(np.log(t.p - 0.5), np.log(t.Ck), 1)[0]
            slS = np.polyfit(np.log(t.p - 0.5), np.log(t.S), 1)[0]
            print(f"L={L}: fitted exponent C/k {sl:.3f} (2beta = {5/18:.3f}); S {slS:.3f} (beta = {5/36:.3f}); n={len(t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    analyze(o) if a.analyze else main(o)
