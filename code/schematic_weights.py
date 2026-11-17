"""Ensemble weights of functions for the Fig. 1 bar panels (simple-contagion limit).

For each random graph G(N, z/(N-1)) with k uniformly placed inputs, every non-input node computes
the OR of the inputs adjacent to its component of G \\ I. We record the fraction of non-input
nodes computing (i) the zero function, (ii) the most common nonzero function (the OR carried by
the largest cluster: the giant above z_c, the incipient cluster at z_c), (iii) any other function,
and average over graphs.

  python schematic_weights.py --out data/schematic_weights
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent


def weights(N, z, k, rng):
    M = rng.poisson(z * N / 2)
    a = rng.integers(0, N, M); b = rng.integers(0, N, M)
    keep = a != b; a, b = a[keep], b[keep]
    inp = rng.choice(N, k, replace=False)
    isin = np.zeros(N, bool); isin[inp] = True
    idx = -np.ones(N, int); idx[inp] = np.arange(k)
    nn = ~isin[a] & ~isin[b]
    g = coo_matrix((np.ones(nn.sum()), (a[nn], b[nn])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    mask = np.zeros(lab.max() + 1, np.int64)
    for u, v in ((a, b), (b, a)):
        e = isin[u] & ~isin[v]
        np.bitwise_or.at(mask, lab[v[e]], (1 << idx[u[e]]).astype(np.int64))
    m = mask[lab[~isin]]
    n = m.size
    zero = np.mean(m == 0)
    nz = m[m > 0]
    if nz.size == 0:
        return zero, 0.0, 0.0
    vals, cnt = np.unique(nz, return_counts=True)
    top = cnt.max() / n
    return zero, top, 1 - zero - top


def curve(out):
    """Weight of the largest responsive cluster's OR against z, for the Fig. 1 order-parameter panel."""
    rng = np.random.default_rng(2027)
    zs = np.round(np.concatenate([np.arange(0.4, 0.9, 0.1), np.arange(0.9, 1.3, 0.05),
                                  np.arange(1.3, 3.01, 0.2)]), 3)
    rows = []
    for N, R in ((10**3, 2000), (10**4, 500), (10**5, 100)):
        for z in zs:
            w = np.array([weights(N, float(z), 3, rng) for _ in range(R)])
            rows.append(dict(N=N, z=float(z), k=3, R=R, top=w[:, 1].mean(),
                             top_se=w[:, 1].std(ddof=1) / np.sqrt(R)))
        print("N", N, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "curve.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/schematic_weights")
    ap.add_argument("--curve", action="store_true")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.curve:
        curve(out)
        return
    rng = np.random.default_rng(2026)
    rows = []
    for N, R in ((10**3, 4000), (10**4, 1000), (10**5, 200)):
        for z in (0.85, 1.0, 1.9):
            w = np.array([weights(N, z, 3, rng) for _ in range(R)])
            rows.append(dict(N=N, z=z, k=3, R=R, zero=w[:, 0].mean(), top=w[:, 1].mean(),
                             other=w[:, 2].mean(), top_se=w[:, 1].std(ddof=1) / np.sqrt(R)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
