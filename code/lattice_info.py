"""Information about the inputs on the 2D square lattice (bond p, periodic), simple contagion
(exact for phi <= 1/4). k random input sites, uniform input patterns.
  I = mean over non-input sites of H(output)      (per-node information)
  J = H(all non-input outputs)                    (joint information, bits)
Question: does J peak at p_c = 1/2, as it peaks at z = 1 on random graphs?

  python lattice_info.py --out data/lattice_info
"""
from __future__ import annotations

import argparse
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
from lattice2d import lattice_edges  # noqa: E402
from info_check import H2  # noqa: E402

K_IN = 8
POP = np.array([bin(x).count("1") for x in range(1 << K_IN)])


def one(task):
    L, p, seed = task
    rng = np.random.default_rng(seed)
    N = L * L
    a, b = lattice_edges(L, p, rng)
    inp = rng.choice(N, K_IN, replace=False)
    isin = np.zeros(N, bool); isin[inp] = True
    idx = -np.ones(N, int); idx[inp] = np.arange(K_IN)
    nn = ~isin[a] & ~isin[b]
    g = coo_matrix((np.ones(nn.sum()), (a[nn], b[nn])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    mask = np.zeros(lab.max() + 1, np.int64)
    for u, v in ((a, b), (b, a)):
        e = isin[u] & ~isin[v]
        np.bitwise_or.at(mask, lab[v[e]], (1 << idx[u[e]]).astype(np.int64))
    m_node = mask[lab[~isin]]
    I = float(H2(2.0 ** -POP[m_node]).mean())
    ms = np.unique(m_node[m_node > 0])
    X = np.arange(1 << K_IN)
    Y = (X[:, None] & ms[None, :]) != 0
    _, cnt = np.unique(Y, axis=0, return_counts=True)
    q = cnt / cnt.sum()
    J = float(-(q * np.log2(q)).sum())
    return L, p, I, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lattice_info")
    ap.add_argument("--fine", action="store_true", help="fine grid near p_c, L up to 1024")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.fine:
        ps = np.round(np.arange(0.440, 0.5051, 0.005), 3)
        reps = {128: 2000, 256: 1200, 512: 600, 1024: 300}
    else:
        ps = np.round(np.concatenate([np.arange(0.30, 0.44, 0.02), np.arange(0.44, 0.60, 0.01),
                                      np.arange(0.60, 0.92, 0.04)]), 3)
        reps = {64: 1000, 128: 800, 256: 500, 512: 300}
    tasks = [(L, float(p), 1000003 * L + 7919 * i + r)
             for L, R in reps.items() for i, p in enumerate(ps) for r in range(R)]
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    with Pool(ncpu) as pool:
        rows = pool.map(one, tasks, chunksize=8)
    df = pd.DataFrame(rows, columns=["L", "p", "I", "J"])
    df.to_csv(out / "samples.csv", index=False)
    s = df.groupby(["L", "p"]).agg(I=("I", "mean"), J=("J", "mean"), J_se=("J", "sem"),
                                   n=("J", "size")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    for L, g in s.groupby("L"):
        jp = g.loc[g.J.idxmax()]; ip = g.loc[g.I.idxmax()]
        print(f"L={L:4d}  J peak p={jp.p:.3f} (J={jp.J:.2f} bits)  I peak p={ip.p:.3f}")


if __name__ == "__main__":
    main()
