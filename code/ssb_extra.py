"""U1 (upper edge, individual inputs at finite density) and D4 (symmetry breaking on the 2D lattice).
Pre-registered in RESEARCH_PROPOSAL_v2.md.

U1  upper: k = hN individual inputs. For counts c (1 .. k, geometric) run R random subsets of c ON inputs (the
    rest clamped off), plus all inputs on. Per network and count: the response (fraction of non-inputs on) of each
    subset. OR of a trigger set predicts P(global | c) = 1 - (1 - P_t)^c, smooth and subset-dependent; a quorum
    (threshold on the count) predicts a step at c*, the same for every subset at a given c.
D4  lattice: bond-diluted L x L square lattice (periodic), phi = 0.1 (simple contagion is exact: degree <= 4),
    k = 3 inputs, exact truth tables; Sym_k order parameter, residual group, P(broken) against P_inf.

  python ssb_extra.py upper   --out data/ssb/upper.parquet   [--smoke]
  python ssb_extra.py lattice --out data/ssb/lattice.parquet [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ssb import FnCache, adjacency, edge_sequence, functions, summarize  # noqa: E402
from lattice2d import lattice_edges  # noqa: E402

R_SUB = 8


def run_patterns(A, inputs, phi, P):
    """P: bool [k, c] patterns -> bool [N, c] fixed points."""
    N = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    thr = np.where(deg > 0, phi * deg - 1e-9, np.inf)
    clamp = np.zeros(N, bool)
    clamp[inputs] = True
    X = np.zeros((N, P.shape[1]), bool)
    X[inputs] = P
    while True:
        new = (A @ X.astype(np.int32)) >= thr[:, None]
        new[clamp] = X[clamp]
        if np.array_equal(new, X):
            return X
        X = new


def counts_for(k):
    c = np.unique(np.round(np.geomspace(1, k, 14)).astype(int))
    return c[c >= 1]


def run_upper(task):
    N, h, phi, z, seed = task
    rng = np.random.default_rng(seed)
    k = max(2, int(round(h * N)))
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    inputs = rng.choice(N, k, replace=False)
    cs = counts_for(k)
    cols, cvals = [], []
    for c in cs:
        for _ in range(R_SUB):
            on = np.zeros(k, bool)
            on[rng.choice(k, c, replace=False)] = True
            cols.append(on)
            cvals.append(c)
    cols.append(np.ones(k, bool))
    P = np.stack(cols, axis=1)
    out = []
    mask = np.ones(N, bool)
    mask[inputs] = False
    X = np.concatenate([run_patterns(A, inputs, phi, P[:, s:s + 128]) for s in range(0, P.shape[1], 128)], axis=1)
    resp = X[mask].mean(axis=0)
    return dict(N=N, h=h, k=k, phi=phi, z=z, seed=seed, counts=np.array(cvals, np.int32),
                resp=resp[:-1].astype(np.float32), resp_allon=float(resp[-1]))


def run_lattice(task):
    L, p, seed = task
    rng = np.random.default_rng(seed)
    N = L * L
    src, dst = lattice_edges(L, p, rng)
    A = adjacency(N, src, dst)
    _, lab = connected_components(A, directed=False)
    k = 3
    inputs = rng.choice(N, k, replace=False)
    rec = summarize(functions(A, inputs, 0.1, k), inputs, lab, k, FnCache(k), census=False)
    rec.update(L=L, N=N, p=p, k=k, seed=seed)
    return rec


def grid(mode, smoke):
    if mode == "upper":
        zs = {0.25: (1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 3.86, 4.2, 4.6, 5.0, 6.0),
              0.2: (1.5, 2.0, 3.0, 4.0, 5.0, 5.5, 5.76, 6.2, 7.0, 8.0, 9.0)}
        cells = [(3000, 2)] if smoke else [(10000, 20), (100000, 6)]
        hs = (0.02,) if smoke else (0.005, 0.02)
        phis = (0.25,) if smoke else (0.25, 0.2)
        g = [(N, h, phi, z, R) for N, R in cells for h in hs for phi in phis
             for z in (zs[phi][::5] if smoke else zs[phi])]
        base, fn = 6 * 10 ** 12, run_upper
    else:
        ps = (0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70, 0.85)
        cells = [(32, 3)] if smoke else [(64, 1000), (128, 500), (256, 200), (512, 100)]
        g = [(L, p, None, None, R) for L, R in cells for p in (ps[::4] if smoke else ps)]
        base, fn = 7 * 10 ** 12, run_lattice
    tasks = []
    for i, cell in enumerate(g):
        for r in range(cell[-1]):
            seed = base + 1_000_003 * i + r
            tasks.append((cell[0], cell[1], cell[2], cell[3], seed) if mode == "upper" else (cell[0], cell[1], seed))
    tasks.sort(key=lambda t: -t[0])
    return tasks, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["upper", "lattice"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tasks, fn = grid(a.mode, a.smoke)
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    res = []
    with Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_parquet(a.out)
    print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
