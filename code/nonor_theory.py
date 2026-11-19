"""Tree (generating-function / message-passing) theory of OR vs non-OR response, m = 2 input groups.

Two groups of seeds, each a fraction h/2 of the N nodes (input density h; k = hN grows with N).
Patterns: x = 10, 01, 11 (00 is all-off).  On a locally tree-like graph the final state of a
monotone threshold cascade is given exactly by messages "is v active when its parent is
inactive" (Gleeson & Cahalane 2007).  With two groups a message is a monotone Boolean function of
(x1, x2): one of 0, x1, x2, OR, AND.  Its distribution p_t obeys

    p = (h/2) d_x1 + (h/2) d_x2 + (1-h) F(p),

F(p)_t = P[ v outputs t ],  v has n ~ Poisson(z) children (ER excess degree), degree n+1,
threshold tau = ceil(phi (n+1)), output on pattern x = [ sum_children t_child(x) >= tau ].
A non-seed node's own function uses n ~ Poisson(z) neighbours (degree n).  The joint law of the
three pattern sums is built by convolving one child at a time (exact, no sampling).

The minimal fixed point (iterate from p = seeds only) is the cascade solution.

  python nonor_theory.py --out data/nonor_theory
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["0", "x1", "x2", "OR", "AND"]
# contribution of a child of each type to the active-neighbour count under patterns (10, 01, 11)
CONTRIB = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]])
NMAX = 40


def classify(a10, a01, a11):
    """output bits on (10, 01, 11) -> type index (monotone: a11 >= a10, a01)."""
    if a10 and a01:
        return 3
    if a10:
        return 1
    if a01:
        return 2
    if a11:
        return 4
    return 0


def outputs(p, z, phi, edge: bool):
    """Distribution over output types for a non-seed vertex whose n ~ Poisson(z) children have
    type law p.  edge=True: degree n+1 (message to a parent); False: degree n (the vertex)."""
    out = np.zeros(5)
    J = np.zeros((1, 1, 1))
    J[0, 0, 0] = 1.0
    pois = math.exp(-z)
    for n in range(NMAX + 1):
        if n > 0:   # add one child
            K = np.zeros((n + 1,) * 3)
            for t in range(5):
                if p[t] == 0:
                    continue
                c = CONTRIB[t]
                K[c[0]:c[0] + n, c[1]:c[1] + n, c[2]:c[2] + n] += p[t] * J
            J = K
            pois *= z / n
        d = n + 1 if edge else n
        if d == 0:
            out[0] += pois
            continue
        tau = math.ceil(phi * d - 1e-12)
        idx = np.arange(n + 1)
        act = idx >= tau
        # P of each (a10, a01, a11) combination
        for a10 in (0, 1):
            m10 = act if a10 else ~act
            for a01 in (0, 1):
                m01 = act if a01 else ~act
                for a11 in (0, 1):
                    m11 = act if a11 else ~act
                    pr = J[np.ix_(m10, m01, m11)].sum()
                    if pr > 0:
                        out[classify(a10, a01, a11)] += pois * pr
    return out


def solve(z, phi, h, tol=1e-13, itmax=20000):
    seed = np.array([0, h / 2, h / 2, 0, 0])
    p = seed + np.array([1 - h, 0, 0, 0, 0])
    for it in range(itmax):
        q = seed + (1 - h) * outputs(p, z, phi, edge=True)
        q /= q.sum()   # total mass 1 is an unstable fixed point of the map for z > 1: renormalise
        if np.abs(q - p).max() < tol:
            p = q
            break
        p = q
    node = (1 - h) * outputs(p, z, phi, edge=False)    # non-seed vertices
    return p, node, it


def task(args):
    phi, h, z = args
    p, node, it = solve(z, phi, h)
    resp = node[1:].sum()
    return dict(phi=phi, h=h, z=z, it=it, w_or=node[1:4].sum(), w_and=node[4],
                w_x=node[1] + node[2], w_OR=node[3], share=node[4] / resp if resp else 0.0,
                m_or=p[3], m_and=p[4])


def main():
    import os
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/nonor_theory")
    ap.add_argument("--phi", type=float, nargs="*", default=[0.25, 0.2])
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / a.out
    out.mkdir(parents=True, exist_ok=True)
    hs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    zs = np.round(np.unique(np.concatenate([np.arange(0.5, 0.8, 0.1), np.arange(0.8, 1.6, 0.02),
                                            np.arange(1.6, 6.01, 0.1)])), 4)
    tasks = [(phi, h, float(z)) for phi in a.phi for h in hs for z in zs]
    tasks.sort(key=lambda t: -abs(1 / (t[2] - 1.1 + 1e-3)))   # slow (near-critical) ones first
    with Pool(int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))) as pool:
        rows = pool.map(task, tasks, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["phi", "h", "z"])
    df.to_csv(out / "theory_m2.csv", index=False)
    print(df.groupby(["phi", "h"]).apply(lambda g: g.loc[g.share.idxmax(), ["z", "share", "w_and"]]))


if __name__ == "__main__":
    main()
