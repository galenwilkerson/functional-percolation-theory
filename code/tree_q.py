"""q_j(s): probability that j uniformly chosen vertices of a uniform random labelled tree on s
vertices admit a non-input region adjacent to all j (=> one function of complexity j).

Finite ER components are uniform random labelled trees given their size (to leading order),
so the corrected subcritical prediction is
    E D_j = sum_s N n_s C(k,j) (s/N)^j (1-s/N)^(k-j) q_j(s).
Monte Carlo via Pruefer sequences.   python tree_q.py --out data/tree_q.csv
"""
from __future__ import annotations

import argparse
import heapq

import numpy as np
import pandas as pd


def prufer_tree(s, rng):
    if s == 1:
        return []
    if s == 2:
        return [(0, 1)]
    seq = rng.integers(0, s, s - 2)
    deg = np.ones(s, int)
    for v in seq:
        deg[v] += 1
    leaves = [i for i in range(s) if deg[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for v in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, int(v)))
        deg[v] -= 1
        if deg[v] == 1:
            heapq.heappush(leaves, int(v))
    a, b = heapq.heappop(leaves), heapq.heappop(leaves)
    edges.append((a, b))
    return edges


def has_full_region(s, edges, marked):
    """Is there a component of (tree minus marked) adjacent to all marked vertices?"""
    mk = set(marked)
    parent = list(range(s))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a not in mk and b not in mk:
            parent[find(a)] = find(b)
    adj = {}
    for a, b in edges:
        if a in mk and b not in mk:
            adj.setdefault(find(b), set()).add(a)
        elif b in mk and a not in mk:
            adj.setdefault(find(a), set()).add(b)
    return any(len(v) == len(mk) for v in adj.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smax", type=int, default=150)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    rows = []
    for j in (2, 3, 4):
        for s in range(j, a.smax + 1):
            hit = 0
            for _ in range(a.reps):
                e = prufer_tree(s, rng)
                m = rng.choice(s, j, replace=False)
                hit += has_full_region(s, e, m.tolist())
            rows.append(dict(j=j, s=s, q=hit / a.reps))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    d = pd.DataFrame(rows)
    print(d[d.s.isin([2, 3, 4, 5, 10, 20, 50, 100, 150])].pivot(index="s", columns="j", values="q").round(3))


if __name__ == "__main__":
    main()
