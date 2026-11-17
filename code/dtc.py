"""Exact decision-tree complexity (DTC) of Boolean functions given as truth tables.

D(f) = 0 if f is constant on the current subcube, else 1 + min_i max(D(f|x_i=0), D(f|x_i=1)),
minimised over the free variables i.  We evaluate it for every subcube of {0,1}^k at once:
a subcube is a ternary index in {0,1,*}^k, stored as an array axis of size 3 (index 2 = *).

Cost per function: O(k 3^k) (k=5: 1.2e3, k=8: 5e4, k=10: 6e5 elementary ops), vectorised
over many functions.  Callers should deduplicate functions first (cache by truth table).
"""
from __future__ import annotations

import numpy as np


def dtc_exact(T: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """T: [F, 2^k] 0/1 truth tables (column p = output on the pattern with input i = bit i of p).
    Returns int8 [F] exact deterministic decision-tree depths."""
    T = np.asarray(T, dtype=np.int8)
    F = T.shape[0]
    if F == 0:
        return np.zeros(0, dtype=np.int8)
    if k == 0:
        return np.zeros(F, dtype=np.int8)
    out = np.empty(F, dtype=np.int8)
    masks = sorted(range(1 << k), key=lambda s: bin(s).count("1"))
    for s0 in range(0, F, chunk):
        Tc = T[s0:s0 + chunk]
        f = Tc.shape[0]
        # array axis a+1 <-> some input variable; which one does not matter (D is symmetric)
        lo = Tc.reshape([f] + [2] * k)
        hi = lo.copy()
        for ax in range(1, k + 1):
            lo = np.concatenate([lo, lo.min(axis=ax, keepdims=True)], axis=ax)
            hi = np.concatenate([hi, hi.max(axis=ax, keepdims=True)], axis=ax)
        D = np.zeros([f] + [3] * k, dtype=np.int8)
        for s in masks:
            if s == 0:
                continue  # points: constant, D = 0
            stars = [a for a in range(k) if (s >> a) & 1]
            base = [slice(None)] + [2 if (s >> a) & 1 else slice(0, 2) for a in range(k)]
            idx = tuple(base)
            best = None
            for a in stars:
                i0 = list(base); i0[a + 1] = 0
                i1 = list(base); i1[a + 1] = 1
                v = np.maximum(D[tuple(i0)], D[tuple(i1)])
                best = v if best is None else np.minimum(best, v)
            const = lo[idx] == hi[idx]
            D[idx] = np.where(const, 0, 1 + best)
        out[s0:s0 + f] = D[tuple([slice(None)] + [2] * k)]
    return out


def dtc_bruteforce(table, k: int) -> int:
    """Reference implementation (slow, recursive). table: sequence of 2^k outputs."""
    from functools import lru_cache

    table = tuple(int(v) for v in table)

    @lru_cache(maxsize=None)
    def rec(fixed):  # fixed: tuple of length k with entries 0, 1 or None
        free = [i for i in range(k) if fixed[i] is None]
        vals = set()
        for p in range(1 << k):
            if all(fixed[i] is None or ((p >> i) & 1) == fixed[i] for i in range(k)):
                vals.add(table[p])
        if len(vals) <= 1:
            return 0
        best = k + 1
        for i in free:
            a = list(fixed); a[i] = 0
            b = list(fixed); b[i] = 1
            best = min(best, max(rec(tuple(a)), rec(tuple(b))))
        return 1 + best

    return rec(tuple([None] * k))
