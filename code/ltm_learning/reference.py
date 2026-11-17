"""Non-vectorized reference cascade: one graph, one input pattern at a time, pure Python.

Asynchronous BFS from the active seeds: when a node switches on, each non-input neighbour's
active count goes up by one, and a neighbour switches on once its count reaches
ceil(phi * degree).  The queue order is irrelevant to the final state (least fixed point);
an optional RNG processes it in random order to exercise that.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import numpy as np


def csr(N: int, src: np.ndarray, dst: np.ndarray):
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    col = v[order]
    deg = np.bincount(u, minlength=N)
    rowptr = np.zeros(N + 1, dtype=np.int64)
    rowptr[1:] = np.cumsum(deg)
    return rowptr, col, deg


def cascade_one(rowptr, col, deg, inputs: Sequence[int], x: Sequence[int], phi,
                rng: Optional[random.Random] = None) -> np.ndarray:
    """Final active set (bool [N]) for one input pattern x; phi scalar or per-node sequence."""
    N = len(deg)
    is_input = np.zeros(N, dtype=bool)
    is_input[list(inputs)] = True
    phis = [float(phi)] * N if np.isscalar(phi) else [float(p) for p in phi]
    tau = [math.ceil(p * int(d) - 1e-9) if d > 0 else math.inf for p, d in zip(phis, deg)]
    active = np.zeros(N, dtype=bool)
    count = [0] * N
    queue = [v for v, xi in zip(inputs, x) if xi]
    active[queue] = True
    rp, cl = rowptr.tolist(), col.tolist()
    while queue:
        u = queue.pop(rng.randrange(len(queue))) if rng else queue.pop()
        for j in range(rp[u], rp[u + 1]):
            v = cl[j]
            if is_input[v] or active[v]:
                continue
            count[v] += 1
            if count[v] >= tau[v]:
                active[v] = True
                queue.append(v)
    return active


def cascade_all_patterns(N: int, src: np.ndarray, dst: np.ndarray, inputs: Sequence[int],
                         k: int, phi, rng: Optional[random.Random] = None) -> np.ndarray:
    """[N, 2^k] 0/1 final states, column p for pattern p (input i = bit i of p)."""
    rowptr, col, deg = csr(N, src, dst)
    out = np.zeros((N, 1 << k), dtype=np.uint8)
    for p in range(1 << k):
        x = [(p >> i) & 1 for i in range(k)]
        out[:, p] = cascade_one(rowptr, col, deg, inputs, x, phi, rng)
    return out
