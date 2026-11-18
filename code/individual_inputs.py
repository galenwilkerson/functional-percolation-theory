"""Exact OR / non-OR classification with k individual inputs at large k (no grouping), ~2k cascades.

For node u let R_u = {i : f_u(e_i) = 1} (inputs that activate u alone; k single-input cascades).
f_u is monotone, so f_u is an OR iff f_u(I minus R_u) = 0: if f_u is not an OR it has a minimal true
set T with |T| >= 2, none of whose elements activates u alone, so T is contained in I minus R_u.
Nodes sharing the same R run one cascade on the pattern I minus R (R = empty: all inputs on).
So every node is classified exactly with k + (#distinct R) cascades instead of 2^k.

Cross-check of grouped_inputs.py: the non-OR share near z_c with k = N^a individual inputs.

  python individual_inputs.py --out data/individual
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

COLS = 64            # patterns per batched cascade
MAX_GROUPS = 4000    # cap on distinct R per graph (sampled with weights if exceeded)


def cascade(A, tau, seeds, on_cols):
    """on_cols: bool [k, P] which seeds are on in each pattern. Returns bool [N, P] final state."""
    N = A.shape[0]
    P = on_cols.shape[1]
    on = np.zeros((N, P), bool)
    on[seeds] = on_cols
    free = np.ones(N, bool)
    free[seeds] = False
    X = on.astype(np.float32)
    while True:
        new = on | (free[:, None] & ((A @ X) >= tau[:, None]))
        Xn = new.astype(np.float32)
        if np.array_equal(Xn, X):
            return new
        X = Xn


def run_graph(N, z, phi, k, rng):
    src, dst = sample_gnp(N, z, rng)
    A = csr_matrix((np.ones(2 * src.size, np.float32), (np.concatenate([src, dst]),
                                                         np.concatenate([dst, src]))), shape=(N, N))
    deg = np.asarray(A.sum(1)).ravel()
    tau = np.ceil(phi * deg - 1e-9)
    tau[deg == 0] = np.inf
    seeds = rng.choice(N, k, replace=False)
    free = np.ones(N, bool)
    free[seeds] = False
    # 1) single-input cascades -> R_u as a list of (node, input) pairs
    pn, pi = [], []
    for s in range(0, k, COLS):
        idx = np.arange(s, min(k, s + COLS))
        on = np.zeros((k, idx.size), bool)
        on[idx, np.arange(idx.size)] = True
        St = cascade(A, tau, seeds, on)
        nn, cc = np.nonzero(St & free[:, None])
        pn.append(nn)
        pi.append(idx[cc])
    pn, pi = np.concatenate(pn), np.concatenate(pi)
    order = np.lexsort((pi, pn))
    pn, pi = pn[order], pi[order]
    # group free nodes by R_u (hash of their sorted input list); R = () for nodes with no single activator
    key = {}
    starts = np.flatnonzero(np.r_[True, pn[1:] != pn[:-1]]) if pn.size else np.array([], int)
    ends = np.r_[starts[1:], pn.size] if pn.size else np.array([], int)
    node_R = {}
    for a, b in zip(starts, ends):
        node_R[int(pn[a])] = tuple(pi[a:b].tolist())
    groups: dict = {}
    for u in np.flatnonzero(free):
        groups.setdefault(node_R.get(int(u), ()), []).append(int(u))
    Rs = list(groups)
    w = np.array([len(groups[R]) for R in Rs], float)
    sampled = False
    if len(Rs) > MAX_GROUPS:     # keep R = () and sample the rest proportional to size
        sampled = True
        rest = [i for i, R in enumerate(Rs) if R != ()]
        pick = rng.choice(rest, MAX_GROUPS - 1, replace=False, p=w[rest] / w[rest].sum())
        keep = [Rs.index(())] + list(pick) if () in groups else list(pick)
        Rs = [Rs[i] for i in keep]
    # 2) test pattern I \ R for each R; R = () is always exact, sampled non-empty groups are rescaled
    e_resp = e_non = s_resp = s_non = 0.0
    for s in range(0, len(Rs), COLS):
        chunk = Rs[s:s + COLS]
        on = np.ones((k, len(chunk)), bool)
        for j, R in enumerate(chunk):
            if R:
                on[list(R), j] = False
        St = cascade(A, tau, seeds, on)
        for j, R in enumerate(chunk):
            us = np.array(groups[R])
            fired = int(St[us, j].sum())
            if R:
                s_resp += us.size
                s_non += fired
            else:
                e_resp += fired
                e_non += fired
    scale = 1.0
    if sampled:
        tot_ne = sum(len(v) for R, v in groups.items() if R != ())
        scale = tot_ne / max(1.0, s_resp)
    n_resp = e_resp + scale * s_resp
    n_non = e_non + scale * s_non
    return dict(n_free=int(free.sum()), n_resp=float(n_resp), n_nonor=float(n_non),
                n_pure=float(e_non), n_groups=len(groups), sampled=sampled)


def cells(stage):
    out = []
    if stage == "GROWK":       # k = N^a individual inputs near z_c
        for N in (10**3, 10**4, 10**5):
            for a in (1 / 3, 0.5, 2 / 3):
                k = max(2, round(N ** a))
                for z in (1.06, 1.08, 1.10, 1.11, 1.12, 1.13, 1.15):
                    out.append(dict(N=N, z=z, phi=0.25, k=k, a=round(a, 3), M=max(5, int(round(2e6 / N)))))
    elif stage == "FIXHI":     # fixed density h = k/N, z across z_c: the figure
        for N, M in ((10**4, 100), (10**5, 10)):
            for h in (1e-3, 3e-3, 1e-2):
                k = max(2, round(h * N))
                for z in (0.9, 1.0, 1.05, 1.08, 1.10, 1.12, 1.15, 1.2, 1.3, 1.5):
                    out.append(dict(N=N, z=z, phi=0.25, k=k, h=h, M=M))
    for c in out:
        c["id"] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
    return out


def run_one(args):
    c, rep, seed = args
    rng = np.random.default_rng([int(c["id"], 16) ^ seed, rep])
    return dict(c, rep=rep, **run_graph(c["N"], c["z"], c["phi"], c["k"], rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="FIXHI", choices=["GROWK", "FIXHI"])
    ap.add_argument("--out", default="data/individual")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(c, r, a.seed) for c in cells(a.stage) for r in range(c["M"])]
    tasks.sort(key=lambda t: -t[0]["N"] * t[0]["k"])          # big graphs first (load balance)
    with Pool(int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))) as pool:
        rows = pool.map(run_one, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{a.stage}.parquet", index=False)
    key = ["N", "k", "z"] + (["h"] if "h" in df else [])
    s = df.groupby(key)[["n_resp", "n_nonor", "n_pure"]].sum()
    s["any"] = s.n_nonor / s.n_resp
    s["pure"] = s.n_pure / s.n_resp
    pd.set_option("display.width", 200)
    print(s[["any", "pure"]].round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
