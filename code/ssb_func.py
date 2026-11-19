"""Functional vs connectivity symmetry breaking (RESEARCH_PROPOSAL_v2, F1-F2).

F1 conn:  same networks, two order parameters.
    functional relevance  m_i = fraction of non-input nodes whose function depends on input i
    connectivity reach    r_i = fraction of non-input nodes reachable from i in G minus the other inputs
  In simple contagion m_i = r_i (Prop. 1 / sandwich lemma).  The functional excess is where they differ:
  which inputs control (m_i >= 1%) vs which are connected (r_i >= 1%), and A_func vs A_conn.
F2 density: finite input density h = k/N (k up to 12, exact truth tables over 2^k patterns, stored as
  bit rows).  For each macroscopic function (>= 1% of nodes): its relevant inputs and whether it is
  symmetric under every transposition of two relevant inputs (true for OR; transpositions generate the
  symmetric group, so this is exact).  "Excess breaking" = macroscopic function not symmetric in its own
  inputs: residual group strictly smaller than Sym_K x Sym_{k-K}.

  python ssb_func.py conn    --out data/ssb/conn.parquet    [--smoke]
  python ssb_func.py density --out data/ssb/density.parquet [--smoke]
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
from spec_card import check_spec, write_with_header  # noqa: E402
from ssb import adjacency, edge_sequence  # noqa: E402

DELTA = 0.01


def truth_rows(A, inputs, phi, k):
    """bool [N, 2^k]: final state of every node on every pattern (input j = bit j of the pattern)."""
    N = A.shape[0]
    P = 1 << k
    deg = np.asarray(A.sum(axis=1)).ravel()
    thr = np.where(deg > 0, phi * deg - 1e-9, np.inf)
    pats = ((np.arange(P)[None, :] >> np.arange(k)[:, None]) & 1).astype(bool)
    X = np.zeros((N, P), bool)
    X[inputs] = pats
    clamp = np.zeros(N, bool)
    clamp[inputs] = True
    while True:
        new = (A @ X.astype(np.int32)) >= thr[:, None]
        new[clamp] = X[clamp]
        if np.array_equal(new, X):
            return X
        X = new


def relevance_rows(X, k):
    """bool [F, k] for truth-table rows X [F, 2^k]."""
    P = 1 << k
    rel = np.zeros((X.shape[0], k), bool)
    for j in range(k):
        p0 = np.array([p for p in range(P) if not (p >> j) & 1])
        rel[:, j] = (X[:, p0] != X[:, p0 | (1 << j)]).any(axis=1)
    return rel


def swap_index(k, i, j):
    p = np.arange(1 << k)
    bi, bj = (p >> i) & 1, (p >> j) & 1
    return p ^ ((bi ^ bj) << i) ^ ((bi ^ bj) << j)


def symmetric_in_relevant(row, rel, k):
    """Is the function (one truth-table row) invariant under every transposition of two relevant inputs?
    Returns (symmetric, fraction of relevant pairs that break it)."""
    R = np.flatnonzero(rel)
    if len(R) < 2:
        return True, 0.0
    bad = tot = 0
    for a in range(len(R)):
        for b in range(a + 1, len(R)):
            tot += 1
            if not np.array_equal(row, row[swap_index(k, R[a], R[b])]):
                bad += 1
    return bad == 0, bad / tot


def is_or_row(row, rel, k):
    if not rel.any():
        return False
    pats = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)
    return bool(np.array_equal((pats & rel[None, :]).any(axis=1), row))


def reach(A, inputs, N):
    """r_i: fraction of non-input nodes reachable from input i when the other inputs are removed."""
    k = len(inputs)
    r = np.zeros(k)
    others = np.zeros(N, bool)
    others[inputs] = True
    n = N - k
    for i, u in enumerate(inputs):
        keep = ~others
        keep[u] = True
        idx = np.flatnonzero(keep)
        sub = A[idx][:, idx]
        _, lab = connected_components(sub, directed=False)
        pos = np.searchsorted(idx, u)
        r[i] = ((lab == lab[pos]).sum() - 1) / n
    return r


def macro_summary(X, inputs, k):
    N = X.shape[0]
    mask = np.ones(N, bool)
    mask[inputs] = False
    Y = X[mask]
    n = len(Y)
    packed = np.packbits(Y, axis=1)
    _, first, cnt = np.unique(packed, axis=0, return_index=True, return_counts=True)
    rows = Y[first]
    nonconst = rows.any(axis=1) & ~rows.all(axis=1)
    rel_u = relevance_rows(rows, k)
    m = (rel_u * cnt[:, None]).sum(axis=0) / n
    macro = np.flatnonzero(nonconst & (cnt >= DELTA * n))
    n_or = n_sym = 0
    frac_bad = []
    share_nonsym = 0.0
    nrel = []
    for f in macro:
        o = is_or_row(rows[f], rel_u[f], k)
        sym, fb = symmetric_in_relevant(rows[f], rel_u[f], k)
        n_or += o
        n_sym += sym
        frac_bad.append(fb)
        nrel.append(int(rel_u[f].sum()))
        if not sym:
            share_nonsym += cnt[f] / n
    return m, dict(macro_n=len(macro), macro_nOR=n_or, macro_nsym=n_sym,
                   macro_nonsym_share=float(share_nonsym),
                   macro_frac_bad_pairs=float(np.mean(frac_bad)) if frac_bad else 0.0,
                   macro_mean_nrel=float(np.mean(nrel)) if nrel else 0.0,
                   macro_share=float(cnt[macro].sum() / n) if len(macro) else 0.0,
                   resp_share=float(cnt[nonconst].sum() / n))


def run_conn(task):
    N, k, phi, z, seed = task
    rng = np.random.default_rng(seed)
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    inputs = rng.choice(N, k, replace=False)
    X = truth_rows(A, inputs, phi, k)
    m, mac = macro_summary(X, inputs, k)
    r = reach(A, inputs, N)
    rec = dict(N=N, k=k, phi=phi, z=z, seed=seed, **mac)
    for i in range(k):
        rec[f"m{i}"], rec[f"r{i}"] = float(m[i]), float(r[i])
    rec["A_func"] = float(((m - m.mean()) ** 2).mean())
    rec["A_conn"] = float(((r - r.mean()) ** 2).mean())
    cf = m >= DELTA
    cc = r >= DELTA
    rec.update(K_func=int(cf.sum()), K_conn=int(cc.sum()), ctrl_differs=bool((cf != cc).any()),
               n_conn_not_func=int((cc & ~cf).sum()), n_func_not_conn=int((cf & ~cc).sum()))
    return rec


def run_density(task):
    N, k, phi, z, seed = task
    rng = np.random.default_rng(seed)
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    inputs = rng.choice(N, k, replace=False)
    X = truth_rows(A, inputs, phi, k)
    m, mac = macro_summary(X, inputs, k)
    return dict(N=N, k=k, h=k / N, phi=phi, z=z, seed=seed, A_func=float(((m - m.mean()) ** 2).mean()),
                K_func=int((m >= DELTA).sum()), **mac)


def single_input_rows(A, inputs, phi, chunk=256):
    """bool [N, k+1]: column i = state with only input i on (others clamped off); last column = all on."""
    N = A.shape[0]
    k = len(inputs)
    deg = np.asarray(A.sum(axis=1)).ravel()
    thr = np.where(deg > 0, phi * deg - 1e-9, np.inf)
    clamp = np.zeros(N, bool)
    clamp[inputs] = True
    cols = []
    patterns = [np.eye(k, dtype=bool)[:, j] for j in range(k)] + [np.ones(k, bool)]
    for s0 in range(0, len(patterns), chunk):
        P = np.stack(patterns[s0:s0 + chunk], axis=1)                 # [k, c]
        X = np.zeros((N, P.shape[1]), bool)
        X[inputs] = P
        while True:
            new = (A @ X.astype(np.int32)) >= thr[:, None]
            new[clamp] = X[clamp]
            if np.array_equal(new, X):
                break
            X = new
        cols.append(X)
    return np.concatenate(cols, axis=1)


def run_finiteh(task):
    N, h, phi, z, seed = task
    rng = np.random.default_rng(seed)
    k = max(2, int(round(h * N)))
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    _, lab = connected_components(A, directed=False)
    inputs = rng.choice(N, k, replace=False)
    X = single_input_rows(A, inputs, phi)
    mask = np.ones(N, bool)
    mask[inputs] = False
    m = X[mask, :k].mean(axis=0)                                       # trigger reach of each input alone
    ctrl = m >= DELTA
    C = float(m[ctrl].mean()) if ctrl.any() else 0.0
    return dict(N=N, h=h, k=k, phi=phi, z=z, seed=seed, A=float(((m - m.mean()) ** 2).mean()),
                K=int(ctrl.sum()), q=float(ctrl.mean()), C=C, psi_allon=float(X[mask, k].mean()),
                gcc=float(np.bincount(lab).max() / N), m_mean=float(m.mean()),
                broken=bool(0 < ctrl.sum() < k))


def grid(mode, smoke):
    if mode == "finiteh":
        zs = (0.6, 0.8, 0.9, 1.0, 1.05, 1.114, 1.2, 1.35, 1.5, 2.0, 3.0)
        # (N, h, networks): both densities at 1e4 and 3e4; only h = 0.002 at 1e5 (k = 200; h = 0.01 costs ~100 s/network)
        cells = ([(10000, 0.002, 60), (10000, 0.01, 60), (30000, 0.002, 30), (30000, 0.01, 30), (100000, 0.002, 8)]
                 if not smoke else [(3000, 0.01, 2)])
        g = [(N, h, phi, z, R) for N, h, R in cells for phi in (0.1, 0.25) for z in (zs if not smoke else (0.8, 2.0))]
        tasks = [(N, h, phi, z, 5 * 10 ** 12 + 1_000_003 * i + r) for i, (N, h, phi, z, R) in enumerate(g)
                 for r in range(R)]
        return tasks, run_finiteh
    if mode == "conn":
        zs = (0.9, 1.114, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5)
        Ns = {2000: 2} if smoke else {4000: 300, 16000: 150, 64000: 60}
        ks = (3,) if smoke else (3, 4)
        g = [(N, k, phi, z, R) for N, R in Ns.items() for k in ks for phi in (0.1, 0.2, 0.25) for z in zs]
        base, fn = 3 * 10 ** 12, run_conn
    else:
        zs = (1.2, 1.5, 2.0, 2.5, 3.0)
        # fixed input density h = k/N: N grows with k (k = 3 .. 12)
        pairs = [(600, 3), (1000, 5), (1600, 8), (2400, 12), (300, 3), (500, 5), (800, 8), (1200, 12)]
        if smoke:
            pairs, zs = [(500, 5), (1200, 12)], (2.0,)
        g = [(N, k, phi, z, 3 if smoke else (100 if k == 12 else 200))
             for N, k in pairs for phi in (0.1, 0.25) for z in zs]
        base, fn = 4 * 10 ** 12, run_density
    tasks = [(N, k, phi, z, base + 1_000_003 * i + r) for i, (N, k, phi, z, R) in enumerate(g) for r in range(R)]
    return tasks, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["conn", "density", "finiteh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tasks, fn = grid(a.mode, a.smoke)
    check_spec(mode=a.mode, phis=sorted({t[2] for t in tasks}), graph="er")
    tasks.sort(key=lambda t: -(t[0] * (1 << t[1]) if isinstance(t[1], int) else t[0] * t[1]))
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    res = []
    with Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_with_header(pd.DataFrame(res), a.out, mode=a.mode, rows=len(res), graph="er")
    print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
