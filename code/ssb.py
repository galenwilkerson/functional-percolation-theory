"""Functional percolation as symmetry breaking (RESEARCH_PROPOSAL_v2, experiments X1-X3).

Standard LTM (Watts, fractional threshold phi), ER graphs, k clamped inputs; exact monotone Boolean
function of every node over all 2^k input patterns (truth table code: bit p = output on pattern p,
input j = bit j of p).  Symmetry: Sym_k relabels the inputs; the ensemble (graph x placement) is
invariant.

Per sample we record
  m_i      relevance of input i: fraction of non-input nodes whose function depends on i
  M_i      m_i - mean(m)              (order parameter, traceless)
  anis     A = mean_i M_i^2
  giant_*  the function carried by the largest connected component (modal function over its
           non-input nodes): code, share of the giant carrying it, #relevant inputs, DTC, is-OR,
           |Stab| = number of input permutations leaving it unchanged (residual symmetry)
  census   all functions with node counts (traj mode)

Modes
  traj  X1: grow ONE graph by adding random edges (every prefix is a uniform G(N,M)); record at each z.
  tilt  X2/X3: independent unbiased samples at fixed (N, z); the extensive field exp(eps N M_0) is
        applied afterwards by reweighting (ssb_analyze.py).

  python ssb.py traj --out data/ssb/traj.parquet [--smoke]
  python ssb.py tilt --out data/ssb/tilt.parquet [--smoke]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spec_card import check_spec, write_with_header  # noqa: E402
from dtc import dtc_exact  # noqa: E402

DEDEKIND = {1: 3, 2: 6, 3: 20, 4: 168, 5: 7581}
DELTA = 0.01          # a function is macroscopic if carried by >= DELTA of the non-input nodes


# ----------------------------------------------------------------------------- core
def edge_sequence(N, m_max, rng):
    """m_max distinct random edges (no self-loops) in random order: every prefix of length M is a
    uniform G(N, M)."""
    need = int(m_max * 1.05) + 100
    while True:
        a = rng.integers(0, N, need)
        b = rng.integers(0, N, need)
        ok = a != b
        lo = np.minimum(a, b)[ok].astype(np.int64)
        hi = np.maximum(a, b)[ok].astype(np.int64)
        _, first = np.unique(lo * N + hi, return_index=True)
        first.sort()
        if len(first) >= m_max:
            first = first[:m_max]
            return lo[first], hi[first]
        need *= 2


def adjacency(N, src, dst):
    return csr_matrix((np.ones(2 * len(src), np.int32),
                       (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(N, N))


def functions(A, inputs, phi, k):
    """Exact truth-table codes [N] of every node (inputs clamped; a non-input node is on iff at least
    a fraction phi of its neighbours is on; iterated to the least fixed point)."""
    N = A.shape[0]
    P = 1 << k
    deg = np.asarray(A.sum(axis=1)).ravel()
    thr = np.where(deg > 0, phi * deg - 1e-9, np.inf)
    pats = ((np.arange(P)[None, :] >> np.arange(k)[:, None]) & 1).astype(bool)   # [k, P]
    X = np.zeros((N, P), bool)
    X[inputs] = pats
    clamp = np.zeros(N, bool)
    clamp[inputs] = True
    while True:
        new = (A @ X.astype(np.int32)) >= thr[:, None]
        new[clamp] = X[clamp]
        if np.array_equal(new, X):
            break
        X = new
    w = np.int64(1) << np.arange(P, dtype=np.int64)
    return (X.astype(np.int64) * w).sum(axis=1)


def code_bits(codes, k):
    return ((np.asarray(codes, np.int64)[:, None] >> np.arange(1 << k)[None, :]) & 1).astype(bool)


def relevance(codes, k):
    """bool [F, k]: does f depend on input j."""
    bits = code_bits(codes, k)
    rel = np.zeros((len(bits), k), bool)
    for j in range(k):
        p0 = np.array([p for p in range(1 << k) if not (p >> j) & 1])
        rel[:, j] = (bits[:, p0] != bits[:, p0 | (1 << j)]).any(axis=1)
    return rel, bits


def is_or(bits, rel, k):
    pats = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)   # [P, k]
    orv = (pats[None, :, :] & rel[:, None, :]).any(axis=2)
    return (orv == bits).all(axis=1) & rel.any(axis=1)          # constants are not ORs


@lru_cache(maxsize=None)
def perm_tables(k):
    """[k!, 2^k]: pattern index after relabelling inputs by each permutation."""
    P = 1 << k
    rows = []
    for pi in itertools.permutations(range(k)):
        rows.append([sum(((p >> j) & 1) << pi[j] for j in range(k)) for p in range(P)])
    return np.array(rows)


def stabilizer_mask(codes, k):
    """bool [F, k!]: which permutations leave each function unchanged."""
    bits = code_bits(codes, k)                           # [F, P]
    T = perm_tables(k)                                   # [k!, P]
    return (bits[:, T] == bits[:, None, :]).all(axis=2)


def stabilizer_size(codes, k):
    """|{pi in Sym_k : f o pi = f}| for each code."""
    return stabilizer_mask(codes, k).sum(axis=1)


class FnCache:
    def __init__(self, k):
        self.k = k
        self.d = {}

    def props(self, codes):
        new = [int(c) for c in np.unique(codes) if int(c) not in self.d]
        if new:
            arr = np.array(new, np.int64)
            rel, bits = relevance(arr, self.k)
            orf = is_or(bits, rel, self.k)
            dt = dtc_exact(bits.astype(np.int8), self.k)
            sm = stabilizer_mask(arr, self.k)
            for i, c in enumerate(new):
                self.d[c] = (rel[i], bool(orf[i]), int(dt[i]), int(sm[i].sum()), sm[i])
        return [self.d[int(c)] for c in codes]


def summarize(codes, inputs, lab, k, fc, census=True):
    N = len(codes)
    mask = np.ones(N, bool)
    mask[inputs] = False
    c = codes[mask]
    n = len(c)
    u, cnt = np.unique(c, return_counts=True)
    pr = fc.props(u)
    rel = np.array([x[0] for x in pr])
    orf = np.array([x[1] for x in pr])
    dt = np.array([x[2] for x in pr])
    full = (np.int64(1) << (1 << k)) - 1
    nonconst = (u != 0) & (u != full)
    m = (rel * cnt[:, None]).sum(axis=0) / n
    M = m - m.mean()
    p = cnt / n
    rec = dict(n_nonconst=int(nonconst.sum()), census_H=float(-(p * np.log2(p)).sum()),
               frac_responsive=float(cnt[nonconst].sum() / n),
               frac_nonOR=float(cnt[nonconst & ~orf].sum() / n), max_dtc=int(dt.max()),
               anis=float((M ** 2).mean()))
    for j in range(k):
        rec[f"m{j}"] = float(m[j])
    # the giant component's function
    big = np.argmax(np.bincount(lab))
    gmask = (lab == big) & mask
    if not gmask.any():                       # largest component holds only inputs (tiny graphs)
        gmask = mask
    gu, gc = np.unique(codes[gmask], return_counts=True)
    j = int(np.argmax(gc))
    g = int(gu[j])
    grel, gor, gdt, gst, _ = fc.props([g])[0]
    rec.update(gcc=float((lab == big).sum() / N), giant_fn=g, giant_share=float(gc[j] / gmask.sum()),
               giant_nrel=int(grel.sum()), giant_dtc=gdt, giant_isOR=gor, giant_stab=gst,
               giant_rel="".join("1" if x else "0" for x in grel),
               giant_resp=float(gc[(gu != 0) & (gu != full)].sum() / gmask.sum()))
    macro = nonconst & (cnt >= DELTA * n)
    if macro.any():
        common = np.logical_and.reduce([pr[i][4] for i in np.flatnonzero(macro)])
        rec.update(macro_n=int(macro.sum()), macro_frac=float(cnt[macro].sum() / n),
                   macro_stab=int(common.sum()), macro_nonOR=int((macro & ~orf).sum()),
                   macro_fns=";".join(f"{int(x)}:{int(y)}" for x, y in zip(u[macro], cnt[macro])))
    else:
        rec.update(macro_n=0, macro_frac=0.0, macro_stab=int(len(perm_tables(k))), macro_nonOR=0, macro_fns="")
    if nonconst.any():
        j = int(np.argmax(np.where(nonconst, cnt, -1)))
        rec.update(top_fn=int(u[j]), top_frac=float(cnt[j] / n), top_nrel=int(rel[j].sum()),
                   top_isOR=bool(orf[j]), top_dtc=int(dt[j]), top_stab=int(pr[j][3]))
    else:
        rec.update(top_fn=0, top_frac=0.0, top_nrel=0, top_isOR=False, top_dtc=0, top_stab=int(len(perm_tables(k))))
    if census:
        rec["census"] = ";".join(f"{int(x)}:{int(y)}" for x, y in zip(u, cnt))
    return rec


# ----------------------------------------------------------------------------- modes
def run_traj(task):
    N, k, phi, zs, seed = task
    rng = np.random.default_rng(seed)
    src, dst = edge_sequence(N, int(round(max(zs) * N / 2)), rng)
    inputs = rng.choice(N, k, replace=False)
    fc = FnCache(k)
    out = []
    for z in zs:
        M = int(round(z * N / 2))
        A = adjacency(N, src[:M], dst[:M])
        _, lab = connected_components(A, directed=False)
        rec = summarize(functions(A, inputs, phi, k), inputs, lab, k, fc)
        rec.update(N=N, k=k, phi=phi, z=z, seed=seed)
        out.append(rec)
    return out


def run_sample(task):
    N, k, phi, z, seed = task
    rng = np.random.default_rng(seed)
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    _, lab = connected_components(A, directed=False)
    inputs = rng.choice(N, k, replace=False)
    fc = FnCache(k)
    codes = functions(A, inputs, phi, k)
    rec = summarize(codes, inputs, lab, k, fc, census=False)
    # quenched local field on THIS network: one extra edge from input 0 to a random non-input node
    nbr = set(A.indices[A.indptr[inputs[0]]:A.indptr[inputs[0] + 1]].tolist())
    while True:
        v = int(rng.integers(N))
        if v not in inputs and v not in nbr:
            break
    A2 = adjacency(N, np.append(src, min(inputs[0], v)), np.append(dst, max(inputs[0], v)))
    codes2 = functions(A2, inputs, phi, k)
    mask = np.ones(N, bool)
    mask[inputs] = False
    rel1 = np.array([x[0] for x in fc.props(codes[mask])])
    rel2 = np.array([x[0] for x in fc.props(codes2[mask])])
    d = rel2.mean(axis=0) - rel1.mean(axis=0)
    rec.update(N=N, k=k, phi=phi, z=z, seed=seed, pert_dm0=float(d[0]), pert_dM0=float(d[0] - d.mean()),
               pert_changed=float((codes2[mask] != codes[mask]).mean()),
               pert_v_in_giant=bool(lab[v] == np.argmax(np.bincount(lab))))
    return [rec]


ZS_TRAJ = [round(x, 3) for x in np.arange(0.1, 5.01, 0.1)]
ZS_TILT = (0.5, 0.8, 0.9, 1.0, 1.05, 1.114, 1.2, 1.5, 2.0, 3.0)


def grid(mode, smoke):
    if mode == "traj":
        if smoke:
            g = [(1000, 3, 0.1, 3), (1000, 4, 0.25, 3)]
        else:
            g = [(N, k, phi, R) for N, R in ((1000, 100), (4000, 50), (16000, 25), (64000, 10))
                 for k in (3, 4, 5) for phi in (0.1, 0.2, 0.25)]
        return [(N, k, phi, ZS_TRAJ, 1_000_000 * i + r) for i, (N, k, phi, R) in enumerate(g)
                for r in range(R)], run_traj
    Ns = {1000: 3} if smoke else {1000: 2000, 4000: 1000, 16000: 500, 64000: 200, 256000: 60}
    g = [(N, k, phi, z, R) for N, R in Ns.items() for k in ((3,) if smoke else (3, 4))
         for phi in (0.1, 0.25) for z in ZS_TILT]
    return [(N, k, phi, z, 10 ** 12 + 2_000_003 * i + 11 + r) for i, (N, k, phi, z, R) in enumerate(g)
            for r in range(R)], run_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["traj", "tilt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tasks, fn = grid(a.mode, a.smoke)
    check_spec(mode=a.mode, phis=sorted({t[2] for t in tasks}), graph="er")
    tasks.sort(key=lambda t: -t[0])                       # big N first, for load balance
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    res = []
    with Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
            res.extend(r)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_with_header(pd.DataFrame(res), a.out, mode=a.mode, rows=len(res), graph="er")
    print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
