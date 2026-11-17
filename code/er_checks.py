"""Numerical checks of the Erdos-Renyi extension (theory.tex §"Extending ER (1960)").

Uses EXACT OR-regime functional classes (theory Cor. 2): components of G with the inputs
deleted, grouped by their adjacent-input set A; the class computes OR_{i in A} x_i and has
complexity |A|. Pure graph computation (scipy), parallel over graphs.

  T1 subcritical complexity thresholds  E D_j = sum_s N n_s C(k,j) (s/N)^j (1-s/N)^(k-j)
  T2 collapse at z_c: E D_j depends on N, k only through c = k N^{-1/3}; ~ c^j at small c
  T3 double jump of the largest non-constant class: log N / N^{2/3} / S N
  T4 full computation (all non-input nodes depend on all k inputs): P ~ exp(-e^{-c}), z = ln N + c

  python er_checks.py --out data/er_checks [--quick]
  python er_checks.py --out data/er_checks --analyze
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

JMAX = 4


def sample_gnp(N, z, rng):
    m = rng.binomial(N * (N - 1) // 2, z / (N - 1))
    extra = int(m * 0.05) + 64
    while True:
        a = rng.integers(0, N, m + extra)
        b = rng.integers(0, N, m + extra)
        ok = a != b
        key = np.unique(np.minimum(a, b)[ok].astype(np.int64) * N + np.maximum(a, b)[ok])
        if key.size >= m:
            key = rng.permutation(key)[:m]
            return key // N, key % N
        extra *= 2


def exact_classes(N, src, dst, inputs):
    """Per non-input node: class id (component of G \\ I) and that class's adjacent-input key.
    Returns (comp label array over all nodes, dict comp -> sorted tuple of adjacent inputs)."""
    inp = np.zeros(N, bool)
    inp[inputs] = True
    keep = ~inp[src] & ~inp[dst]
    A = coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    e1 = inp[src] & ~inp[dst]
    e2 = inp[dst] & ~inp[src]
    comp = np.concatenate([lab[dst[e1]], lab[src[e2]]])
    who = np.concatenate([src[e1], dst[e2]])
    order = np.lexsort((who, comp))
    comp, who = comp[order], who[order]
    keys = {}
    if comp.size:
        cut = np.flatnonzero(np.diff(comp)) + 1
        for c_arr, w_arr in zip(np.split(comp, cut), np.split(who, cut)):
            keys[int(c_arr[0])] = tuple(np.unique(w_arr).tolist())
    return lab, keys, inp


def one_graph(task):
    test, N, z, k, seed = task
    rng = np.random.default_rng(seed)
    src, dst = sample_gnp(N, z, rng)
    inputs = rng.choice(N, k, replace=False)
    lab, keys, inp = exact_classes(N, src, dst, inputs)
    nonin = np.flatnonzero(~inp)
    # node count per component (non-input nodes)
    comp_sizes = np.bincount(lab[nonin], minlength=N)
    # group components with the same key into one class (same function)
    cls = {}
    for c, key in keys.items():
        cls[key] = cls.get(key, 0) + int(comp_sizes[c])
    row = dict(test=test, N=N, z=z, k=k, seed=seed)
    sizes = np.array(list(cls.values()), dtype=np.int64)
    comps = np.array([len(key) for key in cls], dtype=np.int64)
    for j in range(1, JMAX + 1):
        row[f"D{j}"] = int((comps == j).sum())
    row["Lmax"] = int(sizes.max()) if sizes.size else 0
    row["full"] = int(all(len(keys.get(int(c), ())) == k for c in np.unique(lab[nonin]))) \
        if test == "T4" else -1
    return row


def tasks(quick=False):
    rng = np.random.default_rng(2026)
    out = []
    G = {1000: 400, 10000: 200, 100000: 50}
    if quick:
        G = {1000: 20, 10000: 5}
    Ns = sorted(G)
    for N in Ns:
        # T1 subcritical z = 0.5
        for k in sorted({int(x) for x in np.geomspace(2, N / 10, 10)}):
            out += [("T1", N, 0.5, k, int(rng.integers(2**62))) for _ in range(G[N])]
        # T2 critical, k = c N^{1/3}
        for c in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            k = max(2, round(c * N ** (1 / 3)))
            out += [("T2", N, 1.0, k, int(rng.integers(2**62))) for _ in range(G[N])]
    # T3 double jump, k = 5
    for N in ([1000, 3000] if quick else [1000, 3000, 10000, 30000, 100000]):
        for z in (0.5, 1.0, 1.5):
            out += [("T3", N, z, 5, int(rng.integers(2**62))) for _ in range(20 if quick else 200)]
    # T4 full computation, z = ln N + c
    for N in ([1000] if quick else [1000, 10000]):
        for c in (-2, -1, 0, 1, 2, 3):
            out += [("T4", N, math.log(N) + c, 5, int(rng.integers(2**62)))
                    for _ in range(20 if quick else 300)]
    return out


# ------------------------------------------------------------------ predictions
def borel_ns(z, smax):
    s = np.arange(1, smax + 1, dtype=float)
    logP = (s - 1) * np.log(z * s) - z * s - np.array([math.lgamma(x + 1) for x in s])
    return s, np.exp(logP) / s          # n_s = P(|C(v)| = s)/s per vertex


_Q = None


def q_j(j, s):
    """Tree factor q_j(s) from tree_q.py: P(j random vertices of a random s-vertex tree share
    a non-input region adjacent to all of them). Beyond the table, use the last value."""
    global _Q
    if _Q is None:
        t = pd.read_csv(Path(__file__).resolve().parent / "data" / "tree_q.csv")
        _Q = {jj: g.set_index("s").q for jj, g in t.groupby("j")}
    tab = _Q[j]
    smax = int(tab.index.max())
    return tab.reindex(np.minimum(s, smax)).to_numpy()


def pred_Dj(N, z, k, j, tree=True):
    if k < j:
        return 0.0
    s, ns = borel_ns(z, min(N - 1, 20000))   # s < N keeps log(1 - s/N) finite
    p = s / N
    logb = math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(k - j + 1)
    w = N * ns * np.exp(logb + j * np.log(p) + (k - j) * np.log1p(-p))
    if tree and j >= 2:
        w = w * np.nan_to_num(q_j(j, s.astype(int)))   # q_j(s < j) = 0
    return float(w.sum())


def analyze(out: Path):
    df = pd.read_csv(out / "er_checks_raw.csv")
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    res = []
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    # T1
    t1 = df[df.test == "T1"].groupby(["N", "k"]).agg(**{f"D{j}": (f"D{j}", "mean") for j in range(1, JMAX + 1)},
                                                     **{f"D{j}_se": (f"D{j}", "sem") for j in range(1, JMAX + 1)}).reset_index()
    a0 = ax[0, 0]
    for j, col in zip(range(2, JMAX + 1), ["C0", "C1", "C2"]):
        for N, mk in zip(sorted(t1.N.unique()), "os^"):
            d = t1[t1.N == N]
            x = d.k ** j / N ** (j - 1)
            pr = np.array([pred_Dj(N, 0.5, int(k), j) for k in d.k])
            ok = d[f"D{j}"] > 0
            a0.loglog(x[ok], d[f"D{j}"][ok], mk, color=col, ms=3,
                      label=f"j={j}" if N == 1000 else None)
            a0.loglog(x, pr, "-", color=col, lw=0.8, alpha=0.6)
            for kk, mval, se, p in zip(d.k, d[f"D{j}"], d[f"D{j}_se"], pr):
                if mval > 0 and se > 0:
                    res.append(dict(test="T1", N=N, k=kk, j=j, meas=mval, se=se, pred=p,
                                    z_score=(mval - p) / se))
    a0.set_xlabel("k^j / N^(j-1)")
    a0.set_ylabel("E D_j (classes of complexity j)")
    a0.set_title("T1 subcritical (z = 0.5): thresholds k ~ N^((j-1)/j); lines = Borel prediction",
                 fontsize=8)
    a0.legend(fontsize=6)
    # T2
    t2 = df[df.test == "T2"].copy()
    t2["c"] = t2.k / t2.N ** (1 / 3)
    t2 = t2.groupby(["N", "k"]).agg(c=("c", "first"), **{f"D{j}": (f"D{j}", "mean") for j in range(1, JMAX + 1)}).reset_index()
    a1 = ax[0, 1]
    for j, col in zip(range(2, JMAX + 1), ["C0", "C1", "C2"]):
        for N, mk in zip(sorted(t2.N.unique()), "os^"):
            d = t2[(t2.N == N) & (t2[f"D{j}"] > 0)]
            a1.loglog(d.c, d[f"D{j}"], mk + "-", color=col, ms=3, lw=0.7,
                      label=f"j={j}, N={N:.0e}" if j == 2 else None)
        cc = np.geomspace(0.1, 1, 20)
        a1.loglog(cc, 0.3 * cc ** j, "k:", lw=0.8)
    a1.set_xlabel("c = k N^(-1/3)")
    a1.set_ylabel("E D_j")
    a1.set_title("T2 critical (z = 1): collapse in c; dotted ~ c^j", fontsize=8)
    a1.legend(fontsize=6)
    # T3
    t3 = df[df.test == "T3"]
    a2 = ax[1, 0]
    for z, col in [(0.5, "C0"), (1.0, "C3"), (1.5, "C2")]:
        d = t3[(t3.z == z) & (t3.Lmax > 0)].groupby("N").Lmax.mean()
        a2.loglog(d.index, d.values, "o-", color=col, label=f"z = {z}")
    Ns = np.array(sorted(t3.N.unique()), float)
    from scipy.optimize import brentq
    S15 = brentq(lambda S: 1 - math.exp(-1.5 * S) - S, 1e-9, 1)
    a2.loglog(Ns, S15 * Ns, "C2:", lw=1, label="S N")
    a2.loglog(Ns, Ns ** (2 / 3), "C3:", lw=1, label="N^(2/3)")
    a2.loglog(Ns, 3 * np.log(Ns), "C0:", lw=1, label="~ log N")
    a2.set_xlabel("N")
    a2.set_ylabel("largest non-constant class (given present)")
    a2.set_title("T3 double jump of the largest functional class (k = 5)", fontsize=8)
    a2.legend(fontsize=6)
    for z in (0.5, 1.0, 1.5):
        d = t3[(t3.z == z) & (t3.Lmax > 0)].groupby("N").Lmax.mean()
        sl = np.polyfit(np.log(d.index), np.log(d.values), 1)[0]
        res.append(dict(test="T3", z=z, slope=sl,
                        pred={0.5: "~0 (log N)", 1.0: "2/3", 1.5: "1"}[z]))
    # T4
    t4 = df[df.test == "T4"].copy()
    t4["c"] = (t4.z - np.log(t4.N)).round(3)
    d4 = t4.groupby(["N", "c"]).full.mean().reset_index()
    a3 = ax[1, 1]
    for N, mk in zip(sorted(d4.N.unique()), "os"):
        d = d4[d4.N == N]
        a3.plot(d.c, d.full, mk + "-", ms=4, label=f"N = {N:.0e}")
        for c, f in zip(d.c, d.full):
            res.append(dict(test="T4", N=N, c=c, meas=f, pred=math.exp(-math.exp(-c))))
    cc = np.linspace(-2, 3, 100)
    a3.plot(cc, np.exp(-np.exp(-cc)), "k--", lw=1, label="exp(-e^-c) (ER connectivity)")
    a3.set_xlabel("c = z − ln N")
    a3.set_ylabel("P(all nodes depend on all k inputs)")
    a3.set_title("T4 full-computation threshold (k = 5)", fontsize=8)
    a3.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "er_checks.pdf")
    r = pd.DataFrame(res)
    r.to_csv(out / "er_checks_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        t1r = r[r.test == "T1"]
        print("T1: |z-score| median", round(t1r.z_score.abs().median(), 2),
              " frac |z|<3:", round((t1r.z_score.abs() < 3).mean(), 3),
              " median meas/pred:", round((t1r.meas / t1r.pred).median(), 3))
        print(t1r[t1r.N == t1r.N.max()][["k", "j", "meas", "pred", "z_score"]].round(3).to_string(index=False))
        print(r[r.test == "T3"].to_string(index=False))
        print(r[r.test == "T4"].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.analyze:
        analyze(out)
        return
    T = tasks(a.quick)
    ncpu = int(os.environ.get("GQ_CPUS", "4"))
    print(f"{len(T)} graphs on {ncpu} processes", flush=True)
    rows = []
    with Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(one_graph, T, chunksize=8)):
            rows.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(T)}", flush=True)
    tmp = out / "er_checks_raw.csv.part"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out / "er_checks_raw.csv")
    print("done", flush=True)


if __name__ == "__main__":
    main()
