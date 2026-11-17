"""Outline §10.6 robustness: does the input-selection rule change the conclusions?

Selection rules (k = 5 inputs, labels then shuffled):
  uniform   uniform vertices (all other runs)
  edge      endpoints of uniform random edges (degree-biased)
  leaf      uniform among degree-1 vertices
  hub       the k highest-degree vertices
OR regime, exact classes. Only P(input in GCC) = S_in changes; prediction:
  Psi = S[1-(1-S_in)^k],  Lam/k = S S_in,  A = S^2 S_in(1-S_in)(1-1/k),
  H = [1-(1-S_in)^k] H2(S),  Var(c_G) = k S^2 S_in(1-S_in).
  S_in: uniform S; edge 1-(1-S)^2; leaf S; hub -> 1 as N -> infinity (z > 1).

  python input_selection.py --out data/input_selection
  python input_selection.py --out data/input_selection --analyze
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
from er_checks import sample_gnp  # noqa: E402

K = 5
ZS = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]
GRAPHS = {100000: 200, 1000000: 40}
RULES = ["uniform", "edge", "leaf", "hub"]


def pick(rule, N, src, dst, deg, rng):
    if rule == "uniform":
        return rng.choice(N, K, replace=False)
    if rule == "edge":
        out = set()
        while len(out) < K:
            e = rng.integers(len(src))
            out.add(int(src[e] if rng.random() < 0.5 else dst[e]))
        return np.array(sorted(out))
    if rule == "leaf":
        c = np.flatnonzero(deg == 1)
        return rng.choice(c, K, replace=False)
    o = np.argsort(-deg + rng.random(N) * 0.5)          # top degree, random tie-break
    return o[:K]


def one(task):
    N, z, rule, seed = task
    rng = np.random.default_rng(seed)
    src, dst = sample_gnp(N, z, rng)
    deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
    A0 = coo_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    _, lab0 = connected_components(A0, directed=False)
    cs0 = np.bincount(lab0)
    g = int(np.argmax(cs0))
    S = cs0[g] / N
    inputs = rng.permutation(pick(rule, N, src, dst, deg, rng))
    K_in = int((lab0[inputs] == g).sum())
    inp = np.zeros(N, bool)
    inp[inputs] = True
    keep = ~inp[src] & ~inp[dst]
    A = coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    size = np.bincount(lab[~inp], minlength=N)
    e1 = inp[src] & ~inp[dst]
    e2 = inp[dst] & ~inp[src]
    comp = np.concatenate([lab[dst[e1]], lab[src[e2]]])
    who = np.concatenate([src[e1], dst[e2]])
    pair = np.unique(comp.astype(np.int64) * N + who)
    pc, pw = pair // N, pair % N
    r = np.bincount(pw, weights=size[pc], minlength=N)[inputs]
    nn = N - K
    m = r / N
    # function of each responsive component = its adjacent-input set (as a bitmask)
    pos = np.zeros(N, np.int64)
    pos[inputs] = np.arange(K)
    mask = np.zeros(N, np.int64)
    np.bitwise_or.at(mask, pc, 1 << pos[pw])
    comps = np.unique(pc)
    fm = pd.Series(size[comps]).groupby(mask[comps]).sum().values / nn
    p0 = 1 - fm.sum()
    p = np.concatenate([fm, [p0]]) if p0 > 0 else fm
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return dict(N=N, z=z, rule=rule, S=S, K_in=K_in, psi=fm.sum(), lamk=r.sum() / (K * nn),
                A=m.var(), c=r.sum() / nn, H=H, mean_in_deg=deg[inputs].mean())


def main(out):
    rng = np.random.default_rng(1006)
    T = [(N, z, ru, int(rng.integers(2**62))) for N in sorted(GRAPHS, reverse=True)
         for z in ZS for ru in RULES for _ in range(GRAPHS[N])]
    with Pool(int(os.environ.get("GQ_CPUS", "4"))) as pool:
        rows = list(pool.imap_unordered(one, T, chunksize=2))
    pd.DataFrame(rows).to_csv(out / "graphs.csv", index=False)
    print("done", len(rows), flush=True)


def H2(s):
    s = np.clip(s, 1e-300, 1 - 1e-16)
    return -s * np.log(s) - (1 - s) * np.log(1 - s)


def analyze(out):
    d = pd.read_csv(out / "graphs.csv")
    g = d.groupby(["N", "z", "rule"])
    s = g.mean().reset_index()
    s["c_var"] = g.c.var().values
    s["S_in"] = s.K_in / K
    s["H2S"] = H2(s.S)
    S, Si = s.S, s.S_in
    s["psi_p"] = S * (1 - (1 - Si) ** K)
    s["lamk_p"] = S * Si
    s["A_p"] = S ** 2 * Si * (1 - Si) * (1 - 1 / K)
    s["H_p"] = (1 - (1 - Si) ** K) * H2(S)
    s["cvar_p"] = K * S ** 2 * Si * (1 - Si)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s[["N", "z", "rule", "mean_in_deg", "S", "S_in", "psi", "psi_p", "lamk", "lamk_p",
             "A", "A_p", "H", "H_p", "c_var", "cvar_p"]].round(4).to_string(index=False))
    up = s[s.z >= 1.5]
    for q in ("psi", "lamk", "H", "A", "c_var"):
        pcol = {"c_var": "cvar_p"}.get(q, q + "_p")
        rr = (up[q] / up[pcol]).replace([np.inf, -np.inf], np.nan)
        print(f"{q}: median measured/predicted (z>=1.5, all rules) {rr.median():.3f} "
              f"IQR {rr.quantile(.25):.3f}-{rr.quantile(.75):.3f}")
    print("\nonset (z<1): max psi by rule:", s[s.z < 1].groupby("rule").psi.max().round(5).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    analyze(o) if a.analyze else main(o)
