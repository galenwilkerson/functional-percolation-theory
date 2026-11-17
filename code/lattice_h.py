"""Finite input density on the 2D square lattice (simple contagion exact for phi <= 1/4).

k = hN inputs. Predictions (two roles of the inputs):
  * activity: responsive fraction Psi_h > 0 for all p (inputs act as a field): smooth, L-independent
  * relevance: percolation of G \\ I (bond p, sites removed with prob. h) -> sharp transition at
    p_c(h) > 1/2 in the 2D class:
      per-input complexity  C_N/k -> P'^2-like order parameter, vanishing below p_c(h)
      per-node complexity   C_N = sum_i r_i/(N-k) finite below p_c(h), diverging there like a
                            susceptibility; at p_c(h): C_N ~ L^{gamma/nu} = L^{43/24}
      FSS: crossing of (C_N/k) L^{5/24} at p_c(h), collapse in (p - p_c(h)) L^{3/4}

  python lattice_h.py --out data/lattice_h            (production run)
  python lattice_h.py --out data/lattice_h --scan     (locate p_c(h) quickly)
  python lattice_h.py --out data/lattice_h --analyze
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
from lattice2d import lattice_edges  # noqa: E402

HS = [0.05, 0.1]
SAMPLES = {128: 600, 256: 300, 512: 120, 1024: 40}


def one(task):
    L, h, p, seed = task
    rng = np.random.default_rng(seed)
    N = L * L
    k = int(round(h * N))
    src, dst = lattice_edges(L, p, rng)
    inputs = rng.choice(N, k, replace=False)
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
    big = int(np.argmax(size))
    fin = pc != big                                   # exclude the giant of G \ I
    r_fin = np.bincount(pw[fin], weights=size[pc[fin]], minlength=N)[inputs]
    nn = N - k
    psi = size[np.unique(pc)].sum() / nn
    return dict(L=L, h=h, p=p, psi=psi, Ck=r.sum() / (k * nn), Cnode=r.sum() / nn,
                Cfin=r_fin.sum() / nn, Pd=size.max() / N)


def scan():
    rng = np.random.default_rng(5)
    for h in HS:
        for p in np.arange(0.50, 0.66, 0.01):
            v = [one((512, h, p, int(rng.integers(2**62))))["Pd"] for _ in range(6)]
            print(f"h={h} p={p:.2f} P'={np.mean(v):.3f}")


def pgrid(h):
    pc = {0.05: 0.528, 0.1: 0.567}[h]        # set from --scan (refined by the analysis)
    return np.round(np.concatenate([pc + np.array([-0.06, -0.04, -0.03, -0.02, -0.01, -0.005, 0,
                                                   0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.1])]), 4)


def main(out):
    rng = np.random.default_rng(4321)
    T = [(L, h, p, int(rng.integers(2**62))) for L in sorted(SAMPLES, reverse=True) for h in HS
         for p in pgrid(h) for _ in range(SAMPLES[L])]
    with Pool(int(os.environ.get("GQ_CPUS", "4"))) as pool:
        rows = list(pool.imap_unordered(one, T, chunksize=1))
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
    print("done", len(rows), flush=True)


def analyze(out):
    d = pd.read_csv(out / "samples.csv")
    g = d.groupby(["h", "L", "p"])
    s = g.mean().reset_index()
    for c in ("psi", "Ck", "Cnode", "Cfin"):
        s[c + "_se"] = (g[c].std() / np.sqrt(g.size())).values
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for h, t in s.groupby("h"):
        print(f"\n=== h={h}")
        print(t.pivot_table(index="p", columns="L", values="psi").round(4).to_string())
        print("C/k:"); print(t.pivot_table(index="p", columns="L", values="Ck").round(5).to_string())
        print("C_fin (finite-cluster per-node complexity):"); print(t.pivot_table(index="p", columns="L", values="Cfin").round(3).to_string())
        # crossing of (C/k) L^{5/24}: p where curves for successive L cross
        piv = t.pivot_table(index="p", columns="L", values="Ck") * 0
        for L in piv.columns:
            piv[L] = t[t.L == L].set_index("p").Ck * L ** (5 / 24)
        Ls = sorted(piv.columns)
        for a, b in zip(Ls[:-1], Ls[1:]):
            diff = (piv[b] - piv[a]).dropna()
            sgn = np.sign(diff.values)
            idx = np.where(np.diff(sgn) != 0)[0]
            if len(idx):
                i = idx[0]
                p0, p1 = diff.index[i], diff.index[i + 1]
                y0, y1 = diff.values[i], diff.values[i + 1]
                print(f"crossing L={a},{b}: p_c(h) ~ {p0 - y0 * (p1 - p0) / (y1 - y0):.4f}")
        # finite-cluster per-node complexity: susceptibility-like peak
        pk = t.loc[t.groupby("L").Cfin.idxmax(), ["L", "p", "Cfin"]]
        print("C_fin peaks:\n", pk.to_string(index=False))
        sl = np.polyfit(np.log(pk.L), np.log(pk.Cfin), 1)[0]
        print(f"C_fin peak height ~ L^{sl:.3f} (2D gamma/nu = {43 / 24:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    if a.scan:
        scan()
    elif a.analyze:
        analyze(o)
    else:
        main(o)
