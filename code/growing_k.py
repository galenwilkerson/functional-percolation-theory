"""Growing k = k(N): outline §4.2/§6.2 (does C_N vanish for sublinear k?) and §8.5 (limiting
symmetry statement via finite-dimensional marginals of the relevance vector m).

OR regime, exact classes (components of G \\ I with their adjacent inputs).
  C_N   = sum_u |A_u| / (N-k) = sum_i r_i / (N-k),  r_i = # non-inputs for which i is relevant
  bound (z<1): E C_N <= k z / ((N-k)(1-z))   [each input reaches at most its component, mean size 1/(1-z)]
  m_i   = r_i / N.  Prediction (k = hN): m_i -> P * 1[i adjacent to the giant of G \\ I],
        P = (1-h)(1-e^{-zP}),   P(i adjacent) = q = 1 - e^{-zP}, independent across fixed sets of inputs.

  python growing_k.py --out data/growing_k
  python growing_k.py --out data/growing_k --analyze
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from er_checks import sample_gnp  # noqa: E402

ZS = [0.5, 0.8, 0.9, 1.5, 2.0]
GRAPHS = {10000: 200, 100000: 40, 1000000: 8}


def kvals(N):
    return {"a0.25": round(N ** 0.25), "a0.5": round(N ** 0.5), "a0.75": round(N ** 0.75),
            "h0.01": round(0.01 * N), "h0.05": round(0.05 * N)}


def one(task):
    N, z, kname, k, seed = task
    rng = np.random.default_rng(seed)
    src, dst = sample_gnp(N, z, rng)
    inputs = rng.choice(N, k, replace=False)
    inp = np.zeros(N, bool)
    inp[inputs] = True
    keep = ~inp[src] & ~inp[dst]
    A = coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    size = np.bincount(lab[~inp], minlength=N)          # non-input vertices per component
    e1 = inp[src] & ~inp[dst]
    e2 = inp[dst] & ~inp[src]
    comp = np.concatenate([lab[dst[e1]], lab[src[e2]]])
    who = np.concatenate([src[e1], dst[e2]])
    pair = np.unique(comp.astype(np.int64) * N + who)
    pc, pw = pair // N, pair % N
    r = np.bincount(pw, weights=size[pc], minlength=N)[inputs]   # reach of each input
    big = int(np.argmax(size))
    adj_big = np.zeros(N, bool)
    adj_big[pw[pc == big]] = True
    adj = adj_big[inputs]
    m = r / N
    # placement-conditioned sectors
    Pemp = size[big] / N
    on, off = m[adj], m[~adj]
    nresp = size[np.unique(pc)].sum()
    return dict(N=N, z=z, kname=kname, k=k, C=r.sum() / (N - k), psi=nresp / (N - k),
                Pemp=Pemp, q_emp=adj.mean(),
                m_on_mean=on.mean() if on.size else np.nan,
                m_on_relsd=(on.std() / on.mean()) if on.size > 1 else np.nan,
                m_off_mean=off.mean() if off.size else np.nan,
                m_off_max=off.max() if off.size else np.nan,
                pair_corr=np.corrcoef(adj[0::2][: k // 2], adj[1::2][: k // 2])[0, 1]
                if k >= 20 and 0 < adj.mean() < 1 else np.nan)


def main(out: Path):
    rng = np.random.default_rng(85)
    T = [(N, z, kn, k, int(rng.integers(2**62)))
         for N in sorted(GRAPHS, reverse=True) for z in ZS for kn, k in kvals(N).items()
         for _ in range(GRAPHS[N])]
    ncpu = int(os.environ.get("GQ_CPUS", "4"))
    with Pool(ncpu) as pool:
        rows = list(pool.imap_unordered(one, T, chunksize=2))
    pd.DataFrame(rows).to_csv(out / "graphs.csv", index=False)
    print("done", len(rows), flush=True)


def P_h(z, h):
    P = 0.9
    for _ in range(100000):
        Pn = (1 - h) * (1 - math.exp(-z * P))
        if abs(Pn - P) < 1e-14:
            break
        P = Pn
    return P if P > 1e-9 else 0.0


def analyze(out: Path):
    d = pd.read_csv(out / "graphs.csv")
    g = d.groupby(["N", "z", "kname", "k"])
    s = g.mean(numeric_only=True)
    s["C_se"] = g.C.std() / np.sqrt(g.C.count())
    s = s.reset_index()
    s["h"] = s.k / s.N
    s["C_bound"] = np.where(s.z < 1, s.k * s.z / ((s.N - s.k) * (1 - s.z)), np.nan)
    s["P_pred"] = [P_h(z, h) for z, h in zip(s.z, s.h)]
    s["q_pred"] = 1 - np.exp(-s.z * s.P_pred)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("§4.2/§6.2  C_N for z<1 vs bound k z/((N-k)(1-z))")
    t = s[s.z < 1]
    print(t[["z", "kname", "N", "k", "C", "C_se", "C_bound"]]
          .assign(ratio=t.C / t.C_bound).round(5).to_string(index=False))
    print("\n§8.5  sectors of the relevance vector (z>1)")
    t = s[s.z > 1]
    print(t[["z", "kname", "N", "k", "Pemp", "P_pred", "m_on_mean", "m_on_relsd", "m_off_mean",
             "m_off_max", "q_emp", "q_pred", "pair_corr"]].round(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    analyze(o) if a.analyze else main(o)
