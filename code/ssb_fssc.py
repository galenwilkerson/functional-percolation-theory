"""SC-c: cluster-exact finite-size scaling of the Sym_k breaking in the simple-contagion limit (RESEARCH_PROPOSAL_v2).

In simple contagion relevance is cluster membership (Lemma 1): input i is relevant to the nodes of its component,
m_i = x_{c(i)} (component fraction). Per graph G(N, M = zN/2), z = 1 + lambda N^{-1/3}, the average over input
placements (k inputs, independently placed: O(k^2/N) corrections) is computed from the component sizes:
  <A>   = (1 - 1/k) (sum x^3 - (sum x^2)^2)          exact
  chi   = N (sum x^2 - sum x^3)                       exact (one extra edge from input 0 to a random node)
  <|M0|> by Monte Carlo over R_PLACE placements per graph (no cascades, no placement noise across graphs)

  python ssb_fssc.py run --out data/ssb/fssc.parquet [--smoke]
  python ssb_fssc.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ssb import adjacency, edge_sequence  # noqa: E402

K = 3
R_PLACE = 20000
LAMBDAS = np.round(np.arange(-3.0, 4.01, 0.5), 2)
GRAPHS = {1000: 4000, 10000: 2000, 100000: 600, 1000000: 150, 10000000: 30}


def one(task):
    N, lam, seed = task
    rng = np.random.default_rng(seed)
    z = 1 + lam * N ** (-1 / 3)
    src, dst = edge_sequence(N, int(round(z * N / 2)), rng)
    A = adjacency(N, src, dst)
    _, lab = connected_components(A, directed=False)
    x = np.bincount(lab) / N
    s2, s3 = float((x ** 2).sum()), float((x ** 3).sum())
    cum = np.cumsum(x)
    c = np.minimum(np.searchsorted(cum, rng.random((R_PLACE, K)) * cum[-1]), len(x) - 1)
    m = x[c]
    M0 = m[:, 0] - m.mean(axis=1)
    return dict(N=N, lam=float(lam), z=z, seed=seed, absM0=float(np.abs(M0).mean()),
                A=(1 - 1 / K) * (s3 - s2 ** 2), chi=N * (s2 - s3), xmax=float(x.max()), s2=s2, s3=s3)


def run(out, smoke):
    graphs = {1000: 20, 10000: 5} if smoke else GRAPHS
    lams = LAMBDAS[::4] if smoke else LAMBDAS
    tasks = [(N, lam, 10 ** 13 + 1_000_003 * i + r) for i, (N, lam) in enumerate((N, l) for N in graphs for l in lams)
             for r in range(graphs[N])]
    tasks.sort(key=lambda t: -t[0])
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    res = []
    with Pool(ncpu, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_parquet(out)
    print(f"wrote {len(res)} rows to {out}")


def analyze():
    import ssb_fss_analyze as sa
    data = Path(os.environ.get("SSB_DATA", HERE.parent / "data" / "ssb"))
    out = Path(os.environ.get("SSB_OUT", HERE.parent / "summaries" / "ssb"))
    pfx = os.environ.get("SSB_PREFIX", "")
    df = pd.read_parquet(data / f"{pfx}fssc.parquet")
    cm = df.groupby(["N", "lam"])[["absM0", "A", "chi"]].mean().reset_index()
    cm["phi"] = 0.0
    cm.to_csv(out / f"{pfx}scc_cells.csv", index=False)
    al = pd.read_csv(sa.ALDOUS)
    al = al[al.dt == al.dt.min()].sort_values("lam")
    rows = []
    Ns = sorted(cm.N.unique())
    for obs, (a0, th, _) in sa.OBS.items():
        for i in range(len(Ns) - 2):
            win = Ns[i:i + 3]
            a, b, q = sa.fit(cm[cm.N.isin(win)], obs, a0)
            rows.append(dict(obs=obs, N_window=f"{win[0]}-{win[-1]}", a_pred=a0, a_fit=a, b_fit=b, quality=q))
        for N in Ns:
            c = cm[cm.N == N]
            pred = np.interp(c.lam, al.lam, th(al.G2.values, al.G3.values))
            r = (c[obs] * N ** a0 / pred).values
            rows.append(dict(obs=obs, N_window=str(N), ratio_to_aldous_median=float(np.median(r)),
                             ratio_lam0=float(r[np.isclose(c.lam, 0)][0]) if np.isclose(c.lam, 0).any() else np.nan))
    fits = pd.DataFrame(rows)
    fits.to_csv(out / f"{pfx}scc_fits.csv", index=False)
    print(fits.round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for col, (obs, (a0, th, lab)) in enumerate(sa.OBS.items()):
        ax = axs[col]
        for N, c in cm.groupby("N"):
            c = c.sort_values("lam")
            ax.plot(c.lam, c[obs] * N ** a0, "o-", ms=2, lw=0.8, label=f"N=$10^{{{int(np.log10(N))}}}$")
        ax.plot(al.lam, th(al.G2.values, al.G3.values), "k:", lw=1.2, label="Aldous (no fit)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=5.5)
    fig.suptitle("SC-c: cluster-exact placement averages, simple contagion, $k=3$; approach to the Aldous limit",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / f"{pfx}fig_scc_collapse.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--out", default="data/ssb/fssc.parquet")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.out, a.smoke) if a.mode == "run" else analyze()
