"""Rank-ordering of function frequency, colored by DTC (outline §5.1, §5.5; Sci. Rep. 2022 Fig. 3).

  python rank_freq.py --run data/e1c --tag e1c

Uses the per-cell ensemble counts of distinct non-constant functions (<run>/batches/*_ens.npz).
OR-regime prediction (fixed k, N -> inf): a non-input node computes OR over input subset A with
probability S^{|A|+1} (1-S)^{k-|A|}  ->  a staircase of plateaus, one per DTC level j = |A|,
each holding C(k, j) equally likely functions; the expected count of each function in a cell is
M (N-k) S^{j+1} (1-S)^{k-j}.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observables import DTCCache, unpack_words  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_counts(run: Path) -> pd.DataFrame:
    """Rows: (N, k, phi, z, words-bytes, count, words) aggregated over batches."""
    agg = {}
    for f in glob.glob(str(run / "batches" / "*_ens.npz")):
        e = np.load(f)
        N, k = int(e["N"]), int(e["k"])
        for z, phi, w, c in zip(e["z"], e["phi"], e["words"], e["count"]):
            key = (N, k, str(phi), float(z), w.tobytes())
            if key in agg:
                agg[key][0] += int(c)
            else:
                agg[key] = [int(c), w.copy()]
    rows = [dict(N=a[0], k=a[1], phi=a[2], z=a[3], count=v[0], words=v[1])
            for a, v in agg.items()]
    df = pd.DataFrame(rows)
    df["phi"] = df.phi.map(lambda p: p if p == "U01" else f"{float(p):g}")
    return df


def annotate_dtc(df: pd.DataFrame) -> pd.DataFrame:
    caches: dict[int, DTCCache] = {}
    dtc = np.empty(len(df), dtype=int)
    rel = np.empty(len(df), dtype=int)
    for k, idx in df.groupby("k").groups.items():
        idx = np.asarray(idx)
        W = np.stack(df.loc[idx, "words"].to_numpy()).astype(np.int64)
        dtc[idx] = caches.setdefault(k, DTCCache(k)).lookup(W)
        T = unpack_words(W, 1 << k).astype(bool)
        p = np.arange(1 << k)
        rel[idx] = sum((T != T[:, p ^ (1 << i)]).any(1) for i in range(k))
    df["dtc"], df["rel"] = dtc, rel
    return df


def panel(ax, d, k, N, M, z, cmap):
    d = d.sort_values("count", ascending=False).reset_index(drop=True)
    rank = np.arange(1, len(d) + 1)
    sc = ax.scatter(rank, d["count"], c=d["dtc"], cmap=cmap, vmin=0, vmax=k, s=10,
                    edgecolors="none")
    S = giant_fraction(z)
    if S > 0:  # OR staircase: C(k,j) functions at level j, ordered by expected count
        levels = sorted(((M * (N - k) * S ** (j + 1) * (1 - S) ** (k - j), j)
                         for j in range(1, k + 1)), reverse=True)
        r0 = 1
        for cnt, j in levels:
            nj = math.comb(k, j)
            ax.plot([r0, r0 + nj - 1 + 0.6], [cnt, cnt], "k-", lw=1.4, alpha=0.6)
            r0 += nj
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"z = {z:g}  (S = {S:.2f})", fontsize=8)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    cells = pd.read_parquet(glob.glob(str(run / "cells_*.parquet"))[0])
    Mof = {(r.N, r.k, (r.phi if r.phi == "U01" else f"{float(r.phi):g}"), round(r.z, 6)): r.M
           for r in cells.itertuples()}
    df = annotate_dtc(load_counts(run))
    df["z"] = df.z.round(6)
    df.drop(columns="words").to_csv(run / "analysis" / "function_counts.csv", index=False)
    N = int(df.N.max())
    zs = [1.1, 1.5, 2.0, 4.0]
    rowspec = [(5, "0.1"), (8, "0.1"), (5, "0.2"), (8, "0.25")]
    fig, axes = plt.subplots(len(rowspec), len(zs), figsize=(11, 2.5 * len(rowspec)),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, (k, phi) in enumerate(rowspec):
        for j, z in enumerate(zs):
            ax = axes[i, j]
            d = df[(df.N == N) & (df.k == k) & (df.phi == phi) & (df.z == z)]
            if d.empty:
                ax.set_visible(False)
                continue
            sc = panel(ax, d, k, N, Mof[(N, k, phi, z)], z, cmap)
            if j == 0:
                ax.set_ylabel(f"k={k}, φ={phi}\ncount (ensemble)")
            if i == len(rowspec) - 1:
                ax.set_xlabel("rank")
    fig.colorbar(sc, ax=axes, label="decision-tree complexity", shrink=0.6)
    fig.suptitle(f"Rank-ordering of function frequency, colored by DTC (N = {N}); "
                 "black: OR-regime staircase  C(k,j) functions at M(N−k)S^(j+1)(1−S)^(k−j)",
                 fontsize=9)
    fig.savefig(FIG / f"{a.tag}_rank_freq.pdf", bbox_inches="tight")
    plt.close(fig)
    # mean count per function vs DTC (Sci. Rep. Fig. 3 style), N max, phi = 0.1
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, k in zip(axes, (5, 8)):
        for z, col in zip([1.1, 1.5, 2.0, 4.0], ["C0", "C1", "C2", "C3"]):
            d = df[(df.N == N) & (df.k == k) & (df.phi == "0.1") & (df.z == z)]
            if d.empty:
                continue
            tot = d["count"].sum()
            g = d.groupby("dtc")["count"].agg(["mean"]) / tot
            ax.plot(g.index, g["mean"], "o", color=col, label=f"z = {z:g}")
            S = giant_fraction(z)
            js = np.arange(1, k + 1)
            th = S ** (js + 1) * (1 - S) ** (k - js)
            norm = sum(math.comb(k, int(j)) * t for j, t in zip(js, th))
            ax.plot(js, th / norm, "-", color=col, lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("decision-tree complexity j")
        ax.set_title(f"k = {k}, φ = 0.1, N = {N}")
    axes[0].set_ylabel("mean frequency per function")
    axes[0].legend(fontsize=7)
    fig.suptitle("Frequency of a single function vs its complexity (points: simulation; "
                 "lines: S^(j+1)(1−S)^(k−j), normalized)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"{a.tag}_freq_vs_dtc.pdf")
    plt.close(fig)
    print("written", FIG / f"{a.tag}_rank_freq.pdf", FIG / f"{a.tag}_freq_vs_dtc.pdf")


if __name__ == "__main__":
    main()
