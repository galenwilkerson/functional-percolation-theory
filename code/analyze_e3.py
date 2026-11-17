"""E3 analysis: variance decomposition into graph and input-placement parts (outline §7.1, §8.3).

  python analyze_e3.py --run data/e3 --tag e3

For each cell and observable X (Q placements per graph):
  Var_total = Var_graph(E[X | graph]) + E_graph[Var_place(X | graph)]
frac_place = E[Var_place] / Var_total.  Also: is the Sym_k anisotropy A = Var_i m_i explained by
placement through the ER/OR predictor  A_pred = S^2 (j/k)(1 - j/k)  (j = inputs in the GCC)?
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OBS = ["S", "psi", "lamk", "dtc", "aniso", "D_nc", "H_nc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(str(run / "batches" / "*.parquet"))],
                   ignore_index=True)
    df["lamk"] = df.lam / df.k
    df["aniso"] = df["m"].map(lambda v: np.var(np.asarray(v)))
    j = df.k_in_gcc / df.k
    df["aniso_pred"] = df.S ** 2 * j * (1 - j)
    print(f"{len(df)} rows, {df.groupby(['cell_id', 'graph']).ngroups} graphs")
    rows = []
    for (N, k, phi, z), g in df.groupby(["N", "k", "phi", "z"]):
        r = dict(N=N, k=k, phi=phi, z=z, lam_s=(z - 1) * N ** (1 / 3), G=g.graph.nunique())
        for o in OBS:
            x = g[o].astype(float)
            tot = x.var(ddof=1)
            within = g.groupby("graph")[o].var(ddof=1).mean()
            r[f"{o}_var"] = tot
            r[f"{o}_fplace"] = within / tot if tot > 0 else np.nan
        # anisotropy explained by placement (ER predictor), R^2 over realizations
        x, p = g.aniso.to_numpy(), g.aniso_pred.to_numpy()
        ss = ((x - x.mean()) ** 2).sum()
        r["aniso_R2_pred"] = 1 - ((x - p) ** 2).sum() / ss if ss > 0 else np.nan
        r["aniso_mean"], r["aniso_pred_mean"] = x.mean(), p.mean()
        rows.append(r)
    cs = pd.DataFrame(rows)
    (run / "analysis").mkdir(exist_ok=True)
    cs.to_csv(run / "analysis" / "e3_decomposition.csv", index=False)

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), squeeze=False)
    for i, phi in enumerate([0.1, 0.25]):
        for j_, k in enumerate([2, 5, 8]):
            ax = axes[i, j_]
            for o, col in zip(OBS, plt.get_cmap("tab10").colors):
                for N, ls in [(10000, "--"), (100000, "-")]:
                    d = cs[(cs.k == k) & (cs.phi == phi) & (cs.N == N)].sort_values("z")
                    ax.plot(d.z, d[f"{o}_fplace"], ls, color=col, marker="o", ms=2.5, lw=1,
                            label=o if (N == 100000 and i == 0 and j_ == 0) else None)
            ax.axvline(1, color="grey", lw=0.6)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"k = {k}, φ = {phi}", fontsize=8)
            if j_ == 0:
                ax.set_ylabel("fraction of variance\nfrom input placement")
            if i == 1:
                ax.set_xlabel("z")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("E3: graph-to-graph variance split into wiring vs input placement "
                 "(16 placements per graph; dashed N=1e4, solid N=1e5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"{a.tag}_variance_split.pdf")
    plt.close(fig)
    with pd.option_context("display.width", 220):
        cols = ["N", "k", "phi", "z"] + [f"{o}_fplace" for o in OBS] + ["aniso_R2_pred"]
        print(cs[(cs.k == 5)][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
