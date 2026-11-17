"""Analysis of an E1 run (stage C or F): cell summaries, E5 ER-theory overlays, figures.

  python analyze.py --run data/e1c

Writes <run>/analysis/{cells_summary.csv, e5_checks.csv, fig_*.pdf}.
Per-realization rows are aggregated per cell (N, k, phi, z): means with standard errors,
graph-to-graph variances (susceptibility candidates), and the Sym_k anisotropy.
Conditional quantities (cdtc, H_nc, simpson_nc, rho_or) are averaged over realizations
with at least one responsive node only.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

SCALARS = ["S", "S2", "chi_fin", "k_in_gcc", "n_input_comps", "S_vul", "frac_vul", "psi",
           "psi_gcc", "lam", "dtc", "act", "D_all", "H", "simpson", "D_nc", "D_nc_gcc",
           "D_nc_fin", "D_rel", "f1_nc", "f2_nc", "aniso", "aniso_rel", "blocked_frac"]
CONDITIONAL = ["cdtc", "H_nc", "simpson_nc", "rho_or"]


def load(run: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(run / "batches" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dup = df.duplicated(["cell_id", "rep"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate (cell_id, rep) rows: mixed plans in {run}")
    m = df["m"].map(np.asarray)                            # ragged over k
    df["aniso"] = m.map(np.var)                            # Var_i m_i (population)
    mbar = m.map(np.mean).to_numpy()
    df["aniso_rel"] = np.where(mbar > 0, df["aniso"] / np.maximum(mbar, 1e-300) ** 2, np.nan)
    df["blocked_frac"] = np.where(df["n_gcc"] > 0, df["n_gcc_blocked"] / df["n_gcc"].clip(1), 0.0)
    df["dtc_lt_rel_frac"] = np.where(df["n_resp"] > 0, df["n_dtc_lt_rel"] / df["n_resp"].clip(1), np.nan)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (N, k, phi, z), g in df.groupby(["N", "k", "phi", "z"]):
        r = dict(N=N, k=k, phi=phi, z=z, M=len(g))
        for c in SCALARS + ["dtc_lt_rel_frac"]:
            v = g[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            r[c] = v.mean() if len(v) else np.nan
            r[c + "_se"] = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan
            r[c + "_var"] = v.var(ddof=1) if len(v) > 1 else np.nan
        resp = g["n_resp"].to_numpy() > 0
        for c in CONDITIONAL:
            v = g[c].to_numpy(dtype=float)[resp]
            v = v[~np.isnan(v)]
            r[c] = v.mean() if len(v) else np.nan
            r[c + "_se"] = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan
        # DTC distribution over all non-input nodes in the cell
        h = np.stack(g["dtc_hist"].to_numpy()).sum(0)
        for j, x in enumerate(h / h.sum()):
            r[f"pdtc_{j}"] = x
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E5 ER predictions (OR regime)
def er_predictions(z: float, k: int, N: int) -> dict:
    S = giant_fraction(z)
    out = dict(S_th=S, psi_th=S * (1 - (1 - S) ** k), lam_th=k * S * S,
               aniso_th=S ** 3 * (1 - S) * (1 - 1 / k) if k > 1 else 0.0)
    for j in range(k + 1):
        out[f"pdtc_{j}_th"] = math.comb(k, j) * S ** (j + 1) * (1 - S) ** (k - j) if j else np.nan
    if z < 1:  # each input's finite cluster has mean size 1/(1-z) for a random node
        out["psi_sub_th"] = k * (1 / (1 - z) - 1) / (N - k)
    else:
        out["psi_sub_th"] = np.nan
    return out


def add_theory(cs: pd.DataFrame) -> pd.DataFrame:
    th = [er_predictions(r.z, int(r.k), int(r.N)) for r in cs.itertuples()]
    return pd.concat([cs.reset_index(drop=True), pd.DataFrame(th)], axis=1)


# ---------------------------------------------------------------- figures
PHI_ORDER = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.4", "U01"]


def panel_grid(cs, out: Path, obs: list, title: str, theory: dict | None = None,
               logx=True, fname="fig.pdf"):
    Ns = sorted(cs.N.unique())
    ks = sorted(cs.k.unique())
    fig, axes = plt.subplots(len(obs), len(ks), figsize=(4.2 * len(ks), 2.8 * len(obs)),
                             squeeze=False, sharex=True)
    cmap = plt.get_cmap("viridis")
    phis = [p for p in PHI_ORDER if p in set(cs.phi)]
    for j, k in enumerate(ks):
        for i, o in enumerate(obs):
            ax = axes[i, j]
            for pi, phi in enumerate(phis):
                col = "k" if phi == "U01" else cmap(pi / max(1, len(phis) - 2))
                for N, ls in zip(Ns, ["--", "-", ":"]):
                    d = cs[(cs.k == k) & (cs.phi == phi) & (cs.N == N)].sort_values("z")
                    if d.empty or o not in d:
                        continue
                    ax.plot(d.z, d[o], ls=ls, color=col, lw=1.2,
                            label=f"φ={phi}" if (N == Ns[-1] and i == 0 and j == 0) else None)
                    if o + "_se" in d:
                        ax.fill_between(d.z, d[o] - d[o + "_se"], d[o] + d[o + "_se"],
                                        color=col, alpha=0.15, lw=0)
            if theory and o in theory:
                zz = np.geomspace(0.25, 40, 400) if logx else np.linspace(0, 40, 400)
                ax.plot(zz, [theory[o](z, k) for z in zz], color="red", lw=1, alpha=0.8,
                        label="ER/OR theory" if (i == 0 and j == 0) else None)
            ax.axvline(1.0, color="grey", lw=0.6, alpha=0.6)
            if logx:
                ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(o)
            if i == 0:
                ax.set_title(f"k = {k}")
            if i == len(obs) - 1:
                ax.set_xlabel("mean degree z")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{title}   (line style: N = {', '.join(map(str, Ns))} as --, -, :)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    df = load(run)
    print(f"{len(df)} realizations, {df.cell_id.nunique()} cells; "
          f"nonmono={int(df.n_nonmono.sum())} const1={int(df.n_const1.sum())}")
    cs = add_theory(summarize(df))
    cs.to_csv(out / "cells_summary.csv", index=False)

    S = lambda z: giant_fraction(z)  # noqa: E731
    theory = {
        "psi": lambda z, k: S(z) * (1 - (1 - S(z)) ** k),
        "lam": lambda z, k: k * S(z) ** 2,
        "aniso": lambda z, k: S(z) ** 3 * (1 - S(z)) * (1 - 1 / k) if k > 1 else 0.0,
        "S": lambda z, k: S(z),
    }
    panel_grid(cs, out, ["S", "psi", "lam", "aniso"], "Structure, activity and Sym_k anisotropy",
               theory, fname="fig_order_params.pdf")
    panel_grid(cs, out, ["cdtc", "rho_or", "dtc_lt_rel_frac", "blocked_frac"],
               "Beyond the OR regime", fname="fig_beyond_or.pdf")
    panel_grid(cs, out, ["D_nc", "H_nc", "D_nc_gcc", "D_nc_fin"], "Within-graph diversity",
               fname="fig_diversity.pdf")
    for c in ("psi", "dtc"):
        cs[c + "_chi"] = cs[c + "_var"] * cs["N"]
    panel_grid(cs, out, ["psi_chi", "dtc_chi", "S_var"], "Graph-to-graph fluctuations (N·Var)",
               fname="fig_fluct.pdf")

    # E5 quantitative checks in the OR regime (phi <= 0.1, away from z = 1)
    sel = cs[cs.phi.isin(["0.05", "0.1"]) & ((cs.z <= 0.75) | (cs.z >= 1.5))].copy()
    sel["psi_err"] = sel.psi - sel.psi_th
    sel["lam_err"] = sel.lam - sel.lam_th
    sel["aniso_err"] = sel.aniso - sel.aniso_th
    sel["psi_sub_ratio"] = sel.psi / sel.psi_sub_th
    cols = ["N", "k", "phi", "z", "M", "S", "S_th", "psi", "psi_th", "psi_se", "lam", "lam_th",
            "aniso", "aniso_th", "aniso_se", "psi_sub_th", "psi_sub_ratio", "rho_or",
            "blocked_frac", "D_nc"]
    sel[cols].to_csv(out / "e5_checks.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(sel[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
