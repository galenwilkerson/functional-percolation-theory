"""Critical-window (stage W) analysis: classic symmetry-breaking / FSS view at z_c = 1.

  python analyze_window.py --run data/e1w --tag e1w

Order parameters (per realization, averaged per cell):
  Psi   fraction of computing (non-constant) nodes      ER/OR: S[1-(1-S)^k] ~ 4k(z-1)^2  (beta=2)
  Lam/k mean relevant inputs per input                   ER/OR: S^2                       (beta=2)
  A     input anisotropy Var_i m_i                       ER/OR: S^3(1-S)(1-1/k)           (beta=3)
Scaling variable lambda = (z - 1) N^{1/3}; ER window: S ~ N^{-1/3} f(lambda), so
  Psi N^{2/3}, (Lam/k) N^{2/3}, A N  should collapse onto functions of lambda (k fixed, kS << 1).
Large-lambda asymptotes: 4k lambda^2, 4 lambda^2, 8 lambda^3 (1-1/k).
Susceptibilities: chi_X = N Var(X) across realizations; peak position / height vs N.
Writes <run>/analysis/window_summary.csv and ../figures/<tag>_*.pdf.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import load, summarize  # noqa: E402

FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def window_frame(cs: pd.DataFrame) -> pd.DataFrame:
    cs = cs.copy()
    cs["lam_s"] = (cs.z - 1.0) * cs.N ** (1 / 3)
    cs["psi_sc"] = cs.psi * cs.N ** (2 / 3)
    cs["psi_sc_se"] = cs.psi_se * cs.N ** (2 / 3)
    cs["lamk_sc"] = cs.lam / cs.k * cs.N ** (2 / 3)
    cs["lamk_sc_se"] = cs.lam_se / cs.k * cs.N ** (2 / 3)
    cs["A_sc"] = cs.aniso * cs.N
    cs["A_sc_se"] = cs.aniso_se * cs.N
    cs["S_sc"] = cs.S * cs.N ** (1 / 3)
    cs["S_sc_se"] = cs.S_se * cs.N ** (1 / 3)
    for c in ("psi", "S", "aniso", "dtc"):
        cs[f"chi_{c}"] = cs[f"{c}_var"] * cs.N
    return cs


def fig_collapse(cs, tag, phi="0.1"):
    ks = sorted(cs.k.unique())
    show = [k for k in (2, 5, 8) if k in ks] or ks[:3]
    rows = [("S_sc", r"$S\,N^{1/3}$", lambda l, k: 2 * l),
            ("psi_sc", r"$\Psi\,N^{2/3}$", lambda l, k: 4 * k * l ** 2),
            ("A_sc", r"$A\,N$  (anisotropy)", lambda l, k: 8 * l ** 3 * (1 - 1 / k))]
    fig, axes = plt.subplots(len(rows), len(show), figsize=(3.3 * len(show), 2.6 * len(rows)),
                             squeeze=False)
    Ns = sorted(cs.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for j, k in enumerate(show):
        for i, (o, lab, asym) in enumerate(rows):
            ax = axes[i, j]
            for N, c in zip(Ns, cols):
                d = cs[(cs.k == k) & (cs.phi == phi) & (cs.N == N) &
                       (cs.lam_s.abs() <= 4.5)].sort_values("lam_s")
                if d.empty:
                    continue
                ax.errorbar(d.lam_s, d[o], yerr=d[o + "_se"], fmt="o-", ms=2.5, lw=0.9,
                            color=c, capsize=1, label=f"N = {N:.0e}")
            ll = np.linspace(0.5, 4.5, 50)
            ax.plot(ll, [asym(l, k) for l in ll], "k--", lw=1, label="ER asymptote")
            ax.axvline(0, color="grey", lw=0.6)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"k = {k}, φ = {phi}")
            if i == len(rows) - 1:
                ax.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Finite-size-scaling collapse in the critical window "
                 "(ER window scaling; dashed = large-λ ER asymptote)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_collapse.pdf")
    plt.close(fig)


def fig_susceptibility(cs, tag, phi="0.1", k=5):
    obs = [("chi_S", r"$N\,\mathrm{Var}(S)$"), ("chi_psi", r"$N\,\mathrm{Var}(\Psi)$"),
           ("chi_aniso", r"$N\,\mathrm{Var}(A)$")]
    fig, axes = plt.subplots(1, len(obs), figsize=(10, 2.8))
    Ns = sorted(cs.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for ax, (o, lab) in zip(axes, obs):
        for N, c in zip(Ns, cols):
            d = cs[(cs.k == k) & (cs.phi == phi) & (cs.N == N) &
                   (cs.lam_s.abs() <= 4.5)].sort_values("lam_s")
            if not d.empty:
                ax.plot(d.lam_s, d[o], "o-", ms=2.5, lw=0.9, color=c, label=f"N = {N:.0e}")
        ax.set_yscale("log")
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
        ax.set_title(lab)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Susceptibilities across the window (k = {k}, φ = {phi})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_susceptibility.pdf")
    plt.close(fig)


def fig_onsets(cs, tag, k=5):
    """Coincidence of onsets: phi=0.25 has z_lo = 1.114 != 1 (structural vs cascade onset)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    Ns = sorted(cs.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for ax, phi in zip(axes, ["0.1", "0.25"]):
        for N, c in zip(Ns, cols):
            d = cs[(cs.k == k) & (cs.phi == phi) & (cs.N == N) & (cs.z <= 1.6)].sort_values("z")
            if not d.empty:
                ax.plot(d.z, d.psi, "o-", ms=2.5, lw=0.9, color=c, label=f"Ψ, N = {N:.0e}")
                ax.plot(d.z, d.S, ":", lw=0.9, color=c)
        ax.axvline(1.0, color="grey", lw=0.6)
        if phi == "0.25":
            ax.axvline(1.114142, color="cyan", lw=1, label="Watts z_lo")
        ax.set_title(f"φ = {phi}, k = {k}  (dotted: S)")
        ax.set_xlabel("z")
    axes[0].set_ylabel("Ψ")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_onsets.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    (run / "analysis").mkdir(exist_ok=True)
    df = load(run)
    print(f"{len(df)} realizations, {df.cell_id.nunique()} cells; "
          f"nonmono={int(df.n_nonmono.sum())} const1={int(df.n_const1.sum())}")
    cs = window_frame(summarize(df))
    cs["phi"] = cs.phi.map(lambda p: p if p == "U01" else f"{float(p):g}")
    cs.to_csv(run / "analysis" / "window_summary.csv", index=False)
    FIG.mkdir(exist_ok=True)
    fig_collapse(cs, a.tag)
    fig_susceptibility(cs, a.tag)
    fig_onsets(cs, a.tag)
    # peak positions/heights of susceptibilities (phi=0.1)
    out = []
    for (N, k), g in cs[(cs.phi == "0.1") & (cs.lam_s.abs() <= 4.5)].groupby(["N", "k"]):
        r = dict(N=N, k=k)
        for c in ("chi_S", "chi_psi", "chi_aniso"):
            i = g[c].idxmax()
            r[c + "_peak"], r[c + "_lam"] = g.loc[i, c], g.loc[i, "lam_s"]
        out.append(r)
    pk = pd.DataFrame(out)
    pk.to_csv(run / "analysis" / "susceptibility_peaks.csv", index=False)
    print(pk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
