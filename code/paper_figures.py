"""New figures for the paper (Symmetry_Breaking_Basic/paper/figures/).

  fig_fss.pdf        §6.5   Psi N^{2/3}/k vs lambda, with k(chi_N - 1)/N from the same graphs
  fig_fluct.pdf      §7     Var(c_G) above z_c vs k S^3(1-S); N chi_C in the window
  fig_diversity.pdf  §5.2   H_N vs z with [1-(1-S)^k] H2(S) (phi=0.1) and the cascade form (phi=0.25)
  fig_growing_k.pdf  §4.2/§6.2/§8.5  C_N below z_c vs prediction; sector values P, q
  fig_selection.pdf  §10.6  Lam/k and A vs z for four input-selection rules

  python paper_figures.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402
from watts import cascade_size, trigger_prob  # noqa: E402

import os  # noqa: E402
OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
D = Path(os.environ.get("FP_DATA", HERE.parent / "summaries"))  # summary tables; set FP_DATA to use full run outputs
# validated categorical palette (blue, orange, aqua, yellow) + distinct markers (secondary encoding)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK = "#0b0b0b"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "lines.linewidth": 1.2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5})
NS = [1000, 10000, 100000, 1000000]


def H2(s):
    s = np.clip(np.asarray(s, float), 1e-300, 1 - 1e-16)
    return -s * np.log(s) - (1 - s) * np.log(1 - s)


def nlab(N):
    return f"$N=10^{int(round(math.log10(N)))}$"


def panel(ax, t):
    ax.set_title(t, loc="left", fontsize=8.5, color=INK)


def fig_fss():
    w = pd.read_csv(D / "extra/O_psi_window.csv", dtype={"phi": str})
    w = w[(w.phi == "0.1") & (w.k == 5) & (w.lam_s.abs() <= 3.01)]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for i, N in enumerate(NS):
        g = w[w.N == N].sort_values("lam_s")
        ax[0].errorbar(g.lam_s, g.psi * N ** (2 / 3) / 5, yerr=g.psi_se * N ** (2 / 3) / 5,
                       fmt=MK[i], ms=3.5, color=COL[i], lw=0.8, label=nlab(N))
        gp = g[g.k * g.S < 0.3]                      # expansion valid for k x_max << 1
        ax[0].plot(gp.lam_s, gp.psi_pred * N ** (2 / 3) / 5, "-", color=COL[i], lw=0.9)
        ax[1].plot(g.lam_s, g.chiN_sc, MK[i], ms=3.5, color=COL[i], label=nlab(N))
        al = pd.read_csv(D / "aldous/aldous_vs_e1w.csv")
        al = al[(al.N == N) & (al.k == 5) & (al.lam.abs() <= 3.01)].sort_values("lam")
        ax[0].plot(al.lam, al.psi_pred * N ** (2 / 3) / 5, "--", color=COL[i], lw=0.8)
    L = np.linspace(0.5, 3, 50)
    ax[1].plot(L, 4 * L ** 2, "--", color="#52514e", lw=0.9, label=r"$4\lambda^2$")
    L = np.linspace(-3, -0.5, 50)
    ax[1].plot(L, 1 / abs(L), ":", color="#52514e", lw=0.9, label=r"$1/|\lambda|$")
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
    ax[0].set_ylabel(r"$\Psi_N N^{2/3}/k$")
    ax[1].set_ylabel(r"$g(\lambda)=\bar s_N/N^{1/3}$")
    panel(ax[0], "(a) points: measured; solid: same-graph; dashed: Aldous")
    panel(ax[1], "(b) structural mean cluster size")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fss.pdf")


def fig_fluct():
    c = pd.read_csv(D / "extra/C_fluct.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.k == 5)]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    zz = np.linspace(1.001, 3.2, 300)
    S = np.array([giant_fraction(z) for z in zz])
    ax.plot(zz, 5 * S ** 3 * (1 - S), "-", color="#52514e", lw=1.0, label=r"$kS^3(1-S)$")
    for i, N in enumerate(NS):
        g = c[c.N == N].sort_values("z")
        ax.plot(g.z, g.c_var, MK[i], ms=3.2, color=COL[i], label=nlab(N))
    ax.set_xlim(0.4, 3.2)
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\mathrm{Var}(c_G)$")
    panel(ax, "sample fluctuations ($k=5$)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fluct.pdf")


def fig_diversity():
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    zz = np.linspace(0.3, 3.2, 400)
    S = np.array([giant_fraction(z) for z in zz])
    for i, k in enumerate([2, 5, 8]):
        g = w[(w.phi == "0.1") & (w.k == k) & (w.N == 100000)].sort_values("z")
        ax[0].plot(g.z, g.H, MK[i], ms=3.2, color=COL[i], label=f"$k={k}$")
        ax[0].plot(zz, (1 - (1 - S) ** k) * H2(S), "-", color=COL[i], lw=0.9)
    pt = np.array([trigger_prob(z, 0.25) for z in zz])
    sc = np.array([cascade_size(z, 0.25)[0] for z in zz])
    g = w[(w.phi == "0.25") & (w.k == 5) & (w.N == 100000)].sort_values("z")
    ax[1].plot(g.z, g.H, MK[1], ms=3.2, color=COL[1], label=r"measured, $\varphi=0.25$")
    ax[1].plot(zz, (1 - (1 - pt) ** 5) * H2(sc), "-", color=COL[1], lw=0.9,
               label=r"$[1-(1-P_t)^k]H_2(S_c)$")
    ax[1].plot(zz, (1 - (1 - S) ** 5) * H2(S), "--", color="#52514e", lw=0.9,
               label=r"$[1-(1-S)^k]H_2(S)$")
    for a in ax:
        a.set_xlabel("$z$")
        a.set_ylabel(r"$H_N$ (constants included)")
        a.set_xlim(0.3, 3.2)
    panel(ax[0], r"(a) OR regime, $\varphi=0.1$, $N=10^5$")
    panel(ax[1], r"(b) finite threshold, $k=5$, $N=10^5$")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_diversity.pdf")


def fig_growing_k():
    t = pd.read_csv(D / "growing_k/C_subcritical.csv")
    s = pd.read_csv(D / "growing_k/summary.csv")
    s = s[s.z > 1]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    kinds = ["a0.25", "a0.5", "a0.75", "h0.01", "h0.05"]
    labs = [r"$k=N^{1/4}$", r"$k=N^{1/2}$", r"$k=N^{3/4}$", r"$k=0.01N$", r"$k=0.05N$"]
    cols = COL + ["#52514e"]
    mks = MK + ["v"]
    for i, (kn, lb) in enumerate(zip(kinds, labs)):
        g = t[t.kname == kn]
        ax[0].errorbar(g.pred, g.C, yerr=g.C_se, fmt=mks[i], ms=3.5, color=cols[i], lw=0.8, label=lb)
        g = s[s.kname == kn]
        if len(g):
            ax[1].plot(g.P_pred, g.m_on_mean, mks[i], ms=4, color=cols[i], label=lb)
            ax[1].plot(g.q_pred, g.q_emp, mks[i], ms=4, mfc="none", color=cols[i])
    x = np.geomspace(1e-5, 1, 10)
    ax[0].plot(x, x, "-", color="#52514e", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"predicted $\frac{k}{N-k}\frac{z'}{1-z'}$, $z'=z(1-h)$")
    ax[0].set_ylabel(r"measured $C_N$")
    x = np.linspace(0.4, 0.85, 10)
    ax[1].plot(x, x, "-", color="#52514e", lw=0.8)
    ax[1].set_xlabel(r"predicted $P$ (filled), $q$ (open)")
    ax[1].set_ylabel(r"measured $\mathbb{E}[m_i\mid\mathrm{on}]$, $q_{\rm emp}$")
    panel(ax[0], r"(a) complexity below $z_c$ ($z=0.5,0.8,0.9$)")
    panel(ax[1], r"(b) relevance sectors above $z_c$ ($z=1.5,2$)")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_growing_k.pdf")


def fig_selection():
    s = pd.read_csv(D / "input_selection/summary.csv")
    s = s[s.N == 1000000]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for i, ru in enumerate(["uniform", "leaf", "edge", "hub"]):
        g = s[s.rule == ru].sort_values("z")
        ax[0].plot(g.z, g.lamk, MK[i], ms=3.8, color=COL[i], label=ru)
        ax[0].plot(g.z, g.lamk_p, "-", color=COL[i], lw=0.8)
        ax[1].plot(g.z, g.A, MK[i], ms=3.8, color=COL[i], label=ru)
        ax[1].plot(g.z, g.A_p, "-", color=COL[i], lw=0.8)
    for a in ax:
        a.set_xlabel("$z$")
    ax[0].set_ylabel(r"$C_N/k$")
    ax[1].set_ylabel(r"anisotropy $A=\mathrm{Var}_i\,m_i$")
    panel(ax[0], r"(a) lines: $S\,S_{\rm in}$ ($k=5$, $N=10^6$)")
    panel(ax[1], r"(b) lines: $S^2S_{\rm in}(1-S_{\rm in})(1-1/k)$")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_selection.pdf")


def fig_order():
    """§4.3, §5.5: Psi, Lam/k, P[DTC=j] vs the OR benchmark; §4.2 subcritical law."""
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.N == 10000) & (c.z <= 6)]   # below the upper cascade edge
    zz = np.linspace(0.3, 6, 400)
    S = np.array([giant_fraction(z) for z in zz])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for i, k in enumerate([2, 5, 8]):
        g = c[c.k == k].sort_values("z")
        ax[0].errorbar(g.z, g.psi, yerr=g.psi_se, fmt=MK[i], ms=3, color=COL[i], lw=0.7, label=f"$k={k}$")
        ax[0].plot(zz, S * (1 - (1 - S) ** k), "-", color=COL[i], lw=0.9)
        ax[1].errorbar(g.z, g.lam / k, yerr=g.lam_se / k, fmt=MK[i], ms=3, color=COL[i], lw=0.7)
    ax[1].plot(zz, S ** 2, "-", color="#52514e", lw=0.9, label="$S^2$")
    a = pd.read_csv(D / "extra/A_subcritical_bound.csv", dtype={"phi": str})
    xcol, ycol = "ER_finite", "psi"
    a = a[(a[xcol] > 0) & (a[ycol] > 0) & a.phi.isin(["0.05", "0.1"])]
    ax[2].loglog(a[xcol], a[ycol], ".", ms=2, color=COL[0], alpha=0.5)
    x = np.geomspace(a[xcol].min(), a[xcol].max(), 10)
    ax[2].loglog(x, x, "-", color="#52514e", lw=0.8)
    for q in ax[:2]:
        q.set_xscale("log")
        q.set_xlabel("$z$")
        q.axvline(1, color="#9a9994", lw=0.6)
    ax[0].set_ylabel(r"$\Psi$")
    ax[1].set_ylabel(r"$\Lambda/k$")
    ax[2].set_xlabel(r"$\frac{k}{N-k}\frac{z}{1-z}$")
    ax[2].set_ylabel(r"$\Psi_N$ ($z<1$)")
    panel(ax[0], r"(a) lines: $S[1-(1-S)^k]$")
    panel(ax[1], r"(b) complexity per input")
    panel(ax[2], r"(c) subcritical, all cells")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_order.pdf")


def fig_symmetry():
    """§8.3-8.4: anisotropy, sector values of m_1, placement share of variance."""
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.N == 10000)]
    zz = np.linspace(0.3, 5, 400)
    S = np.array([giant_fraction(z) for z in zz])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for i, k in enumerate([2, 5, 8]):
        g = c[(c.k == k) & (c.z <= 5)].sort_values("z")
        ax[0].errorbar(g.z, g.aniso, yerr=g.aniso_se, fmt=MK[i], ms=3, color=COL[i], lw=0.7, label=f"$k={k}$")
        ax[0].plot(zz, S ** 3 * (1 - S) * (1 - 1 / k), "-", color=COL[i], lw=0.9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel(r"$A=\mathrm{Var}_i\,m_i$")
    f = pd.read_csv(D / "extra/F_input1.csv", dtype={"phi": str})
    f = f[(f.phi == "0.1") & (f.k == 5) & (f.N == 100000) & (f.z >= 0.5)].sort_values("z")
    ax[1].plot(f.z, f.m0_in, MK[0], ms=3, color=COL[0], label=r"$\mathbb{E}[m_1\mid 1\in\mathrm{GCC}]$")
    ax[1].plot(f.z, np.maximum(f.m0_out, 1e-7), MK[1], ms=3, color=COL[1],
               label=r"$\mathbb{E}[m_1\mid 1\notin\mathrm{GCC}]$")
    ax[1].plot(zz, S, "-", color="#52514e", lw=0.9, label="$S$")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_ylim(1e-7, 2)
    ax[1].set_xlabel("$z$")
    ax[1].set_ylabel("relevance of input 1")
    e = pd.read_csv(D / "e3/analysis/e3_decomposition.csv", dtype={"phi": str})
    e = e[(e.phi == "0.1") & (e.k == 5) & (e.N == 100000)].sort_values("z")
    for i, (col, lab) in enumerate([("aniso_fplace", "$A$"), ("dtc_fplace", r"$c_G$"),
                                    ("psi_fplace", r"$\Psi$"), ("S_fplace", "$S$")]):
        ax[2].plot(e.z, e[col], MK[i] + "-", ms=3, lw=0.8, color=COL[i], label=lab)
    ax[2].set_xlabel("$z$")
    ax[2].set_ylabel("placement share of variance")
    ax[2].set_ylim(-0.05, 1.05)
    panel(ax[0], r"(a) lines: $S^3(1-S)(1-1/k)$")
    panel(ax[1], r"(b) two sectors ($N=10^5$)")
    panel(ax[2], r"(c) wiring vs placement")
    ax[0].legend(fontsize=6.5)
    ax[1].legend(fontsize=6.5, loc="center right")
    ax[2].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_symmetry.pdf")


def fig_kS():
    """§6.4: Psi/S = 1-(1-S)^k = 1-exp(-x), x = -k ln(1-S)."""
    b = pd.read_csv(D / "extra/B_kS_crossover.csv", dtype={"phi": str})
    b = b[b.phi.isin(["0.05", "0.1"]) & (b.kS > 0)]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, src in enumerate(sorted(b.src.unique())):
        g = b[b.src == src]
        lab = {"E1-C": "coarse map", "E1-W": "critical window", "probe": "probe"}[src]
        ax.loglog(-g.k * np.log1p(-g.S), g.psi / g.S, MK[i], ms=2.5, color=COL[i], alpha=0.6, label=lab)
    x = np.geomspace(1e-3, 1e4, 200)
    ax.loglog(x, 1 - np.exp(-x), "-", color="#52514e", lw=1.0, label=r"$1-e^{-y}$")
    ax.set_xlabel(r"$y=-k\ln(1-S)$")
    ax.set_ylabel(r"$\Psi/S$")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kS.pdf")


if __name__ == "__main__" and not os.environ.get("PAPER"):
    fig_order()
    fig_symmetry()
    fig_kS()
    fig_fss()
    fig_fluct()
    fig_diversity()
    fig_growing_k()
    fig_selection()
    print("ok")


# ---------------------------------------------------------------- paper_v3 figures
def fig_field_v3():
    """Ghost-field equation of state (field run, phi = 0.1)."""
    f = pd.read_csv(D / "e2f/analysis/field_summary.csv")
    f = f[f.phi == 0.1]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    # (a) order of limits at N = 1e6
    g = f[f.N == 1000000]
    zz = np.linspace(0.5, 1.45, 300)
    S = np.array([giant_fraction(z) for z in zz])
    ax[0].plot(zz, S, "--", color=INK, lw=1.1, label="$S$", zorder=5)
    for i, (law, par, lab) in enumerate([("rho", 0.01, "$h=10^{-2}$"), ("rho", 0.001, "$h=10^{-3}$"),
                                         ("fixed", 5.0, "$k=5$"), ("fixed", 1.0, "$k=1$")]):
        t = g[(g.law == law) & (g.par == par)].sort_values("z")
        t = t[t.z <= 1.45]
        ax[0].plot(t.z, t["mean"], MK[i], ms=2.6, color=COL[i], label=lab)
        import finite_h as fh
        if law == "rho":
            th = [fh.theory(z, par)[0] for z in zz]
        else:
            th = S * (1 - (1 - S) ** par)
        ax[0].plot(zz, th, "-", color=COL[i], lw=0.7)
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel(r"$\Psi$")
    ax[0].legend(fontsize=5.5, loc="upper left")
    # (b) linear response below z_c
    for i, N in enumerate(NS):
        t = f[(f.N == N) & (f.z < 0.99) & (f.law == "rho")]
        for hv, fill in ((1e-3, COL[i]), (1e-2, "none")):
            u = t[np.isclose(t.h, hv)]
            ax[1].loglog(u.z / (1 - u.z), u["mean"] / u.h, MK[i], ms=2.6, color=COL[i], mfc=fill,
                         label=nlab(N) if hv == 1e-3 else None)
    x = np.geomspace(0.9, 120, 20)
    ax[1].loglog(x, x, "-", color=INK, lw=0.8)
    ax[1].set_xlabel(r"$z/(1-z)$")
    ax[1].set_ylabel(r"$\Psi/h$")
    ax[1].legend(fontsize=5.5)
    # (c) critical isotherm
    for i, N in enumerate(NS):
        t = f[(f.N == N) & (abs(f.z - 1) < 1e-9)]
        ax[2].loglog(t.h, t["mean"], MK[i], ms=2.6, color=COL[i])
    hh = np.geomspace(1e-6, 1e-1, 20)
    from scipy.optimize import brentq
    iso = [1 - np.exp(-brentq(lambda P: P - 1 + (1 - x) * np.exp(-P), 1e-12, 1)) for x in hh]
    ax[2].loglog(hh, iso, "-", color=INK, lw=0.8, label="Eq. (3)")
    ax[2].loglog(hh, np.sqrt(2 * hh), "--", color=INK, lw=0.7, label=r"$\sqrt{2h}$")
    ax[2].set_xlabel("$h=k/N$")
    ax[2].set_ylabel(r"$\Psi(z=1)$")
    ax[2].legend(fontsize=5.5)
    # (d) scaled-field collapse
    for j, c in enumerate([0.3, 1.0, 10.0]):
        for i, N in enumerate(NS):
            t = f[(f.N == N) & (f.law == "scaled") & (f.par == c) & (f.lam_s.abs() <= 4.01)].sort_values("lam_s")
            ax[3].semilogy(t.lam_s, t["mean"] * N ** (1 / 3), MK[i] + "-", ms=2.2, lw=0.5, color=COL[j],
                           label=(f"$c={c:g}$" if i == 3 else None))
    ax[3].set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
    ax[3].set_ylabel(r"$\Psi N^{1/3}$")
    ax[3].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a)", "(b)", "(c)", "(d)"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_field.pdf")


def fig_fss_v3():
    w = pd.read_csv(D / "extra/O_psi_window.csv", dtype={"phi": str})
    w = w[(w.phi == "0.1") & (w.k == 5) & (w.lam_s.abs() <= 3.01)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for i, N in enumerate(NS):
        g = w[w.N == N].sort_values("lam_s")
        ax[0].plot(g.lam_s, g.S * N ** (1 / 3), MK[i], ms=3, color=COL[i], label=nlab(N))
        ax[1].errorbar(g.lam_s, g.psi * N ** (2 / 3) / 5, yerr=g.psi_se * N ** (2 / 3) / 5,
                       fmt=MK[i], ms=3, color=COL[i], lw=0.7)
        gp = g[g.k * g.S < 0.3]
        ax[1].plot(gp.lam_s, gp.psi_pred * N ** (2 / 3) / 5, "-", color=COL[i], lw=0.8)
        ax[2].plot(g.lam_s, g.chiN_sc, MK[i], ms=3, color=COL[i])
    L = np.linspace(0.5, 3, 50)
    ax[2].plot(L, 4 * L ** 2, "--", color="#52514e", lw=0.8, label=r"$4\lambda^2$")
    L = np.linspace(-3, -0.5, 50)
    ax[2].plot(L, 1 / abs(L), ":", color="#52514e", lw=0.8, label=r"$1/|\lambda|$")
    ax[0].set_ylabel(r"$S_N N^{1/3}$")
    ax[1].set_ylabel(r"$\Psi_N N^{2/3}/k$")
    ax[2].set_ylabel(r"$g(\lambda)=\bar s_N/N^{1/3}$")
    for a in ax:
        a.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=6)
    ax[2].legend(fontsize=6)
    for a, t in zip(ax, ["(a) order parameter", "(b) functional response", "(c) mean cluster size"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_fss.pdf")


def fig_composites_v3():
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.N == 10000) & (c.z <= 6)]
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[w.phi == "0.1"]
    cf = pd.read_csv(D / "extra/C_fluct.csv", dtype={"phi": str})
    cf = cf[(cf.phi == "0.1") & (cf.k == 5)]
    zz = np.linspace(0.3, 6, 400)
    S = np.array([giant_fraction(z) for z in zz])
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for i, k in enumerate([2, 5, 8]):
        g = c[c.k == k].sort_values("z")
        ax[0].plot(g.z, g.lam / k, MK[i], ms=2.6, color=COL[i], label=f"$k={k}$")
        g = w[(w.k == k) & (w.N == 100000)].sort_values("z")
        g = g[g.z <= 3.2]
        ax[1].plot(g.z, g.H, MK[i], ms=2.6, color=COL[i])
        ax[1].plot(zz, (1 - (1 - S) ** k) * H2(S), "-", color=COL[i], lw=0.7)
        two = g.simpson - ((1 - g.psi) ** 2 + g.psi_var)
        ax[2].plot(g.z, two, MK[i], ms=2.6, color=COL[i])
        ax[2].plot(zz, S ** 2 * (1 - (1 - S) ** k), "-", color=COL[i], lw=0.7)
    ax[0].plot(zz, S ** 2, "-", color=INK, lw=0.8, label="$S^2$")
    ax[0].set_xscale("log")
    for i, N in enumerate(NS):
        g = cf[cf.N == N].sort_values("z")
        ax[3].plot(g.z, g.c_var, MK[i], ms=2.6, color=COL[i], label=nlab(N))
    zz2 = np.linspace(1.001, 3.2, 300)
    S2 = np.array([giant_fraction(z) for z in zz2])
    ax[3].plot(zz2, 5 * S2 ** 3 * (1 - S2), "-", color=INK, lw=0.8)
    for a in ax[1:]:
        a.set_xlim(0.4, 3.2)
    for a in ax:
        a.set_xlabel("$z$")
    ax[0].set_ylabel(r"$C_N/k$")
    ax[1].set_ylabel(r"$H_N$")
    ax[2].set_ylabel(r"$G_2$")
    ax[3].set_ylabel(r"$\mathrm{Var}(c_G)$")
    ax[0].legend(fontsize=5.5)
    ax[3].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a) complexity/input", "(b) entropy", "(c) two-point", "(d) fluctuations"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "fig_composites.pdf")


if __name__ == "__main__" and os.environ.get("V3"):
    fig_field_v3()
    fig_fss_v3()
    fig_composites_v3()
    print("v3 ok")


def fig_lattice_v4():
    """2D square lattice, k = 5: mapping C/k = P^2 and 2D exponents at p_c."""
    s = pd.read_csv(D / "lattice2d/summary.csv")
    Ls = sorted(s.L.unique())
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    cols = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(Ls)))
    for c, L in zip(cols, Ls):
        t = s[(s.L == L) & (s.p >= 0.5)]                       # supercritical: infinite cluster carries the law
        ax[0].plot(t.S ** 2, t.Ck, "o", ms=2.8, color=c, label=f"$L={L}$")
        t = s[s.L == L]
        t = t.sort_values("p")
        ax[2].semilogy((t.p - 0.5) * L ** 0.75, t.Ck * L ** (5 / 24), "o-", ms=2.6, lw=0.5, color=c)
    x = np.linspace(0, 1, 10)
    ax[0].plot(x, x, "-", color=INK, lw=0.8)
    c0 = s[s.p == 0.5].sort_values("L")
    out = {}
    for name, col, ex, mk, cc in (("Ck", "Ck", -5 / 24, "o", COL[0]), ("S", "S", -5 / 48, "s", COL[1])):
        lx, ly = np.log(c0.L.values), np.log(c0[col].values)
        w = 1 / (c0[col + "_se"].values / c0[col].values) ** 2
        A = np.vstack([lx, np.ones_like(lx)]).T
        W = np.diag(w)
        cov = np.linalg.inv(A.T @ W @ A)
        beta = cov @ A.T @ W @ ly
        out[name] = (beta[0], np.sqrt(cov[0, 0]))
        ax[1].loglog(c0.L, c0[col], mk, ms=4, color=cc, label=(r"$C_N/k$" if name == "Ck" else r"$P_\infty$"))
        ax[1].loglog(c0.L, np.exp(beta[1]) * c0.L ** ex, "-", color=cc, lw=0.8)
    ax[0].set_xlabel(r"$P_\infty^2$")
    ax[0].set_ylabel(r"$C_N/k$")
    ax[1].set_xlabel("$L$")
    ax[1].set_ylabel(r"value at $p_c=1/2$")
    ax[2].set_xlabel(r"$(p-p_c)L^{3/4}$")
    ax[2].set_ylabel(r"$(C_N/k)\,L^{5/24}$")
    ax[2].set_xlim(-30, 60)
    ax[0].legend(fontsize=5.5)
    ax[1].legend(fontsize=6)
    for a, t in zip(ax, ["(a) $C_N/k=P_\\infty^2$ ($p\\geq p_c$)", "(b) 2D exponents at $p_c$", "(c) collapse"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_lattice.pdf")
    print("fits at p_c: C/k slope %.4f +- %.4f (pred %.4f); P slope %.4f +- %.4f (pred %.4f)"
          % (out["Ck"][0], out["Ck"][1], -5 / 24, out["S"][0], out["S"][1], -5 / 48))


if __name__ == "__main__" and os.environ.get("V4"):
    fig_lattice_v4()


def fig_dtc_v4():
    """Distribution of decision-tree complexity vs binom(k,j) S^{j+1}(1-S)^{k-j} (paper style)."""
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.N == 10000) & (c.k == 5)]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    j = np.arange(1, 6)
    for a, z in zip(ax, [1.5, 2.0, 3.0]):
        r = c[np.isclose(c.z, z)].iloc[0]
        meas = [r[f"pdtc_{i}"] for i in j]
        th = [r[f"pdtc_{i}_th"] for i in j]
        a.bar(j - 0.18, meas, width=0.34, color=COL[0], label="measured")
        a.bar(j + 0.18, th, width=0.34, color=COL[1], label="prediction")
        a.set_yscale("log")
        a.set_xlabel("decision-tree complexity $j$")
        panel(a, f"$z={z:g}$ ($S={r.S_th:.3f}$)")
    ax[0].set_ylabel(r"$\Pr[C_{\rm DT}=j]$")
    ax[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUT / "fig_dtc.pdf")


if __name__ == "__main__" and os.environ.get("V4"):
    fig_dtc_v4()


def fig_finiteh_v5():
    """(a,b) finite input density: activity smoothed, complexity sharp; (c) phi = 0.25 (non-exact)."""
    s = pd.read_csv(D / "finite_h/summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    zz = np.linspace(0.6, 2.0, 300)
    import finite_h as fh
    for i, h in enumerate([0.05, 0.1, 0.2]):
        t = s[(s.N == 1000000) & np.isclose(s.h, h)].sort_values("z")
        th = np.array([fh.theory(z, h) for z in zz])
        ax[0].plot(t.z, t.psi, MK[i], ms=2.6, mfc="none", color=COL[i])
        ax[0].plot(zz, th[:, 0], "--", color=COL[i], lw=0.7)
        ax[0].plot(t.z, t.Ck, MK[i], ms=2.6, color=COL[i], label=f"$h={h:g}$")
        ax[0].plot(zz, th[:, 1], "-", color=COL[i], lw=0.8)
        ax[0].axvline(1 / (1 - h), color=COL[i], lw=0.5, alpha=0.6)
    for i, N in enumerate(NS[1:]):
        t = s[(s.N == N) & np.isclose(s.h, 0.1)].sort_values("z")
        ax[1].semilogy(t.z * 0.9, t.Ck, MK[i + 1] + "-", ms=2.6, lw=0.6, color=COL[i + 1], label=nlab(N))
    ax[1].axvline(1.0, color="#9a9994", lw=0.6)
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[(w.phi == "0.25") & (w.k == 5) & (w.z <= 3.0)]
    z3 = np.linspace(0.9, 3.0, 200)
    pt = np.array([trigger_prob(z, 0.25) for z in z3])
    sc = np.array([cascade_size(z, 0.25)[0] for z in z3])
    S3 = np.array([giant_fraction(z) for z in z3])
    for i, N in enumerate([100000, 1000000]):
        t = w[w.N == N].sort_values("z")
        ax[2].plot(t.z, t.lam / t.k, MK[i + 2], ms=2.6, color=COL[i + 2], label=nlab(N))
    ax[2].plot(z3, sc * pt, "-", color=INK, lw=0.8, label=r"$S_cP_t$")
    ax[2].plot(z3, S3 ** 2, "--", color="#52514e", lw=0.8, label=r"$S^2$")
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel(r"$\Psi$ (open), $C_N/k$ (filled)")
    ax[1].set_xlabel(r"$z(1-h)$")
    ax[1].set_ylabel(r"$C_N/k$ ($h=0.1$)")
    ax[2].set_xlabel("$z$")
    ax[2].set_ylabel(r"$C_N/k$ ($\varphi=0.25$)")
    ax[0].legend(fontsize=5.5)
    ax[1].legend(fontsize=5.5)
    ax[2].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a) finite input density", "(b) sharp in $N$", "(c) non-exact case"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_finiteh.pdf")


def fig_fss_v5():
    """Two-panel FSS (g(lambda) moved to SM)."""
    w = pd.read_csv(D / "extra/O_psi_window.csv", dtype={"phi": str})
    w = w[(w.phi == "0.1") & (w.k == 5) & (w.lam_s.abs() <= 3.01)]
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.1))
    for i, N in enumerate(NS):
        g = w[w.N == N].sort_values("lam_s")
        ax[0].plot(g.lam_s, g.S * N ** (1 / 3), MK[i], ms=2.4, color=COL[i], label=nlab(N))
        ax[1].plot(g.lam_s, g.psi * N ** (2 / 3) / 5, MK[i], ms=2.4, color=COL[i])
        gp = g[g.k * g.S < 0.3]
        ax[1].plot(gp.lam_s, gp.psi_pred * N ** (2 / 3) / 5, "-", color=COL[i], lw=0.7)
    ax[1].set_yscale("log")
    ax[0].set_ylabel(r"$S_N N^{1/3}$")
    ax[1].set_ylabel(r"$\Psi_N N^{2/3}/k$")
    for a in ax:
        a.set_xlabel(r"$\lambda$")
    ax[0].legend(fontsize=5)
    panel(ax[0], "(a)")
    panel(ax[1], "(b)")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUT / "fig_fss_col.pdf")


if __name__ == "__main__" and os.environ.get("V5"):
    fig_finiteh_v5()
    fig_fss_v5()
    print("v5 ok")


def fig_finiteh_v6():
    """Finite input density: (a) ER activity vs complexity; (b) ER per-node complexity diverges;
    (c) 2D lattice at h = 0.1; (d) non-exact phi = 0.25."""
    import finite_h as fh
    s = pd.read_csv(D / "finite_h/summary.csv")
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    zz = np.linspace(0.6, 2.0, 300)
    for i, h in enumerate([0.05, 0.1, 0.2]):
        t = s[(s.N == 1000000) & np.isclose(s.h, h)].sort_values("z")
        th = np.array([fh.theory(z, h) for z in zz])
        ax[0].plot(t.z, t.psi, MK[i], ms=2.2, mfc="none", color=COL[i])
        ax[0].plot(zz, th[:, 0], "--", color=COL[i], lw=0.7)
        ax[0].plot(t.z, t.Ck, MK[i], ms=2.2, color=COL[i], label=f"$h={h:g}$")
        ax[0].plot(zz, th[:, 1], "-", color=COL[i], lw=0.8)
    for i, N in enumerate(NS[1:]):
        t = s[(s.N == N) & np.isclose(s.h, 0.1)].sort_values("z")
        ax[1].semilogy(t.z * 0.9, t.Ck * 0.1 * N, MK[i + 1], ms=2.4, color=COL[i + 1], label=nlab(N))
    zp = np.linspace(0.5, 0.97, 100)
    ax[1].semilogy(zp, 0.1 / 0.9 * zp / (1 - zp), "-", color=INK, lw=0.8)
    ax[1].axvline(1.0, color="#9a9994", lw=0.6)
    lh = pd.read_csv(D / "lattice_h/summary.csv")
    lh = lh[np.isclose(lh.h, 0.1)]
    cols = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, 4))
    for c, L in zip(cols, sorted(lh.L.unique())):
        t = lh[lh.L == L].sort_values("p")
        ax[2].plot(t.p, t.psi, "--", color=c, lw=0.8)
        ax[2].plot(t.p, t.Ck, "o-", ms=2.2, lw=0.6, color=c, label=f"$L={L}$")
    ax[2].axvline(0.571, color="#9a9994", lw=0.6)
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[(w.phi == "0.25") & (w.k == 5) & (w.z <= 3.0)]
    z3 = np.linspace(0.9, 3.0, 200)
    pt = np.array([trigger_prob(z, 0.25) for z in z3])
    sc = np.array([cascade_size(z, 0.25)[0] for z in z3])
    S3 = np.array([giant_fraction(z) for z in z3])
    for i, N in enumerate([100000, 1000000]):
        t = w[w.N == N].sort_values("z")
        ax[3].plot(t.z, t.lam / t.k, MK[i + 2], ms=2.4, color=COL[i + 2], label=nlab(N))
    ax[3].plot(z3, sc * pt, "-", color=INK, lw=0.8, label=r"$S_cP_t$")
    ax[3].plot(z3, S3 ** 2, "--", color="#52514e", lw=0.8, label=r"$S^2$")
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel(r"$\Psi$ (open), $C_N/k$ (filled)")
    ax[1].set_xlabel(r"$z(1-h)$")
    ax[1].set_ylabel(r"$C_N$ per node ($h=0.1$)")
    ax[2].set_xlabel("$p$")
    ax[2].set_ylabel(r"$\Psi$ (dashed), $C_N/k$")
    ax[3].set_xlabel("$z$")
    ax[3].set_ylabel(r"$C_N/k$ ($\varphi=0.25$)")
    for a in ax:
        a.legend(fontsize=4.8)
    for a, t in zip(ax, ["(a) random graph", "(b) random graph", "(c) 2D, $h=0.1$", "(d) non-exact"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(OUT / "fig_finiteh.pdf")


if __name__ == "__main__" and os.environ.get("V6"):
    fig_finiteh_v6()
    print("v6 ok")


def fig_phase_onset_v6():
    """SM restyle: (a,b) responsive fraction and entropy in the (z, phi) plane; (c,d) onsets."""
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.k == 5) & (c.N == 10000) & (c.phi != "U01")].copy()
    c["phif"] = c.phi.astype(float)
    phis = sorted(c.phif.unique())
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for a, col, lab in ((ax[0], "psi", r"$\Psi$"), (ax[1], "H", r"$H_N$")):
        for j, ph in enumerate(phis):
            t = c[c.phif == ph].sort_values("z")
            zs = t.z.values
            edges = np.concatenate([[zs[0] * 0.95], np.sqrt(zs[1:] * zs[:-1]), [zs[-1] * 1.05]])
            for i in range(len(zs)):
                v = t[col].values[i]
                a.fill_between([edges[i], edges[i + 1]], j - 0.45, j + 0.45,
                               color=plt.get_cmap("Blues")(0.08 + 0.9 * min(v / (0.7 if col == "H" else 1.0), 1)),
                               linewidth=0)
        a.set_xscale("log")
        a.set_yticks(range(len(phis)))
        a.set_yticklabels([f"{p:g}" for p in phis], fontsize=6)
        a.set_xlabel("$z$")
        a.set_ylabel(r"$\varphi$")
        a.axvline(1.0, color=INK, lw=0.6)
        a.grid(False)
        panel(a, f"({'a' if col == 'psi' else 'b'}) {lab}, $k=5$, $N=10^4$")
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    for a, ph, zl, lab in ((ax[2], "0.1", 1.0, "(c)"), (ax[3], "0.25", 1.1141, "(d)")):
        t0 = w[(w.phi == ph) & (w.k == 5) & (w.z >= 0.5) & (w.z <= 1.5)]
        for i, N in enumerate(NS):
            t = t0[t0.N == N].sort_values("z")
            a.plot(t.z, t.psi, MK[i], ms=2.2, color=COL[i], label=nlab(N))
        if ph == "0.1":
            zz = np.linspace(1.0, 1.5, 100)
            Sz = np.array([giant_fraction(z) for z in zz])
            a.plot(zz, Sz * (1 - (1 - Sz) ** 5), "-", color="#52514e", lw=0.7, label="$S[1-(1-S)^5]$")
        else:
            import watts
            zz = np.linspace(1.1145, 1.5, 60)
            pr = [watts.predict(z, 0.25, 5)["psi_w"] for z in zz]
            a.plot(zz, pr, "-", color="#52514e", lw=0.7, label="$S_c[1-(1-P_t)^5]$")
        a.axvline(zl, color=INK, lw=0.6)
        a.set_xlabel("$z$")
        a.set_ylabel(r"$\Psi$")
        panel(a, rf"{lab} $\varphi={ph}$" + (" (onset $z=1$)" if ph == "0.1" else r" (onset $z_{\rm lo}$)"))
    ax[2].legend(fontsize=4.8)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUT / "fig_phase_onset.pdf")


if __name__ == "__main__" and os.environ.get("V6"):
    fig_phase_onset_v6()
    print("phase ok")


def fig_composites3_v7():
    c = pd.read_csv(D / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.phi == "0.1") & (c.N == 10000) & (c.z <= 6)]
    w = pd.read_csv(D / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[w.phi == "0.1"]
    zz = np.linspace(0.3, 6, 400)
    S = np.array([giant_fraction(z) for z in zz])
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 2.1))
    for i, k in enumerate([2, 5, 8]):
        g = c[c.k == k].sort_values("z")
        ax[0].plot(g.z, g.lam / k, MK[i], ms=2.6, color=COL[i], label=f"$k={k}$")
        g = w[(w.k == k) & (w.N == 100000)].sort_values("z")
        g = g[g.z <= 3.2]
        ax[1].plot(g.z, g.H, MK[i], ms=2.6, color=COL[i])
        ax[1].plot(zz, (1 - (1 - S) ** k) * H2(S), "-", color=COL[i], lw=0.7)
        two = g.simpson - ((1 - g.psi) ** 2 + g.psi_var)
        ax[2].plot(g.z, two, MK[i], ms=2.6, color=COL[i])
        ax[2].plot(zz, S ** 2 * (1 - (1 - S) ** k), "-", color=COL[i], lw=0.7)
    ax[0].plot(zz, S ** 2, "-", color=INK, lw=0.8, label="$S^2$")
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.5, 1, 2, 4])
    ax[0].set_xticklabels(["0.5", "1", "2", "4"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for a in ax[1:]:
        a.set_xlim(0.4, 3.2)
    for a in ax:
        a.set_xlabel("$z$")
    ax[0].set_ylabel(r"$C_N/k$")
    ax[1].set_ylabel(r"$H_N$")
    ax[2].set_ylabel(r"$G_2$")
    ax[0].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "fig_composites3.pdf")


if __name__ == "__main__" and os.environ.get("V7"):
    fig_composites3_v7()
    print("v7 ok")


def fig_info_v8():
    from info_check import theory
    er = pd.read_csv(D / "er_info/summary.csv")
    lat = pd.concat([pd.read_csv(D / "lattice_info/summary.csv"),
                     pd.read_csv(D / "lattice_info_fine/summary.csv")])
    lat["w"] = lat.J * lat.n
    lat = lat.groupby(["L", "p"]).agg(w=("w", "sum"), n=("n", "sum")).reset_index()
    lat["J"] = lat.w / lat.n
    zz = np.linspace(0.2, 5, 600)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for i, k in enumerate([4, 8, 12]):
        th = np.array([theory(z, k) for z in zz])
        g = er[(er.k == k) & (er.N == er.N.max())].sort_values("z")
        ax[0].plot(g.z, g.J / k, MK[i], ms=2.6, color=COL[i], label=f"$k={k}$")
        ax[0].plot(zz, th[:, 1] / k, "-", color=COL[i], lw=0.7)
        ax[1].plot(g.z, g.I, MK[i], ms=2.6, color=COL[i])
        ax[1].plot(zz, th[:, 0], "-", color=COL[i], lw=0.7)
    zf = np.array([z if z <= 1 else z * (1 - giant_fraction(z)) for z in zz])
    ax[0].plot(zz, 1 - np.exp(-zf), "--", color=INK, lw=0.8, label=r"$1-e^{-z_f}$")
    for a in ax[:2]:
        a.axvline(1, color="0.6", lw=0.5)
        a.set_xlabel("$z$")
    ax[0].set_ylabel("$J/k$ (bits)")
    ax[1].set_ylabel("$I$ (bits)")
    ax[0].legend(fontsize=5.5)
    for i, L in enumerate([128, 256, 512, 1024]):
        g = lat[lat.L == L].sort_values("p")
        ax[2].plot(g.p, g.J / 8, "-" + MK[i], ms=2.0, lw=0.6, color=COL[i], label=f"$L={L}$")
    ax[2].axvline(0.5, color="0.6", lw=0.5)
    ax[2].set_xlim(0.3, 0.8)
    ax[2].set_xlabel("$p$")
    ax[2].set_ylabel("$J/k$ (bits)")
    ax[2].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_info.pdf")


if __name__ == "__main__" and os.environ.get("V8"):
    fig_info_v8()
    print("v8 ok")


def _packed_layout(G, comps, seed, sublayout="spring"):
    """Lay out each component on its own and pack them: largest in the centre, the rest on a ring
    of grid cells around it, so small clusters stay visible."""
    import networkx as nx
    pos = {}
    big = comps[0]
    if len(big) > 8:
        sub = (nx.kamada_kawai_layout(G.subgraph(big)) if sublayout == "kamada"
               else nx.spring_layout(G.subgraph(big), seed=seed, iterations=300))
        m = np.mean([np.asarray(v) for v in sub.values()], axis=0)
        r = max(np.abs(np.asarray(v) - m).max() for v in sub.values())
        for u, xy in sub.items():
            pos[u] = 0.5 * (np.asarray(xy) - m) / r
        rest = comps[1:]
    else:
        rest = comps
    cells = [(x, y) for y in np.linspace(-1, 1, 7) for x in np.linspace(-1, 1, 7)
             if (len(big) <= 8) or max(abs(x), abs(y)) > 0.6]
    rng = np.random.default_rng(seed)
    cells = [cells[i] for i in rng.permutation(len(cells))]
    for c, (cx, cy) in zip(rest, cells):
        c = sorted(c)
        if len(c) == 1:
            pos[c[0]] = np.array([cx, cy])
            continue
        sub = nx.spring_layout(G.subgraph(c), seed=seed, iterations=100)
        for u, xy in sub.items():
            pos[u] = np.array([cx, cy]) + 0.11 * np.asarray(xy)
    for u in G:
        if u not in pos:
            pos[u] = rng.uniform(-1, 1, 2)
    return pos


def _gv_layout(G, engine):
    """Positions from a graphviz engine (neato, fdp, sfdp), components packed by graphviz."""
    import subprocess
    lines = ["graph G {", "graph [pack=true, overlap=false, splines=false, sep=0.3];",
             "node [shape=point, width=0.05];"]
    lines += [f"{u};" for u in G.nodes()]
    lines += [f"{u} -- {v};" for u, v in G.edges()]
    lines.append("}")
    out = subprocess.run([engine, "-Tplain"], input="\n".join(lines), capture_output=True,
                         text=True, check=True).stdout
    pos = {}
    for ln in out.splitlines():
        f = ln.split()
        if f and f[0] == "node":
            pos[int(f[1])] = np.array([float(f[2]), float(f[3])])
    return pos


def _layout(G, comps, seed, name):
    import networkx as nx
    if name == "packed":
        return _packed_layout(G, comps, seed)
    if name in ("neato", "fdp", "sfdp"):
        return _gv_layout(G, name)
    if name == "spring":
        return nx.spring_layout(G, seed=seed, k=0.6 / np.sqrt(G.number_of_nodes()), iterations=500)
    if name == "kamada":
        # Kamada-Kawai per component, then packed like "packed"
        return _packed_layout(G, comps, seed, sublayout="kamada")
    raise ValueError(name)


def fig_schematic_v9(layout="neato", fname="fig_schematic.pdf", gvstyle=False):
    """Fig. 1: structural vs functional percolation (simple contagion, k inputs)."""
    import networkx as nx
    GREY, OR_C, DUST, INP = "#a8a8a8", COL[1], COL[0], INK
    fig = plt.figure(figsize=(3.4, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.35, wspace=0.12)
    for col, (z, seed) in enumerate([(0.85, 3), (1.9, 5)]):
        rng = np.random.default_rng(seed)
        N, k = 70, 3
        G = nx.gnp_random_graph(N, z / (N - 1), seed=seed)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        giant = comps[0] if (z > 1 and len(comps[0]) > 0.2 * N) else set()
        # inputs: in the above-threshold panel put two in the giant and one in a small cluster
        if giant:
            small = [c for c in comps[1:] if len(c) >= 2]
            inp = list(rng.choice(sorted(giant), 3, replace=False))
        else:
            multi = [c for c in comps if len(c) >= 2]
            inp = [sorted(multi[i])[0] for i in range(k)]
        inp = set(inp)
        color, size, shape = {}, {}, {}
        for c in comps:
            has_in = bool(c & inp)
            for u in c:
                if u in inp:
                    color[u] = INP
                elif c is comps[0] and giant:
                    color[u] = OR_C
                elif has_in:
                    color[u] = DUST
                else:
                    color[u] = GREY
        pos = _layout(G, comps, seed, layout)
        ax = fig.add_subplot(gs[0, col])
        others = [u for u in G if u not in inp]
        if gvstyle:   # graphviz look: outlined circles, uniform thin edges
            nx.draw_networkx_edges(G, pos, ax=ax, width=0.55, edge_color="#8a8a8a")
            nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=15, ax=ax,
                                   node_color=[color[u] for u in others], linewidths=0.45,
                                   edgecolors="#3a3a3a")
            nx.draw_networkx_nodes(G, pos, nodelist=sorted(inp), node_size=26, node_shape="s", ax=ax,
                                   node_color=INP, linewidths=0.45, edgecolors=INP)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, width=0.55, edge_color="0.35")
            nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=18, ax=ax,
                                   node_color=[color[u] for u in others], linewidths=0)
            nx.draw_networkx_nodes(G, pos, nodelist=sorted(inp), node_size=34, node_shape="s", ax=ax,
                                   node_color=INP, linewidths=0)
        ax.set_axis_off()
        ax.set_title("below threshold" if col == 0 else "above threshold", fontsize=7.5)
        panel(ax, "(a)" if col == 0 else "(b)")
        # function space
        ax2 = fig.add_subplot(gs[1, col])
        S = 0.0 if z <= 1 else giant_fraction(z)
        labels = ["0", "OR", "other"]
        w = [1.0, 0.0, 0.0] if z <= 1 else [1 - S, S, 0.0]
        cols = [GREY, OR_C, DUST]
        for i, (lab, wi, ci) in enumerate(zip(labels, w, cols)):
            ax2.bar(i, max(wi, 0.008), width=0.6, color=ci, edgecolor="none")
        ax2.grid(False)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylim(0, 1.12)
        ax2.set_xlim(-0.6, 2.6)
        if col == 0:
            ax2.set_ylabel("weight", fontsize=7.5)
        else:
            ax2.set_yticklabels([])
            ax2.text(0, 1 - S + 0.06, r"$1-S$", ha="center", fontsize=6.5)
            ax2.text(1, S + 0.06, r"$S$", ha="center", fontsize=6.5)
        ax2.text(2, 0.06, r"$O(k/N)$", ha="center", fontsize=6)
        ax2.set_xlabel("function", fontsize=7.5)
        panel(ax2, "(c)" if col == 0 else "(d)")
    fig.savefig(OUT / fname, bbox_inches="tight")


def fig_info_si_v9():
    """SM figure: information on random graphs (corrected; finite-k maxima above threshold)."""
    from info_check import theory
    er = pd.read_csv(D / "er_info/summary.csv")
    zz = np.linspace(0.2, 5, 600)
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.1))
    for i, k in enumerate([4, 8, 12]):
        th = np.array([theory(z, k) for z in zz])
        g = er[(er.k == k) & (er.N == er.N.max())].sort_values("z")
        ax[0].plot(g.z, g.J / k, MK[i], ms=2.6, color=COL[i], label=f"$k={k}$")
        ax[0].plot(zz, th[:, 1] / k, "-", color=COL[i], lw=0.7)
        ax[1].plot(g.z, g.I, MK[i], ms=2.6, color=COL[i])
        ax[1].plot(zz, th[:, 0], "-", color=COL[i], lw=0.7)
    zf = np.array([z if z <= 1 else z * (1 - giant_fraction(z)) for z in zz])
    ax[0].plot(zz, 1 - np.exp(-zf), "--", color=INK, lw=0.8, label=r"$1-e^{-z_f}$ ($k\to\infty$)")
    for a in ax:
        a.axvline(1, color="0.6", lw=0.5)
        a.set_xlabel("$z$")
    ax[0].set_ylabel("$J/k$ (bits)")
    ax[1].set_ylabel("$I$ (bits)")
    ax[0].legend(fontsize=5.5)
    for a, t in zip(ax, ["(a)", "(b)"]):
        panel(a, t)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_info_si.pdf")


def _gv_render(G, color, inp, engine, path, seed):
    """Render a network panel with graphviz itself (graphviz look and feel) to PNG."""
    import subprocess
    lines = ["graph G {",
             f"graph [pack=true, overlap=false, splines=true, sep=0.25, start={seed}, bgcolor=transparent];",
             'node [shape=circle, style=filled, width=0.14, fixedsize=true, label="", penwidth=0.7, '
             'color="#333333"];',
             'edge [color="#7a7a7a", penwidth=0.8];']
    for u in G.nodes():
        if u in inp:
            lines.append(f'{u} [shape=square, width=0.2, fillcolor="#0b0b0b", color="#0b0b0b"];')
        else:
            lines.append(f'{u} [fillcolor="{color[u]}"];')
    lines += [f"{u} -- {v};" for u, v in G.edges()]
    lines.append("}")
    subprocess.run([engine, "-Tpng", "-Gdpi=500", "-o", str(path)], input="\n".join(lines),
                   text=True, check=True)


def fig_schematic_gviz(fname="fig_schematic_gviz.pdf", engine="neato"):
    """Fig. 1 with network panels rendered by graphviz."""
    import networkx as nx
    GREY, OR_C, DUST = "#d4d4d4", COL[1], COL[0]
    fig = plt.figure(figsize=(3.4, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.35, wspace=0.12)
    for col, (z, seed) in enumerate([(0.85, 3), (1.9, 5)]):
        rng = np.random.default_rng(seed)
        N, k = 70, 3
        G = nx.gnp_random_graph(N, z / (N - 1), seed=seed)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        giant = comps[0] if (z > 1 and len(comps[0]) > 0.2 * N) else set()
        if giant:
            inp = set(rng.choice(sorted(giant), 3, replace=False))
        else:
            multi = [c for c in comps if len(c) >= 2]
            inp = {sorted(multi[i])[0] for i in range(k)}
        color = {}
        for c in comps:
            has_in = bool(c & inp)
            for u in c:
                color[u] = OR_C if (giant and c is comps[0]) else (DUST if has_in else GREY)
        png = OUT / f"_gv_panel_{col}.png"
        _gv_render(G, color, inp, engine, png, seed)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(plt.imread(png), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title("below threshold" if col == 0 else "above threshold", fontsize=7.5)
        panel(ax, "(a)" if col == 0 else "(b)")
        png.unlink()
        ax2 = fig.add_subplot(gs[1, col])
        S = 0.0 if z <= 1 else giant_fraction(z)
        w = [1.0, 0.0, 0.0] if z <= 1 else [1 - S, S, 0.0]
        for i, (wi, ci) in enumerate(zip(w, ["#c9c9c9", OR_C, DUST])):
            ax2.bar(i, max(wi, 0.008), width=0.6, color=ci, edgecolor="none")
        ax2.grid(False)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(["0", "OR", "other"], fontsize=7)
        ax2.set_ylim(0, 1.12)
        ax2.set_xlim(-0.6, 2.6)
        if col == 0:
            ax2.set_ylabel("weight", fontsize=7.5)
        else:
            ax2.set_yticklabels([])
            ax2.text(0, 1 - S + 0.06, r"$1-S$", ha="center", fontsize=6.5)
            ax2.text(1, S + 0.06, r"$S$", ha="center", fontsize=6.5)
        ax2.text(2, 0.06, r"$O(k/N)$", ha="center", fontsize=6)
        ax2.set_xlabel("function", fontsize=7.5)
        panel(ax2, "(c)" if col == 0 else "(d)")
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=500)


if __name__ == "__main__" and os.environ.get("V9GVIZ"):
    fig_schematic_v9("neato", "fig_schematic_gvstyle.pdf", gvstyle=True)
    fig_schematic_gviz()
    fig_schematic_gviz("fig_schematic_gviz_sfdp.pdf", "sfdp")
    print("gviz ok")


if __name__ == "__main__" and os.environ.get("V9LAYOUTS"):
    for name in ("packed", "kamada", "spring", "neato", "fdp", "sfdp"):
        fig_schematic_v9(name, f"fig_schematic_{name}.pdf")
    print("layouts ok")


if __name__ == "__main__" and os.environ.get("V9"):
    fig_info_si_v9()
    fig_schematic_v9()
    print("v9 ok")


def fig_schematic3_v9(fname="fig_schematic3.pdf"):
    """Fig. 1, three phases (below, at, above z_c): example graphs (neato) and ensemble weights of
    functions averaged over random graphs and input placements (schematic_weights.py)."""
    import networkx as nx
    from matplotlib.colors import to_rgb
    GREY, OR_C, DUST, INP = "#a8a8a8", COL[1], COL[0], INK
    wts = pd.read_csv(D / "schematic_weights/summary.csv")
    cases = [(0.5, 3, r"$z<z_c$"), (1.0, 26, r"$z=z_c$"), (1.9, 5, r"$z>z_c$")]
    Ns = [10**3, 10**4, 10**5]
    tops = []
    fig = plt.figure(figsize=(6.8, 3.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0], hspace=0.32, wspace=0.1)
    for col, (z, seed, title) in enumerate(cases):
        rng = np.random.default_rng(seed)
        N = 130
        G = nx.gnp_random_graph(N, z / (N - 1), seed=seed)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        multi = [c for c in comps if len(c) >= 2]
        if z > 1:      # typical: all inputs in the giant (probability S^3 ~ 0.45)
            inp = set(rng.choice(sorted(comps[0]), 3, replace=False))
        elif z == 1:   # typical at this N: one input in the largest (incipient) cluster
            inp = {sorted(multi[i])[len(multi[i]) // 2] for i in (0, 2, 3)}
        else:
            inp = {sorted(multi[i])[len(multi[i]) // 2] for i in range(3)}
        # one rule for all panels: largest responsive cluster orange, other responsive blue
        resp = [c for c in comps if c & inp]
        color = {}
        for c in comps:
            for u in c:
                color[u] = OR_C if (resp and c is resp[0]) else (DUST if c & inp else GREY)
        pos = _gv_layout(G, "neato")
        ax = fig.add_subplot(gs[0, col])
        others = [u for u in G if u not in inp]
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.55, edge_color="0.35")
        nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=14, ax=ax,
                               node_color=[color[u] for u in others], linewidths=0)
        nx.draw_networkx_nodes(G, pos, nodelist=sorted(inp), node_size=30, node_shape="s", ax=ax,
                               node_color=INP, linewidths=0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8)
        panel(ax, "(" + "abc"[col] + ")")
        tops.append(ax)
    # order-parameter panel: weight of the largest responsive cluster's OR against z
    cur = pd.read_csv(D / "schematic_weights/curve.csv")
    ax2 = fig.add_subplot(gs[1, :])
    for ni, Nn in enumerate([10**3, 10**4, 10**5]):
        g = cur[cur.N == Nn].sort_values("z")
        shade = 0.45 + 0.55 * ni / 2
        c = tuple(1 - shade * (1 - x) for x in to_rgb(OR_C))
        ax2.plot(g.z, g.top, "-o", ms=2.2, lw=0.9, color=c, label=f"$N=10^{ni + 3}$")
    zz = np.linspace(0.0, 2.3, 400)
    Sz = np.array([giant_fraction(x) if x > 1 else 0.0 for x in zz])
    ax2.plot(zz, Sz * (1 - (1 - Sz) ** 3), "--", color=INK, lw=0.8, label=r"$N\to\infty$")
    for zc, lab in ((0.5, "(a)"), (1.0, "(b)"), (1.9, "(c)")):
        ax2.axvline(zc, color="0.6", lw=0.5, zorder=0)
        ax2.text(zc, 0.86, lab, ha="center", fontsize=6.5, color="0.35",
                 bbox=dict(fc="white", ec="none", pad=0.6))
    ax2.grid(False)
    # place z_c = 1 under the centre of panel (b); axis ends under the right edge of panel (c)
    ZMIN, ZMAX = 0.0, 2.3
    pb, pc = tops[1].get_position(), tops[2].get_position()
    xc, R = 0.5 * (pb.x0 + pb.x1), pc.x1
    W = (R - xc) / (1 - (1 - ZMIN) / (ZMAX - ZMIN))
    p2 = ax2.get_position()
    ax2.set_position([R - W, p2.y0, W, p2.height])
    ax2.set_xlim(ZMIN, ZMAX)
    ax2.set_ylim(0, 0.95)
    ax2.set_xlabel("mean degree $z$", fontsize=7.5)
    ax2.set_ylabel("fraction computing the\nlargest responsive\ncluster's OR", fontsize=7)
    ax2.legend(fontsize=6, frameon=False, loc="center right")
    panel(ax2, "(d)")
    fig.savefig(OUT / fname, bbox_inches="tight")


if __name__ == "__main__" and os.environ.get("V9THREE"):
    fig_schematic3_v9()
    print("three ok")

# ---------------------------------------------------------------- the figures of the paper
PAPER_FIGURES = ["fig_schematic3_v9", "fig_field_v3", "fig_fss", "fig_composites3_v7", "fig_kS",
                 "fig_growing_k", "fig_phase_onset_v6", "fig_info_si_v9", "fig_lattice_v4", "fig_finiteh_v6",
                 "fig_symmetry", "fig_selection", "fig_dtc_v4", "fig_fluct"]

if __name__ == "__main__" and os.environ.get("PAPER"):
    for name in PAPER_FIGURES:      # plus fig_synergy.py (separate script)
        globals()[name]()
    print("paper figures ok")
