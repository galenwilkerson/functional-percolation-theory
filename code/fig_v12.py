"""New main-text figures for paper v12 (outline v8 rev c).
  fig_phase_diagram.pdf  functional phase diagram in (z, phi): z_c = 1, z_lo(phi), z_hi(phi) (staircase in n = floor(1/phi))
  fig_ssb.pdf            input-symmetry breaking at z_lo: trajectories, atoms, P(broken) (no fit), one-edge response,
                         finite input density, finite-size scaling (cluster-exact, Aldous)
  fig_beyond.pdf         beyond connectivity: trigger set vs connected set (phi = 0.25); upper-edge quorum

  FIG_OUT=../paper_v12/figures python fig_v12.py
Data: data/ssb (SSB_DATA), ssb_results (SSB_OUT), data/aldous/aldous_moments.csv (ALDOUS_CSV).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssb_theory as th  # noqa: E402
import watts  # noqa: E402
from spec_card import watts_lhs, z_lo as z_lo_n  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("SSB_DATA", HERE.parent / "data" / "ssb"))
RES = Path(os.environ.get("SSB_OUT", HERE.parent / "summaries" / "ssb"))
ALDOUS = Path(os.environ.get("ALDOUS_CSV", HERE.parent / "summaries" / "aldous" / "aldous_moments.csv"))
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7d4fc2"]
MK = ["o", "s", "^", "D", "v"]
INK = "#0b0b0b"
plt.rcParams.update({"font.size": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "lines.linewidth": 1.2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5})


def panel(ax, t):
    ax.set_title(t, loc="left", fontsize=9.5, color=INK)


def nlab(N):
    e = math.log10(N)
    if abs(e - round(e)) < 1e-9:
        return f"$N=10^{int(round(e))}$"
    p = int(math.floor(e))
    return f"$N={N / 10**p:g}\\times10^{p}$"


def z_hi_n(n):
    zs = np.linspace(2.0, 40.0, 20000)
    f = np.array([watts_lhs(z, n) for z in zs])
    idx = np.where((f[:-1] >= 1) & (f[1:] < 1))[0]
    return float(zs[idx[-1]]) if len(idx) else np.nan


# ============================================================================ phase diagram
def fig_phase_diagram():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    C_WIN, C_STRIP, C_HI = COL[0], "#f08a4b", "#8e5bb5"
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.45), gridspec_kw=dict(width_ratios=[1.45, 1]))
    for ax, (xlo, xhi, logx) in zip(axs, ((0.5, 20.0, True), (0.95, 1.25, False))):
        for n in range(4, 21):
            p_lo, p_hi = 1 / (n + 1), 1 / n
            zl, zh = z_lo_n(n), z_hi_n(n)
            ax.fill_betweenx([p_lo, p_hi], zl, zh, color=C_WIN, alpha=0.18, lw=0)
            ax.fill_betweenx([p_lo, p_hi], 1.0, zl, color=C_STRIP, alpha=0.55, lw=0)
            ax.plot([zl, zl], [p_lo, p_hi], color=C_WIN, lw=2.4)
            ax.plot([zh, zh], [p_lo, p_hi], color=C_HI, lw=2.4)
        ax.axvline(1.0, color=INK, lw=0.9, ls="--")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.04, 0.3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("mean degree $z$")
    axs[0].set_ylabel(r"threshold $\varphi$")
    axs[0].text(1.04, 0.285, "$z_c=1$", fontsize=8.8)
    axs[0].text(5.0, 0.272, "no window ($\\varphi>1/4$)", fontsize=8.3, color="#52514e")
    panel(axs[0], r"(a) functional phase diagram")
    # zoom: the strip between z_c and z_lo, with the two data points
    axs[1].text(1.122, 0.24, "$n=4$: $z_{\\rm lo}=1.114$", fontsize=8)
    axs[1].text(1.03, 0.183, "$n=5$: $z_{\\rm lo}=1.021$", fontsize=8)
    axs[1].text(1.03, 0.093, "$n=10$: $z_{\\rm lo}=1+10^{-6}$", fontsize=8)
    axs[1].text(1.057, 0.226, "connected,\nno function", fontsize=7.6, color="#a8441a", ha="center", va="bottom")
    g = pd.read_csv(RES / "g_finiteh.csv")
    for (z, ok) in ((1.05, False), (1.2, True)):
        s = g[(g.phi == 0.25) & (g.h == 0.002) & (g.N == 100000) & np.isclose(g.z, z)]
        if len(s):
            axs[1].plot(z, 0.215, "s", ms=6, mfc=C_WIN if ok else C_STRIP, mec=INK, mew=0.8, zorder=5)
    panel(axs[1], r"(b) zoom near $z_c$")
    hs = [Patch(color=C_WIN, alpha=0.3, label="cascade window: macroscopic function, input symmetry broken"),
          Patch(color=C_STRIP, alpha=0.6, label="$z_c<z<z_{\\rm lo}$: connected, no function, symmetric"),
          Line2D([], [], color=INK, lw=0.9, ls="--", label="$z_c=1$ (graph percolation)"),
          Line2D([], [], color=C_WIN, lw=2.4, label="$z_{\\rm lo}(\\varphi)$ (functional onset)"),
          Line2D([], [], color=C_HI, lw=2.4, label="$z_{\\rm hi}(\\varphi)$ (upper edge)"),
          Line2D([], [], ls="", marker="s", ms=6, mfc=C_STRIP, mec=INK, label="data: no input controls ($z=1.05$)"),
          Line2D([], [], ls="", marker="s", ms=6, mfc=C_WIN, mec=INK, label="data: every network broken ($z=1.2$)")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.23, 1, 1))
    fig.savefig(OUT / "fig_phase_diagram.pdf")
    plt.close(fig)


def fig_ssb():
    t = pd.read_parquet(DATA / "traj.parquet")
    s = pd.read_parquet(DATA / "tilt.parquet")
    for df in (t, s):
        mc = [c for c in df.columns if c.startswith("m") and c[1:].isdigit()]
        df["M0"] = df.m0 - df[mc].mean(axis=1)
        df["K"] = (df[mc] >= 0.01).sum(axis=1)
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.6))
    # (a) one grown network
    g0 = t[(t.k == 3) & (t.phi == 0.1) & (t.N == 64000)]
    g = g0[g0.seed == sorted(g0.seed.unique())[0]].sort_values("z")
    ax = axs[0, 0]
    for j in range(3):
        ax.plot(g.z, g[f"m{j}"], color=COL[j], marker=MK[j], ms=2.2, lw=1.1, label=f"input {j + 1}")
    ax.plot(g.z, g.gcc, color=INK, ls=":", lw=0.9, label="giant component")
    ax.set_xlabel("$z$ (edges added)")
    ax.set_ylabel("relevance $m_i$")
    ax.legend(fontsize=6.3, loc="center right")
    panel(ax, r"(a) one network, $\varphi=0.1$")
    # (b) atoms
    ax = axs[0, 1]
    for i, N in enumerate((1000, 16000, 256000)):
        x = s[(s.k == 3) & (s.phi == 0.1) & (s.z == 1.5) & (s.N == N)].M0
        ax.hist(x, bins=np.linspace(-0.45, 0.45, 91), histtype="step", density=True, color=COL[i], label=nlab(N))
    S = float(th.S_er(1.5)[0])
    for _, M, _, _ in th.mixture(S, 3):
        ax.axvline(M, color="#52514e", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("$M_0=m_0-\\bar m$")
    ax.set_ylabel("$P(M_0)$")
    ax.legend(fontsize=6.3)
    panel(ax, r"(b) broken states, $z=1.5$")
    # (c) P(broken), parameter-free
    ax = axs[0, 2]
    zz = np.linspace(0.5, 3.0, 200)
    for i, phi in enumerate((0.1, 0.25)):
        q = th.S_er(zz) if phi == 0.1 else np.array([watts.trigger_prob(z, phi) if z > 1.114 else 0 for z in zz])
        ax.plot(zz, 1 - q ** 3 - (1 - q) ** 3, color=COL[i], lw=1.0, ls="--")
        d = s[(s.k == 3) & (s.phi == phi) & (s.N == 64000)]
        pb = d.groupby("z").K.apply(lambda K: float(K.between(1, 2).mean()))
        nn = d.groupby("z").K.size()
        se = np.sqrt(pb * (1 - pb) / nn)
        ax.errorbar(pb.index, pb.values, yerr=se.values, fmt=MK[i], color=COL[i], ms=3.5, lw=0.8, capsize=1.5,
                    label=f"$\\varphi={phi}$")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$P(\\mathrm{broken})$, $k=3$")
    ax.legend(fontsize=6.3)
    panel(ax, r"(c) no fit: $1-q^3-(1-q)^3$")
    # (d) one-edge response
    ax = axs[1, 0]
    q6 = pd.read_csv(RES / "sb6b_quenched.csv")
    for i, N in enumerate((1000, 16000, 256000)):
        d = q6[(q6.phi == 0.1) & (q6.k == 3) & (q6.N == N)].sort_values("z")
        ax.plot(d.z, d.chi, MK[i], color=COL[i], ms=3, label=nlab(N))
        zz = np.linspace(1.15, 3.0, 50)
        Sz = th.S_er(zz)
        ax.plot(zz, (N - 3) * (1 - Sz) * Sz ** 2, color=COL[i], lw=0.8, ls="--")
    zb = np.linspace(0.45, 0.95, 50)
    ax.plot(zb, 1 / (1 - zb), color=INK, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\\chi_1=N\\langle\\Delta m_0\\rangle$")
    ax.legend(fontsize=6.3)
    panel(ax, r"(d) one added edge")
    # (e) finite input density
    ax = axs[1, 1]
    gf = pd.read_csv(RES / "g_finiteh.csv")
    for i, N in enumerate((10000, 30000, 100000)):
        d = gf[(gf.phi == 0.1) & (gf.h == 0.002) & (gf.N == N)].sort_values("z")
        ax.plot(d.z, d.median_A, "-" + MK[i], color=COL[i], ms=2.5, label=nlab(N))
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 0.3)
    ax.set_xlabel("$z$")
    ax.set_ylabel("median anisotropy $A$")
    ax.legend(fontsize=6.3)
    panel(ax, r"(e) input density $h=0.002$")
    # (f) finite-size scaling, cluster-exact, against Aldous
    ax = axs[1, 2]
    cm = pd.read_csv(RES / "scc_cells.csv")
    al = pd.read_csv(ALDOUS)
    al = al[al.dt == al.dt.min()].sort_values("lam")
    for i, N in enumerate((1000, 100000, 10000000)):
        d = cm[cm.N == N].sort_values("lam")
        ax.plot(d.lam, d.absM0 * N ** (2 / 3), "-" + MK[i], color=COL[i], ms=2.5, label=nlab(N))
    ax.plot(al.lam, 2 * (1 - 1 / 3) * al.G2, color=INK, ls=":", lw=1.2, label="Aldous limit")
    ax.set_yscale("log")
    ax.set_xlabel("$\\lambda_N=(z-1)N^{1/3}$")
    ax.set_ylabel("$\\langle|M_0|\\rangle N^{2/3}$")
    ax.legend(fontsize=6.0)
    panel(ax, r"(f) finite-size scaling")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssb.pdf")
    plt.close(fig)


# ============================================================================ beyond connectivity
def fig_beyond():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    c = pd.read_parquet(DATA / "conn.parquet")
    c = c[(c.phi == 0.25) & (c.k == 3) & (c.N == 64000)]
    pf = c.groupby("z").K_func.apply(lambda K: float(K.between(1, 2).mean()))
    pc = c.groupby("z").K_conn.apply(lambda K: float(K.between(1, 2).mean()))
    ax.plot(pf.index, pf.values, MK[0], color=COL[0], ms=4, label="function (controlling inputs)")
    ax.plot(pc.index, pc.values, MK[1], color=COL[1], ms=4, label="connectivity (connected inputs)")
    zz = np.linspace(1.12, 3.6, 120)
    pt = np.array([watts.trigger_prob(z, 0.25) for z in zz])
    S = th.S_er(zz)
    ax.plot(zz, 1 - pt ** 3 - (1 - pt) ** 3, color=COL[0], ls="--", lw=0.9)
    ax.plot(zz, 1 - S ** 3 - (1 - S) ** 3, color=COL[1], ls="--", lw=0.9)
    ax.axvline(1.114, color="#52514e", lw=0.6)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$P(\\mathrm{broken})$, $k=3$")
    ax.legend(fontsize=6.5, loc="lower right")
    panel(ax, r"(a) $\varphi=0.25$: function $\neq$ connectivity")
    ax = axs[1]
    uf = pd.read_parquet(DATA / "upperfine.parquet")
    from ssb_extra import cstar_theory
    s0 = uf[(uf.phi == 0.25) & (uf.z == 4.6)]
    for i, (N, s) in enumerate(s0.groupby("N")):
        cs = cstar_theory(int(N), float(s.h.iloc[0]), 0.25, 4.6)
        recs = []
        for _, r in s.iterrows():
            cnt = np.asarray(r.counts)
            glob = (np.asarray(r.resp) >= 0.5 * r.resp_allon) & (r.resp_allon >= 0.1)
            recs += [dict(c=cc, p=glob[cnt == cc].mean()) for cc in np.unique(cnt)]
        cur = pd.DataFrame(recs).groupby("c").p.mean()
        ax.plot(cur.index / cs, cur.values, "-" + MK[i], color=COL[i], ms=3, label=nlab(N))
    ax.axvline(1.0, color="#52514e", lw=0.6)
    ax.set_xlabel("ON inputs $c$ / predicted quorum $c^*$")
    ax.set_ylabel("P(global response)")
    ax.legend(fontsize=6.5)
    panel(ax, r"(b) past $z_{\rm hi}$: a quorum ($\varphi=0.25$, $z=4.6$)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_beyond.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_phase_diagram()
    fig_ssb()
    fig_beyond()
    print("wrote", OUT)
