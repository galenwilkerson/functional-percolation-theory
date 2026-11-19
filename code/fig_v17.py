"""Figures for paper_v17.

fig_ssb4      main text, 4 panels: (a) one grown network, (b) broken states, (c) fraction of broken networks by
              function and by connectivity against the no-fit theory, (d) finite input density.
fig_ssb_si    SM: (a) one-edge response, (b) finite-size scaling against the Aldous limit.
fig_quorum_si SM: response past z_hi against the number of ON inputs.
Data and helpers come from fig_v12 (same paths, env overrides FIG_OUT, SSB_DATA, SSB_OUT, ALDOUS_CSV).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("FIG_OUT", str(HERE.parent / "figures"))
import fig_v12 as F  # noqa: E402
from fig_v12 import plt, th, watts, COL, MK, INK, panel, nlab, DATA, RES, ALDOUS, OUT  # noqa: E402


def _prep(df):
    mc = [c for c in df.columns if c.startswith("m") and c[1:].isdigit()]
    df["M0"] = df.m0 - df[mc].mean(axis=1)
    df["K"] = (df[mc] >= 0.01).sum(axis=1)
    return df


def _pb(K):
    return float(K.between(1, 2).mean())


def p_traj(ax, t):
    g0 = t[(t.k == 3) & (t.phi == 0.1) & (t.N == 64000)]
    g = g0[g0.seed == sorted(g0.seed.unique())[0]].sort_values("z")
    for j in range(3):
        ax.plot(g.z, g[f"m{j}"], color=COL[j], marker=MK[j], ms=2.2, lw=1.1, label=f"input {j}")
    ax.plot(g.z, g.gcc, color=INK, ls=":", lw=0.9, label="giant component")
    ax.set_xlabel("mean degree $z$ (growing network)")
    ax.set_ylabel("relevance $m_i$")
    # direct labels next to each input's jump (no legend: the jumps cross any legend position)
    for j in range(3):
        zj = float(g.z[g[f"m{j}"] > 0.5].min())
        ax.text(zj + 0.08, 0.33 - 0.13 * (j % 2), f"input {j}", color=COL[j], fontsize=8, ha="left", va="center")
    ax.text(0.95, 0.42, "giant\ncomponent", color=INK, fontsize=8, ha="right", va="center")
    panel(ax, r"(a) a growing network commits to inputs ($\varphi=0.1$)")


def p_atoms(ax, s):
    for i, N in enumerate((1000, 16000, 256000)):
        x = s[(s.k == 3) & (s.phi == 0.1) & (s.z == 1.5) & (s.N == N)].M0
        ax.hist(x, bins=np.linspace(-0.45, 0.45, 91), histtype="step", density=True, color=COL[i], label=nlab(N))
    S = float(th.S_er(1.5)[0])
    for _, M, _, _ in th.mixture(S, 3):
        ax.axvline(M, color="#52514e", lw=0.5, ls=":")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 40)
    ax.set_xlabel("$M_0=m_0-\\bar m$")
    ax.set_ylabel("$P(M_0)$")
    ax.legend(fontsize=7.6, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2, borderaxespad=0.2)
    panel(ax, r"(b) broken states are sharp atoms ($z=1.5$)")


def p_func_conn(ax, s):
    """Fraction of broken networks (k = 3): by function and by connectivity at phi = 0.25, by function at
    phi = 0.1 (where the two coincide); dashed: 1 - q^3 - (1-q)^3 with q = P_t or S, no fit."""
    c = pd.read_parquet(DATA / "conn.parquet")
    c = c[(c.phi == 0.25) & (c.k == 3) & (c.N == 64000)]
    rows = [("K_func", c, COL[0], "s", r"$\varphi=0.25$, controlling inputs"),
            ("K_conn", c, COL[1], "o", r"$\varphi=0.25$, connected inputs")]
    d1 = s[(s.k == 3) & (s.phi == 0.1) & (s.N == 64000)]
    rows.append(("K", d1, COL[2], "^", r"$\varphi=0.1$ (the two coincide)"))
    for col, d, colr, mk, lab in rows:
        pb = d.groupby("z")[col].apply(_pb)
        nn = d.groupby("z")[col].size()
        se = np.sqrt(pb * (1 - pb) / nn)
        ax.errorbar(pb.index, pb.values, yerr=se.values, fmt=mk, color=colr, ms=3.5, lw=0.8, capsize=1.5,
                    mfc="white" if col == "K_conn" else colr, label=lab)
    zz = np.linspace(1.0, 3.6, 200)
    S = th.S_er(zz)
    pt = np.array([watts.trigger_prob(z, 0.25) if z > 1.114 else 0.0 for z in zz])
    ax.plot(zz, 1 - pt ** 3 - (1 - pt) ** 3, color=COL[0], ls="--", lw=0.9)
    ax.plot(zz, 1 - S ** 3 - (1 - S) ** 3, color=COL[1], ls="--", lw=0.9)
    ax.set_xlabel("$z$")
    ax.set_ylabel("fraction of broken networks, $k=3$")
    ax.axvline(1.114, color="#52514e", lw=0.6)
    ax.text(1.14, 1.12, r"$z_{\rm lo}$", fontsize=8.5, color="#52514e", va="top")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=7.6, loc="upper right", ncol=1, borderaxespad=0.2)
    panel(ax, r"(c) broken by function, not connectivity")


def p_finiteh(ax):
    gf = pd.read_csv(RES / "g_finiteh.csv")
    for i, N in enumerate((10000, 30000, 100000)):
        d = gf[(gf.phi == 0.1) & (gf.h == 0.002) & (gf.N == N)].sort_values("z")
        ax.plot(d.z, d.median_A, "-" + MK[i], color=COL[i], ms=2.5, label=nlab(N))
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 0.3)
    ax.set_xlabel("$z$")
    ax.set_ylabel("median anisotropy $A$")
    ax.legend(fontsize=7.6)
    panel(ax, r"input density $h=0.002$, $\varphi=0.1$")


def p_edge(ax, lab):
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
    ax.legend(fontsize=7.6)
    panel(ax, lab)


def p_fss(ax, lab):
    cm = pd.read_csv(RES / "scc_cells.csv")
    al = pd.read_csv(ALDOUS)
    al = al[al.dt == al.dt.min()].sort_values("lam")
    for i, N in enumerate((1000, 100000, 10000000)):
        d = cm[cm.N == N].sort_values("lam")
        ax.plot(d.lam, d.absM0 * N ** (2 / 3), "-" + MK[i], color=COL[i], ms=2.5, label=nlab(N))
    ax.plot(al.lam, 2 * (1 - 1 / 3) * al.G2, color=INK, ls=":", lw=1.2, label="Aldous limit")
    ax.set_yscale("log")
    ax.set_xlabel("$t_N=(z-1)N^{1/3}$")
    ax.set_ylabel("$\\langle|M_0|\\rangle N^{2/3}$")
    ax.legend(fontsize=7.6)
    panel(ax, lab)


def fig_ssb4():
    t = _prep(pd.read_parquet(DATA / "traj.parquet"))
    s = _prep(pd.read_parquet(DATA / "tilt.parquet"))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.8))
    p_traj(axs[0, 0], t)
    p_atoms(axs[0, 1], s)
    p_func_conn(axs[1, 0], s)
    p_finiteh(axs[1, 1])
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssb4.pdf")
    plt.close(fig)


def fig_ssb3():
    """Main text: (a) grown network, (b) broken states, (c) broken by function vs connectivity, in one row."""
    t = _prep(pd.read_parquet(DATA / "traj.parquet"))
    s = _prep(pd.read_parquet(DATA / "tilt.parquet"))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    p_traj(axs[0], t)
    p_atoms(axs[1], s)
    p_func_conn(axs[2], s)
    panel(axs[0], r"(a) commitment ($\varphi=0.1$)")
    panel(axs[1], r"(b) broken states ($z=1.5$)")
    axs[2].set_title(r"(c) controlling vs connected", loc="left", fontsize=9, color=INK)
    # (a): move the giant-component label off its curve, into the empty upper left
    for txt in list(axs[0].texts):
        if "giant" in txt.get_text():
            txt.remove()
    axs[0].text(2.05, 0.62, "giant\ncomponent\n(dotted)", fontsize=8, color=INK, ha="left", va="center")
    # (b): legend in clear headroom above the atoms
    lo, _ = axs[1].get_ylim()
    for ln in list(axs[1].lines):
        ln.remove()
    S = float(th.S_er(1.5)[0])
    for _, M, _, _ in th.mixture(S, 3):
        axs[1].vlines(M, lo, 80, color="#52514e", lw=0.5, ls=":")
    axs[1].set_ylim(lo, 3e4)
    axs[1].legend(fontsize=7.4, loc="upper center", ncol=1, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
    # (c): legend in a band above the data; z_lo label at the foot of its line
    for txt in list(axs[2].texts):
        txt.remove()
    axs[2].text(1.06, 1.2, r"$z_{\rm lo}$", fontsize=8.5, color="#52514e", ha="right", va="center")
    axs[2].set_ylim(0, 1.3)
    axs[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    h, _ = axs[2].get_legend_handles_labels()
    axs[2].legend(h, [r"controlling, $\varphi=0.25$", r"connected, $\varphi=0.25$", r"$\varphi=0.1$ (both)"],
                  fontsize=7.4, loc="upper right", borderaxespad=0.1, labelspacing=0.15, handlelength=1.2)
    axs[2].set_ylabel("fraction broken, $k=3$")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_ssb3.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_finiteh_si():
    """SM: single-input anisotropy at finite input density (zero-field observable)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    p_finiteh(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_finiteh_A_si.pdf")
    plt.close(fig)


def fig_ssb_si():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    p_edge(axs[0], r"(a) one added edge")
    p_fss(axs[1], r"(b) finite-size scaling")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ssb_si.pdf")
    plt.close(fig)


def fig_quorum_si():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    F_axes = axs
    # reuse the quorum panel of fig_beyond by drawing it into the second axis, then dropping the first
    uf = pd.read_parquet(DATA / "upperfine.parquet")
    from ssb_extra import cstar_theory
    ax = F_axes[0]
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
    ax.legend(fontsize=7.6)
    panel(ax, r"past $z_{\rm hi}$: a quorum ($\varphi=0.25$, $z=4.6$)")
    F_axes[1].remove()
    fig.tight_layout()
    fig.savefig(OUT / "fig_quorum_si.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_ssb3()
    fig_finiteh_si()
    fig_ssb_si()
    fig_quorum_si()
    print("wrote", OUT)
