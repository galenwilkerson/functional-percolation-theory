"""Analysis of ssb.py runs against RESEARCH_PROPOSAL_v2 (SB1-SB7).
Cuts (SPEC S9): K_ctrl uses relevance >= 1% of nodes; macroscopic functions are classes on >= 1% of nodes; use these
only well away from z_lo. Columns *_fitted_* use quantities measured on the same samples (they test independence only).  Writes figures and tables to
../ssb_results/.   python ssb_analyze.py"""
import json
from math import comb, factorial
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ssb_theory as th

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SSB_OUT", HERE.parent / "summaries" / "ssb"))
DATA = Path(os.environ.get("SSB_DATA", HERE.parent / "data" / "ssb"))
ALDOUS = Path(os.environ.get("ALDOUS_CSV", HERE.parent / "summaries" / "aldous" / "aldous_moments.csv"))
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
import os
PFX = os.environ.get("SSB_PREFIX", "")              # "smoke_" to test on the smoke runs
if PFX:
    OUT = OUT / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_parquet(DATA / f"{PFX}traj.parquet")
s = pd.read_parquet(DATA / f"{PFX}tilt.parquet")
for df in (t, s):
    df["M0"] = df.m0 - df[[c for c in df.columns if c.startswith("m") and c[1:].isdigit()]].mean(axis=1)
# M0 above uses all m_j columns present; for k=3 rows m3.. are NaN -> mean skips NaN
for df in (t, s):
    df["Mvec"] = np.sqrt(df.k * df.anis)                       # |M| in the standard representation
    mcols = [c for c in df.columns if c.startswith("m") and c[1:].isdigit()]
    df["K_ctrl"] = (df[mcols] >= 0.01).sum(axis=1)             # inputs that control a macroscopic share
    df["C_ctrl"] = df[mcols].max(axis=1)


def qC(df):
    """Measured probability an input controls the macroscopic function, and its size."""
    q = float((df.K_ctrl / df.k).mean())
    live = df[df.K_ctrl > 0]
    C = float(live.C_ctrl.mean()) if len(live) else 0.0
    return q, C
KEY = {}
NCOL = {1000: "C0", 4000: "C1", 16000: "C2", 64000: "C3", 256000: "C4"}
Z_FP = {0.1: 1.0, 0.2: 1.0, 0.25: 1.114}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


# =========================================================================== SB1 symmetric ensemble
sb1 = []
for (phi, k), g in t.groupby(["phi", "k"]):
    firsts = []
    for _, tr in g.groupby(["N", "seed"]):
        tr = tr.sort_values("z")
        hit = tr[(tr.gcc >= 0.05) & (tr.giant_nrel > 0)]
        if len(hit):
            firsts.append(hit.iloc[0].giant_rel)
    cnt = np.array([sum(int(f[j]) for f in firsts) for j in range(k)])
    p = stats.chisquare(cnt).pvalue
    sb1.append(dict(test="first-committed inputs", phi=phi, k=k, counts=" ".join(map(str, cnt)), p=p))
for (phi, k, z), g in s.groupby(["phi", "k", "z"]):
    if z not in (1.5, 2.0):
        continue
    M = np.stack([g[f"m{j}"] - g[[f"m{i}" for i in range(k)]].mean(axis=1) for j in range(k)], axis=1)
    tt = [stats.ttest_1samp(M[:, j], 0).pvalue for j in range(k)]
    sb1.append(dict(test=f"E M_i = 0 (z={z})", phi=phi, k=k, counts="", p=float(min(1.0, min(tt) * k))))
sb1 = pd.DataFrame(sb1)
sb1.to_csv(OUT / "sb1_symmetry.csv", index=False)
KEY["SB1_min_p"] = float(sb1.p.min())

# =========================================================================== SB2 two phases (anisotropy)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
rows = []
for ax, phi in zip(axs, (0.1, 0.2, 0.25)):
    g0 = t[(t.k == 3) & (t.phi == phi)]
    for N, g in g0.groupby("N"):
        med = g.groupby("z").anis.median()
        ax.plot(med.index, med, color=NCOL[N], lw=1.1, label=f"N={N}")
    if phi == 0.1:
        zz = np.linspace(1.001, 5, 300)
        S = th.S_er(zz)
        ax.plot(zz, th.anis_mean(S, 3), "k:", lw=0.9, label="theory mean")
        ax.plot(zz, [th.anis_quantile(x, 3, 0.5) for x in S], "k--", lw=0.7, label="theory median")
    ax.axvline(Z_FP[phi], color="gray", lw=0.4)
    ax.set_title(f"$\\varphi={phi}$, $k=3$")
    ax.set_xlabel("$z$")
axs[0].set_ylabel("median anisotropy $A$")
axs[0].legend(fontsize=6)
savefig(fig, "fig_sb2_anisotropy.pdf")
for (phi, k, N, z), g in t.groupby(["phi", "k", "N", "z"]):
    if any(np.isclose(z, v) for v in (0.5, 0.9, 1.5, 2.0, 3.0)):
        S = float(th.S_er(z)[0])
        er = phi == 0.1          # the ER-S closed forms apply only in simple contagion (SPEC issue 4)
        rows.append(dict(phi=phi, k=k, N=N, z=round(z, 2), median_A=g.anis.median(), mean_A=g.anis.mean(),
                         broken_frac=float((g.anis > 1e-3).mean()),
                         theory_mean=th.anis_mean(S, k) if er else np.nan,
                         theory_median=th.anis_quantile(S, k, 0.5) if er else np.nan,
                         n_rows=len(g), n_networks=int(g.seed.nunique())))
sb2 = pd.DataFrame(rows)
sb2.to_csv(OUT / "sb2_anisotropy.csv", index=False)
# restated SB2 (after review): P(A>0) = 1 - q^k - (1-q)^k, from the tilt samples (independent graphs)
rows = []
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, k in zip(axs, (3, 4, 5)):
    src = s if k in s.k.unique() else t
    for phi, mk in ((0.1, "o"), (0.25, "s")):
        for N, g0 in src[(src.k == k) & (src.phi == phi)].groupby("N"):
            zs, pb, pth = [], [], []
            for z, g in g0.groupby("z"):
                q, C = qC(g)
                zs.append(z)
                pb.append(float(g.K_ctrl.between(1, k - 1).mean()))
                pth.append(th.p_broken(q, k))
                rows.append(dict(k=k, phi=phi, N=N, z=z, P_broken=pb[-1], theory_fitted_q=pth[-1], q_measured=q, C_measured=C, n=len(g)))
            if N in (4000, 64000) or (src is t and N in (4000, 16000)):
                ax.plot(zs, pb, mk + "-", ms=2.5, lw=0.8, color=NCOL[N], label=f"$\\varphi={phi}$, N={N}")
                ax.plot(zs, pth, ":", color=NCOL[N], lw=0.8)
    ax.set_title(f"$k={k}$")
    ax.set_xlabel("$z$")
axs[0].set_ylabel("P(broken) = P(1 $\\leq$ K $\\leq$ k-1)")
axs[0].legend(fontsize=5)
savefig(fig, "fig_sb2_pbroken.pdf")
pd.DataFrame(rows).to_csv(OUT / "sb2_pbroken.csv", index=False)

# =========================================================================== SB3 broken states
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey="row")
for r, phi in enumerate((0.1, 0.25)):
    for c, z in enumerate((0.9, 1.2, 1.5, 2.0)):
        ax = axs[r, c]
        for N, g in s[(s.k == 3) & (s.phi == phi) & (s.z == z)].groupby("N"):
            ax.hist(g.M0, bins=np.linspace(-0.7, 0.7, 141), histtype="step", density=True, color=NCOL[N],
                    label=f"N={N}")
        if phi == 0.1 and z > 1:
            S = float(th.S_er(z)[0])
            for _, M, _, _ in th.mixture(S, 3):
                ax.axvline(M, color="k", lw=0.4, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"$\\varphi={phi}$, $z={z}$")
        if r == 1:
            ax.set_xlabel("$M_0$")
axs[0, 0].legend(fontsize=5.5)
axs[0, 0].set_ylabel("$P(M_0)$")
axs[1, 0].set_ylabel("$P(M_0)$")
savefig(fig, "fig_sb3_PM0.pdf")
# peak width: residual from the predicted state value S_bar (b0 - K/k), K and b0 from the giant function
rows = []
for (phi, k, z, N), g in s.groupby(["phi", "k", "z", "N"]):
    if z < 1.5:
        continue
    # predicted state value C (b0 - K/k): K, b0 = inputs controlling >= 1% of nodes, C = their mean reach
    ctrl = np.stack([g[f"m{j}"] >= 0.01 for j in range(k)], axis=1)
    b0 = ctrl[:, 0].astype(int)
    K = ctrl.sum(axis=1)
    C = qC(g)[1]
    pred = C * (b0 - K / k)
    res = g.M0 - pred
    rows.append(dict(phi=phi, k=k, z=z, N=N, width=float(res.std()), width_mad=float(stats.median_abs_deviation(res)),
                     n=len(g)))
sb3 = pd.DataFrame(rows)
sb3.to_csv(OUT / "sb3_peak_width.csv", index=False)
for phi in (0.1, 0.25):
    w = sb3[(sb3.phi == phi) & (sb3.k == 3) & (sb3.z == 2.0)].sort_values("N")
    KEY[f"SB3_width_slope_phi{phi}"] = slope(w.N, w.width_mad)

# =========================================================================== SB4 pattern of breaking
# macroscopic state = all non-constant functions on >= 1% of nodes; residual group = common stabiliser
rows = []
for (phi, k, N), g in t.groupby(["phi", "k", "N"]):
    h = g[(g.z >= 1.5) & (g.macro_n > 0)]
    if not len(h):
        continue
    pred = h.K_ctrl.apply(lambda K: factorial(K) * factorial(k - K))
    rows.append(dict(phi=phi, k=k, N=N, n_rows=len(h), n_networks=int(h.seed.nunique()), frac_top_OR=float(h.top_isOR.mean()),
                     frac_stab_eq_KkK=float((h.macro_stab == pred).mean()),
                     mean_states=float((factorial(k) / h.macro_stab).mean()),
                     frac_with_nonOR_macro=float((h.macro_nonOR > 0).mean()),
                     mean_macro_n=float(h.macro_n.mean())))
sb4 = pd.DataFrame(rows)
sb4.to_csv(OUT / "sb4_pattern.csv", index=False)
# residual-group distribution vs Bin(k,S) at phi=0.1 (tilt samples; K=0 or k -> full Sym_k)
rows = []
for (k, z, N), g in s[s.phi == 0.1].groupby(["k", "z", "N"]):
    if z < 1.5:
        continue
    S = float(th.S_er(z)[0])
    pd_ = th.stab_dist(S, k)
    emp = g.macro_stab.value_counts(normalize=True).to_dict()
    tv = 0.5 * sum(abs(emp.get(a, 0) - pd_.get(a, 0)) for a in set(emp) | set(pd_))
    rows.append(dict(k=k, z=z, N=N, TV_distance=tv, n=len(g)))
sb4b = pd.DataFrame(rows)
sb4b.to_csv(OUT / "sb4_stab_dist.csv", index=False)
# higher threshold: is the macroscopic state ever more than ORs, and what residual group then?
h = t[(t.phi >= 0.2) & (t.macro_n > 0)]
tab = h.groupby(["phi", "k", "N"]).agg(n=("macro_stab", "size"),
                                      frac_nonOR_macro=("macro_nonOR", lambda x: float((x > 0).mean())),
                                      frac_top_nonOR=("top_isOR", lambda x: float((~x.astype(bool)).mean())),
                                      mean_macro_n=("macro_n", "mean")).reset_index()
tab.to_csv(OUT / "sb4_highphi.csv", index=False)
hh = h[h.macro_nonOR > 0]
tab2 = hh.groupby(["phi", "k"]).agg(n=("macro_stab", "size"), mean_stab=("macro_stab", "mean")).reset_index()
tab2.to_csv(OUT / "sb4_highphi_nonOR_stab.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, k in zip(axs, (3, 4, 5)):
    for phi, col in ((0.1, "C0"), (0.2, "C1"), (0.25, "C3")):
        g = t[(t.k == k) & (t.phi == phi) & (t.N == 16000)]
        orb = (factorial(k) / g.macro_stab).groupby(g.z).mean()
        ax.plot(orb.index, orb, color=col, label=f"$\\varphi={phi}$")
    ax.set_title(f"$k={k}$ ($k!={factorial(k)}$)")
    ax.set_xlabel("$z$")
axs[0].set_ylabel("equivalent broken states $k!/|\\mathrm{Stab}|$")
axs[0].legend(fontsize=6)
savefig(fig, "fig_sb4_orbits.pdf")

# =========================================================================== SB5 onset, exponent, FSS
rows = []
for (phi, k, z, N), g in s.groupby(["phi", "k", "z", "N"]):
    S = float(th.S_er(z)[0])
    q, C = qC(g)
    m2, m4 = float((g.M0 ** 2).mean()), float((g.M0 ** 4).mean())
    rows.append(dict(phi=phi, k=k, z=z, N=N, mean_absM0=float(g.M0.abs().mean()),
                     theory_absM0_ER=sum(p * abs(M) for p, M, _, _ in th.mixture(S, k)) if phi == 0.1 else np.nan,
                     theory_absM0_fitted_qC=sum(p * abs(M) for p, M, _, _ in th.mixture(C, k, q)),
                     q=q, C=C, gcc=float(g.gcc.mean()), mean_Mvec=float(g.Mvec.mean()),
                     binder=1 - m4 / (3 * m2 ** 2) if m2 > 0 else np.nan,
                     median_A=float(g.anis.median()), n=len(g)))
sb5 = pd.DataFrame(rows)
sb5.to_csv(OUT / "sb5_order.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
for phi, mk in ((0.1, "o"), (0.25, "s")):
    for k, col in ((3, "C0"), (4, "C1")):
        g = sb5[(sb5.phi == phi) & (sb5.k == k) & np.isclose(sb5.z, Z_FP[phi])].sort_values("N")
        axs[0].loglog(g.N, g.mean_absM0, mk + "-", color=col, ms=3, label=f"$\\varphi={phi}$, $k={k}$")
        KEY[f"SB5_FSS_slope_phi{phi}_k{k}"] = slope(g.N, g.mean_absM0)
        KEY[f"SB5_FSS_slope_Mvec_phi{phi}_k{k}"] = slope(g.N, g.mean_Mvec)
NN = np.array([1e3, 2.56e5])
axs[0].loglog(NN, 0.015 * (NN / 1e3) ** (-2 / 3), "k:", lw=0.8, label="$N^{-2/3}$")
axs[0].set_xlabel("$N$")
axs[0].set_ylabel("$\\langle|M_0|\\rangle$ at the FP point")
axs[0].legend(fontsize=5.5)
for phi, mk in ((0.1, "o"), (0.25, "s")):
    g = sb5[(sb5.phi == phi) & (sb5.k == 3) & (sb5.N == 64000)].sort_values("z")
    axs[1].plot(g.z, g.mean_absM0, mk + "-", ms=3, label=f"$\\varphi={phi}$, $N=64000$")
zz = np.linspace(0.5, 3, 300)
S = th.S_er(zz)
axs[1].plot(zz, [sum(p * abs(M) for p, M, _, _ in th.mixture(x, 3)) for x in S], "k:", label="theory ($\\varphi\\to0$)")
axs[1].set_xlabel("$z$")
axs[1].set_ylabel("$\\langle|M_0|\\rangle$ ($k=3$)")
axs[1].legend(fontsize=6)
savefig(fig, "fig_sb5_order_fss.pdf")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, phi in zip(axs, (0.1, 0.25)):
    for N in sorted(sb5.N.unique()):
        g = sb5[(sb5.phi == phi) & (sb5.k == 3) & (sb5.N == N)].sort_values("z")
        ax.plot(g.z, g.binder, "o-", ms=2.5, lw=0.8, color=NCOL[N], label=f"N={N}")
    ax.axvline(Z_FP[phi], color="gray", lw=0.4)
    ax.set_title(f"$\\varphi={phi}$, $k=3$")
    ax.set_xlabel("$z$")
axs[0].set_ylabel("Binder cumulant of $M_0$")
axs[0].legend(fontsize=6)
savefig(fig, "fig_sb5_binder.pdf")

# =========================================================================== SB6 extensive field
rows = []
eps_list = (0.001, 0.003, 0.01, 0.03)
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
xg = np.logspace(-1, 3.5, 60)
for r, phi in enumerate((0.1, 0.25)):
    for c, z in enumerate((0.9, 1.2, 1.5, 2.0)):
        ax = axs[r, c]
        for N, g in s[(s.k == 3) & (s.phi == phi) & (s.z == z)].groupby("N"):
            ax.plot(xg, th.tilted_empirical(g.M0.values, xg), color=NCOL[N], lw=1, label=f"N={N}")
        if z > 1:
            ax.plot(xg, th.tilted_M0(float(th.S_er(z)[0]), 3, xg), "k:", lw=1, label="theory")
        ax.set_xscale("log")
        ax.set_title(f"$\\varphi={phi}$, $z={z}$")
        if r == 1:
            ax.set_xlabel("$x=\\varepsilon N$")
axs[0, 0].set_ylabel("$\\langle M_0\\rangle_\\varepsilon$")
axs[1, 0].set_ylabel("$\\langle M_0\\rangle_\\varepsilon$")
axs[0, 0].legend(fontsize=5)
savefig(fig, "fig_sb6_collapse.pdf")
for (phi, k, z, N), g in s.groupby(["phi", "k", "z", "N"]):
    S = float(th.S_er(z)[0])
    mix = th.mixture(S, k)
    p_top = sum(p for p, M, _, _ in mix if np.isclose(M, S * (1 - 1 / k)))
    rng = np.random.default_rng(0)
    for eps in eps_list:
        y = float(th.tilted_empirical(g.M0.values, eps * N)[0])
        bs = [float(th.tilted_empirical(g.M0.values[rng.integers(0, len(g), len(g))], eps * N)[0])
              for _ in range(200)]
        rows.append(dict(phi=phi, k=k, z=z, N=N, eps=eps, M0_eps=y, boot_sd=float(np.std(bs)),
                         limit=S * (1 - 1 / k) if phi == 0.1 else np.nan,
                         theory_finiteN=float(th.tilted_M0(S, k, eps * N)[0]) if phi == 0.1 else np.nan,
                         expected_top_peak_count=p_top * len(g), n=len(g)))
sb6 = pd.DataFrame(rows)
sb6.to_csv(OUT / "sb6_field.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, phi in zip(axs, (0.1, 0.25)):
    for N in sorted(s.N.unique()):
        g = sb6[(sb6.phi == phi) & (sb6.k == 3) & (sb6.N == N) & (sb6.eps == 0.01)].sort_values("z")
        ax.errorbar(g.z, g.M0_eps, yerr=g.boot_sd, fmt="o-", ms=2.5, lw=0.8, color=NCOL[N], label=f"N={N}")
    zz = np.linspace(0.5, 3, 300)
    ax.plot(zz, th.S_er(zz) * (2 / 3), "k:", label="$(1-1/k)S$")
    ax.axvline(Z_FP[phi], color="gray", lw=0.4)
    ax.set_title(f"$\\varphi={phi}$, $k=3$, $\\varepsilon=0.01$")
    ax.set_xlabel("$z$")
axs[0].set_ylabel("$\\langle M_0\\rangle_\\varepsilon$")
axs[0].legend(fontsize=5.5)
savefig(fig, "fig_sb6_order_vs_z.pdf")

# =========================================================================== SB6b quenched local field
# one extra edge from input 0 on a FIXED network; chi = N E[dm0] (nodes whose dependence on input 0 changes)
rows = []
for (phi, k, z, N), g in s.groupby(["phi", "k", "z", "N"]):
    nk = N - k                     # dm0 is a fraction of the N - k non-input nodes (SPEC issue 8)
    rows.append(dict(phi=phi, k=k, z=z, N=N, chi=float(nk * g.pert_dm0.mean()), chi_se=float(nk * g.pert_dm0.sem()),
                     dm0_over=float(g.pert_dm0.mean()), frac_flip_macro=float((g.pert_dm0.abs() > 0.01).mean()),
                     changed=float(g.pert_changed.mean()), n=len(g)))
sb6b = pd.DataFrame(rows)
sb6b.to_csv(OUT / "sb6b_quenched.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
for phi, mk in ((0.1, "o"), (0.25, "s")):
    for N in sorted(sb6b.N.unique()):
        g = sb6b[(sb6b.phi == phi) & (sb6b.k == 3) & (sb6b.N == N)].sort_values("z")
        axs[0].plot(g.z, g.chi / N, mk + "-", ms=2.5, lw=0.8, color=NCOL[N],
                    label=f"N={N}" if phi == 0.1 else None)
        axs[1].plot(g.z, g.chi, mk + "-", ms=2.5, lw=0.8, color=NCOL[N])
zz = np.linspace(1.001, 3, 200)
S = th.S_er(zz)
axs[0].plot(zz, (1 - S) * S * S, "k:", label="$(1-S)S^2$")
axs[0].set_ylabel("$\\chi/N=\\langle\\Delta m_0\\rangle$ (one edge)")
axs[1].set_yscale("log")
axs[1].set_ylabel("$\\chi=N\\langle\\Delta m_0\\rangle$")
for ax in axs:
    ax.set_xlabel("$z$")
axs[0].legend(fontsize=5.5)
savefig(fig, "fig_sb6b_quenched.pdf")
for phi in (0.1, 0.25):
    for z in (0.8, 2.0):
        g = sb6b[(sb6b.phi == phi) & (sb6b.k == 3) & np.isclose(sb6b.z, z)].sort_values("N")
        KEY[f"chi_slope_phi{phi}_z{z}"] = slope(g.N, g.chi)

# =========================================================================== SB7 re-selection
rows = []
events = []
for (N, k, phi, seed), g in t.groupby(["N", "k", "phi", "seed"]):
    g = g.sort_values("z")
    prev = None
    losses = collapses = 0
    for _, r in g.iterrows():
        cur = set(i for i in range(k) if r[f"m{i}"] >= 0.01) or None    # inputs controlling >= 1% of nodes
        if prev is not None:
            if cur is None:
                collapses += 1
            elif prev - cur:
                losses += 1
                events.append(dict(N=N, k=k, phi=phi, z=r.z))
        prev = cur if cur is not None else None
    rows.append(dict(N=N, k=k, phi=phi, seed=seed, losses=losses, collapses=collapses))
sb7 = pd.DataFrame(rows)
sb7s = sb7.groupby(["phi", "k", "N"]).agg(mean_losses=("losses", "mean"), se=("losses", "sem"),
                                         frac_any=("losses", lambda x: (x > 0).mean()),
                                         mean_collapses=("collapses", "mean"), n=("losses", "size")).reset_index()
sb7s.to_csv(OUT / "sb7_reselection.csv", index=False)
ev = pd.DataFrame(events)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
for phi, col in ((0.1, "C0"), (0.2, "C1"), (0.25, "C3")):
    for k, ls in ((3, ":"), (4, "--"), (5, "-")):
        g = sb7s[(sb7s.phi == phi) & (sb7s.k == k)].sort_values("N")
        axs[0].errorbar(g.N, g.mean_losses, yerr=g.se, fmt="o" + ls, ms=2.5, color=col, lw=0.8,
                        label=f"$\\varphi={phi}$, $k={k}$")
axs[0].set_xscale("log")
axs[0].set_xlabel("$N$")
axs[0].set_ylabel("re-selections per grown network")
axs[0].legend(fontsize=5, ncol=2)
for phi, col in ((0.2, "C1"), (0.25, "C3")):
    e = ev[(ev.phi == phi) & (ev.N >= 4000)] if len(ev) else ev
    if len(e):
        axs[1].hist(e.z, bins=np.arange(0.5, 5.1, 0.1), histtype="step", color=col, density=True,
                    label=f"$\\varphi={phi}$")
axs[1].set_xlabel("$z$ of re-selection")
axs[1].set_ylabel("density")
axs[1].legend(fontsize=6)
savefig(fig, "fig_sb7_reselection.pdf")

# =========================================================================== trajectories + function space
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
for r, phi in enumerate((0.1, 0.25)):
    g0 = t[(t.k == 3) & (t.phi == phi) & (t.N == 64000)]
    for c, seed in enumerate(sorted(g0.seed.unique())[:3]):
        g = g0[g0.seed == seed].sort_values("z")
        ax = axs[r, c]
        for j in range(3):
            ax.plot(g.z, g[f"m{j}"], lw=1.3, label=f"input {j}")
        ax.plot(g.z, g.gcc, "k:", lw=0.8, label="giant")
        ax.set_title(f"$\\varphi={phi}$, network {c + 1}")
        if c == 0:
            ax.set_ylabel("$m_i$")
        if r == 1:
            ax.set_xlabel("$z$ (edges added)")
axs[0, 0].legend(fontsize=6)
savefig(fig, "fig_traj.pdf")
rows = []
for (phi, k, N), g in t.groupby(["phi", "k", "N"]):
    allf = set()
    for c in g.census:
        allf.update(int(x.split(":")[0]) for x in c.split(";") if x)
    rows.append(dict(phi=phi, k=k, N=N, n_functions=len(allf), dedekind={3: 20, 4: 168, 5: 7581}[k], max_giant_dtc=int(g.giant_dtc.max()),
                     mean_distinct_per_net=float(g[np.isclose(g.z, 2.0)].n_nonconst.mean())))
fs = pd.DataFrame(rows)
fs.to_csv(OUT / "function_space.csv", index=False)

json.dump(KEY, open(OUT / "key_numbers.json", "w"), indent=1)
print(json.dumps(KEY, indent=1))
