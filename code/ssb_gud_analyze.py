"""Analysis of G (finite input density), U1 (upper edge, individual inputs) and D4 (lattice).
Writes figures/tables to ../ssb_results/.   python ssb_gud_analyze.py"""
import json
import os
from math import ceil, comb, factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

import ssb_theory as th
import watts

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SSB_OUT", HERE.parent / "summaries" / "ssb"))
DATA = Path(os.environ.get("SSB_DATA", HERE.parent / "data" / "ssb"))
ALDOUS = Path(os.environ.get("ALDOUS_CSV", HERE.parent / "summaries" / "aldous" / "aldous_moments.csv"))
OUT.mkdir(parents=True, exist_ok=True)
PFX = os.environ.get("SSB_PREFIX", "")
if PFX:
    OUT = OUT / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
KEY = {}


def seeded_response(z, phi, rho0, blocked, dmax=80):
    """Gleeson-Cahalane tree theory with ON seeds (fraction rho0) and blocked OFF inputs (fraction blocked):
    final active fraction of the other nodes."""
    d = np.arange(dmax + 1)
    pd_ = poisson.pmf(d, z)
    need = np.array([ceil(phi * x - 1e-9) if x > 0 else 10 ** 9 for x in d])
    q = rho0
    for _ in range(5000):
        s = 0.0
        for dd in range(1, dmax + 1):
            m = np.arange(need[dd], dd)
            if len(m):
                s += dd * pd_[dd] / z * sum(comb(dd - 1, int(mm)) * q ** mm * (1 - q) ** (dd - 1 - mm) for mm in m)
        qn = rho0 + (1 - rho0 - blocked) * s
        if abs(qn - q) < 1e-12:
            break
        q = qn
    f = 0.0
    for dd in range(1, dmax + 1):
        m = np.arange(need[dd], dd + 1)
        f += pd_[dd] * sum(comb(dd, int(mm)) * q ** mm * (1 - q) ** (dd - mm) for mm in m)
    return f


# =============================================================================== G
g = pd.read_parquet(DATA / f"{PFX}finiteh.parquet")
rows = []
for (phi, h, N, z), s in g.groupby(["phi", "h", "N", "z"]):
    qth = float(th.S_er(z)[0]) if phi == 0.1 else float(watts.trigger_prob(z, phi))
    rows.append(dict(phi=phi, h=h, N=N, z=z, n=len(s), median_A=float(s.A.median()), mean_A=float(s.A.mean()),
                     cv_A=float(s.A.std() / s.A.mean()) if s.A.mean() > 0 else np.nan,
                     P_broken=float(s.broken.mean()), q=float(s.q.mean()), q_theory=qth, C=float(s.C.mean()),
                     psi_allon=float(s.psi_allon.mean()), gcc=float(s.gcc.mean())))
G = pd.DataFrame(rows)
G.to_csv(OUT / "g_finiteh.csv", index=False)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
cols = {10000: "C0", 30000: "C2", 100000: "C3"}
for r, phi in enumerate((0.1, 0.25)):
    for (h, N), s in G[G.phi == phi].groupby(["h", "N"]):
        s = s.sort_values("z")
        ls = "-" if h == 0.002 else "--"
        lab = f"h={h}, N={N}"
        axs[r, 0].plot(s.z, s.median_A, "o" + ls, ms=2.5, lw=0.8, color=cols.get(N, "k"), label=lab)
        axs[r, 1].plot(s.z, s.P_broken, "o" + ls, ms=2.5, lw=0.8, color=cols.get(N, "k"))
        axs[r, 2].plot(s.z, s.psi_allon, "o" + ls, ms=2.5, lw=0.8, color=cols.get(N, "k"))
    axs[r, 0].set_yscale("symlog", linthresh=1e-5)
    axs[r, 0].set_ylabel(f"$\\varphi={phi}$\nmedian anisotropy $A$")
    axs[r, 1].set_ylabel("P(broken)")
    axs[r, 2].set_ylabel("$\\Psi$ (all inputs on)")
    for ax in axs[r]:
        ax.axvline(1.0 if phi == 0.1 else 1.114, color="gray", lw=0.4)
for ax in axs[1]:
    ax.set_xlabel("$z$")
axs[0, 0].legend(fontsize=5)
fig.suptitle("G: finite input density. The symmetry-breaking order parameter stays sharp at the onset; the "
             "responsive fraction is rounded by the field", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_g_finiteh.pdf")
plt.close(fig)
for phi in (0.1, 0.25):
    zb = 0.8 if phi == 0.1 else 0.9
    za = 2.0
    for h in (0.002, 0.01):
        s = G[(G.phi == phi) & (G.h == h)]
        KEY[f"G_phi{phi}_h{h}_medA_z{zb}_vsN"] = s[np.isclose(s.z, zb)].sort_values("N").median_A.tolist()
        KEY[f"G_phi{phi}_h{h}_cvA_z{za}_vsN"] = s[np.isclose(s.z, za)].sort_values("N").cv_A.round(3).tolist()
        KEY[f"G_phi{phi}_h{h}_Pbroken_z{za}_vsN"] = s[np.isclose(s.z, za)].sort_values("N").P_broken.round(3).tolist()

# =============================================================================== U1
u = pd.read_parquet(DATA / f"{PFX}upper.parquet")
recs = []
for _, r in u.iterrows():
    cnt = np.asarray(r.counts)
    resp = np.asarray(r.resp)
    glob = (resp >= 0.5 * r.resp_allon) & (r.resp_allon >= 0.1)
    for c in np.unique(cnt):
        sel = glob[cnt == c]
        recs.append(dict(N=r.N, h=r.h, k=r.k, phi=r.phi, z=r.z, seed=r.seed, c=int(c), frac=c / r.k,
                         p_global=float(sel.mean()), disagree=bool(0 < sel.mean() < 1), allon=r.resp_allon))
U = pd.DataFrame(recs)
cells = []
for (phi, h, N, z), s in U.groupby(["phi", "h", "N", "z"]):
    curve = s.groupby("c").agg(p=("p_global", "mean"), dis=("disagree", "mean"), frac=("frac", "first")).reset_index()
    k = int(s.k.iloc[0])
    pt = float(watts.trigger_prob(z, phi))
    # OR-of-trigger-set prediction, and the quorum point from seeded tree theory
    curve["p_OR"] = 1 - (1 - pt) ** curve.c
    allon_th = seeded_response(z, phi, h, 0.0)
    cstar = np.nan
    if allon_th > 0.1:
        for c in curve.c:
            if seeded_response(z, phi, c / N, (k - c) / N) >= 0.5 * allon_th:
                cstar = c
                break
    # empirical quorum point and width (in c/k) of the 10-90% rise
    def cross(level):
        above = curve[curve.p >= level]
        return float(above.c.min()) if len(above) else np.nan
    c10, c50, c90 = cross(0.1), cross(0.5), cross(0.9)
    cells.append(dict(phi=phi, h=h, N=N, z=z, k=k, P_t=pt, allon=float(s.allon.mean()), allon_theory=allon_th,
                      c50=c50, c_star_theory=cstar, width_rel=(c90 - c10) / c50 if c50 == c50 and c50 > 0 else np.nan,
                      p_single=float(curve.p.iloc[0]), max_disagree=float(curve.dis.max()),
                      rms_OR=float(np.sqrt(((curve.p - curve.p_OR) ** 2).mean()))))
UC = pd.DataFrame(cells)
UC.to_csv(OUT / "u1_upper.csv", index=False)
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0), sharey=True)
for r, phi in enumerate((0.25, 0.2)):
    zsel = sorted(UC[UC.phi == phi].z.unique())
    if not zsel:
        continue
    zshow = [zsel[i] for i in np.unique(np.linspace(0, len(zsel) - 1, 4).round().astype(int))]
    for c_, z in enumerate(zshow):
        ax = axs[r, c_]
        for (h, N), s in U[(U.phi == phi) & (U.z == z)].groupby(["h", "N"]):
            cur = s.groupby("frac").p_global.mean()
            ax.plot(cur.index, cur.values, "o-" if N == 100000 else "o--", ms=2, lw=0.8,
                    color="C0" if h == 0.005 else "C3", label=f"h={h}, N={N}")
        ax.set_xscale("log")
        ax.set_title(f"$\\varphi={phi}$, $z={z}$", fontsize=7)
        if r == 1:
            ax.set_xlabel("fraction of inputs on, $c/k$")
    axs[r, 0].set_ylabel("P(global response)")
axs[0, 0].legend(fontsize=5)
fig.suptitle("U1: inside the window any trigger input suffices (OR); past $z_{hi}$ the network fires only above a "
             "count of ON inputs", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_u1_upper.pdf")
plt.close(fig)

# =============================================================================== D4
L = pd.read_parquet(DATA / f"{PFX}lattice.parquet")
mcols = ["m0", "m1", "m2"]
L["K"] = (L[mcols] >= 0.01).sum(axis=1)
L["broken"] = L.K.between(1, 2)
L["stab_ok"] = L.macro_stab == L.K.apply(lambda K: factorial(K) * factorial(3 - K))
rows = []
for (Ls, p), s in L.groupby(["L", "p"]):
    P = float(s.gcc.mean())
    rows.append(dict(L=Ls, p=p, n=len(s), P_inf=P, P_broken=float(s.broken.mean()), theory=1 - P ** 3 - (1 - P) ** 3,
                     median_A=float(s.anis.median()), stab_ok=float(s.stab_ok[s.macro_n > 0].mean()) if (s.macro_n > 0).any() else np.nan))
DL = pd.DataFrame(rows)
DL.to_csv(OUT / "d4_lattice.csv", index=False)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for Ls, s in DL.groupby("L"):
    s = s.sort_values("p")
    ax.plot(s.p, s.P_broken, "o-", ms=2.5, lw=0.8, label=f"L={Ls}")
    ax.plot(s.p, s.theory, ":", lw=0.8, color=ax.lines[-1].get_color())
ax.axvline(0.5, color="gray", lw=0.4)
ax.set_xlabel("bond probability $p$")
ax.set_ylabel("P(broken), $k=3$")
ax.legend(fontsize=6)
ax.set_title("D4: square lattice; dotted $1-P_\\infty^3-(1-P_\\infty)^3$", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig_d4_lattice.pdf")
plt.close(fig)
json.dump(KEY, open(OUT / "key_numbers_GUD.json", "w"), indent=1)
print(json.dumps(KEY, indent=1))
