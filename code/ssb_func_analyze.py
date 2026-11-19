"""Analysis of ssb_func.py (F1 functional vs connectivity; F2 finite input density).
Writes to ../ssb_results/.   python ssb_func_analyze.py"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SSB_OUT", HERE.parent / "summaries" / "ssb"))
DATA = Path(os.environ.get("SSB_DATA", HERE.parent / "data" / "ssb"))
ALDOUS = Path(os.environ.get("ALDOUS_CSV", HERE.parent / "summaries" / "aldous" / "aldous_moments.csv"))
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
NCOL = {4000: "C0", 16000: "C2", 64000: "C3"}
KEY = {}
import os
PFX = os.environ.get("SSB_PREFIX", "")
if PFX:
    OUT = OUT / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ F1
c = pd.read_parquet(DATA / f"{PFX}conn.parquet")
c["broken_func"] = c.K_func.between(1, c.k - 1)
c["broken_conn"] = c.K_conn.between(1, c.k - 1)
f1 = c.groupby(["phi", "k", "z", "N"]).agg(
    n=("seed", "size"), q_func=("K_func", "mean"), q_conn=("K_conn", "mean"),
    P_broken_func=("broken_func", "mean"), P_broken_conn=("broken_conn", "mean"),
    A_func=("A_func", "mean"), A_conn=("A_conn", "mean"), P_ctrl_differs=("ctrl_differs", "mean"),
    conn_not_func=("n_conn_not_func", "mean"), func_not_conn=("n_func_not_conn", "mean")).reset_index()
f1["q_func"] /= f1.k
f1["q_conn"] /= f1.k
f1["A_excess"] = f1.A_func - f1.A_conn
f1.to_csv(OUT / "f1_func_vs_conn.csv", index=False)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
for c_, phi in enumerate((0.1, 0.2, 0.25)):
    for N in sorted(f1.N.unique()):
        g = f1[(f1.phi == phi) & (f1.k == 3) & (f1.N == N)].sort_values("z")
        axs[0, c_].plot(g.z, g.P_broken_func, "o-", ms=2.5, lw=1, color=NCOL.get(N, "k"), label=f"function, N={N}")
        axs[0, c_].plot(g.z, g.P_broken_conn, "x--", ms=3, lw=0.8, color=NCOL.get(N, "k"), label=f"connectivity, N={N}")
        axs[1, c_].plot(g.z, g.A_func, "o-", ms=2.5, lw=1, color=NCOL.get(N, "k"))
        axs[1, c_].plot(g.z, g.A_conn, "x--", ms=3, lw=0.8, color=NCOL.get(N, "k"))
    axs[0, c_].set_title(f"$\\varphi={phi}$, $k=3$")
    axs[1, c_].set_xlabel("$z$")
axs[0, 0].set_ylabel("P(broken)")
axs[1, 0].set_ylabel("mean anisotropy $A$")
axs[0, 0].legend(fontsize=5)
fig.tight_layout()
fig.savefig(OUT / "fig_f1_func_vs_conn.pdf")
plt.close(fig)
for phi in (0.1, 0.25):
    g = f1[(f1.phi == phi) & (f1.k == 3) & (f1.z == 3.0)].sort_values("N")
    KEY[f"F1_phi{phi}_z3_Pbroken_func"] = g.P_broken_func.round(3).tolist()
    KEY[f"F1_phi{phi}_z3_Pbroken_conn"] = g.P_broken_conn.round(3).tolist()
    KEY[f"F1_phi{phi}_max_Pdiffers"] = float(f1[(f1.phi == phi)].P_ctrl_differs.max())

# ------------------------------------------------------------------ F2
d = pd.read_parquet(DATA / f"{PFX}density.parquet")
d["has_nonsym"] = d.macro_n > d.macro_nsym
d["has_nonOR"] = d.macro_n > d.macro_nOR
f2 = d.groupby(["phi", "h", "k", "N", "z"]).agg(
    n=("seed", "size"), nonsym_share=("macro_nonsym_share", "mean"), P_nonsym=("has_nonsym", "mean"),
    P_nonOR=("has_nonOR", "mean"), macro_n=("macro_n", "mean"), macro_share=("macro_share", "mean"),
    frac_bad_pairs=("macro_frac_bad_pairs", "mean"), K=("K_func", "mean"), A=("A_func", "mean"),
    nrel=("macro_mean_nrel", "mean")).reset_index()
f2.to_csv(OUT / "f2_density.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for phi, ls in ((0.1, ":"), (0.25, "-")):
    for h, col in ((0.005, "C0"), (0.01, "C3")):
        for ax, z in zip(axs, (1.5, 2.0, 3.0)):
            g = f2[(f2.phi == phi) & np.isclose(f2.h, h) & (f2.z == z)].sort_values("N")
            ax.plot(g.N, g.nonsym_share, "o" + ls, ms=3, color=col, label=f"$\\varphi={phi}$, $h={h}$")
for ax, z in zip(axs, (1.5, 2.0, 3.0)):
    ax.set_xscale("log")
    ax.set_title(f"$z={z}$")
    ax.set_xlabel("$N$ ($k=hN$)")
axs[0].set_ylabel("share of nodes on non-symmetric\nmacroscopic functions")
axs[0].legend(fontsize=5.5)
fig.tight_layout()
fig.savefig(OUT / "fig_f2_density.pdf")
plt.close(fig)
for phi in (0.1, 0.25):
    for h in (0.005, 0.01):
        for z in (1.5, 2.0):
            g = f2[(f2.phi == phi) & np.isclose(f2.h, h) & (f2.z == z)].sort_values("N")
            KEY[f"F2_phi{phi}_h{h}_z{z}_nonsym_share_vsN"] = g.nonsym_share.round(4).tolist()
json.dump(KEY, open(OUT / "key_numbers_F.json", "w"), indent=1)
print(json.dumps(KEY, indent=1))
