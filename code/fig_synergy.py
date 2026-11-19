"""SM figure: pure-synergy share of responsive nodes, individual inputs at density h (phi = 0.25).

(a) share vs z: tree theory (solid, valid for z <= z_c) and exact simulation, N = 1e5 (points, bootstrap 1 s.e.; h = 1e-3 theory only).
(b) theory share vs h at z_c and at fixed z < z_c: tends to 1 - sqrt(1 - r) at z_c, to 0 elsewhere.

  FIG_OUT=../paper_v11/figures python fig_synergy.py
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
from synergy_theory import solve, zc_and_r  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "figures"))
D = Path(os.environ.get("FP_DATA", HERE.parent / "summaries"))
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK = "#0b0b0b"
plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "lines.linewidth": 1.2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5})


def boot(g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    r, p = g.n_resp.to_numpy(), g.n_pure.to_numpy()
    idx = rng.integers(0, len(r), (n, len(r)))
    s = p[idx].sum(1) / np.maximum(r[idx].sum(1), 1)
    return p.sum() / max(r.sum(), 1), s.std()


def main():
    zc, nu1, mu2 = zc_and_r(0.25)
    pc = 1 - math.sqrt(1 - mu2 / nu1)
    th = pd.read_csv(D / "synergy_theory" / "synergy_theory.csv")
    th = th[(th.phi == 0.25) & (th.z <= zc + 1e-6)]
    sim = pd.read_parquet(D / "individual" / "FIXHI.parquet")
    sim = sim[sim.N == 10**5]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for i, h in enumerate((1e-2, 3e-3, 1e-3)):
        t = th[th.h == h]
        ax[0].plot(t.z, t.pure, "-", color=COL[i], lw=1.3, label=f"$h={h:g}$")
        if h == 1e-3:      # only 10 graphs at N = 1e5: too noisy to show
            continue
        rows = [(z, *boot(g)) for z, g in sim[np.isclose(sim.h, h)].groupby("z")]
        z, m, e = np.array(rows).T
        ax[0].errorbar(z, m, yerr=e, fmt=MK[i], ms=3.2, lw=0.7, color=COL[i])
    t = th[th.h == 1e-6]
    ax[0].plot(t.z, t.pure, "--", color="#52514e", lw=1.5, label=r"$h=10^{-6}$ (theory)")
    ax[0].plot([zc], [pc], "*", ms=8, color=INK, label=r"$1-\sqrt{1-\kappa}$")
    ax[0].axvline(zc, color="#52514e", lw=0.8)
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel("pure-synergy share")
    ax[0].set_title(r"(a) it peaks near the onset $z_{\rm lo}$", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    hh, ll = ax[0].get_legend_handles_labels()
    hh.append(Line2D([], [], ls="", marker="o", color="#52514e", ms=3.5)); ll.append("simulation (points)")
    ax[0].legend(hh, ll, fontsize=7.5, loc="upper right")
    hs = np.logspace(-7, -2, 21)
    for i, (lab, z) in enumerate(((r"$z=z_{\rm lo}$", zc), ("$z=1.05$", 1.05), ("$z=0.9$", 0.9))):
        v = []
        for h in hs:
            qn, _, wn, _ = solve(z, 0.25, h)
            v.append(wn / qn)
        ax[1].semilogx(hs, v, MK[i] + "-", ms=3, lw=1.1, color=COL[i], label=lab)
    ax[1].axhline(pc, color="#52514e", lw=0.6, ls="--")
    ax[1].set_xlabel("input density $h$")
    ax[1].set_ylabel("pure-synergy share")
    ax[1].set_title(r"(b) as $h\to0$: 25% at $z_{\rm lo}$, zero elsewhere", loc="left", fontsize=9)
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_synergy.pdf")
    print("saved", OUT / "fig_synergy.pdf")


if __name__ == "__main__":
    main()
