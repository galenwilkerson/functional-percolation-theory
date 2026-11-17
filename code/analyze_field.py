"""E2/E4 field analysis: inputs as the ghost field h = k/N (classic SB view of functional percolation).

  python analyze_field.py --run data/e2f --tag e2f

Exact ER ghost-field theory (N -> inf at fixed h): a branch carries no source with probability
u = (1-h) exp(-z(1-u)); a non-input node computes (non-constant function) with probability
    Psi_h(z) = 1 - exp(-z(1-u))         (OR regime).
Limits: h -> 0: Psi -> S (beta = 1);  z < 1: Psi ~ z h/(1-z) (chi ~ 1/(1-z), gamma = 1);
        z = 1: Psi ~ sqrt(2h) (delta = 2).
Fixed k (h -> 0 BEFORE N -> inf): Psi = S[1-(1-S)^k] ~ k S^2 (apparent beta = 2).
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
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def psi_ghost(z: float, h: float) -> float:
    if h <= 0:
        return giant_fraction(z)
    f = lambda u: (1 - h) * math.exp(-z * (1 - u)) - u  # noqa: E731
    u = brentq(f, 0.0, 1.0 - 1e-15) if f(0.0) > 0 else 0.0
    # pick the root continuously connected to u=1-h at small z (the physical, smallest-Psi root)
    return 1 - math.exp(-z * (1 - u))


def psi_fixed_k(z, k):
    S = giant_fraction(z)
    return S * (1 - (1 - S) ** k)


def load(run):
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(run) / "batches" / "*.parquet"))],
                   ignore_index=True)
    g = df.groupby(["N", "phi", "law", "par", "k", "z"])
    cs = g.psi.agg(["mean", "std", "count"]).reset_index()
    cs["se"] = cs["std"] / np.sqrt(cs["count"])
    cs["var"] = g.psi.var().values
    cs["S"] = g.S.mean().values
    cs["h"] = cs.k / cs.N
    cs["lam_s"] = (cs.z - 1) * cs.N ** (1 / 3)
    return df, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    df, cs = load(a.run)
    print(f"{len(df)} realizations; N: {sorted(cs.N.unique())}")
    out = Path(a.run) / "analysis"
    out.mkdir(exist_ok=True)
    cs["psi_th_ghost"] = [psi_ghost(z, h) for z, h in zip(cs.z, cs.h)]
    cs["psi_th_fixedk"] = [psi_fixed_k(z, k) for z, k in zip(cs.z, cs.k)]
    cs.to_csv(out / "field_summary.csv", index=False)
    FIG.mkdir(exist_ok=True)
    phi = 0.1
    Nmax = int(cs.N.max())
    Nbig = 100000 if 100000 in set(cs.N) else Nmax

    fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))
    # (a) order of limits
    a0 = ax[0, 0]
    zz = np.linspace(0.5, 3, 300)
    a0.plot(zz, [giant_fraction(z) for z in zz], "k-", lw=2, label="S (structural)")
    for law, par, col in [("rho", 0.01, "C0"), ("rho", 0.001, "C1"), ("fixed", 5, "C3"),
                          ("fixed", 1, "C4")]:
        d = cs[(cs.N == Nbig) & (cs.phi == phi) & (cs.law == law) & (cs.par == par)].sort_values("z")
        if d.empty:
            continue
        lab = f"h = {par:g} (k = ρN)" if law == "rho" else f"k = {par:g} fixed"
        a0.errorbar(d.z, d["mean"], yerr=d.se, fmt="o", ms=2.5, color=col, label=lab)
        if law == "rho":
            a0.plot(zz, [psi_ghost(z, par) for z in zz], "-", color=col, lw=1)
        else:
            a0.plot(zz, [psi_fixed_k(z, par) for z in zz], "--", color=col, lw=1)
    a0.axvline(1, color="grey", lw=0.6)
    a0.set_xlabel("z")
    a0.set_ylabel("Ψ")
    a0.set_title(f"(a) Order of limits, N = {Nbig:.0e}: fixed h → S (β=1); fixed k → kS² (β=2)",
                 fontsize=8)
    a0.legend(fontsize=6)
    # (b) linear response below z_c
    a1 = ax[0, 1]
    d = cs[(cs.phi == phi) & (cs.z < 0.97) & (cs.law == "rho")]
    for N, mk in zip(sorted(d.N.unique()), "osD^"):
        dd = d[d.N == N]
        a1.loglog(dd.z / (1 - dd.z), dd["mean"] / dd.h, mk, ms=3, label=f"N = {N:.0e}")
    xx = np.geomspace(0.3, 40, 50)
    a1.loglog(xx, xx, "k--", lw=1, label="χ − 1 = z/(1−z)")
    a1.set_xlabel("z/(1−z)")
    a1.set_ylabel("Ψ / h")
    a1.set_title("(b) Linear response below z_c: γ = 1", fontsize=8)
    a1.legend(fontsize=6)
    # (c) critical isotherm z = 1
    a2 = ax[1, 0]
    d = cs[(cs.phi == phi) & (cs.z.round(6) == 1.0)]
    for N, mk in zip(sorted(d.N.unique()), "osD^"):
        dd = d[d.N == N].sort_values("h")
        a2.loglog(dd.h, dd["mean"], mk, ms=3, label=f"N = {N:.0e}")
    hh = np.geomspace(1e-6, 2e-2, 50)
    a2.loglog(hh, np.sqrt(2 * hh), "k--", lw=1, label="√(2h)  (δ = 2)")
    a2.set_xlabel("field h = k/N")
    a2.set_ylabel("Ψ at z = 1")
    a2.set_title("(c) Critical isotherm (finite N saturates at small h)", fontsize=8)
    a2.legend(fontsize=6)
    # (d) scaled-field FSS collapse at fixed c = k N^{-1/3}
    a3 = ax[1, 1]
    d = cs[(cs.phi == phi) & (cs.law == "scaled") & (cs.lam_s.abs() <= 4.5)]
    cols = {0.1: "C0", 1.0: "C1", 10.0: "C2"}
    for c, col in cols.items():
        for N, mk in zip(sorted(d.N.unique()), "osD^"):
            dd = d[(d.par == c) & (d.N == N)].sort_values("lam_s")
            if not dd.empty:
                a3.plot(dd.lam_s, dd["mean"] * N ** (1 / 3), mk + "-", ms=2.5, lw=0.7, color=col,
                        label=f"c={c:g}, N={N:.0e}" if N in (1000, Nmax) else None)
    a3.set_yscale("log")
    a3.set_xlabel("λ = (z−1)N^{1/3}")
    a3.set_ylabel("Ψ N^{1/3}")
    a3.set_title("(d) Scaled-field collapse, c = k N^{-1/3} = h N^{2/3}", fontsize=8)
    a3.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / f"{a.tag}_field.pdf")
    plt.close(fig)

    # numeric checks
    chk = cs[(cs.phi == phi) & (cs.law == "rho")].copy()
    chk["rel_err"] = (chk["mean"] - chk.psi_th_ghost) / chk.psi_th_ghost
    print(chk[chk.N == chk.N.max()][["N", "par", "z", "mean", "se", "psi_th_ghost", "rel_err"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
