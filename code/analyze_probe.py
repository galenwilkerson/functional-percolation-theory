"""E2-probe analysis: growing k (field h = k/N): complexity, scale-free functional classes, hubs.

  python analyze_probe.py --run data/e2p --tag e2p

(a) Lam/k (probability an input is relevant to a node) vs z for fixed k and fixed field h:
    OR/ER without blocking gives S^2; extensive inputs block paths (clamped-off inputs).
(b) within-graph size distribution of the finite functional classes (log2 bins, all classes
    except the condensate), at z = 1 vs off-critical, vs s^{-3/2} (size-biased clusters) and
    s^{-5/2} (all clusters input-bearing, s > 1/h).
(c) support vs complexity of finite classes (complexity = |Rel ∩ R| scaled by k/r).
(d) hub-placed vs random inputs (N = 1e4).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(str(run / "batches" / "*.parquet"))],
                   ignore_index=True)
    df["h"] = df.k / df.N
    df["zr"] = df.z.round(6)
    print(f"{len(df)} realizations")
    g = df.groupby(["N", "phi", "sel", "law", "par", "k", "zr"])
    cs = g[["psi", "lamk", "lamk_or", "aniso", "mass_max", "mass_2nd", "chi_F", "D_sig", "clam"]] \
        .mean().reset_index()
    cs["chi_F_var_psi"] = g.psi.var().values * cs.N
    (run / "analysis").mkdir(exist_ok=True)
    cs.to_csv(run / "analysis" / "probe_summary.csv", index=False)

    N = int(df.N.max())
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    # (a) Lam/k vs z
    a0 = ax[0, 0]
    zz = np.linspace(0.5, 3, 200)
    a0.plot(zz, [giant_fraction(z) ** 2 for z in zz], "k-", lw=2, label="S² (ER/OR, no blocking)")
    for (law, par), col in zip([("fixed", 5), ("scaled", 1.0), ("scaled", 10.0), ("rho", 0.001),
                                ("rho", 0.01)], ["C0", "C1", "C2", "C3", "C4"]):
        d = cs[(cs.N == N) & (cs.phi == 0.1) & (cs.sel == "random") & (cs.law == law) &
               (cs.par == par)].sort_values("zr")
        if d.empty:
            continue
        lab = {"fixed": f"k = {par:g}", "scaled": f"k = {par:g}·N^(1/3) = {int(d.k.iloc[0])}",
               "rho": f"k = {par:g}·N = {int(d.k.iloc[0])}"}[law]
        a0.plot(d.zr, d.lamk, "o-", ms=2.5, lw=1, color=col, label=lab)
    a0.axvline(1, color="grey", lw=0.6)
    a0.set_xlabel("z")
    a0.set_ylabel("Λ/k = P(input relevant to node)")
    a0.set_title(f"(a) Complexity per input Λ/k ≈ S² for every k (N = {N}, φ = 0.1)", fontsize=8)
    a0.legend(fontsize=6)
    # (b) class-size distribution within graphs
    a1 = ax[0, 1]
    for law, par, z, col, mk in [("rho", 0.01, 1.0, "C3", "o"), ("scaled", 10.0, 1.0, "C2", "s"),
                                 ("rho", 0.01, 0.5, "C0", "^"), ("rho", 0.01, 2.0, "C1", "v")]:
        d = df[(df.N == N) & (df.phi == 0.1) & (df.sel == "random") & (df.law == law) &
               (df.par == par) & (df.zr == round(z, 6))]
        if d.empty:
            continue
        H = np.stack(d.supp_hist.to_numpy()).sum(0).astype(float) / len(d)  # classes per graph per bin
        s_lo = 2.0 ** np.arange(len(H))
        dens = H / s_lo                         # per unit s (bin width = s_lo)
        ok = dens > 0
        a1.loglog(s_lo[ok] * 1.5, dens[ok], mk + "-", ms=3, lw=1, color=col,
                  label=f"z = {z:g}, k = {int(d.k.iloc[0])}")
    ss = np.geomspace(1, 1e4, 50)
    a1.loglog(ss, 30 * ss ** -1.5, "k--", lw=1, label="s^-3/2")
    a1.loglog(ss, 30 * ss ** -2.5, "k:", lw=1, label="s^-5/2")
    a1.set_xlabel("class support s (nodes computing the function)")
    a1.set_ylabel("finite classes per graph per unit s")
    a1.set_title("(b) Within-graph functional class sizes (excl. condensate)", fontsize=8)
    a1.legend(fontsize=6)
    # (c) support vs complexity of classes
    a2 = ax[1, 0]
    for z, col in [(1.0, "C3"), (0.75, "C0"), (1.5, "C1")]:
        d = df[(df.N == N) & (df.phi == 0.1) & (df.sel == "random") & (df.law == "rho") &
               (df.par == 0.01) & (df.zr == round(z, 6))]
        if d.empty:
            continue
        cnt = np.stack(d.cplx_hist.to_numpy()).sum(0).astype(float)
        sl2 = np.stack(d.supp_by_cplx.to_numpy()).sum(0)
        r = int(d.r.iloc[0])
        k = int(d.k.iloc[0])
        j = np.arange(len(cnt))
        ok = (cnt > 0) & (j > 0)   # j = 0: no relevant input in the probed subset (unresolved)
        a2.semilogy(j[ok] * k / r, 2 ** (sl2[ok] / cnt[ok]), "o-", ms=3, color=col, label=f"z = {z:g}")
    a2.set_xlabel("class complexity |Rel| (estimated: |Rel ∩ R|·k/r)")
    a2.set_ylabel("geometric-mean class support")
    a2.set_title(f"(c) Support grows with complexity (k = 0.01 N = {int(0.01 * N)})", fontsize=8)
    a2.legend(fontsize=6)
    # (d) hub vs random inputs
    a3 = ax[1, 1]
    for sel, ls in [("random", "-"), ("hub", "--")]:
        for (law, par), col in [(("fixed", 5), "C0"), (("scaled", 1.0), "C1")]:
            d = cs[(cs.N == 10000) & (cs.phi == 0.1) & (cs.sel == sel) & (cs.law == law) &
                   (cs.par == par)].sort_values("zr")
            if not d.empty:
                a3.plot(d.zr, d.psi, ls, color=col, lw=1.2, label=f"Ψ, {sel}, k = {int(d.k.iloc[0])}")
    a3.plot(zz, [giant_fraction(z) for z in zz], "k:", lw=1, label="S")
    a3.axvline(1, color="grey", lw=0.6)
    a3.set_xlabel("z")
    a3.set_ylabel("Ψ")
    a3.set_title("(d) Input selection: hubs vs random (N = 1e4, φ = 0.1)", fontsize=8)
    a3.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / f"{a.tag}_probe.pdf")
    plt.close(fig)
    with pd.option_context("display.width", 220):
        d = cs[(cs.N == N) & (cs.phi == 0.1) & (cs.sel == "random") & cs.zr.isin([0.5, 1.0, 1.5, 2.0, 3.0])]
        print(d[["law", "par", "k", "zr", "psi", "lamk", "clam", "mass_max", "chi_F", "D_sig"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
