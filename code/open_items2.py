"""Remaining outline gaps, on existing E1-W v2 data.

  1  §7.1/§10.5  Var(h_G): above z_c, h_G = 1[K>=1] H2(S), K ~ Bin(k, S)
                 -> Var(h_G) = H2(S)^2 (1-S)^k [1-(1-S)^k]
  2  §4.6        finite phi: onset at z_lo(phi) = percolation of the vulnerable subgraph,
                 sum_d v_d d(d-1) p_d / z = 1. FSS collapse in (z - z_lo) N^{1/3} at phi = 0.25.
  3  §5.6        diversity beyond OR: H -> [1-(1-P_t)^k] H2(S_c)  (P_t trigger prob., S_c cascade size)
  4  §7.2        placement-conditioned susceptibility chi_C = E_K Var(c_G | K): peak location,
                 height, width vs N.

  python open_items2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402
from watts import cascade_size, trigger_prob  # noqa: E402  (vulnerable-cluster fixed points of this model)

OUT = HERE / "data" / "extra"


def H2(s):
    s = np.clip(s, 1e-300, 1 - 1e-16)
    return -s * np.log(s) - (1 - s) * np.log(1 - s)


def z_lo(phi):
    def f(z):
        d = np.arange(1, 200)
        p = poisson.pmf(d, z)
        v = (d <= 1 / phi + 1e-9)
        return (v * d * (d - 1) * p).sum() / z - 1
    return brentq(f, 0.5, 3.0)


def main():
    w = pd.read_csv(HERE / "data/e1w/analysis/window_summary.csv", dtype={"phi": str})
    pd.set_option("display.width", 200)

    # ---- 1. Var(h_G) --------------------------------------------------------------
    a = w[w.phi.isin(["0.05", "0.1"]) & (w.z >= 1.5)].copy()
    S = np.array([giant_fraction(z) for z in a.z])
    a["Hvar_pred"] = H2(S) ** 2 * (1 - S) ** a.k * (1 - (1 - S) ** a.k)
    a["r"] = a.H_var / a.Hvar_pred
    a.to_csv(OUT / "O2_hvar.csv", index=False)
    print("1  Var(h_G) / H2(S)^2 (1-S)^k [1-(1-S)^k], z>=1.5, phi<=0.1, k>=2")
    for N, g in a[a.k >= 2].groupby("N"):
        print(f"   N={N:>7}: cells={len(g):3d} median {g.r.median():.3f} "
              f"(IQR {g.r.quantile(.25):.3f}-{g.r.quantile(.75):.3f})")
    pk = (w[(w.phi == "0.1") & (w.k == 5)].sort_values("z").groupby("N")
          .apply(lambda g: g.loc[g.H_var.idxmax(), ["z", "H_var"]]))
    print("   raw Var(h_G) peak (k=5, phi=0.1):", pk.round(4).to_dict("index"))

    # ---- 2. finite-phi onset ---------------------------------------------------------
    print("\n2  vulnerable-subgraph threshold z_lo(phi):",
          {p: round(z_lo(p), 4) for p in (0.1, 0.125, 0.15, 0.2, 0.25)})
    zl = z_lo(0.25)
    b = w[(w.phi == "0.25") & (w.k == 5) & (w.z > 0.9) & (w.z < 1.25)].copy()
    b["lp"] = (b.z - zl) * b.N ** (1 / 3)
    b["Ssc"] = b.S_vul * b.N ** (1 / 3)
    b["Lsc"] = b.lam / b.k * b.N ** (2 / 3)
    b["Psc"] = b.psi * b.N ** (2 / 3) / b.k
    b.to_csv(OUT / "O2_zlo.csv", index=False)
    print(f"   phi=0.25, k=5: scaled vs lambda' = (z - {zl:.4f}) N^(1/3)")
    for N, g in b[b.N >= 10000].groupby("N"):
        g = g[(g.lp > -3.5) & (g.lp < 4.5)].sort_values("lp")
        print(f"   N={N}: " + "  ".join(f"[{r.lp:+.2f}: S_vul {r.Ssc:.2f}, Lam/k {r.Lsc:.2f}, Psi {r.Psc:.2f}]"
                                       for r in g.itertuples()))
    # same metrics at z_c, phi=0.1, for comparison of scaling functions
    c = w[(w.phi == "0.1") & (w.k == 5) & (w.N == 1000000)].copy()
    c["lp"] = (c.z - 1) * c.N ** (1 / 3)
    c = c[(c.lp > -3.5) & (c.lp < 4.5)].sort_values("lp")
    print("   phi=0.1 at z_c, N=1e6: " + "  ".join(
        f"[{r.lp:+.2f}: S {r.S * r.N ** (1 / 3):.2f}, Lam/k {r.lam / r.k * r.N ** (2 / 3):.2f}]"
        for r in c.itertuples()))
    # mean-field slopes just above z_lo from the tree-level fixed points
    for dz in (0.01, 0.02, 0.04):
        z = zl + dz
        print(f"   z_lo+{dz}: P_t/dz = {trigger_prob(z, 0.25) / dz:.3f}, "
              f"S_c/dz = {cascade_size(z, 0.25)[0] / dz:.3f}")

    # ---- 3. diversity beyond OR --------------------------------------------------------
    d = w[w.phi.isin(["0.2", "0.25"]) & w.z.isin([1.2, 1.3, 1.4, 1.5, 2.0, 3.0])].copy()
    cache = {}
    for z, p in set(zip(d.z, d.phi)):
        cache[(z, p)] = (trigger_prob(z, float(p)), cascade_size(z, float(p))[0])
    d["Pt"] = [cache[(z, p)][0] for z, p in zip(d.z, d.phi)]
    d["Sc"] = [cache[(z, p)][1] for z, p in zip(d.z, d.phi)]
    d["H_pred"] = (1 - (1 - d.Pt) ** d.k) * H2(d.Sc)
    Sor = np.array([giant_fraction(z) for z in d.z])
    d["H_or"] = (1 - (1 - Sor) ** d.k) * H2(Sor)
    d.to_csv(OUT / "O2_H_finitephi.csv", index=False)
    print("\n3  H_N beyond OR (phi=0.2, 0.25): measured / [1-(1-P_t)^k] H2(S_c)  (vs OR formula)")
    for (p, N), g in d.groupby(["phi", "N"]):
        print(f"   phi={p} N={N:>7}: cells={len(g):2d} median {np.median(g.H / g.H_pred):.3f}"
              f"  (OR formula {np.median(g.H / g.H_or):.3f})")

    # ---- 4. placement-conditioned susceptibility --------------------------------------
    e = pd.read_parquet(HERE / "data/extra/e1w_compact.parquet",
                        columns=["N", "k", "z", "phi", "dtc", "k_in_gcc"])
    e = e[(e.phi == "0.1") & (e.k == 5)]
    g = e.groupby(["N", "z", "k_in_gcc"]).dtc.agg(["var", "count"]).reset_index()
    g = g[g["count"] >= 5]
    g["wv"] = g["var"] * g["count"]
    s = g.groupby(["N", "z"]).agg(wv=("wv", "sum"), n=("count", "sum")).reset_index()
    s["chiC"] = s.wv / s.n
    raw = e.groupby(["N", "z"]).dtc.var().rename("rawvar").reset_index()
    s = s.merge(raw, on=["N", "z"])
    s.to_csv(OUT / "O2_chiC.csv", index=False)
    print("\n4  chi_C = E_K Var(c_G | K), k=5, phi=0.1")
    rows = []
    for N, t in s.groupby("N"):
        t = t.sort_values("z")
        t = t[(t.z > 0.8) & (t.z < 1.5)]
        i = t.chiC.idxmax()
        zp, hp = t.loc[i, "z"], t.loc[i, "chiC"]
        above = t[t.chiC >= hp / 2]
        width = above.z.max() - above.z.min()
        rows.append(dict(N=N, z_peak=zp, lam_peak=(zp - 1) * N ** (1 / 3), height=hp,
                         height_N43=hp * N ** (4 / 3), fwhm=width, fwhm_N13=width * N ** (1 / 3)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "O2_chiC_peaks.csv", index=False)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    lN = np.log(r.N)
    print(f"   fitted: height ~ N^{np.polyfit(lN, np.log(r.height), 1)[0]:.3f}, "
          f"fwhm ~ N^{np.polyfit(lN, np.log(r.fwhm), 1)[0]:.3f}, "
          f"|z_peak-1| ~ N^{np.polyfit(lN, np.log(abs(r.z_peak - 1)), 1)[0]:.3f}")


if __name__ == "__main__":
    main()
