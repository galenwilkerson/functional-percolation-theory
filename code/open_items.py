"""Checks for the four open outline items, on existing E1-W v2 cell summaries.

  §6.5  Psi_N from the full component-size distribution:
        Psi_N = E sum_j x_j [1-(1-x_j)^k]  ~=  k E sum_j x_j^2   (k x_j << 1)
        E sum_j x_j^2 = E[S^2] + E[(1-S) chi_fin]/N   (chi_fin = mean finite cluster size)
        -> Psi_N N^{2/3}/k = chi_N / N^{1/3},  chi_N = N E sum_j x_j^2 (mean cluster size of a vertex)
  §5.2  primary diversity observable H_N (constants included, per graph, then averaged):
        fixed k, z > 1:  E H_N -> [1-(1-S)^k] H2(S),  H2(S) = -S ln S - (1-S) ln(1-S)

  python open_items.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

OUT = HERE / "data" / "extra"


def main():
    w = pd.read_csv(HERE / "data/e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[w.phi.isin(["0.05", "0.1"])].copy()

    # --- §6.5: Psi from the component-size distribution, same graphs -------------
    sum_x2 = w.S ** 2 + w.S_var + (1 - w.S) * w.chi_fin / w.N
    # input's own vertex not counted (-1/N); second order in x_j (giant term dominates sum x^3)
    sum_x3 = w.S ** 3 + 3 * w.S * w.S_var
    w["psi_lin"] = w.k * sum_x2
    w["psi_pred"] = w.k * (sum_x2 - 1 / w.N) - w.k * (w.k - 1) / 2 * sum_x3
    w["chiN_sc"] = w.N * sum_x2 / w.N ** (1 / 3)
    win = w[(w.lam_s.abs() <= 3.01)].copy()
    win["r"] = win.psi / win.psi_pred
    win["dev_se"] = (win.psi - win.psi_pred) / win.psi_se
    win.to_csv(OUT / "O_psi_window.csv", index=False)
    print("§6.5  Psi / [k E sum x_j^2], window |lam|<=3, phi<=0.1")
    for N, g in win.groupby("N"):
        sm = g[g.k * g.S < 0.3]
        print(f"  N={N:>7}: kS<0.3 cells={len(sm):3d}  median ratio {sm.r.median():.3f} "
              f"(IQR {sm.r.quantile(.25):.3f}-{sm.r.quantile(.75):.3f})  "
              f"|dev|<2se {np.mean(sm.dev_se.abs() < 2):.2f}")
    # collapse of the scaling function chi_N / N^{1/3} vs lambda
    c = win[(win.k == 5) & (win.phi == "0.1")].pivot_table(index="lam_s", columns="N",
                                                            values="chiN_sc")
    print("  chi_N/N^{1/3} at k=5 (rows lam):")
    print(c.loc[[l for l in (-3, -2, -1, 0, 1, 2, 3) if l in c.index]].round(3).to_string())

    # --- §5.2: H_N including constants -------------------------------------------
    up = w[(w.z >= 1.5) & (w.lam_s.isna() | (w.z >= 1.5))].copy()
    up["Sinf"] = [giant_fraction(z) for z in up.z]
    H2 = -up.Sinf * np.log(up.Sinf) - (1 - up.Sinf) * np.log(1 - up.Sinf)
    up["H_pred"] = (1 - (1 - up.Sinf) ** up.k) * H2
    up["r"] = up.H / up.H_pred
    up.to_csv(OUT / "O_H.csv", index=False)
    print("\n§5.2  H_N / ([1-(1-S)^k] H2(S)), z>=1.5, phi<=0.1")
    for N, g in up.groupby("N"):
        print(f"  N={N:>7}: cells={len(g):3d}  median {g.r.median():.4f} "
              f"(IQR {g.r.quantile(.25):.4f}-{g.r.quantile(.75):.4f})")
    sub = w[w.z <= 0.8]
    print("  z<=0.8: N * H_N / k (finite-class entropy per input) median by N:",
          sub.assign(x=sub.H * sub.N / sub.k).groupby("N").x.median().round(2).to_dict())
    for nm in ("D_all", "D_nc"):
        t = w[(w.k == 5) & (w.phi == "0.1") & w.z.isin([0.5, 1.0, 2.0])]
        print(f"  {nm} at k=5 by (z, N):",
              t.pivot_table(index="z", columns="N", values=nm).round(2).to_dict("index"))


if __name__ == "__main__":
    main()
