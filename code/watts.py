"""Watts generalization of the fixed-k functional laws (theory.tex, section Watts).

Poisson(z) degrees, threshold tau(d) = ceil(phi d) (d >= 1; isolated nodes never fire).
Two quantities replace the giant fraction S of the OR regime:
  P_t  probability that one ON input triggers a global cascade
       = P(seed adjacent to the infinite *vulnerable* cluster);   v_d = 1[tau(d) = 1]
       u = sum_d (d p_d / z) [1 - v_d + v_d u^(d-1)],   P_t = 1 - exp(-z (1 - u))
  S_c  final active fraction of a global cascade (Gleeson-Cahalane, seed fraction -> 0+)
       q = sum_d (d p_d / z) P[Bin(d-1, q) >= tau(d)],   S_c = sum_d p_d P[Bin(d, q) >= tau(d)]
Every node of the global cascade computes the OR of the triggering inputs, so
  Psi = S_c [1 - (1 - P_t)^k],   Lam/k = S_c P_t,   A = S_c^2 P_t (1 - P_t)(1 - 1/k),
  P[DTC = j] = S_c C(k,j) P_t^j (1 - P_t)^(k-j),    Var(c_G) = k S_c^2 P_t (1 - P_t).
For phi -> 0 both reduce to S.

  python watts.py            compares with E1-C (N = 1e4) and E1-W context cells
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

HERE = Path(__file__).resolve().parent
DMAX = 200


def _deg(z):
    d = np.arange(0, DMAX + 1)
    return d, poisson.pmf(d, z)


def tau(d, phi):
    return np.ceil(phi * d - 1e-9).astype(int)


def trigger_prob(z, phi, it=20000):
    d, p = _deg(z)
    t = tau(d, phi)
    v = ((d >= 1) & (t <= 1)).astype(float)
    w = d * p / z
    u = 0.0
    for _ in range(it):
        un = np.sum(w[1:] * (1 - v[1:] + v[1:] * u ** (d[1:] - 1)))
        if abs(un - u) < 1e-14:
            break
        u = un
    return 1 - math.exp(-z * (1 - u))


def cascade_size(z, phi, rho0=1e-10, it=200000):
    d, p = _deg(z)
    t = tau(d, phi)
    w = d * p / z
    q = rho0
    for _ in range(it):
        # P[Bin(d-1, q) >= tau(d)] = sf(tau-1)
        f = binom.sf(t[1:] - 1, d[1:] - 1, q)
        qn = rho0 + (1 - rho0) * np.sum(w[1:] * f)
        if abs(qn - q) < 1e-13:
            q = qn
            break
        q = qn
    return float(np.sum(p[1:] * binom.sf(t[1:] - 1, d[1:], q))), q


def predict(z, phi, k):
    Pt = trigger_prob(z, phi)
    Sc, _ = cascade_size(z, phi)
    if Pt < 1e-9 or Sc < 1e-7:
        Pt, Sc = 0.0, 0.0
    return dict(Pt=Pt, Sc=Sc, psi_w=Sc * (1 - (1 - Pt) ** k), lamk_w=Sc * Pt,
                aniso_w=Sc ** 2 * Pt * (1 - Pt) * (1 - 1 / k) if k > 1 else 0.0,
                cvar_w=k * Sc ** 2 * Pt * (1 - Pt))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ltm_learning.closed_forms import cascade_window, giant_fraction
    out = HERE / "data" / "extra"
    out.mkdir(exist_ok=True)
    c = pd.read_csv(HERE / "data/e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[c.phi != "U01"].copy()
    rows = []
    cache = {}
    for r in c.itertuples():
        key = (r.z, r.phi)
        if key not in cache:
            cache[key] = (trigger_prob(r.z, float(r.phi)), cascade_size(r.z, float(r.phi))[0])
        Pt, Sc = cache[key]
        k = int(r.k)
        S = giant_fraction(r.z)
        w = cascade_window(float(r.phi), zmax=40.0)
        rows.append(dict(N=r.N, k=k, phi=r.phi, z=r.z, M=r.M, Pt=Pt, Sc=Sc, S=S,
                         zlo=w[0] if w else np.nan, zup=w[1] if w else np.nan,
                         psi=r.psi, psi_se=r.psi_se, lam=r.lam / k, lam_se=r.lam_se / k,
                         aniso=r.aniso, aniso_se=r.aniso_se,
                         psi_w=Sc * (1 - (1 - Pt) ** k), lamk_w=Sc * Pt,
                         aniso_w=Sc ** 2 * Pt * (1 - Pt) * (1 - 1 / k),
                         psi_or=S * (1 - (1 - S) ** k), lamk_or=S * S,
                         aniso_or=S ** 3 * (1 - S) * (1 - 1 / k)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "W_watts_vs_e1c.csv", index=False)
    # interior of the cascade window, away from both edges (finite-N rounding there)
    inn = d[(d.z >= 1.25 * d.zlo) & (d.z <= 0.8 * d.zup) & (d.N == 10000)]
    for phi, g in inn.groupby("phi"):
        e_w = (g.psi / g.psi_w - 1).abs().median()
        e_o = (g.psi / g.psi_or - 1).abs().median()
        l_w = (g.lam / g.lamk_w - 1).abs().median()
        l_o = (g.lam / g.lamk_or - 1).abs().median()
        a = g[g.k > 1]
        a_w = (a.aniso - a.aniso_w).abs().median() / a.aniso.abs().median()
        a_o = (a.aniso - a.aniso_or).abs().median() / a.aniso.abs().median()
        print(f"phi={phi}: n={len(g)}  median rel err  Psi: Watts {e_w:.3f} vs OR {e_o:.3f}   "
              f"Lam/k: Watts {l_w:.3f} vs OR {l_o:.3f}   aniso: Watts {a_w:.3f} vs OR {a_o:.3f}")
    show = inn[inn.phi.isin(["0.2", "0.25"]) & (inn.k == 5)]
    print(show[["phi", "z", "S", "Pt", "Sc", "psi", "psi_w", "psi_or", "lam", "lamk_w",
                "lamk_or", "aniso", "aniso_w", "aniso_or"]].round(4).to_string(index=False))
    # Var(c_G) generalization, E1-W context cells
    f = pd.read_csv(HERE / "data/extra/C_fluct.csv", dtype={"phi": str})
    f = f[f.phi.isin(["0.2", "0.25"]) & f.z.isin([1.5, 2.0, 3.0]) & (f.N <= 100000)].copy()
    pr = [predict(z, float(p), int(k)) for z, p, k in zip(f.z, f.phi, f.k)]
    f["cvar_w"] = [x["cvar_w"] for x in pr]
    f["ratio_w"] = f.c_var / f.cvar_w
    f["ratio_or"] = f.c_var / f.c_var_pred
    f.to_csv(out / "W_cvar.csv", index=False)
    for phi, g in f.groupby("phi"):
        print(f"Var(c_G) phi={phi}: median ratio Watts {g.ratio_w.median():.3f} "
              f"(IQR {g.ratio_w.quantile(.25):.2f}-{g.ratio_w.quantile(.75):.2f})  "
              f"OR {g.ratio_or.median():.3f}")


if __name__ == "__main__":
    main()
