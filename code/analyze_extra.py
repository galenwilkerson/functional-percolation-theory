"""Analysis-only items left in the outline (no new simulation).

  python analyze_extra.py            (reads data/e1c, data/e1w, data/e2p; ~minutes, 12 procs)

  A  §4.2  subcritical bound  Psi <= k/((N-1)(1-z)),  C <= k Psi           (E1-C, E1-W, z < 1)
  B  §6.4  kS crossover       Psi/S = 1-(1-S)^k ~ 1-exp(-kS)                 (E1-W, E1-C, probe)
  C  §7    sample fluctuations of c_G (mean DTC) and h_G (H_nc): which a_N gives a finite
           susceptibility; peak position vs N                                (E1-W)
  D  §5.6  function-space class statistics in the window: chi_F, mass_2nd / mass_max,
           finite-class size law n(s) from supp_hist vs lambda               (E1-W)
  E  §5.1  rank-frequency of functions across the window                     (E1-W ensembles)
  F  §8.4  conditioning on the symmetry-breaking event "input 1 in the GCC"   (E1-W)
  G  RQ4   onset of non-OR functions (rho_or < 1, DTC < |Rel|) vs phi         (E1-C, E1-W)

Writes data/extra/*.csv and ../figures/x_*.pdf.
"""
from __future__ import annotations

import glob
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

DATA = HERE / "data"
OUT = DATA / "extra"
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ZLO = {"0.05": 1.0, "0.1": 1.0, "0.2": 1.0, "0.25": 1.114142, "U01": 1.0}
ZUP = {"0.05": 27.5186, "0.1": 13.6593, "0.15": 7.4771, "0.2": 5.7647, "0.25": 3.8631}
SCOLS = ["N", "k", "z", "phi", "rep", "S", "S2", "psi", "dtc", "lam", "H_nc", "D_nc", "chi_F",
         "mass_max", "mass_2nd", "rho_or", "n_resp", "n_dtc_lt_rel", "k_in_gcc"]


def phi_lab(p):
    return p if p == "U01" else f"{float(p):g}"


# ------------------------------------------------------------------ E1-W compact loader
def _read_one(f):
    d = pd.read_parquet(f, columns=SCOLS + ["m", "in_gccbit", "supp_hist"])
    d["m0"] = d["m"].map(lambda a: float(a[0]))
    d["g0"] = d["in_gccbit"].map(lambda a: int(a[0]))
    d["phi"] = d.phi.map(phi_lab)
    sh = {}
    for key, g in d.groupby(["N", "k", "phi", "z"]):
        sh[key] = np.stack(g["supp_hist"].to_numpy()).sum(0)
    return d.drop(columns=["m", "in_gccbit", "supp_hist"]), sh


def load_w():
    cache = OUT / "e1w_compact.parquet"
    shc = OUT / "e1w_supp_hist.csv"
    if cache.exists() and shc.exists():
        return pd.read_parquet(cache), pd.read_csv(shc)
    files = sorted(glob.glob(str(DATA / "e1w" / "batches" / "*.parquet")))
    with Pool(12) as p:
        res = p.map(_read_one, files, chunksize=40)
    df = pd.concat([r[0] for r in res], ignore_index=True)
    agg = {}
    for _, sh in res:
        for key, h in sh.items():
            agg[key] = agg.get(key, 0) + h
    rows = [dict(N=k[0], k=k[1], phi=k[2], z=k[3], **{f"b{i}": int(x) for i, x in enumerate(h)})
            for k, h in agg.items()]
    shd = pd.DataFrame(rows)
    df.to_parquet(cache)
    shd.to_csv(shc, index=False)
    return df, shd


def lam_of(z, N, phi):
    return (z / ZLO[phi] - 1.0) * N ** (1 / 3)


# ------------------------------------------------------------------ A  §4.2 bound
def item_A():
    rows = []
    for tag, f in (("e1c", "e1c/analysis/cells_summary.csv"),
                   ("e1w", "e1w/analysis/window_summary.csv")):
        c = pd.read_csv(DATA / f, dtype={"phi": str})
        c = c[c.z < 1].copy()
        c["bound"] = np.minimum(1, c.k / ((c.N - 1) * (1 - c.z)))
        c["r_psi"] = c.psi / c.bound
        c["r_psi_hi"] = (c.psi + 2 * c.psi_se) / c.bound
        c["r_C"] = np.where(c.psi > 0, c.dtc / (c.k * c.psi), np.nan)
        c["ER_finite"] = c.k * c.z / ((1 - c.z) * (c.N - c.k))     # sharper ER (finite-cluster)
        c["r_ER"] = c.psi / c.ER_finite
        c["src"] = tag
        rows.append(c[["src", "N", "k", "phi", "z", "M", "psi", "psi_se", "bound", "r_psi",
                       "r_psi_hi", "dtc", "r_C", "ER_finite", "r_ER"]])
    a = pd.concat(rows)
    a.to_csv(OUT / "A_subcritical_bound.csv", index=False)
    lo = a[a.z <= 0.75]
    print("[A] subcritical cells:", len(a), " max Psi/bound =", round(a.r_psi.max(), 3),
          " max C/(k Psi) =", round(a.r_C.max(), 3),
          " cells with mean above bound:", int((a.r_psi > 1).sum()),
          " of which > 2se above:", int(((a.psi - 2 * a.psi_se) > a.bound).sum()))
    print("    z<=0.75, phi<=0.1: Psi / ER finite-cluster law  median",
          round(lo[lo.phi.isin(['0.05', '0.1'])].r_ER.median(), 3),
          " IQR", lo[lo.phi.isin(['0.05', '0.1'])].r_ER.quantile([.25, .75]).round(3).tolist())
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for src, mk in (("e1c", "o"), ("e1w", "s")):
        d = a[a.src == src]
        axes[0].scatter(1 - d.z, d.r_psi, s=6, marker=mk, c=np.log10(d.N), cmap="viridis",
                        label=src.upper())
        axes[1].scatter(1 - d.z, d.r_C, s=6, marker=mk, c=np.log10(d.N), cmap="viridis")
    axes[0].axhline(1, color="r", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("1 − z")
    axes[0].set_ylabel(r"$\Psi_N\,/\,\min\{1,k/((N-1)(1-z))\}$")
    axes[0].legend(fontsize=7)
    axes[1].axhline(1, color="r", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("1 − z")
    axes[1].set_ylabel(r"$C_N\,/\,(k\Psi_N)$")
    fig.suptitle("§4.2 subcritical bounds (all z < 1 cells; colour = log10 N). "
                 "Red line = bound.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_A_bound.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ B  §6.4 kS crossover
def item_B(cw):
    pts = []
    d = cw[(cw.z >= 1.0) & (cw.S > 0)].copy()
    d["kS"] = d.k * d.S
    d["ratio"] = d.psi / d.S
    d["src"] = "E1-W"
    pts.append(d[["src", "N", "k", "phi", "z", "S", "kS", "psi", "ratio"]])
    c = pd.read_csv(DATA / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c = c[(c.z >= 1.0) & (c.S > 0)].copy()
    c["kS"], c["ratio"], c["src"] = c.k * c.S, c.psi / c.S, "E1-C"
    pts.append(c[["src", "N", "k", "phi", "z", "S", "kS", "psi", "ratio"]])
    pr = pd.concat([pd.read_parquet(f, columns=["N", "k", "z", "phi", "sel", "psi"])
                    for f in glob.glob(str(DATA / "e2p" / "batches" / "*.parquet"))])
    pr = pr[pr.sel == "random"].groupby(["N", "k", "phi", "z"], as_index=False).psi.mean()
    pr["phi"] = pr.phi.map(lambda p: f"{p:g}")
    pr["S"] = pr.z.map(giant_fraction)
    pr = pr[(pr.z >= 1.2) & (pr.S > 0)]
    pr["kS"], pr["ratio"], pr["src"] = pr.k * pr.S, pr.psi / pr.S, "probe"
    pts.append(pr[["src", "N", "k", "phi", "z", "S", "kS", "psi", "ratio"]])
    b = pd.concat(pts, ignore_index=True)
    b = b[b.z <= b.phi.map(ZUP).fillna(np.inf)]   # inside the cascade window only
    b["pred"] = 1 - (1 - b.S) ** b.k
    b["pred_exp"] = 1 - np.exp(-b.kS)
    b["resid"] = b.ratio - b.pred
    b.to_csv(OUT / "B_kS_crossover.csv", index=False)
    orr = b[b.phi.isin(["0.05", "0.1"])]
    big = orr[orr.N >= 10000]
    print("[B] kS crossover, phi<=0.1, N>=1e4: n =", len(big), " median |Psi/S - pred| =",
          round(big.resid.abs().median(), 4), " 90% =", round(big.resid.abs().quantile(.9), 4))
    for (lo, hi) in ((0, .3), (.3, 3), (3, 1e9)):
        s = big[(big.kS >= lo) & (big.kS < hi)]
        print(f"    kS in [{lo},{hi}): n={len(s)}  median Psi/(kS^2) = "
              f"{(s.psi / (s.k * s.S ** 2)).median():.3f}   median Psi/S = {s.ratio.median():.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    xx = np.geomspace(1e-3, 1e3, 300)
    for (src, mk), col in zip((("E1-W", "o"), ("E1-C", "s"), ("probe", "^")), ("C0", "C1", "C2")):
        for phi, a_ in (("0.1", 0.8), ("0.2", 0.35)):
            s = b[(b.src == src) & (b.phi == phi) & (b.N >= 10000)]
            lab = f"{src}, φ={phi}"
            axes[0].scatter(s.kS, s.ratio, s=7, marker=mk, color=col, alpha=a_, label=lab)
            axes[1].scatter(s.kS, s.psi / (s.k * s.S ** 2), s=7, marker=mk, color=col, alpha=a_)
    axes[0].plot(xx, 1 - np.exp(-xx), "k-", lw=1, label=r"$1-e^{-kS}$")
    axes[1].plot(xx, (1 - np.exp(-xx)) / xx, "k-", lw=1, label=r"$(1-e^{-kS})/kS$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("kS  (expected number of inputs in the GCC)")
    axes[0].set_ylabel(r"$\Psi/S$")
    axes[1].set_ylabel(r"$\Psi/(kS^2)$")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=7)
    fig.suptitle("§6.4 crossover in kS: Ψ ~ kS² for kS ≪ 1, Ψ ~ S for kS ≫ 1 "
                 "(N ≥ 10⁴; probe k up to 1000)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_B_kS.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C  §7 sample fluctuations
def item_C(df):
    g = df.groupby(["N", "k", "phi", "z"])
    s = g.agg(M=("dtc", "size"), c=("dtc", "mean"), c_var=("dtc", "var"),
              h=("H_nc", "mean"), h_var=("H_nc", "var"), S=("S", "mean"), S_var=("S", "var"),
              kin=("k_in_gcc", "mean")).reset_index()
    s["lam"] = [lam_of(z, N, p) for z, N, p in zip(s.z, s.N, s.phi)]
    # conditional on the number of inputs in the GCC: removes placement (Sym_k) fluctuations
    w = df.copy()
    w["cell"] = list(zip(w.N, w.k, w.phi, w.z))
    cond = w.groupby(["N", "k", "phi", "z", "k_in_gcc"]).dtc.agg(["var", "size"]).reset_index()
    cond["wv"] = cond["var"].fillna(0) * (cond["size"] - 1)
    cv = cond.groupby(["N", "k", "phi", "z"]).apply(
        lambda x: x.wv.sum() / max(1, x["size"].sum() - len(x)), include_groups=False)
    s = s.merge(cv.rename("c_var_within").reset_index(), on=["N", "k", "phi", "z"])
    s["c_var_between"] = s.c_var - s.c_var_within
    # placement prediction (OR regime, z > 1 fixed): c_G ~ S * Bin(k, S)  ->  Var = k S^3 (1-S)
    s["S_th"] = s.z.map(giant_fraction)
    s["c_var_pred"] = s.k * s.S_th ** 3 * (1 - s.S_th)
    s.to_csv(OUT / "C_fluct.csv", index=False)
    chk = s[s.phi.isin(["0.05", "0.1"]) & (s.z >= 1.5) & (s.N <= 100000)]
    r = chk.c_var / chk.c_var_pred
    print(f"[C] Var(c_G)/[k S^3 (1-S)], z>=1.5, phi<=0.1, N<=1e5: n={len(r)} median "
          f"{r.median():.3f}  IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}")
    # exponent of Var vs N at fixed lambda (k=5, phi=0.1; N=1e4..1e6)
    ex = []
    for (k, phi), gg in s[s.phi.isin(["0.1", "0.25"])].groupby(["k", "phi"]):
        for lam in sorted(set(gg.lam.round(2))):
            h = gg[(gg.lam.round(2) == lam) & (gg.N >= 10000)].sort_values("N")
            if len(h) < 2 or (h.c_var <= 0).any():
                continue
            lx = np.log(h.N)
            ex.append(dict(k=k, phi=phi, lam=lam, nN=len(h),
                           a_c=np.polyfit(lx, np.log(h.c_var), 1)[0],
                           a_c_within=np.polyfit(lx, np.log(h.c_var_within.clip(1e-300)), 1)[0],
                           a_S=np.polyfit(lx, np.log(h.S_var), 1)[0]))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / "C_var_exponents.csv", index=False)
    e5 = ex[(ex.k == 5) & (ex.phi == "0.1") & ex.lam.isin([-2, -1, 0, 1, 2, 4])]
    print("[C] Var ~ N^a at fixed lambda (k=5, phi=0.1):")
    print(e5.round(3).to_string(index=False))
    pk = []
    for (N, k, phi), gg in s.groupby(["N", "k", "phi"]):
        i = gg.c_var.idxmax()
        j = gg.c_var_within.idxmax()
        pk.append(dict(N=N, k=k, phi=phi, z_peak_c=gg.loc[i, "z"], lam_peak_c=gg.loc[i, "lam"],
                       z_peak_within=gg.loc[j, "z"], lam_peak_within=gg.loc[j, "lam"],
                       z_peak_h=gg.loc[gg.h_var.idxmax(), "z"]))
    pk = pd.DataFrame(pk)
    pk.to_csv(OUT / "C_peaks.csv", index=False)
    print(pk[(pk.k == 5)].round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    Ns = sorted(s.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for N, col in zip(Ns, cols):
        d = s[(s.k == 5) & (s.phi == "0.1") & (s.N == N)].sort_values("z")
        axes[0].plot(d.z, d.c_var, "o-", ms=2.5, lw=0.9, color=col, label=f"N = {N:.0e}")
        axes[1].plot(d.z, d.c_var_within * N ** (1 / 3), "o-", ms=2.5, lw=0.9, color=col)
        axes[2].plot(d.z, d.h_var, "o-", ms=2.5, lw=0.9, color=col)
    zz = np.linspace(1.02, 3, 100)
    Sz = np.array([giant_fraction(z) for z in zz])
    axes[0].plot(zz, 5 * Sz ** 3 * (1 - Sz), "k--", lw=1.2, label=r"$kS^3(1-S)$")
    axes[0].set_title(r"Var$(c_G)$  (no $N$ factor)")
    axes[1].set_title(r"$N^{1/3}\,$Var$(c_G\,|\,k_{\rm in\,GCC})$  (placement removed)")
    axes[2].set_title(r"Var$(h_G)$, $h_G=H_{\rm nc}$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1, color="grey", lw=0.6)
        ax.set_xlabel("z")
    axes[0].legend(fontsize=6)
    fig.suptitle("§7 sample fluctuations (k = 5, φ = 0.1). Above z_c, Var(c_G) = O(1) = kS³(1−S) "
                 "(input placement); with placement removed the peak sits in the window.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_C_fluct.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ D  §5.6 class statistics
def item_D(df, shd):
    g = df.groupby(["N", "k", "phi", "z"])
    s = g.agg(chiF=("chi_F", "mean"), chiF_se=("chi_F", "sem"), m1=("mass_max", "mean"),
              m2=("mass_2nd", "mean"), S=("S", "mean"), S2=("S2", "mean")).reset_index()
    s["lam"] = [lam_of(z, N, p) for z, N, p in zip(s.z, s.N, s.phi)]
    s.to_csv(OUT / "D_classes.csv", index=False)
    pk = []
    for (N, k, phi), gg in s.groupby(["N", "k", "phi"]):
        i = gg.chiF.idxmax()
        pk.append(dict(N=N, k=k, phi=phi, z_peak=gg.loc[i, "z"], lam_peak=gg.loc[i, "lam"],
                       chiF_peak=gg.loc[i, "chiF"]))
    pk = pd.DataFrame(pk)
    pk.to_csv(OUT / "D_chiF_peaks.csv", index=False)
    print("[D] chi_F peaks (phi=0.1):")
    print(pk[pk.phi == "0.1"].round(3).to_string(index=False))
    # finite-class size law from supp_hist: n(s) ~ count_b / 2^b, slope over s in [2, 2^10]
    nb = 25
    sl = []
    for r in shd.itertuples():
        c = np.array([getattr(r, f"b{i}") for i in range(nb)], float)
        b = np.arange(nb)
        dens = c / 2.0 ** b
        sel = (b >= 1) & (b <= 10) & (c >= 5)
        slope = np.polyfit(b[sel] * np.log(2), np.log(dens[sel]), 1)[0] if sel.sum() >= 4 else np.nan
        sl.append(dict(N=r.N, k=r.k, phi=phi_lab(str(r.phi)), z=r.z, slope=slope,
                       n_classes=c.sum()))
    sl = pd.DataFrame(sl)
    sl["lam"] = [lam_of(z, N, p) for z, N, p in zip(sl.z, sl.N, sl.phi)]
    sl.to_csv(OUT / "D_class_size_slopes.csv", index=False)
    t = sl[(sl.phi == "0.1") & (sl.k == 5)].pivot_table(index="z", columns="N", values="slope")
    print("    class-size slope (k=5, phi=0.1) by z x N:")
    print(t.round(2).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    Ns = sorted(s.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for N, col in zip(Ns, cols):
        d = s[(s.k == 5) & (s.phi == "0.1") & (s.N == N) & (s.lam.abs() <= 4.5)].sort_values("lam")
        axes[0].plot(d.lam, d.chiF, "o-", ms=2.5, lw=0.9, color=col, label=f"N = {N:.0e}")
        axes[1].plot(d.lam, d.m2 * N ** (1 / 3), "o-", ms=2.5, lw=0.9, color=col)
        axes[1].plot(d.lam, d.S2 * N ** (1 / 3), ":", lw=0.9, color=col)
    axes[0].set_ylabel(r"$\chi_F$ (no rescaling: stays $O(1)$)")
    axes[1].set_ylabel(r"$N^{1/3}\times$ 2nd largest")
    axes[1].set_title("solid: function class; dotted: component", fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel(r"$\lambda=(z-1)N^{1/3}$")
        ax.axvline(0, color="grey", lw=0.6)
    axes[0].legend(fontsize=6)
    N = int(shd.N.max())
    zz = sorted(shd[(shd.N == N) & (shd.k == 5) & (shd.phi.astype(str) == "0.1")].z.unique())
    pick = [z for z in zz if abs(z - 0.75) < 1e-6 or abs(z - 1.0) < 1e-6 or abs(z - 1.5) < 1e-6
            or abs(z - 3.0) < 1e-6]
    for z, col in zip(pick, ("C0", "C1", "C2", "C3")):
        r = shd[(shd.N == N) & (shd.k == 5) & (shd.phi.astype(str) == "0.1") &
                (np.isclose(shd.z, z))].iloc[0]
        c = np.array([r[f"b{i}"] for i in range(nb)], float)
        x = 2.0 ** np.arange(nb)
        m = c > 0
        axes[2].loglog(x[m], c[m] / x[m] / c.sum(), "o-", ms=2.5, color=col, label=f"z = {z:g}")
    xs = np.geomspace(2, 2 ** 12, 10)
    axes[2].loglog(xs, 0.5 * xs ** -1.5, "k--", lw=1, label=r"$s^{-3/2}$")
    axes[2].set_xlabel("class size s (nodes)")
    axes[2].set_ylabel("n(s) (normalised)")
    axes[2].set_title(f"finite functional classes, N = {N:.0e}, k = 5", fontsize=8)
    axes[2].legend(fontsize=6)
    fig.suptitle("§5.6 function-space cluster statistics in the critical window (φ = 0.1)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_D_classes.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ E  §5.1 rank-frequency
def item_E():
    from rank_freq import annotate_dtc, load_counts, panel
    run = DATA / "e1w"
    cache = OUT / "E_function_counts.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["words"] = df.words.map(np.asarray)
    else:
        df = annotate_dtc(load_counts(run))
        df["z"] = df.z.round(6)
        df.to_parquet(cache)
    cells = pd.read_parquet(run / "cells_W.parquet")
    Mof = {(r.N, r.k, phi_lab(r.phi), round(r.z, 6)): r.M for r in cells.itertuples()}
    N = 100000
    zs = sorted(df[(df.N == N) & (df.k == 5) & (df.phi == "0.1")].z.unique())
    lam_pick = [-1, 0, 1, 4]
    zsel = [min(zs, key=lambda z: abs(lam_of(z, N, "0.1") - L)) for L in lam_pick] + [2.0]
    fig, axes = plt.subplots(2, len(zsel), figsize=(13, 5), squeeze=False)
    cmap = plt.get_cmap("viridis")
    summ = []
    for i, k in enumerate((5, 8)):
        for j, z in enumerate(zsel):
            d = df[(df.N == N) & (df.k == k) & (df.phi == "0.1") & (df.z == z)]
            ax = axes[i, j]
            if d.empty:
                ax.set_visible(False)
                continue
            sc = panel(ax, d, k, N, Mof[(N, k, "0.1", z)], z, cmap)
            ax.set_title(f"k={k}, z={z:.4g} (λ={lam_of(z, N, '0.1'):.2g})", fontsize=8)
            ax.set_xlim(0.8, 2 ** k)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            tot = d["count"].sum()
            top = d.sort_values("count", ascending=False)
            summ.append(dict(k=k, z=z, lam=lam_of(z, N, "0.1"), n_functions=len(d),
                             top_share=top["count"].iloc[0] / tot,
                             top_dtc=int(top["dtc"].iloc[0]), mean_dtc=(d.dtc * d["count"]).sum() / tot))
    fig.colorbar(sc, ax=axes, label="DTC", shrink=0.6)
    fig.suptitle(f"Rank-frequency across the critical window (N = 10⁵, φ = 0.1); "
                 "black = OR staircase", fontsize=9)
    fig.savefig(FIG / "x_E_rank_freq.pdf", bbox_inches="tight")
    plt.close(fig)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "E_rank_summary.csv", index=False)
    print("[E] rank-frequency summary (N=1e5, phi=0.1):")
    print(summ.round(4).to_string(index=False))


# ------------------------------------------------------------------ F  §8.4 input-1 conditioning
def item_F(df):
    g = df.groupby(["N", "k", "phi", "z"])
    s = g.agg(S=("S", "mean"), p_in=("g0", "mean"), m0=("m0", "mean")).reset_index()
    inn = df[df.g0 == 1].groupby(["N", "k", "phi", "z"]).agg(
        m0_in=("m0", "mean"), m0_in_se=("m0", "sem"), S_in=("S", "mean"),
        psi_in=("psi", "mean")).reset_index()
    out = df[df.g0 == 0].groupby(["N", "k", "phi", "z"]).agg(
        m0_out=("m0", "mean"), S_out=("S", "mean")).reset_index()
    s = s.merge(inn, how="left").merge(out, how="left")
    s["lam"] = [lam_of(z, N, p) for z, N, p in zip(s.z, s.N, s.phi)]
    s["S_th"] = s.z.map(giant_fraction)
    s.to_csv(OUT / "F_input1.csv", index=False)
    hi = s[(s.z >= 1.5) & s.phi.isin(["0.05", "0.1"]) & (s.N >= 100000)]
    print("[F] input-1 conditioning, z>=1.5, phi<=0.1, N>=1e5:")
    print("    median P(1 in GCC)/S =", round((hi.p_in / hi.S).median(), 4),
          "  median E[m1|in]/S_in =", round((hi.m0_in / hi.S_in).median(), 4),
          "  max E[m1|out] =", f"{hi.m0_out.max():.2e}")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    Ns = sorted(s.N.unique())
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(Ns)))
    for N, col in zip(Ns, cols):
        d = s[(s.k == 5) & (s.phi == "0.1") & (s.N == N)].sort_values("z")
        w = d[d.lam.abs() <= 4.5]
        axes[0].plot(w.lam, w.p_in * N ** (1 / 3), "o-", ms=2.5, color=col, label=f"N = {N:.0e}")
        axes[0].plot(w.lam, w.S * N ** (1 / 3), ":", color=col)
        axes[1].plot(w.lam, w.m0_in * N ** (1 / 3), "o-", ms=2.5, color=col)
        axes[1].plot(w.lam, w.S_in * N ** (1 / 3), ":", color=col)
        axes[2].plot(d.z, d.m0_in, "o-", ms=2.5, color=col)
        axes[2].plot(d.z, d.m0_out, "x--", ms=3, lw=0.7, color=col)
    zz = np.linspace(1.001, 3, 100)
    axes[2].plot(zz, [giant_fraction(z) for z in zz], "k-", lw=1, label="S(z)")
    axes[0].set_ylabel(r"$N^{1/3}P(1\in$GCC$)$  (dotted: $N^{1/3}S$)")
    axes[1].set_ylabel(r"$N^{1/3}E[m_1\,|\,1\in$GCC$]$  (dotted: $N^{1/3}E[S|\cdot]$)")
    axes[2].set_ylabel(r"$E[m_1\,|\,\cdot]$  (o: in GCC, x: not)")
    for ax in axes[:2]:
        ax.set_xlabel(r"$\lambda$")
        ax.axvline(0, color="grey", lw=0.6)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("z")
    axes[2].legend(fontsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle("§8.4 conditioning on the symmetry-breaking event (input 1 lands in the GCC), "
                 "k = 5, φ = 0.1", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_F_input1.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ G  RQ4 non-OR onset
def item_G(df):
    d = df[df.n_resp > 0].copy()
    d["nonor"] = (1 - d.rho_or) * d.n_resp          # number of non-disjunctive responsive nodes
    d["dlt"] = d.n_dtc_lt_rel
    g = d.groupby(["N", "k", "phi", "z"]).agg(nonor=("nonor", "mean"), dlt=("dlt", "mean"),
                                               resp=("n_resp", "mean")).reset_index()
    allg = df.groupby(["N", "k", "phi", "z"]).agg(M=("S", "size"), S=("S", "mean"),
                                                   psi=("psi", "mean")).reset_index()
    g = allg.merge(g, how="left").fillna({"nonor": 0, "dlt": 0, "resp": 0})
    frac = df[["N", "k", "phi", "z", "n_resp"]].assign(pos=lambda x: x.n_resp > 0)
    g["nonor_density"] = g.nonor * (frac.groupby(["N", "k", "phi", "z"]).pos.mean().values) / g.N
    g["lam"] = [lam_of(z, N, p) for z, N, p in zip(g.z, g.N, g.phi)]
    g.to_csv(OUT / "G_nonor.csv", index=False)
    c = pd.read_csv(DATA / "e1c/analysis/cells_summary.csv", dtype={"phi": str})
    c["nonor_density"] = (1 - c.rho_or.fillna(1)) * c.psi
    c["dlt_density"] = c.dtc_lt_rel_frac.fillna(0) * c.psi
    c.to_csv(OUT / "G_nonor_e1c.csv", index=False)
    print("[G] non-OR density (per non-input node), E1-C, k=5, by phi x z (N=1e4):")
    t = c[(c.k == 5) & (c.N == 10000)].pivot_table(index="z", columns="phi", values="nonor_density")
    print(t.map(lambda v: f"{v:.1e}").to_string())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    cmap = plt.get_cmap("viridis")
    phis = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.4", "U01"]
    for i, phi in enumerate(phis):
        dd = c[(c.k == 5) & (c.N == 10000) & (c.phi == phi)].sort_values("z")
        if dd.empty:
            continue
        col = "k" if phi == "U01" else cmap(i / 6)
        axes[0].plot(dd.z, dd.nonor_density.clip(1e-7), "o-", ms=2.5, color=col, label=f"φ={phi}")
        axes[0].plot(dd.z, dd.psi, ":", color=col, lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axvline(1, color="grey", lw=0.6)
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("non-OR nodes / N  (dotted: Ψ)")
    axes[0].set_title("E1-C, k = 5, N = 10⁴", fontsize=8)
    axes[0].legend(fontsize=6, ncol=2)
    for i, phi in enumerate(phis[:5]):
        for N, ls in ((1000, ":"), (10000, "-")):
            dd = c[(c.k == 5) & (c.N == N) & (c.phi == phi)].sort_values("z")
            axes[1].plot(dd.z / ZUP[phi], dd.nonor_density.clip(1e-7), ls, marker="o", ms=2,
                         color=cmap(i / 6), lw=1, label=f"φ={phi}" if N == 10000 else None)
    axes[1].axvline(1, color="red", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$z / z_{\rm up}(\varphi)$")
    axes[1].set_ylabel("non-OR nodes / N")
    axes[1].set_title("E1-C, k = 5 (solid N = 10⁴, dotted 10³): onset at the upper edge",
                      fontsize=8)
    axes[1].legend(fontsize=6)
    fig.suptitle("RQ4: do non-disjunctive functions have their own onset?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "x_G_nonor.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    item_A()
    df, shd = load_w()
    print(f"E1-W compact: {len(df)} realizations")
    cw = pd.read_csv(DATA / "e1w/analysis/window_summary.csv", dtype={"phi": str})
    cw["phi"] = cw.phi.map(phi_lab)
    item_B(cw)
    item_C(df)
    item_D(df, shd)
    item_F(df)
    item_G(df)
    item_E()


if __name__ == "__main__":
    main()
