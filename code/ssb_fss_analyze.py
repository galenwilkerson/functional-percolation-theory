"""SC analysis: scaling collapse of the Sym_k breaking against Aldous' excursion limit, with free-exponent fits.
python ssb_fss_analyze.py   (figures/tables to ../ssb_results/)"""
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
PFX = os.environ.get("SSB_PREFIX", "")
if PFX:
    OUT = OUT / "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
K = 3
OBS = {  # name: (column builder, predicted exponent a in y N^a, theory from Aldous, label)
    "absM0": (2 / 3, lambda G2, G3: 2 * (1 - 1 / K) * G2, r"$\langle|M_0|\rangle N^{2/3}$"),
    "A": (1.0, lambda G2, G3: (1 - 1 / K) * G3, r"$\langle A\rangle N$"),
    "chi": (-1 / 3, lambda G2, G3: G2, r"$\chi N^{-1/3}$"),
}


def cell_means(df):
    df = df.copy()
    df["M0"] = df.m0 - df[["m0", "m1", "m2"]].mean(axis=1)
    df["absM0"] = df.M0.abs()
    df["A"] = df.anis
    df["chi"] = df.N * df.pert_dm0
    return df.groupby(["phi", "N", "lam"])[["absM0", "A", "chi"]].mean().reset_index()


def quality(cm, obs, a, b):
    """Mean squared difference of log(y N^a) between sizes, as functions of x = lam N^(b - 1/3)."""
    cur = {}
    for N, s in cm.groupby("N"):
        s = s.sort_values("lam")
        y = s[obs].values * N ** a
        ok = y > 0
        cur[N] = (s.lam.values[ok] * N ** (b - 1 / 3), np.log(y[ok]))
    tot, n = 0.0, 0
    Ns = sorted(cur)
    for i in range(len(Ns)):
        for j in range(i + 1, len(Ns)):
            x1, y1 = cur[Ns[i]]
            x2, y2 = cur[Ns[j]]
            lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
            sel = (x1 >= lo) & (x1 <= hi)
            if sel.sum() < 2:
                continue
            tot += ((np.interp(x1[sel], x2, y2) - y1[sel]) ** 2).sum()
            n += sel.sum()
    return tot / n if n else np.inf


def fit(cm, obs, a0):
    """Coarse grid, then a fine grid around the coarse optimum."""
    def scan(A_, B_, best):
        for a in A_:
            for b in B_:
                q = quality(cm, obs, a, b)
                if best is None or q < best[2]:
                    best = (float(a), float(b), float(q))
        return best
    best = scan(np.linspace(a0 - 0.35, a0 + 0.35, 15), np.linspace(0.15, 0.55, 9), None)
    a, b = best[:2]
    return scan(np.linspace(a - 0.05, a + 0.05, 11), np.linspace(b - 0.05, b + 0.05, 11), best)


def per_sample(df):
    df = df.copy()
    df["M0"] = df.m0 - df[["m0", "m1", "m2"]].mean(axis=1)
    df["absM0"] = df.M0.abs()
    df["A"] = df.anis
    df["chi"] = df.N * df.pert_dm0
    return df


def boot_cells(groups, rng):
    """groups: list of (phi, N, lam, array[n, 3]); one bootstrap resample of the cell means."""
    rows = []
    for phi, N, lam, arr in groups:
        idx = rng.integers(0, len(arr), len(arr))
        m = arr[idx].mean(axis=0)
        rows.append(dict(phi=phi, N=N, lam=lam, absM0=m[0], A=m[1], chi=m[2]))
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(DATA / f"{PFX}fss.parquet")
    al = pd.read_csv(ALDOUS)
    al = al[al.dt == al.dt.min()].sort_values("lam")
    cm = cell_means(df)
    ps = per_sample(df)
    groups = {phi: [(phi, N, lam, g[["absM0", "A", "chi"]].values) for (N, lam), g in s.groupby(["N", "lam"])]
              for phi, s in ps.groupby("phi")}
    cm.to_csv(OUT / "sc_cells.csv", index=False)
    key, rows = {}, []
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for r, phi in enumerate(sorted(cm.phi.unique())):
        c = cm[cm.phi == phi]
        for col, (obs, (a0, th, lab)) in enumerate(OBS.items()):
            ax = axs[r, col]
            for N, s in c.groupby("N"):
                s = s.sort_values("lam")
                ax.plot(s.lam, s[obs] * N ** a0, "o-", ms=2.5, lw=0.8, label=f"N={N}")
            if phi == 0.1:
                ax.plot(al.lam, th(al.G2.values, al.G3.values), "k:", lw=1.2, label="Aldous (no fit)")
            ax.set_yscale("log")
            ax.set_title(f"$\\varphi={phi}$", fontsize=7)
            ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel(r"$\lambda=(z-z_c)N^{1/3}$")
            for drop in (False, True):
                cc = c[c.N > c.N.min()] if drop else c
                a, b, q = fit(cc, obs, a0)
                boots = []
                for _ in range(40):
                    bc = boot_cells(groups[phi], rng)
                    bc = bc[bc.N > bc.N.min()] if drop else bc
                    boots.append(fit(bc, obs, a0)[:2])
                boots = np.array(boots)
                rows.append(dict(phi=phi, obs=obs, drop_smallest=drop, a_pred=a0, a_fit=a, a_lo=np.percentile(boots[:, 0], 2.5),
                                 a_hi=np.percentile(boots[:, 0], 97.5), b_pred=1 / 3, b_fit=b,
                                 b_lo=np.percentile(boots[:, 1], 2.5), b_hi=np.percentile(boots[:, 1], 97.5), quality=q))
    axs[0, 0].legend(fontsize=5)
    fig.suptitle("SC: finite-size scaling of the symmetry breaking ($k=3$). Dotted: Aldous excursion limit, no fit "
                 "($\\varphi=0.1$)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sc_collapse.pdf")
    plt.close(fig)
    # theory ratio at phi = 0.1, largest N
    c = cm[(cm.phi == 0.1) & (cm.N == cm.N.max())]
    for obs, (a0, th, _) in OBS.items():
        pred = np.interp(c.lam, al.lam, th(al.G2.values, al.G3.values))
        key[f"SC_{obs}_ratio_to_Aldous_Nmax"] = dict(zip(c.lam.round(2).astype(str), (c[obs] * c.N ** a0 / pred).round(3)))
    fits = pd.DataFrame(rows)
    fits.to_csv(OUT / "sc_fits.csv", index=False)
    json.dump(key, open(OUT / "key_numbers_SC.json", "w"), indent=1)
    print(fits.round(3).to_string(index=False))
    print(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
