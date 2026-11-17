"""Critical-window scaling functions from Aldous' (1997) excursion limit (theory.tex, FSS).

For z = 1 + lam N^{-1/3}, N^{-2/3}(|C_1|, |C_2|, ...) -> excursion lengths (xi_j) of
B(t) = W(t) + lam t - t^2/2 above its running minimum. In the OR regime at fixed k,
with x_j = xi_j N^{-1/3} the component fractions:
  S N^{1/3}            -> E xi_max
  N Var(S) N^{-1/3}    -> Var xi_max
  Psi                  =  E sum_j x_j [1 - (1 - x_j)^k]   (~ k N^{-2/3} G2)
  (Lam/k) N^{2/3}      -> G2 = E sum xi_j^2
  A N                  -> (1 - 1/k) (G3 - N^{-1/3} G2^2),   G3 = E sum xi_j^3
Parameter-free: no fit to the LTM data. Tail beyond the simulated horizon T (drift
mu = t - lam): sum xi^2 per unit time = 1/mu^2, sum xi^3 = 3/mu^4 -> add 1/(T-lam), 1/(T-lam)^3.

  python aldous.py --out data/aldous            simulate
  python aldous.py --out data/aldous --analyze  compare with E1-W (phi = 0.1)
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NS = [1000, 10000, 100000, 1000000]
KS = [1, 2, 5, 8]
LAMS = [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4]


def run(task):
    lam, dt, npaths, seed = task
    rng = np.random.default_rng(seed)
    T = max(lam, 0.0) + 8.0
    n = int(round(T / dt))
    t = np.arange(1, n + 1) * dt
    drift = lam * t - t * t / 2
    acc = dict(xmax=[], g2=[], g3=[])
    psi = {(N, k): [] for N in NS for k in KS}
    B = 200
    for b0 in range(0, npaths, B):
        nb = min(B, npaths - b0)
        W = np.cumsum(rng.standard_normal((nb, n), dtype=np.float32) * np.float32(np.sqrt(dt)),
                      axis=1) + drift.astype(np.float32)
        rm = np.minimum.accumulate(W, axis=1)
        newmin = np.empty_like(W, dtype=bool)
        newmin[:, 0] = W[:, 0] < 0
        newmin[:, 1:] = W[:, 1:] < rm[:, :-1]
        for i in range(nb):
            idx = np.flatnonzero(newmin[i])
            ends = np.concatenate([[0], idx, [n]])
            L = np.diff(ends) * dt
            L = L[L > 0]
            acc["xmax"].append(L.max())
            acc["g2"].append((L ** 2).sum() + 1 / (T - lam))
            acc["g3"].append((L ** 3).sum() + 1 / (T - lam) ** 3)
            for N in NS:
                x = L * N ** (-1 / 3)
                tail = 1 / (T - lam) * N ** (-2 / 3)
                for k in KS:
                    psi[(N, k)].append((x * (1 - (1 - np.minimum(x, 1)) ** k)).sum() + k * tail)
    xm = np.array(acc["xmax"])
    r = dict(lam=lam, dt=dt, paths=npaths, xmax=xm.mean(), xmax_se=xm.std(ddof=1) / np.sqrt(len(xm)),
             xmax_var=xm.var(ddof=1), G2=np.mean(acc["g2"]),
             G2_se=np.std(acc["g2"], ddof=1) / np.sqrt(len(xm)), G3=np.mean(acc["g3"]),
             G3_se=np.std(acc["g3"], ddof=1) / np.sqrt(len(xm)))
    for (N, k), v in psi.items():
        r[f"psi_N{N}_k{k}"] = float(np.mean(v))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.analyze:
        analyze(out)
        return
    rng = np.random.default_rng(1997)
    T = [(lam, 1e-4, 40000, int(rng.integers(2**62))) for lam in LAMS]
    T += [(lam, 4e-4, 20000, int(rng.integers(2**62))) for lam in (-2, 0, 2)]   # dt check
    ncpu = int(os.environ.get("GQ_CPUS", "4"))
    with Pool(ncpu) as pool:
        rows = list(pool.imap_unordered(run, T))
    pd.DataFrame(rows).sort_values(["dt", "lam"]).to_csv(out / "aldous_moments.csv", index=False)
    print(pd.DataFrame(rows)[["lam", "dt", "xmax", "G2", "G3"]].sort_values(["lam", "dt"])
          .round(4).to_string(index=False), flush=True)


def analyze(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    m = pd.read_csv(out / "aldous_moments.csv")
    m = m[m.dt == 1e-4].set_index("lam")
    w = pd.read_csv(HERE / "data/e1w/analysis/window_summary.csv", dtype={"phi": str})
    w = w[(w.phi == "0.1") & (w.lam_s.abs() <= 4.01)].copy()
    w["lamr"] = w.lam_s.round(2)
    w = w[w.lamr.isin(m.index)]
    rows = []
    for r in w.itertuples():
        a = m.loc[r.lamr]
        N, k = int(r.N), int(r.k)
        rows.append(dict(N=N, k=k, lam=r.lamr,
                         S_sc=r.S * N ** (1 / 3), S_pred=a.xmax,
                         chiS=r.S_var * N / N ** (1 / 3), chiS_pred=a.xmax_var,
                         psi=r.psi, psi_pred=a.get(f"psi_N{N}_k{k}", np.nan),
                         lamk_sc=r.lam / k * N ** (2 / 3), lamk_pred=a.G2,
                         A_sc=r.aniso * N,
                         A_pred=(1 - 1 / k) * (a.G3 - N ** (-1 / 3) * a.G2 ** 2) if k > 1 else 0))
    d = pd.DataFrame(rows)
    d.to_csv(out / "aldous_vs_e1w.csv", index=False)
    for N, g in d.groupby("N"):
        print(f"N={N:>7}: median |ratio-1|  S {np.median(abs(g.S_sc / g.S_pred - 1)):.3f}  "
              f"Psi {np.median(abs(g.psi / g.psi_pred - 1)):.3f}  "
              f"Lam/k {np.median(abs(g.lamk_sc / g.lamk_pred - 1)):.3f}  "
              f"A {np.median(abs(g[g.k > 1].A_sc / g[g.k > 1].A_pred - 1)):.3f}  "
              f"NVar(S) {np.median(abs(g.chiS / g.chiS_pred - 1)):.3f}")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    cols = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 4))
    for N, col in zip(sorted(d.N.unique()), cols):
        g = d[(d.N == N) & (d.k == 5)].sort_values("lam")
        ax[0].plot(g.lam, g.S_sc, "o", ms=3, color=col, label=f"N = {N:.0e}")
        ax[1].plot(g.lam, g.psi * N ** (2 / 3) / 5, "o", ms=3, color=col)
        ax[1].plot(g.lam, g.psi_pred * N ** (2 / 3) / 5, "-", lw=0.8, color=col)
        ax[2].plot(g.lam, g.lamk_sc, "o", ms=3, color=col)
        ax[3].plot(g.lam, g.A_sc, "o", ms=3, color=col)
        ax[3].plot(g.lam, g.A_pred, "-", lw=0.8, color=col)
    L = m.index.values
    ax[0].plot(L, m.xmax, "k-", lw=1.3, label=r"Aldous $E\xi_{\max}$")
    ax[2].plot(L, m.G2, "k-", lw=1.3, label=r"$G_2=E\sum\xi_j^2$")
    ax[0].set_ylabel(r"$S\,N^{1/3}$")
    ax[1].set_ylabel(r"$\Psi N^{2/3}/k$ (lines: Aldous at that $N$)")
    ax[2].set_ylabel(r"$(\Lambda/k)\,N^{2/3}$")
    ax[3].set_ylabel(r"$A\,N$ (lines: $(1-1/k)(G_3-N^{-1/3}G_2^2)$)")
    for a_ in ax:
        a_.set_xlabel(r"$\lambda$")
        a_.set_yscale("log")
    ax[0].legend(fontsize=6)
    ax[2].legend(fontsize=6)
    fig.suptitle("Critical-window scaling functions = moments of Aldous' excursion lengths "
                 "(no fit; k = 5, φ = 0.1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE.parent / "figures" / "aldous_fss.pdf")


if __name__ == "__main__":
    main()
