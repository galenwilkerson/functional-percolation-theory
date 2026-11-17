"""Finite functional class-size law at z = 1 with field h (theory.tex, class-size law).

Exact OR-regime classes (er_checks.exact_classes): components of G \\ I grouped by their
adjacent-input set. k = h N inputs, h = 0.01, N = 1e5, 1e6, 1e7 (N^{2/3} = 2.2e3, 1e4, 4.6e4),
so the crossover at 1/h = 100 is separated from the finite-size cutoff N^{2/3}.

Prediction (infinite N): G \\ I is ER with z' = z(1-h) (site dilution), Borel n'_s with cutoff
2/h^2 = 2e4; a region of size s touches Poisson(z h s) inputs, so classes with >= 2 inputs have
density n'_s [1 - e^{-zhs}(1 + zhs)] ~ s^{-5/2} for 1/h << s << 1/h^2; single-input classes
(all regions hanging off one input merge) dominate for s << 1/h with density ~ h s^{-3/2}.

  python class_dilution.py --out data/class_dilution
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from er_checks import exact_classes, sample_gnp  # noqa: E402

H = 0.01
NB = 30


def one(task):
    N, seed = task
    rng = np.random.default_rng(seed)
    src, dst = sample_gnp(N, 1.0, rng)
    k = int(round(H * N))
    inputs = rng.choice(N, k, replace=False)
    lab, keys, inp = exact_classes(N, src, dst, inputs)
    nonin = np.flatnonzero(~inp)
    cs = np.bincount(lab[nonin], minlength=N)
    cls1, clsm = {}, {}
    for c, key in keys.items():
        d = cls1 if len(key) == 1 else clsm
        d[key] = d.get(key, 0) + int(cs[c])
    out = []
    for kind, d in (("single", cls1), ("multi", clsm)):
        s = np.array(list(d.values()), dtype=np.int64)
        s = s[s > 0]
        b = np.floor(np.log2(s)).astype(int).clip(max=NB - 1)
        out.append((kind, np.bincount(b, minlength=NB)))
    return N, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G = {100000: 4000, 1000000: 800, 10000000: 200}
    rng = np.random.default_rng(11)
    T = [(N, int(rng.integers(2**62))) for N in sorted(G, reverse=True) for _ in range(G[N])]
    ncpu = int(os.environ.get("GQ_CPUS", "4"))
    acc = {}
    with Pool(ncpu) as pool:
        for i, (N, res) in enumerate(pool.imap_unordered(one, T, chunksize=1)):
            for kind, h in res:
                acc[(N, kind)] = acc.get((N, kind), 0) + h
            if i % 200 == 0:
                print(f"  {i}/{len(T)}", flush=True)
    rows = [dict(N=N, kind=kind, graphs=G[N], **{f"b{j}": int(x) for j, x in enumerate(h)})
            for (N, kind), h in acc.items()]
    pd.DataFrame(rows).to_csv(out / "class_hist.csv", index=False)
    print("done", flush=True)


def borel(zp, smax):
    s = np.arange(1, smax + 1, dtype=float)
    lg = np.array([math.lgamma(x + 1) for x in s])
    return s, np.exp((s - 1) * np.log(zp * s) - zp * s - lg) / s


def analyze(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    d = pd.read_csv(out / "class_hist.csv")
    b = np.arange(NB)
    lo, hi = 2.0 ** b, 2.0 ** (b + 1)
    # prediction for multi-input classes (per vertex), binned
    s, ns = borel(1 - H, 400000)
    cm = ns * (1 - np.exp(-H * s) * (1 + H * s))
    pred_bins = np.array([cm[(s >= l) & (s < u)].sum() for l, u in zip(lo, hi)])
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for N, col in zip(sorted(d.N.unique()), ("C0", "C1", "C2")):
        tot = 0
        for kind, ls in (("single", ":"), ("multi", "-")):
            r = d[(d.N == N) & (d.kind == kind)].iloc[0]
            c = np.array([r[f"b{j}"] for j in range(NB)], float)
            dens = c / (r.graphs * N) / (hi - lo)       # per vertex per unit s
            m = c > 0
            ax[0].loglog(np.sqrt(lo * hi)[m], dens[m], ls, marker="o", ms=2.5, color=col,
                         label=f"N={N:.0e} {kind}")
            tot = tot + c
        dens = tot / (r.graphs * N) / (hi - lo)
        for (a_, b_) in ((2, 64), (256, 2048), (256, 8192)):
            sel = (lo >= a_) & (hi <= b_ * 2) & (tot >= 10)
            if sel.sum() >= 3:
                sl = np.polyfit(np.log(np.sqrt(lo * hi)[sel]), np.log(dens[sel]), 1)[0]
                rows.append(dict(N=N, range=f"[{a_},{b_}]", slope=sl, n_bins=int(sel.sum())))
        ax[1].loglog(np.sqrt(lo * hi)[tot > 0], dens[tot > 0], "o-", ms=2.5, color=col,
                     label=f"N = {N:.0e}  (N^(2/3) = {N ** (2 / 3):.0f})")
    pdens = pred_bins / (hi - lo)
    m = pdens > 0
    ax[0].loglog(np.sqrt(lo * hi)[m], pdens[m], "k--", lw=1, label="pred. multi (N=∞)")
    sel = (lo >= 256) & (hi <= 4096)
    psl = np.polyfit(np.log(np.sqrt(lo * hi)[sel]), np.log(pdens[sel]), 1)[0]
    rows.append(dict(N=np.inf, range="[256,2048] pred multi", slope=psl, n_bins=int(sel.sum())))
    xs = np.geomspace(2, 64, 5)
    ax[1].loglog(xs, 0.02 * xs ** -1.5, "k:", label=r"$s^{-3/2}$")
    xs = np.geomspace(200, 2e4, 5)
    ax[1].loglog(xs, 3 * xs ** -2.5, "k--", label=r"$s^{-5/2}$")
    for a_ in ax:
        a_.axvline(1 / H, color="grey", lw=0.6)
        a_.set_xlabel("class size s")
    ax[0].set_ylabel("classes per vertex per unit s")
    ax[0].legend(fontsize=5)
    ax[1].legend(fontsize=6)
    ax[0].set_title("single- vs multi-input classes", fontsize=8)
    ax[1].set_title("all classes: crossover at 1/h, cutoff at N^(2/3)", fontsize=8)
    fig.suptitle("Functional class-size law at z = 1, h = 0.01 (exact OR classes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE.parent / "figures" / "class_dilution.pdf")
    r = pd.DataFrame(rows)
    r.to_csv(out / "class_slopes.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze(Path(sys.argv[sys.argv.index("--out") + 1]))
    else:
        main()
