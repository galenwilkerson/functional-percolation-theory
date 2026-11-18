"""Tree theory of synergy with individual inputs at density h (k = hN), ER, homogeneous phi.

Messages (parent inactive):  q = P(responsive: active with all inputs on),
  y = P(non-OR: fires with its single activators off),  w = P(pure synergy: responsive, no single
  input activates it).  On a tree a single input reaches a node through one branch, so a node needing
  t >= 2 active neighbours has no single activator; a node with t = 1 is non-OR iff some child is, and
  pure iff all its responsive children are pure.
    q = h + (1-h) sum_n p_n P[Bin(n,q) >= tau(n+1)]
    y = (1-h) sum_n p_n { 1-(1-y)^n                 if tau = 1;  P[Bin(n,q) >= tau]  otherwise }
    w = (1-h) sum_n p_n { (1-q+w)^n - (1-q)^n       if tau = 1;  P[Bin(n,q) >= tau]  otherwise }
Node values use degree n (no parent).  Shares: y/q (any synergy), w/q (pure synergy).
Exact for z <= z_c (above z_c a single input in the giant reaches a node along many paths).
Second-order closed forms at z_c (h -> 0): any = sqrt(r), pure = 1 - sqrt(1-r), r = mu_2/nu_1.

  python synergy_theory.py --out data/synergy_theory
"""
from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, poisson

HERE = Path(__file__).resolve().parent
NS = np.arange(0, 60)


def solve(z, phi, h, tol=1e-15, itmax=2_000_000):
    pn = poisson.pmf(NS, z)
    te = np.ceil(phi * (NS + 1) - 1e-12)
    tv = np.ceil(phi * np.maximum(NS, 1) - 1e-12)
    vul = te == 1
    q, y, w = h, 0.0, 0.0
    for it in range(itmax):
        hi = binom.sf(te - 1, NS, q)
        q2 = h + (1 - h) * np.sum(pn * hi)
        y2 = (1 - h) * np.sum(pn * np.where(vul, 1 - (1 - y) ** NS, hi))
        w2 = (1 - h) * np.sum(pn * np.where(vul, (1 - q + w) ** NS - (1 - q) ** NS, hi))
        done = max(abs(q2 - q), abs(y2 - y), abs(w2 - w)) < tol
        q, y, w = q2, y2, w2
        if done:
            break
    vv = (tv == 1) & (NS > 0)
    hv = np.where(NS > 0, binom.sf(tv - 1, NS, q), 0.0)
    qn = np.sum(pn * hv)
    yn = np.sum(pn * np.where(vv, 1 - (1 - y) ** NS, hv))
    wn = np.sum(pn * np.where(vv, (1 - q + w) ** NS - (1 - q) ** NS, hv))
    return qn, yn, wn, it


def zc_and_r(phi):
    lam = lambda z: sum(poisson.pmf(n, z) * n * (math.ceil(phi * (n + 1) - 1e-12) == 1)
                        for n in range(60))
    zc = brentq(lambda z: lam(z) - 1, 0.5, 2.0)
    nu1 = sum(poisson.pmf(n, zc) * n * (n - 1) * (math.ceil(phi * (n + 1) - 1e-12) == 1) for n in range(60))
    mu2 = sum(poisson.pmf(n, zc) * n * (n - 1) * (math.ceil(phi * (n + 1) - 1e-12) == 2) for n in range(60))
    return zc, nu1, mu2


def task(a):
    phi, h, z = a
    qn, yn, wn, it = solve(z, phi, h)
    return dict(phi=phi, h=h, z=z, q=qn, any=yn / qn if qn else 0.0, pure=wn / qn if qn else 0.0, it=it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synergy_theory")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for phi in (0.25, 0.2):
        zc, nu1, mu2 = zc_and_r(phi)
        r = mu2 / nu1
        print(f"phi={phi}: z_c={zc:.5f} r={r:.4f} any(z_c)={math.sqrt(r):.4f} "
              f"pure(z_c)={1 - math.sqrt(1 - r):.4f}", flush=True)
        zs = np.unique(np.round(np.concatenate([np.linspace(0.8, zc, 60), zc + np.linspace(0, 0.4, 30)]), 6))
        for h in (1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6):
            tasks += [(phi, h, float(z)) for z in zs]
    with Pool(int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))) as pool:
        rows = pool.map(task, tasks, chunksize=4)
    pd.DataFrame(rows).to_csv(out / "synergy_theory.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
