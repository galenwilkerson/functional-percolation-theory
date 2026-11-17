"""Double-check the theory notes (theory.tex): symbolic (sympy) and numerical checks.

  python verify_theory.py            # prints PASS/FAIL per check

D1  exact OR classes = components of G \\ I labelled by adjacent input sets  (vs truth tables)
D2  fixed-k laws (binomial sums)
D3  subcritical Psi_N = k z / ((1-z) N)
D4  field equation: Psi_h = (S-h)/(1-h); gamma (z/(1-z)), delta (sqrt(2h))
D5  Landau expansion with field
D6  anisotropy E Var_i m_i = S^3 (1-S)(1-1/k)
D7  Lambda/k at finite field with blocking: (1-e^{-zP})^2, P = (1-h)(1-e^{-zP})  (vs probe data)
D8  exact class sizes at large k (CPU, scipy) vs c_s = n_s(z') x P(adjacent to >= 1 input)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observables import DTCCache, engine, function_stats, structure_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltm_learning.closed_forms import giant_fraction  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


# ------------------------------------------------------------------ symbolic
def symbolic():
    S, z, h, k, j = sp.symbols("S z h k j", positive=True)
    # D5 Landau with field
    Phi = S**2 / 2 - S - (1 - h) / z * sp.exp(-z * S)
    ser = sp.series(Phi, S, 0, 4).removeO()
    target = -(1 - h) / z - h * S + (1 - (1 - h) * z) / 2 * S**2 + (1 - h) * z**2 / 6 * S**3
    check("D5 Landau expansion with field", sp.simplify(sp.expand(ser - target)) == 0,
          f"Phi ≈ {sp.simplify(target)}")
    check("D5 stationarity <=> S = 1-(1-h)e^{-zS}",
          sp.simplify(sp.diff(Phi, S) - (S - 1 + (1 - h) * sp.exp(-z * S))) == 0)
    # D4 Psi_h = (S-h)/(1-h): S = 1-u with u=(1-h)e^{-zS}; Psi = 1 - e^{-zS}
    u = (1 - h) * sp.exp(-z * S)
    check("D4 Psi_h = (S-h)/(1-h)",
          sp.simplify((1 - sp.exp(-z * S)) - ((1 - u) - h) / (1 - h)) == 0)
    # D6 E[(j/k)(1-j/k)] for j ~ Bin(k, S)  (use E j = kS, E j^2 = kS(1-S) + k^2 S^2)
    Ej, Ej2 = k * S, k * S * (1 - S) + k**2 * S**2
    expr = (Ej / k - Ej2 / k**2) * S**2
    check("D6 anisotropy E Var_i m_i = S^3(1-S)(1-1/k)",
          sp.simplify(expr - S**3 * (1 - S) * (1 - 1 / k)) == 0)
    # D2 sum_j C(k,j) S^{j+1}(1-S)^{k-j} = S(1-(1-S)^k)
    kk = 6
    s = sum(sp.binomial(kk, jj) * S**(jj + 1) * (1 - S)**(kk - jj) for jj in range(1, kk + 1))
    check("D2 sum_j P[DTC=j] = Psi (k=6)", sp.simplify(s - S * (1 - (1 - S)**kk)) == 0)
    s2 = sum(jj * sp.binomial(kk, jj) * S**(jj + 1) * (1 - S)**(kk - jj) for jj in range(1, kk + 1))
    check("D2 sum_j j P[DTC=j] = k S^2 (k=6)", sp.simplify(s2 - kk * S**2) == 0)


# ------------------------------------------------------------------ numerical (closed forms)
def psi_field(z, h):
    f = lambda S: 1 - (1 - h) * math.exp(-z * S) - S  # noqa: E731
    S = brentq(f, 1e-15, 1.0)
    return (S - h) / (1 - h)


def numeric_closed_forms():
    # D3 Borel mean cluster size 1/(1-z)
    for z in (0.3, 0.6, 0.9):
        s = np.arange(1, 4000)
        logp = (s - 1) * np.log(z * s) - z * s - np.array([math.lgamma(x + 1) for x in s])
        mean = float((s * np.exp(logp)).sum())
        check(f"D3 Borel E|C| = 1/(1-z) at z={z}", abs(mean - 1 / (1 - z)) < 2e-3 / (1 - z) ** 2,
              f"{mean:.5f} vs {1/(1-z):.5f}")
    # D4 gamma and delta from the exact field equation
    for z in (0.5, 0.8):
        h = 1e-7
        check(f"D4 gamma: Psi/h -> z/(1-z) at z={z}", abs(psi_field(z, h) / h - z / (1 - z)) < 1e-3 * z / (1 - z),
              f"{psi_field(z, h)/h:.5f} vs {z/(1-z):.5f}")
    for h in (1e-4, 1e-6):
        r = psi_field(1.0, h) / math.sqrt(2 * h)
        check(f"D4 delta: Psi(z=1)/sqrt(2h) -> 1 at h={h}", abs(r - 1) < 3 * math.sqrt(h) + 1e-3, f"{r:.5f}")
    check("D4 h->0 limit = S", abs(psi_field(1.5, 1e-9) - giant_fraction(1.5)) < 1e-6)


# ------------------------------------------------------------------ D1 / D8 exact OR classes
def or_classes(N, src, dst, inputs):
    """Exact OR-regime functional classes: components of G minus inputs; each component's
    function = OR of its adjacent inputs. Returns (label per non-input node, key per label)."""
    inp = np.zeros(N, bool)
    inp[inputs] = True
    keep = ~inp[src] & ~inp[dst]
    A = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    # adjacent inputs per component
    e1 = inp[src] & ~inp[dst]
    e2 = inp[dst] & ~inp[src]
    comp = np.concatenate([lab[dst[e1]], lab[src[e2]]])
    who = np.concatenate([src[e1], dst[e2]])
    df = pd.DataFrame(dict(c=comp, i=who)).drop_duplicates()
    keys = df.groupby("c").i.apply(lambda v: tuple(sorted(v))).to_dict()
    return lab, keys


def check_D1():
    """Exact-class construction reproduces the truth-table partition (OR regime, k <= 6)."""
    dev = torch.device("cpu")
    gen = torch.Generator().manual_seed(5)
    N, k, B = 3000, 6, 24
    rng = np.random.default_rng(5)
    zs = rng.uniform(0.6, 3.0, B)
    m = torch.as_tensor(rng.binomial(N * (N - 1) // 2, zs / (N - 1)))
    g = engine.sample_gnm_batch(N, m, dev, gen)
    inputs = engine.sample_inputs(N, k, B, dev, gen)
    phi = 0.05
    L, _ = engine.cascade_frontier(g, inputs, k, phi)
    T = (L > 0).numpy()
    src, dst = g.src.numpy(), g.dst.numpy()
    ok_all, n_or = True, 0
    for b in range(B):
        sel = (src // N) == b
        s_, d_ = src[sel] - b * N, dst[sel] - b * N
        deg = np.bincount(np.concatenate([s_, d_]), minlength=N)
        if (deg > 1 / phi).any():   # OR regime requires all degrees <= 1/phi (else skip)
            continue
        n_or += 1
        inp = (inputs[b] - b * N).numpy()
        lab, keys = or_classes(N, s_, d_, inp)
        Tb = T[b * N:(b + 1) * N]
        nonin = np.setdiff1d(np.arange(N), inp)
        pos = {int(v): q for q, v in enumerate(inp)}
        for u in nonin[::7]:
            A = keys.get(lab[u], ())
            pred = np.array([any((p >> pos[i]) & 1 for i in A) for p in range(1 << k)])
            if not np.array_equal(pred, Tb[u].astype(bool)):
                ok_all = False
                break
    check("D1 exact OR classes = truth tables (graphs with max degree <= 1/phi)",
          ok_all and n_or > 5, f"{n_or} graphs checked")


def check_D7_D8():
    """Large-k exact classes on CPU: Lambda/k with blocking, and class-size distribution."""
    rng = np.random.default_rng(11)
    N, h, G = 100000, 0.01, 12
    k = int(h * N)
    out = {}
    for z in (1.0, 1.5):
        sizes, lamk = [], []
        for _ in range(G):
            m = rng.binomial(N * (N - 1) // 2, z / (N - 1))
            a = rng.integers(0, N, int(m * 1.02) + 10)
            b = rng.integers(0, N, int(m * 1.02) + 10)
            keep = a != b
            key = np.unique(np.minimum(a, b)[keep].astype(np.int64) * N + np.maximum(a, b)[keep])[:m]
            src, dst = key // N, key % N
            inputs = rng.choice(N, k, replace=False)
            lab, keys = or_classes(N, src, dst, inputs)
            inp = np.zeros(N, bool); inp[inputs] = True
            nonin = np.where(~inp)[0]
            # classes = components grouped by their adjacent-input key (non-empty)
            klist = pd.Series([keys.get(c, ()) for c in lab[nonin]])
            cnt = klist[klist.map(len) > 0].value_counts()
            sizes.append(cnt.to_numpy()[1:] if len(cnt) > 1 else np.array([]))   # drop condensate
            lamk.append(sum(len(keys.get(c, ())) for c in lab[nonin]) / (len(nonin) * k))
        out[z] = (np.concatenate(sizes), float(np.mean(lamk)))
    # D7: Lambda/k with blocking
    for z in (1.5,):
        P = brentq(lambda P: (1 - h) * (1 - math.exp(-z * P)) - P, 1e-12, 1.0)
        pred = (1 - math.exp(-z * P)) ** 2
        S2 = giant_fraction(z) ** 2
        meas = out[z][1]
        check(f"D7 Lambda/k with blocking at z={z}, h={h}", abs(meas - pred) < 0.01,
              f"exact-classes {meas:.4f}  vs blocking theory {pred:.4f}  (no-blocking S^2 = {S2:.4f})")
    # D8: class-size distribution at z = 1 (exact classes)
    s = out[1.0][0]
    bins = 2.0 ** np.arange(0, 15)
    H, _ = np.histogram(s, bins=bins)
    dens = H / np.diff(bins) / G
    mid = bins[:-1] * 1.5
    for lo, hi_ in [(2, 64), (128, 4096)]:
        msk = (mid >= lo) & (mid <= hi_) & (dens > 0)
        sl = np.polyfit(np.log(mid[msk]), np.log(dens[msk]), 1)[0] if msk.sum() >= 3 else float("nan")
        print(f"      exact classes z=1, h={h}: slope over s in [{lo},{hi_}] = {sl:.3f}")
        out[f"slope_{lo}"] = sl
    check("D8 small classes ~ s^-3/2 at z_c (exact classes, s in [2,64])",
          abs(out["slope_2"] + 1.5) < 0.15, f"slope {out['slope_2']:.3f}")
    print(f"      large classes s in [128,4096]: slope {out['slope_128']:.3f} "
          f"(prediction -5/2 if crossover at 1/h = {1/h:.0f}; site-diluted z' = {1*(1-h):.3f})")
    pd.DataFrame(dict(bin_lo=bins[:-1], dens=dens)).to_csv(HERE / "data" / "verify_D8_exact_classes.csv",
                                                            index=False)


def main():
    symbolic()
    numeric_closed_forms()
    check_D1()
    check_D7_D8()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
