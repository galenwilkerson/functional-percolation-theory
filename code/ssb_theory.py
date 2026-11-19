"""Closed-form predictions for Sym_k breaking in the simple-contagion (OR) regime, N -> infinity, fixed k.

By Prop. 1 the giant component computes OR_A, A = inputs in the giant component, each input
independently in it with probability S (S = 1 - exp(-z S)).  Then m_i -> S 1[i in A] and
    M_0 = S (b_0 - K/k),  K = |A| ~ Bin(k, S),  b_0 = 1[0 in A]
    A_anis = S^2 (K/k)(1 - K/k),        E A_anis = S^3 (1-S)(1-1/k)
    |Stab(OR_A)| = K! (k-K)!            (residual symmetry Sym_K x Sym_{k-K})
Extensive field exp(x M_0/S... ) : with x = eps N,
    <M_0>_x = sum_{b0,K'} p(b0,K') M w / sum p w,  w = exp(x M),  M = S(b0 - (b0+K')/k),
    K' ~ Bin(k-1, S) inputs other than 0 in A.   x -> inf gives S(1-1/k).
"""
from math import ceil, comb, factorial

import numpy as np
from scipy.stats import poisson


def S_er(z, iters=2000):
    z = np.atleast_1d(np.asarray(z, float))
    S = np.where(z > 1, 0.9, 0.0)
    for _ in range(iters):
        S = np.where(z > 1, 1 - np.exp(-z * S), 0.0)
    return S


def mixture(S, k, q=None):
    """List of (prob, M0, anis, K) over (b0, K').  Amplitude S (size of the macroscopic function), each
    input independently controls it with probability q (= S in the OR regime; at higher phi q is the
    probability that a seed triggers the global cascade and S the cascade size)."""
    q = S if q is None else q
    out = []
    for b0 in (0, 1):
        for Kp in range(k):
            p = (q if b0 else 1 - q) * comb(k - 1, Kp) * q ** Kp * (1 - q) ** (k - 1 - Kp)
            K = b0 + Kp
            out.append((p, S * (b0 - K / k), S ** 2 * (K / k) * (1 - K / k), K))
    return out


def tilted_M0(S, k, x, q=None):
    """<M_0> under weight exp(x M_0); x = eps N (array ok)."""
    mix = mixture(S, k, q)
    p = np.array([a[0] for a in mix])
    M = np.array([a[1] for a in mix])
    x = np.atleast_1d(x)[:, None]
    e = x * M[None, :]
    w = p[None, :] * np.exp(e - e.max(axis=1, keepdims=True))
    return (w * M).sum(axis=1) / w.sum(axis=1)


def tilted_empirical(M, x):
    """Reweight unbiased samples M (array) by exp(x M); x scalar or array.  Returns <M>_x."""
    M = np.asarray(M, float)
    x = np.atleast_1d(x)[:, None]
    e = x * M[None, :]
    w = np.exp(e - e.max(axis=1, keepdims=True))
    return (w * M).sum(axis=1) / w.sum(axis=1)


def anis_mean(S, k, q=None):
    q = S if q is None else q
    return S ** 2 * q * (1 - q) * (1 - 1 / k)


def p_broken(q, k):
    """P(A > 0): some but not all inputs control the macroscopic function."""
    return 1 - q ** k - (1 - q) ** k


def anis_quantile(S, k, qq, q=None):
    mix = sorted(mixture(S, k, q), key=lambda a: a[2])
    c = 0.0
    for p, _, a, _ in mix:
        c += p
        if c >= qq:
            return a
    return mix[-1][2]


def stab_dist(S, k):
    """{|Stab|: prob} for the giant's function OR_A."""
    d = {}
    for K in range(k + 1):
        s = factorial(K) * factorial(k - K)
        d[s] = d.get(s, 0) + comb(k, K) * S ** K * (1 - S) ** (k - K)
    return d


def seeded_response(z, phi, rho0, blocked, dmax=80):
    """Gleeson-Cahalane tree theory with ON seeds (fraction rho0) and blocked OFF inputs (fraction blocked):
    final active fraction of the other nodes."""
    d = np.arange(dmax + 1)
    pd_ = poisson.pmf(d, z)
    need = np.array([ceil(phi * x - 1e-9) if x > 0 else 10 ** 9 for x in d])
    q = rho0
    for _ in range(5000):
        s = 0.0
        for dd in range(1, dmax + 1):
            m = np.arange(need[dd], dd)
            if len(m):
                s += dd * pd_[dd] / z * sum(comb(dd - 1, int(mm)) * q ** mm * (1 - q) ** (dd - 1 - mm) for mm in m)
        qn = rho0 + (1 - rho0 - blocked) * s
        if abs(qn - q) < 1e-12:
            break
        q = qn
    f = 0.0
    for dd in range(1, dmax + 1):
        m = np.arange(need[dd], dd + 1)
        f += pd_[dd] * sum(comb(dd, int(mm)) * q ** mm * (1 - q) ** (dd - mm) for mm in m)
    return f
