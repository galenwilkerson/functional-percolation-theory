"""Closed forms used as validation gates (proposal Appendix A2-A4)."""
from __future__ import annotations

import math

from scipy.optimize import brentq


def giant_fraction(z: float) -> float:
    """ER giant-component fraction: root of S = 1 - exp(-zS), zero for z <= 1."""
    if z <= 1.0:
        return 0.0
    return brentq(lambda s: 1.0 - math.exp(-z * s) - s, 1e-12, 1.0)


def psi(z: float, k: int) -> float:
    """Disjunctive-regime order parameter: P[node function non-constant] = S[1-(1-S)^k]."""
    S = giant_fraction(z)
    return S * (1.0 - (1.0 - S) ** k)


def function_entropy(z: float, k: int) -> float:
    """H_F(z) in bits from P[Rel = A] = S^(|A|+1)(1-S)^(k-|A|), P[empty] = 1 - psi."""
    S = giant_fraction(z)
    ps = [1.0 - psi(z, k)]
    for j in range(1, k + 1):
        ps += [S ** (j + 1) * (1 - S) ** (k - j)] * math.comb(k, j)
    return -sum(p * math.log2(p) for p in ps if p > 0)


def watts_G(z: float, phi: float) -> float:
    """Vulnerable-cluster function for Poisson degree, D = floor(1/phi)."""
    D = math.floor(1.0 / phi + 1e-12)
    return math.exp(-z) / z * sum(z ** d / math.factorial(d - 2) for d in range(2, D + 1))


def watts_G_uniform(z: float, phi_mean: float, width: float, dmax: int = 400) -> float:
    """Vulnerable-cluster function with per-node phi ~ U[phi_mean(1-w), phi_mean(1+w)].

    G = sum_d d(d-1) p_d P(phi <= 1/d) / z, Poisson p_d; width = 0 recovers watts_G.
    """
    if width == 0:
        return watts_G(z, phi_mean)
    lo, hi = phi_mean * (1 - width), phi_mean * (1 + width)
    total, logp = 0.0, -z  # log Poisson(0)
    for d in range(1, dmax + 1):
        logp += math.log(z) - math.log(d)
        if d >= 2:
            v = min(1.0, max(0.0, (1.0 / d - lo) / (hi - lo)))
            if v == 0.0:
                break
            total += d * (d - 1) * math.exp(logp) * v
    return total / z


def watts_G_normal(z: float, phi_mean: float, sigma: float, dmax: int = 400) -> float:
    """Vulnerable-cluster function with per-node phi ~ N(phi_mean, sigma), clipped to
    [PHI_MIN, 1] (clipping keeps P(phi <= 1/d) unchanged for d >= 2).  sigma = 0 recovers watts_G."""
    if sigma == 0:
        return watts_G(z, phi_mean)
    total, logp = 0.0, -z
    for d in range(1, dmax + 1):
        logp += math.log(z) - math.log(d)
        if d >= 2:
            v = 0.5 * math.erfc(-(1.0 / d - phi_mean) / (sigma * math.sqrt(2.0)))
            if v < 1e-15:
                break
            total += d * (d - 1) * math.exp(logp) * v
    return total / z


PHI_MIN = 0.01


def cascade_window(phi: float, zmax: float = 40.0, n: int = 40000, width: float = 0.0,
                   sigma: float = 0.0):
    """Roots of G(z) = 1 as (lower, upper), or None if the window is empty.

    width > 0 uses the uniform threshold-disorder version of G; sigma > 0 the normal one.
    """
    if sigma > 0:
        G = lambda z: watts_G_normal(z, phi, sigma)  # noqa: E731
    elif width == 0:
        G = lambda z: watts_G(z, phi)  # noqa: E731
    else:
        G = lambda z: watts_G_uniform(z, phi, width)  # noqa: E731
    zs = [1e-3 + i * (zmax - 1e-3) / n for i in range(n + 1)]
    f = [G(z) - 1.0 for z in zs]
    roots = [brentq(lambda z: G(z) - 1.0, zs[i], zs[i + 1])
             for i in range(n) if f[i] * f[i + 1] < 0]
    return (roots[0], roots[-1]) if len(roots) >= 2 else None
