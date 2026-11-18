"""Exact mean cluster size of G(N, p) at z = 1 + lam N^{-1/3} (SM Sec. S1, finite-size check).

Uses the connectivity recursion C_k = 1 - sum_{j<k} binom(k-1, j-1) C_j q^{j(k-j)} (q = 1 - p) and
P(|C(v)| = k) = binom(N-1, k-1) C_k q^{k(N-k)}. Prints E|C(v)| / N^{1/3}, to compare with the Aldous
limit g(lam) (aldous.py). High precision is needed: N = 300 works at 80 digits, N = 1000 needs ~400.

  python exact_cluster_size.py 300 0 80      # -> g = 1.683
  python exact_cluster_size.py 1000 0 400    # -> g = 1.720
"""
import sys

import mpmath as mp


def mean_cluster_size(n, lam):
    z = 1 + lam * n ** (-1 / 3)
    p = mp.mpf(z) / (n - 1)
    logq = mp.log(1 - p)
    C = [mp.mpf(0), mp.mpf(1)]                    # C[k]: probability that G(k, p) is connected
    for k in range(2, n + 1):
        s, b = mp.mpf(0), mp.mpf(1)               # b = binom(k-1, j-1)
        for j in range(1, k):
            s += b * C[j] * mp.e ** (logq * j * (k - j))
            b = b * (k - j) / j
        C.append(1 - s)
    E, tot, b = mp.mpf(0), mp.mpf(0), mp.mpf(1)   # b = binom(n-1, k-1)
    for k in range(1, n + 1):
        pk = b * C[k] * mp.e ** (logq * k * (n - k))
        E += k * pk
        tot += pk
        b = b * (n - k) / k
    return E, tot


if __name__ == "__main__":
    n, lam = int(sys.argv[1]), float(sys.argv[2])
    mp.mp.dps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    E, tot = mean_cluster_size(n, lam)
    print(f"N={n} lambda={lam}: sum P = {mp.nstr(tot, 10)} (must be 1), "
          f"E|C| = {mp.nstr(E, 10)}, g_N = {mp.nstr(E / n ** (1 / 3), 8)}")
