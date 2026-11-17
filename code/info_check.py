"""Information carried by the functions, simple-contagion limit, uniform inputs.
Per-node I = mean over non-inputs of H(output); joint J = H(all non-input outputs).
Theory (N->inf, fixed k): I = S E[H2(2^-K)], J = k(e^{-zS}-e^{-z}) + E[H2(2^-K)], K~Bin(k,S)."""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def H2(q):
    q = np.asarray(q, float)
    out = np.zeros_like(q)
    m = (q > 0) & (q < 1)
    out[m] = -q[m] * np.log2(q[m]) - (1 - q[m]) * np.log2(1 - q[m])
    return out


def S_of(z):
    return 0.0 if z <= 1 else brentq(lambda s: s - 1 + np.exp(-z * s), 1e-12, 1)


def theory(z, k):
    s = S_of(z)
    K = np.arange(1, k + 1)
    eh = float((binom.pmf(K, k, s) * H2(2.0 ** -K)).sum())
    # joint: input readable alone iff it has a neighbour in a finite cluster of G\I
    pr = 1 - np.exp(-z * (1 - s))
    pa, pb = s * pr, s * (1 - pr)          # in GCC & readable / in GCC & not readable
    Jor = 0.0
    from math import comb
    for a in range(k + 1):
        for b in range(k + 1 - a):
            w = comb(k, a) * comb(k - a, b) * pa**a * pb**b * (1 - pa - pb)**(k - a - b)
            if b:
                Jor += w * 2.0**-a * float(H2(np.array([2.0**-b]))[0])
    return s * eh, k * pr + Jor


def sample(N, z, k, rng):
    M = rng.poisson(z * N / 2)
    a = rng.integers(0, N, M); b = rng.integers(0, N, M)
    keep = a != b; a, b = a[keep], b[keep]
    inp = rng.choice(N, k, replace=False)
    isin = np.zeros(N, bool); isin[inp] = True
    idx = -np.ones(N, int); idx[inp] = np.arange(k)
    nn = ~isin[a] & ~isin[b]
    g = coo_matrix((np.ones(nn.sum()), (a[nn], b[nn])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    mask = np.zeros(lab.max() + 1, np.int64)
    for u, v in ((a, b), (b, a)):  # input u adjacent to non-input v
        e = isin[u] & ~isin[v]
        np.bitwise_or.at(mask, lab[v[e]], (1 << idx[u[e]]).astype(np.int64))
    m_node = mask[lab[~isin]]
    pc = np.array([bin(int(x)).count("1") for x in range(1 << k)]) if k <= 16 else None
    I = float(H2(2.0 ** -pc[m_node]).mean())
    ms = np.unique(m_node[m_node > 0])
    X = np.arange(1 << k)
    Y = ((X[:, None] & ms[None, :]) != 0)
    _, cnt = np.unique(Y, axis=0, return_counts=True)
    p = cnt / cnt.sum()
    J = float(-(p * np.log2(p)).sum())
    return I, J


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print("theory peaks (z grid 0.2..6):")
    zg = np.linspace(0.2, 6, 2901)
    for k in (4, 8, 16, 32, 64):
        T = np.array([theory(z, k) for z in zg])
        print(f"  k={k:3d}  per-node peak z={zg[T[:,0].argmax()]:.3f}  joint peak z={zg[T[:,1].argmax()]:.3f}"
              f"  J_max={T[:,1].max():.2f} bits")
    print(f"simulation N={N}, {reps} graphs per point, k=8:")
    for z in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        r = np.array([sample(N, z, 8, rng) for _ in range(reps)])
        tI, tJ = theory(z, 8)
        print(f"  z={z:.1f}  I sim {r[:,0].mean():.4f} th {tI:.4f} | J sim {r[:,1].mean():.3f} th {tJ:.3f}")
