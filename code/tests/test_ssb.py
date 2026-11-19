"""Tests for ssb.py / ssb_theory.py.  pytest Code/tests/test_ssb.py"""
import sys
from math import comb, factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ssb  # noqa: E402
import ssb_theory as th  # noqa: E402


def naive_functions(N, edges, inputs, phi, k):
    nb = [[] for _ in range(N)]
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)
    codes = np.zeros(N, np.int64)
    for p in range(1 << k):
        on = [False] * N
        for j, u in enumerate(inputs):
            on[u] = bool((p >> j) & 1)
        changed = True
        while changed:                       # sequential (asynchronous) updates
            changed = False
            for u in range(N):
                if u in inputs or on[u] or not nb[u]:
                    continue
                if sum(on[v] for v in nb[u]) >= phi * len(nb[u]) - 1e-9:
                    on[u] = True
                    changed = True
        for u in range(N):
            if on[u]:
                codes[u] |= np.int64(1) << p
    return codes


def code_of(f, k):
    return sum(int(bool(f([(p >> j) & 1 for j in range(k)]))) << p for p in range(1 << k))


def test_cascade_matches_naive():
    rng = np.random.default_rng(1)
    for trial in range(40):
        N = 30
        k = int(rng.integers(2, 5))
        phi = float(rng.choice([0.1, 0.2, 0.25, 0.34, 0.5]))
        src, dst = ssb.edge_sequence(N, int(rng.integers(15, 60)), rng)
        inputs = [int(x) for x in rng.choice(N, k, replace=False)]
        A = ssb.adjacency(N, src, dst)
        got = ssb.functions(A, np.array(inputs), phi, k)
        want = naive_functions(N, list(zip(src, dst)), inputs, phi, k)
        assert np.array_equal(got, want), (trial, k, phi)


def test_edge_sequence_simple():
    rng = np.random.default_rng(2)
    s, d = ssb.edge_sequence(500, 1500, rng)
    assert len(s) == 1500 and (s != d).all() and (s < d).all()
    assert len(np.unique(s * 500 + d)) == 1500


def test_function_properties():
    k = 3
    cases = {   # name: (f, n_relevant, is_or, dtc, |stab|)
        "zero": (lambda x: 0, 0, False, 0, 6),
        "x0": (lambda x: x[0], 1, True, 1, 2),
        "x0|x1": (lambda x: x[0] | x[1], 2, True, 2, 2),
        "or3": (lambda x: x[0] | x[1] | x[2], 3, True, 3, 6),
        "and3": (lambda x: x[0] & x[1] & x[2], 3, False, 3, 6),
        "maj": (lambda x: x[0] + x[1] + x[2] >= 2, 3, False, 3, 6),
        "x0|(x1&x2)": (lambda x: x[0] | (x[1] & x[2]), 3, False, 3, 2),
        "x0&x1": (lambda x: x[0] & x[1], 2, False, 2, 2),
    }
    codes = np.array([code_of(c[0], k) for c in cases.values()])
    rel, bits = ssb.relevance(codes, k)
    orf = ssb.is_or(bits, rel, k)
    st = ssb.stabilizer_size(codes, k)
    from dtc import dtc_exact
    dt = dtc_exact(bits.astype(np.int8), k)
    for i, (name, (_, nr, o, d, s)) in enumerate(cases.items()):
        assert rel[i].sum() == nr, name
        assert orf[i] == o, name
        assert dt[i] == d, name
        assert st[i] == s, name


def test_or_stabilizer_general():
    for k in (4, 5):
        for K in range(1, k + 1):
            c = code_of(lambda x: any(x[:K]), k)
            assert ssb.stabilizer_size(np.array([c]), k)[0] == factorial(K) * factorial(k - K)


def test_theory_limits():
    for k in (3, 4, 5):
        for S in (0.3, 0.6, 0.9):
            mix = th.mixture(S, k)
            assert abs(sum(a[0] for a in mix) - 1) < 1e-12
            assert abs(sum(a[0] * a[1] for a in mix)) < 1e-12           # symmetric ensemble
            assert abs(sum(a[0] * a[2] for a in mix) - th.anis_mean(S, k)) < 1e-12
            assert abs(th.tilted_M0(S, k, 0)[0]) < 1e-12
            assert abs(th.tilted_M0(S, k, 1e5)[0] - S * (1 - 1 / k)) < 1e-9
            sd = th.stab_dist(S, k)
            assert abs(sum(sd.values()) - 1) < 1e-12
    assert abs(th.S_er(2.0)[0] - 0.7968121300) < 1e-8


def test_tilted_empirical():
    M = np.array([-1.0, 1.0])
    x = np.array([0.0, 1.0, 50.0])
    got = th.tilted_empirical(M, x)
    assert np.allclose(got, np.tanh(x))


def test_giant_or_regime():
    """phi small enough that every vertex is vulnerable: the giant's function is OR of the inputs in it."""
    rng = np.random.default_rng(3)
    N, k = 3000, 4
    src, dst = ssb.edge_sequence(N, 3000, rng)          # z = 2
    A = ssb.adjacency(N, src, dst)
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(A, directed=False)
    for _ in range(5):
        inputs = rng.choice(N, k, replace=False)
        codes = ssb.functions(A, inputs, 1e-6, k)        # simple contagion
        rec = ssb.summarize(codes, inputs, lab, k, ssb.FnCache(k), census=False)
        big = np.argmax(np.bincount(lab))
        inA = [int(lab[u] == big) for u in inputs]
        # clamped-off inputs inside the giant can cut it; the modal function must still be OR
        assert rec["giant_isOR"] or rec["giant_fn"] == 0
        assert rec["giant_nrel"] <= sum(inA)


def test_common_stabiliser():
    k = 3
    a = code_of(lambda x: x[0] | x[1], k)          # stab {e, (01)}
    b = code_of(lambda x: x[1] | x[2], k)          # stab {e, (12)}
    m = ssb.stabilizer_mask(np.array([a, b]), k)
    assert m[0].sum() == 2 and m[1].sum() == 2 and np.logical_and(m[0], m[1]).sum() == 1


def test_sample_record_and_perturbation():
    rec = ssb.run_sample((2000, 3, 0.1, 2.0, 5))[0]
    for key in ("macro_stab", "top_stab", "pert_dm0", "pert_changed", "giant_stab"):
        assert key in rec
    assert -1 <= rec["pert_dm0"] <= 1 and 0 <= rec["pert_changed"] <= 1
    # adding an edge from input 0 can only add reach in simple contagion: dm0 >= 0
    rec = ssb.run_sample((2000, 3, 1e-6, 2.0, 6))[0]
    assert rec["pert_dm0"] >= -1e-12


def test_theory_q():
    S, q, k = 0.8, 0.3, 4
    mix = th.mixture(S, k, q)
    assert abs(sum(a[0] * a[2] for a in mix) - th.anis_mean(S, k, q)) < 1e-12
    pb = sum(a[0] for a in mix if a[2] > 0)
    assert abs(pb - th.p_broken(q, k)) < 1e-12
