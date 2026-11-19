"""Tests for ssb_func.py.  pytest Code/tests/test_ssb_func.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ssb  # noqa: E402
import ssb_func as sf  # noqa: E402


def row_of(f, k):
    return np.array([bool(f([(p >> j) & 1 for j in range(k)])) for p in range(1 << k)])


def test_truth_rows_match_codes():
    rng = np.random.default_rng(0)
    for phi in (0.1, 0.25):
        src, dst = ssb.edge_sequence(400, 500, rng)
        A = ssb.adjacency(400, src, dst)
        inputs = rng.choice(400, 4, replace=False)
        X = sf.truth_rows(A, inputs, phi, 4)
        codes = ssb.functions(A, inputs, phi, 4)
        assert np.array_equal((X.astype(np.int64) * (1 << np.arange(16))).sum(axis=1), codes)


def test_symmetry_and_or():
    k = 4
    cases = [(lambda x: x[0] | x[1] | x[2], True, True),
             (lambda x: (x[0] + x[1] + x[2]) >= 2, False, True),     # majority: symmetric, not OR
             (lambda x: x[0] | (x[1] & x[2]), False, False),
             (lambda x: (x[0] & x[1]) | (x[2] & x[3]), False, False)]
    for f, o, sym in cases:
        r = row_of(f, k)
        rel = sf.relevance_rows(r[None, :], k)[0]
        assert sf.is_or_row(r, rel, k) == o
        assert sf.symmetric_in_relevant(r, rel, k)[0] == sym


def test_reach_equals_relevance_in_simple_contagion():
    rng = np.random.default_rng(1)
    N, k = 3000, 4
    src, dst = ssb.edge_sequence(N, 3000, rng)
    A = ssb.adjacency(N, src, dst)
    inputs = rng.choice(N, k, replace=False)
    X = sf.truth_rows(A, inputs, 1e-6, k)
    m, _ = sf.macro_summary(X, inputs, k)
    r = sf.reach(A, inputs, N)
    assert np.allclose(m, r)


def test_relevance_never_exceeds_reach():
    rng = np.random.default_rng(2)
    for _ in range(5):
        rec = sf.run_conn((2000, 3, 0.25, 2.0, int(rng.integers(1e9))))
        for i in range(3):
            assert rec[f"m{i}"] <= rec[f"r{i}"] + 1e-12


def test_single_input_rows_match_truth_table():
    rng = np.random.default_rng(4)
    for phi in (0.1, 0.25):
        src, dst = ssb.edge_sequence(500, 600, rng)
        A = ssb.adjacency(500, src, dst)
        k = 5
        inputs = rng.choice(500, k, replace=False)
        X = sf.truth_rows(A, inputs, phi, k)
        Y = sf.single_input_rows(A, inputs, phi, chunk=2)
        for i in range(k):
            assert np.array_equal(Y[:, i], X[:, 1 << i])
        assert np.array_equal(Y[:, k], X[:, (1 << k) - 1])
