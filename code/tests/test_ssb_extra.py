"""Tests for ssb_extra.py.  pytest Code/tests/test_ssb_extra.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ssb  # noqa: E402
import ssb_extra as se  # noqa: E402
import ssb_func as sf  # noqa: E402
from lattice2d import lattice_edges  # noqa: E402


def test_run_patterns_match_truth_table():
    rng = np.random.default_rng(5)
    for phi in (0.2, 0.25):
        src, dst = ssb.edge_sequence(600, 1500, rng)
        A = ssb.adjacency(600, src, dst)
        k = 4
        inputs = rng.choice(600, k, replace=False)
        X = sf.truth_rows(A, inputs, phi, k)
        pats = ((np.arange(16)[None, :] >> np.arange(k)[:, None]) & 1).astype(bool)
        Y = se.run_patterns(A, inputs, phi, pats)
        assert np.array_equal(X, Y)


def test_lattice_degree_and_simple_contagion():
    rng = np.random.default_rng(6)
    src, dst = lattice_edges(32, 0.7, rng)
    A = ssb.adjacency(32 * 32, src, dst)
    assert np.asarray(A.sum(axis=1)).max() <= 4
    rec = se.run_lattice((32, 0.7, 11))
    assert rec["top_isOR"] or rec["top_fn"] == 0          # phi = 0.1 on degree <= 4: only ORs


def test_counts_cover_range():
    c = se.counts_for(200)
    assert c[0] == 1 and c[-1] == 200 and len(c) >= 10
