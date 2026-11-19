"""Tests for the SC collapse fit.  pytest Code/tests/test_ssb_fss.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ssb_fss as sf  # noqa: E402
import ssb_fss_analyze as sa  # noqa: E402


def test_fit_recovers_known_exponents():
    rng = np.random.default_rng(0)
    rows = []
    for N in (1000, 4000, 16000, 64000):
        for lam in np.arange(-3, 4.01, 0.5):
            x = lam * N ** (0.40 - 1 / 3)           # true b = 0.40
            y = np.exp(0.6 * x) * N ** (-0.7)       # true a = 0.7
            rows.append(dict(phi=0.1, N=N, lam=lam, absM0=y * (1 + 0.01 * rng.standard_normal()), A=y, chi=y))
    cm = pd.DataFrame(rows)
    a, b, q = sa.fit(cm, "absM0", 2 / 3)
    assert abs(a - 0.7) < 0.03 and abs(b - 0.40) < 0.03


def test_z_grid():
    assert abs(sf.z_of(0.1, 0.0, 1000) - 1.0) < 1e-12
    assert abs(sf.z_of(0.25, 1.0, 1000) - (1.114 + 0.1)) < 1e-9
    t = sf.tasks(smoke=True)
    assert all(len(x) == 6 for x in t)
