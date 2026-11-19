"""Spec card for the symmetry-breaking runs (SPEC.md, confirmed by Galen 2026-11-18).

check_spec(**run) asserts the card facts for a run and fails loudly; write_with_header() writes a parquet file whose
schema metadata records the facts as actually used. Every ssb* run script calls both.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

CARD = {
    "graph_er": "G(N,M), M = round(zN/2), no self-loops, no multi-edges",      # S1
    "graph_lattice": "LxL square lattice, periodic, bond probability p",      # S4
    "rule": "ON iff #ON neighbours >= phi*deg (tol 1e-9); deg-0 never ON; least fixed point from all-OFF",  # S5
    "inputs": "uniform without replacement, clamped ON/OFF, excluded from node observables",  # S6
    "phis_er": (0.1, 0.2, 0.25),                                              # S7
    "phi_lattice": 0.1,
    "delta": 0.01,                                                            # S9
}
# S18: functional onset z_lo and upper edge z_hi (Watts' condition), keyed by n = floor(1/phi)
Z_LO = {10: 1.000, 6: 1.004, 5: 1.021, 4: 1.114}
Z_HI = {10: 13.66, 5: 5.76, 4: 3.86}


def watts_lhs(z: float, n: int) -> float:
    """(e^{-z}/z) sum_{d=2}^{n} z^d/(d-2)!  (= 1 at z_lo, z_hi)."""
    return math.exp(-z) / z * sum(z ** d / math.factorial(d - 2) for d in range(2, n + 1))


def z_lo(n: int) -> float:
    lo, hi = 0.5, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if watts_lhs(mid, n) < 1 else (lo, mid)
    return (lo + hi) / 2


def check_spec(*, mode: str, phis=(), graph="er", delta=None):
    """Fail loudly if a run departs from the card."""
    allowed = CARD["phis_er"] if graph == "er" else (CARD["phi_lattice"],)
    for phi in phis:
        if not any(abs(phi - a) < 1e-12 for a in allowed) and phi > 1e-5:
            raise AssertionError(f"SPEC S7: phi={phi} not on the card for graph={graph} (mode {mode})")
    if delta is not None and abs(delta - CARD["delta"]) > 1e-12:
        raise AssertionError(f"SPEC S9: DELTA={delta} != {CARD['delta']}")
    for n, v in Z_LO.items():                       # S18: the onset values the analysis uses must match Watts
        if abs(z_lo(n) - v) > 6e-4:
            raise AssertionError(f"SPEC S18: z_lo(n={n}) = {z_lo(n):.4f} != card {v}")


def git_hash() -> str:
    """git short hash if in a repo, else sha1 of the ssb*/spec code files (code fingerprint)."""
    here = Path(__file__).resolve().parent
    try:
        h = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        if h:
            return "git:" + h
    except Exception:
        pass
    import hashlib
    sha = hashlib.sha1()
    for f in sorted(here.glob("ssb*.py")) + [here / "spec_card.py"]:
        sha.update(f.read_bytes())
    return "sha1:" + sha.hexdigest()[:12]


def write_with_header(df, path, **facts):
    """Parquet with the card facts as used in the schema metadata (key 'spec')."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)
    meta = dict(table.schema.metadata or {})
    meta[b"spec"] = json.dumps({"card": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CARD.items()},
                                "used": facts, "code": git_hash()}, default=str).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_header(path):
    import pyarrow.parquet as pq
    md = pq.read_schema(path).metadata or {}
    return json.loads(md[b"spec"]) if b"spec" in md else None
