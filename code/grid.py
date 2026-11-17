"""Pre-registered parameter grid for experiment E1 (RESEARCH_PROPOSAL.md, section E1).

A cell is (N, k, z, phi).  phi is a homogeneous threshold fraction, or "U01" (per-node
phi ~ U[0,1], the Sci. Rep. 2022 setting).  The z grid is adapted to each phi's analytic
cascade window [z_lo, z_up] (Watts vulnerable-cluster condition, Poisson degrees):
  * a global base grid, truncated a little beyond z_up,
  * the lower critical window  z = z_lo (1 + lam N^{-1/3}),
  * points straddling the upper edge  z = z_up (1 + d).
"""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE_SRC = Path(__file__).resolve().parent
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))
from ltm_learning.closed_forms import cascade_window  # noqa: E402

PHIS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, "U01"]
KS = [5, 1, 2, 3, 4, 6, 8]           # priority order: k=5 (published setting) first
NS = [1000, 10000, 100000]
M_OF_N = {1000: 1000, 10000: 400, 100000: 100}   # realizations per cell

BASE_Z = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
          8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
ZMAX_FACTOR = 2.0   # finite N and k push the observed upper window edge well above z_up
LOWER_LAMBDA = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
# Upper cascade-window edge: deferred by the outline (§11.5 open extensions) -> no refinement.
UPPER_DELTA: list = []
DEAD_ZMAX = 12.0                       # phi with no window (and U01): base grid up to here


def window(phi):
    if phi == "U01":
        return None
    return cascade_window(float(phi), zmax=40.0)


def _dedupe(zs, keep_extra, rel=0.005):
    """Sort and drop base-grid points within `rel` of a window/edge point (keep_extra)."""
    zs = sorted(set(zs))
    out = []
    for z in zs:
        if z in keep_extra:
            out.append(z)
        elif not any(abs(z - e) <= rel * e for e in keep_extra):
            out.append(z)
    # collapse exact near-duplicates among the extras too
    final = []
    for z in out:
        if not final or abs(z - final[-1]) > 1e-4 * z:
            final.append(z)
    return final


def z_grid(N: int, phi, coarse: bool = False) -> list[float]:
    w = window(phi)
    if coarse:
        zmax = DEAD_ZMAX if w is None else max(4.0, ZMAX_FACTOR * w[1])
        return [z for z in BASE_Z if z <= zmax]
    if w is None:
        base = [z for z in BASE_Z if z <= DEAD_ZMAX]
        extra = [1.0 + lam * N ** (-1 / 3) for lam in LOWER_LAMBDA]  # structural window
    else:
        zlo, zup = w
        zmax = max(4.0, ZMAX_FACTOR * zup)
        base = [z for z in BASE_Z if z <= zmax]
        extra = [zlo * (1 + lam * N ** (-1 / 3)) for lam in LOWER_LAMBDA]
        if abs(zlo - 1.0) > 1e-3:  # structural window too: the coincidence-of-onsets test
            extra += [1.0 + lam * N ** (-1 / 3) for lam in LOWER_LAMBDA]
        extra += [zup * (1 + d) for d in UPPER_DELTA]
    extra = [round(z, 6) for z in extra if z > 0]
    return _dedupe([round(z, 6) for z in base] + extra, set(extra))


def phi_label(phi) -> str:
    return phi if isinstance(phi, str) else f"{phi:g}"


# Stage C (coarse map) and stage F (full / refined).  Cell ids are fixed per stage.
STAGES = {
    "C": dict(Ns=[1000, 10000], Ks=[5, 2, 8], phis=PHIS, coarse=True,
              M={1000: 300, 10000: 100}),
    "F": dict(Ns=NS, Ks=KS, phis=PHIS, coarse=False, M=M_OF_N),
    # W = critical-window stage (replaces F after E1-C; Galen 2026-11-15: "we mostly care what
    # happens at z_c" -> classic symmetry-breaking analysis: order parameters, beta, gamma, FSS)
    "W": dict(Ns=[1000, 10000, 100000, 1000000], Ks=KS, phis=[0.05, 0.1, 0.2, 0.25, "U01"],
              window=True, M={1000: 2000, 10000: 1000, 100000: 300, 1000000: 100},
              skip=lambda N, k: N >= 1000000 and k not in (2, 5)),
}

WINDOW_LAMBDA = [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
CONTEXT_Z = [0.5, 0.75, 1.5, 2.0, 3.0]


def z_grid_window(N: int, phi) -> list[float]:
    """Dense lambda grid around z_c = 1 (and around z_lo if the cascade onset differs)."""
    w = window(phi)
    centers = [1.0]
    if w is not None and abs(w[0] - 1.0) > 1e-3:
        centers.append(w[0])
    zs = [c * (1 + lam * N ** (-1 / 3)) for c in centers for lam in WINDOW_LAMBDA]
    extra = {round(z, 6) for z in zs if z > 0}
    return _dedupe([round(z, 6) for z in CONTEXT_Z] + sorted(extra), extra)


def _is_base(z):
    return any(abs(z - b) < 1e-9 for b in BASE_Z)


def cells(stage: str = "F"):
    """All cells of a stage, in run-priority order: by k (Ks order), then N ascending."""
    s = STAGES[stage]
    out = []
    for k in s["Ks"]:
        for N in s["Ns"]:
            for phi in s["phis"]:
                if "skip" in s and s["skip"](N, k):
                    continue
                zs = z_grid_window(N, phi) if s.get("window") else \
                    z_grid(N, phi, coarse=s["coarse"])
                for z in zs:
                    out.append(dict(N=N, k=k, z=z, phi=phi_label(phi), M=s["M"][N]))
    for i, c in enumerate(out):
        c["cell_id"] = i
    return out


if __name__ == "__main__":
    for st in STAGES:
        cs = cells(st)
        print(f"stage {st}: cells {len(cs)}  realizations {sum(c['M'] for c in cs)}")
    for phi in PHIS:
        print(phi_label(phi), window(phi), len(z_grid(10000, phi)), z_grid(10000, phi))
