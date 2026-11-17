"""E2/E4 field experiment: the k inputs as the symmetry-breaking (ghost) field h = k/N.

For monotone LTM functions with f(0) = 0, a node computes a non-constant function iff it is
active when ALL inputs are on.  So Psi needs one cascade per graph (the all-on pattern), and k
can be arbitrary (up to rho N): this is percolation from k random sources, i.e. the classic
ghost-field construction with field h = k/N.

  python field.py --out data/e2f              # full grid
  python field.py --out data/e2f --fit-only   # gpufit smoke (in gq debug)

Per realization: S (GCC fraction), Psi (all-on activity among non-inputs), Psi_gcc, n_src_gcc.
Grid: N in {1e3..1e6}; z in the critical window (lambda in [-4, 4]) + context; phi in
{0.05, 0.1, 0.2}; field via k in {1, 2, 5} (h -> 0 first), k = c N^{1/3} (scaled field
c = h N^{2/3} in {0.1, 0.3, 1, 3, 10}) and k = rho N (rho in {1e-3, 1e-2}; finite field).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observables import components, engine  # noqa: E402
from sweep import TOOLS, atomic_parquet, batch_seed, should_stop, sha256  # noqa: E402

torch.set_grad_enabled(False)

NS = [1000, 10000, 100000, 1000000]
PHIS = [0.05, 0.1, 0.2]
LAMBDA = [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
CONTEXT_Z = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0]
K_FIXED = [1, 2, 5]
C_SCALED = [0.1, 0.3, 1.0, 3.0, 10.0]          # k = c N^{1/3}
RHO = [1e-3, 1e-2]                             # k = rho N
M_OF_N = {1000: 2000, 10000: 1000, 100000: 300, 1000000: 100}


def k_values(N):
    ks = [("fixed", k, k) for k in K_FIXED]
    ks += [("scaled", c, max(1, round(c * N ** (1 / 3)))) for c in C_SCALED]
    ks += [("rho", r, max(1, round(r * N))) for r in RHO]
    return ks


def cells():
    out = []
    for N in NS:
        zs = sorted({round(z, 6) for z in CONTEXT_Z + [1 + l * N ** (-1 / 3) for l in LAMBDA]})
        for phi in PHIS:
            for law, par, k in k_values(N):
                for z in zs:
                    out.append(dict(N=N, z=z, phi=phi, law=law, par=par, k=k, M=M_OF_N[N]))
    for i, c in enumerate(out):
        c["cell_id"] = i
    return out


@contextlib.contextmanager
def all_on_setup():
    """Make engine.cascade_frontier run ONE pattern with every input clamped on."""
    orig = engine._setup

    def setup(g, inputs, k, phi, dtype):
        tau = engine.thresholds(g.deg, phi).unsqueeze(1)
        flat = inputs.reshape(-1)
        L = torch.zeros(g.n_nodes, 1, dtype=dtype, device=g.src.device)
        L[flat] = 1
        upd = torch.ones(g.n_nodes, 1, dtype=torch.bool, device=g.src.device)
        upd[flat] = False
        return tau, L, upd

    engine._setup = setup
    try:
        yield
    finally:
        engine._setup = orig


def sample_sources(N, ks, dev, gen):
    """Distinct uniform sources per graph, ragged k: returns (flat global ids, graph id)."""
    B = len(ks)
    kmax = int(max(ks))
    r = torch.rand(B, N, device=dev, generator=gen)
    idx = r.topk(kmax, dim=1).indices                       # [B, kmax] distinct per row
    kt = torch.as_tensor(ks, device=dev)
    keep = torch.arange(kmax, device=dev).unsqueeze(0) < kt.unsqueeze(1)
    gid = torch.arange(B, device=dev).unsqueeze(1).expand(B, kmax)[keep]
    return idx[keep] + gid * N, gid


def run_batch(bid, N, items, cell_by_id, dev, base_seed):
    B = len(items)
    seed = batch_seed(base_seed, bid)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cs = [cell_by_id[c] for c, _ in items]
    zs = np.array([c["z"] for c in cs])
    m = rng.binomial(N * (N - 1) // 2, zs / (N - 1))
    g = engine.sample_gnm_batch(N, torch.as_tensor(m, device=dev), dev, gen)
    src, sgid = sample_sources(N, [c["k"] for c in cs], dev, gen)
    phi_node = torch.tensor([c["phi"] for c in cs], dtype=torch.float64,
                            device=dev).repeat_interleave(N)
    # cascade_frontier needs inputs as a 2-D tensor; ragged k -> pass [1, total] (only flat use)
    with all_on_setup():
        L, rounds = engine.cascade_frontier(g, src.unsqueeze(0), 1, phi_node)
    act = L[:, 0] > 0
    n = g.n_nodes
    nid = torch.arange(n, device=dev)
    gid = nid // N
    is_src = torch.zeros(n, dtype=torch.bool, device=dev)
    is_src[src] = True
    lab = components(n, g.row, g.col)
    size_of = torch.bincount(lab, minlength=n)[lab]
    key = size_of * n + (n - 1 - lab)
    best = torch.full((B,), -1, dtype=torch.long, device=dev).scatter_reduce_(
        0, gid, key, reduce="amax")
    gcc_lab = n - 1 - best % n
    in_gcc = lab == gcc_lab[gid]

    def gsum(x):
        return torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, gid, x.double())

    nni = gsum(~is_src)
    rows = dict(
        cell_id=[c for c, _ in items], rep=[r for _, r in items], N=N, z=zs,
        phi=[c["phi"] for c in cs], law=[c["law"] for c in cs], par=[c["par"] for c in cs],
        k=[c["k"] for c in cs], h=[c["k"] / N for c in cs], m_edges=m, batch=bid, seed=seed,
        rounds_batch=rounds,
        S=(best // n).double().cpu().numpy() / N,
        psi=(gsum(act & ~is_src) / nni).cpu().numpy(),
        psi_gcc=(gsum(act & ~is_src & in_gcc) / nni).cpu().numpy(),
        n_src_gcc=torch.zeros(B, device=dev).index_add_(
            0, sgid, in_gcc[src].float()).cpu().numpy(),
    )
    return pd.DataFrame(rows), rounds


def plan(cs, node_budget, edge_budget):
    batches = []
    for N in NS:
        sub = sorted([c for c in cs if c["N"] == N], key=lambda c: (c["z"], c["phi"], c["k"]))
        bmax = max(1, int(node_budget // N))
        cur, idx = [], 0

        def close(cur, idx):
            h = hashlib.sha256(json.dumps([(c_, r_) for c_, r_ in cur]).encode()).hexdigest()[:10]
            batches.append((f"F_N{N}_b{idx:05d}_{h}", N, cur))

        for c in sub:
            e = c["z"] * N / 2 * 1.05 + 64
            for r in range(c["M"]):
                if cur and (len(cur) >= bmax or (len(cur) + 1) * e > edge_budget):
                    close(cur, idx)
                    cur, idx = [], idx + 1
                cur.append((c["cell_id"], r))
        if cur:
            close(cur, idx)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--M", type=int)
    ap.add_argument("--node-budget", type=float, default=4e7)
    ap.add_argument("--edge-budget", type=float, default=6e7)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--max-batches", type=int)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    cs = [c for c in cells() if not a.N or c["N"] in a.N]
    if a.M:
        cs = [dict(c, M=a.M) for c in cs]
    by = {c["cell_id"]: c for c in cs}
    batches = plan(cs, a.node_budget, a.edge_budget)
    print(f"cells {len(cs)} realizations {sum(c['M'] for c in cs)} batches {len(batches)}",
          flush=True)
    if a.fit_only:  # one quick fit at the costliest N, batch = graphs
        sys.path.insert(0, str(TOOLS))
        import gpufit
        N = max(c["N"] for c in cs)
        ids = [c["cell_id"] for c in cs if c["N"] == N]
        rng = np.random.default_rng(0)

        def step(b):
            items = sorted([(int(rng.choice(ids)), 0) for _ in range(b)],
                           key=lambda it: by[it[0]]["z"])
            run_batch("fit", N, items, by, dev, a.seed)

        rep = gpufit.fit(step, name=f"FP-basic: field N{N}", project="RES-28",
                         units_total=sum(c["M"] for c in cs if c["N"] == N),
                         key_files=[__file__, str(HERE / "observables.py"), engine.__file__],
                         start=max(1, int(a.node_budget // N) // 8), seconds_per_size=1.0,
                         max_batch=max(1, int(a.node_budget // N)))
        print("FIT", rep["best_batch"], rep["units_per_s"], rep["peak_vram_gb"],
              rep["util_mean"], rep["projected_min"], rep["verdict"], flush=True)
        return
    meta = dict(args=vars(a), engine_sha256=sha256(Path(engine.__file__)),
                code_sha256={p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / f"meta_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(meta, indent=1))
    pd.DataFrame(cs).to_parquet(out / "cells_field.parquet", index=False)
    t0, done = time.time(), 0
    for bi, (bid, N, items) in enumerate(batches):
        if a.max_batches is not None and done >= a.max_batches:
            break
        pq = out / "batches" / f"{bid}.parquet"
        if pq.exists():
            continue
        if should_stop():
            print("STOPPED: stop file seen; exiting cleanly (resume by re-running)", flush=True)
            sys.exit(3)
        df, rounds = run_batch(bid, N, items, by, dev, a.seed)
        atomic_parquet(df, pq)
        done += 1
        if bi % 20 == 0:
            print(f"[{bi+1}/{len(batches)}] {bid} B={len(items)} rounds={rounds} "
                  f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
