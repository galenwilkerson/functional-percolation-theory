"""E3: graph disorder vs input-placement disorder (outline §7.1, §8.3).

Each sampled graph is copied Q times with Q independent input placements, so every observable
X splits as  Var(X) = Var_graph(E[X|graph]) + E_graph[Var_placement(X|graph)].
If the Sym_k anisotropy and the near-z_c fluctuations are placement-dominated, the input
asymmetry is quenched selection by placement, not spontaneous symmetry breaking.

  python e3.py --out data/e3            # full grid
  python e3.py --out data/e3 --fit-only # gpufit smoke (gq debug)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observables import DTCCache, engine, function_stats, structure_stats  # noqa: E402
from sweep import TOOLS, atomic_parquet, batch_seed, sha256, should_stop  # noqa: E402

torch.set_grad_enabled(False)

Q = 16
NS = [10000, 100000]
KS = [5, 2, 8]
PHIS = [0.1, 0.25]
LAMBDA = [-2, -1, 0, 1, 2, 4]
CONTEXT_Z = [1.5, 2.0, 3.0]
G_OF_N = {10000: 200, 100000: 50}        # graphs per cell (x Q placements)


def cells():
    out = []
    for k in KS:
        for N in NS:
            zs = sorted({round(z, 6) for z in CONTEXT_Z + [1 + l * N ** (-1 / 3) for l in LAMBDA]})
            for phi in PHIS:
                for z in zs:
                    out.append(dict(N=N, k=k, z=z, phi=phi, G=G_OF_N[N]))
    for i, c in enumerate(out):
        c["cell_id"] = i
    return out


def replicate(g, N, G, Q):
    """GraphBatch of G graphs -> G*Q graphs: copy (g, q) at block g*Q + q."""
    src_l = g.src % N
    dst_l = g.dst % N
    gi = g.src // N
    blocks = (gi.unsqueeze(1) * Q + torch.arange(Q, device=g.src.device)).reshape(-1)
    src = (src_l.unsqueeze(1).expand(-1, Q).reshape(-1)) + blocks * N
    dst = (dst_l.unsqueeze(1).expand(-1, Q).reshape(-1)) + blocks * N
    return engine.graph_from_edges(N, G * Q, src, dst)


def run_batch(bid, N, k, items, by, dev, base_seed, cache):
    G = len(items)
    B = G * Q
    seed = batch_seed(base_seed, bid)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cs = [by[c] for c, _ in items]
    zs = np.array([c["z"] for c in cs])
    m = rng.binomial(N * (N - 1) // 2, zs / (N - 1))
    g = replicate(engine.sample_gnm_batch(N, torch.as_tensor(m, device=dev), dev, gen), N, G, Q)
    inputs = engine.sample_inputs(N, k, B, dev, gen)
    phi_node = torch.tensor([c["phi"] for c in cs], dtype=torch.float64,
                            device=dev).repeat_interleave(Q * N)
    tau = engine.thresholds(g.deg, phi_node)
    L, rounds = engine.cascade_frontier(g, inputs, k, phi_node)
    st, in_gcc = structure_stats(g, inputs, tau, N, B)
    cell_of = torch.as_tensor([c for c, _ in items], device=dev).repeat_interleave(Q)
    fs, _ = function_stats(L, inputs, in_gcc, N, B, k, cache, cell_of, k_in_gcc=st["k_in_gcc"])
    del L
    rows = dict(cell_id=np.repeat([c for c, _ in items], Q), graph=np.repeat([r for _, r in items], Q),
                place=np.tile(np.arange(Q), G), N=N, k=k, z=np.repeat(zs, Q),
                phi=np.repeat([c["phi"] for c in cs], Q), batch=bid, seed=seed, rounds_batch=rounds)
    for d in (st, fs):
        for key, v in d.items():
            a = v.cpu().numpy()
            rows[key] = list(a) if a.ndim == 2 else a
    return pd.DataFrame(rows)


def plan(cs, node_budget, edge_budget):
    batches = []
    groups = {}
    for c in cs:
        groups.setdefault((c["N"], c["k"]), []).append(c)
    for (N, k), sub in groups.items():
        sub = sorted(sub, key=lambda c: (c["z"], c["phi"]))
        gmax = max(1, int(node_budget // (N * Q)))
        gmax = max(1, min(gmax, int(3e8 // (N * Q * (1 << k)))))   # entry budget B*N*2^k
        cur, idx = [], 0

        def close(cur, idx):
            h = hashlib.sha256(json.dumps(cur).encode()).hexdigest()[:10]
            batches.append((f"E3_N{N}_k{k}_b{idx:05d}_{h}", N, k, cur))

        for c in sub:
            e = (c["z"] * N / 2 * 1.05 + 64) * Q
            for r in range(c["G"]):
                if cur and (len(cur) >= gmax or (len(cur) + 1) * e > edge_budget):
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
    ap.add_argument("--k", type=int, nargs="*")
    ap.add_argument("--G", type=int, help="override graphs per cell")
    ap.add_argument("--node-budget", type=float, default=2.5e7)
    ap.add_argument("--edge-budget", type=float, default=6e7)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    cs = [c for c in cells() if (not a.N or c["N"] in a.N) and (not a.k or c["k"] in a.k)]
    if a.G:
        cs = [dict(c, G=a.G) for c in cs]
    by = {c["cell_id"]: c for c in cs}
    batches = plan(cs, a.node_budget, a.edge_budget)
    print(f"cells {len(cs)} graphs {sum(c['G'] for c in cs)} x Q={Q} batches {len(batches)}",
          flush=True)
    caches = {}
    if a.fit_only:  # quick smoke on the costliest group: one full-size batch's memory/util
        sys.path.insert(0, str(TOOLS))
        import gpufit
        N, k = max(NS), max(KS)
        sub = [b for b in batches if b[1] == N and b[2] == k]
        gmax = max(len(b[3]) for b in sub)
        ids = [c["cell_id"] for c in cs if c["N"] == N and c["k"] == k]
        rng = np.random.default_rng(0)
        cache = DTCCache(k)

        def step(b):
            items = sorted([(int(rng.choice(ids)), 0) for _ in range(b)], key=lambda it: by[it[0]]["z"])
            run_batch("fit", N, k, items, by, dev, a.seed, cache)

        rep = gpufit.fit(step, name=f"FP-basic: E3 N{N} k{k}", project="RES-28",
                         units_total=sum(c["G"] for c in cs),
                         key_files=[__file__, str(HERE / "observables.py"), engine.__file__],
                         start=max(1, gmax // 4), seconds_per_size=1.0, max_batch=gmax)
        print("FIT", rep["best_batch"], rep["units_per_s"], rep["peak_vram_gb"], rep["util_mean"],
              rep["projected_min"], rep["verdict"], flush=True)
        return
    meta = dict(args=vars(a), Q=Q, engine_sha256=sha256(Path(engine.__file__)),
                code_sha256={p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / f"meta_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    for bi, (bid, N, k, items) in enumerate(batches):
        pq = out / "batches" / f"{bid}.parquet"
        if pq.exists():
            continue
        if should_stop():
            print("STOPPED: stop file seen; exiting cleanly (resume by re-running)", flush=True)
            sys.exit(3)
        engine.EDGE_CHUNK = max(1 << 14, int(2.5e8 // (1 << k)))
        df = run_batch(bid, N, k, items, by, dev, a.seed, caches.setdefault(k, DTCCache(k)))
        atomic_parquet(df, pq)
        if bi % 10 == 0:
            print(f"[{bi+1}/{len(batches)}] {bid} G={len(items)} elapsed={(time.time()-t0)/60:.1f}min",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
