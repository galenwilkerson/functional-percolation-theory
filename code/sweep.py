"""E1 sweep runner: (N, k, z, phi) -> per-realization observables.

Batches mix cells freely at fixed (N, k): each graph gets its own edge count (its z) and its
own per-node thresholds (its phi), so one cascade call advances many sweep points at once.
Work is split into deterministic batches; each batch writes
    <out>/batches/<id>.parquet   one row per realization
    <out>/batches/<id>_ens.npz   per-cell counts of distinct non-constant functions
atomically (tmp + os.replace).  Re-running skips finished batches, so the job is resumable;
it checks $GQ_STOP_FILE between batches and exits cleanly.

Usage:
  python sweep.py --out data/e1                    # full E1 grid
  python sweep.py --out data/smoke --N 100000 --k 8 --max-batches 2   # timing / VRAM check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid  # noqa: E402
from observables import DTCCache, engine, function_stats, structure_stats  # noqa: E402

torch.set_grad_enabled(False)

PHI_MIN = 1e-6   # U01: keep tau >= 1 (a node must never fire with zero active neighbours)


def should_stop() -> bool:
    f = os.environ.get("GQ_STOP_FILE")
    return bool(f) and os.path.exists(f)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIT_PROJECT = "RES-28"
TOOLS = Path(os.environ.get("GPUFIT_PATH", "."))  # optional GPU-fit helper, used only with --fit-only / --use-fit


def fit_name(stage: str, N: int, k: int) -> str:
    return f"FP-basic: {stage} N{N} k{k}"


def key_files():
    return [str(HERE / f) for f in ("sweep.py", "observables.py", "dtc.py", "grid.py")] + \
           [str(Path(engine.__file__))]


def groups_of(cells):
    """(N, k) -> list of cells, in priority order."""
    g: dict = {}
    for c in cells:
        g.setdefault((c["N"], c["k"]), []).append(c)
    return g


def run_fits(stage, sel, cell_by_id, dev, args):
    """One gpufit smoke per (N, k) group; step(b) = one batch of b realizations sampled
    uniformly from the group's cells (a z-mixed batch: conservative for throughput)."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import gpufit
    total = 0.0
    for (N, k), cs in groups_of(sel).items():
        rng = np.random.default_rng(0)
        ids = [c["cell_id"] for c in cs]
        engine.EDGE_CHUNK = max(1 << 14, int(args.gather_budget // (1 << k)))
        cache = DTCCache(k)
        zmax = max(c["z"] for c in cs)
        # never let a sampled batch exceed the edge budget used in the real run
        bcap = max(1, min(int(args.edge_budget // (zmax * N / 2 * 1.05 + 64)),
                          int(args.node_budget // N)))

        def step(b):
            items = [(int(rng.choice(ids)), 0) for _ in range(b)]
            items.sort(key=lambda it: cell_by_id[it[0]]["z"])
            run_batch("fit", N, k, items, cell_by_id, dev, args.seed, cache)

        # quick smoke: start near the budgeted batch size (not at 1) and time each size ~1 s
        guess = max(1, min(bcap, int(args.entry_budget // (N * (1 << k)))))
        rep = gpufit.fit(step, name=fit_name(stage, N, k),
                         start=max(1, guess // 8), seconds_per_size=1.0,
                         units_total=sum(c["M"] for c in cs), project=FIT_PROJECT,
                         key_files=key_files(),
                         extra=dict(stage=stage, N=N, k=k, edge_budget=args.edge_budget,
                                    gather_budget=args.gather_budget, n_cells=len(cs)),
                         max_batch=min(bcap, sum(c["M"] for c in cs)))
        total += rep["projected_min"] or 0
        print(f"FIT {fit_name(stage, N, k)}: batch {rep['best_batch']} "
              f"{rep['units_per_s']} real/s {rep['peak_vram_gb']} GB util {rep['util_mean']}% "
              f"~{rep['projected_min']} min [{rep['verdict']}]", flush=True)
    print(f"FIT TOTAL projected {total:.1f} min", flush=True)


def plan_batches(cells, entry_budget: float, edge_budget: float, bmax_of=None,
                 node_budget: float = 2.5e7, stage: str = "C"):
    """Group realizations into batches at fixed (N, k), z-sorted, within memory budgets.
    bmax_of(N, k) overrides the entry-budget batch cap (e.g. from a gpufit report).
    Returns list of (batch_id, N, k, [(cell_id, rep), ...])."""
    batches = []
    groups: dict = {}
    for c in cells:
        groups.setdefault((c["N"], c["k"]), []).append(c)
    order = []
    for c in cells:  # preserve priority order of groups
        if (c["N"], c["k"]) not in order:
            order.append((c["N"], c["k"]))
    for (N, k) in order:
        cs = sorted(groups[(N, k)], key=lambda c: (c["z"], c["phi"]))
        P = 1 << k
        bmax = (bmax_of(N, k) if bmax_of else None) or max(1, int(entry_budget // (N * P)))
        bmax = max(1, min(bmax, int(node_budget // N)))
        cur, idx = [], 0
        cell_z = {c["cell_id"]: c["z"] for c in cs}
        cell_phi = {c["cell_id"]: c["phi"] for c in cs}

        def close(cur, idx):
            # id = stage, group, index and a hash of the exact (cell, z, phi, rep) contents,
            # so a changed plan can never silently reuse an old batch file
            h = hashlib.sha256(json.dumps([(c_, cell_z[c_], cell_phi[c_], r_) for c_, r_ in cur])
                               .encode()).hexdigest()[:10]
            batches.append((f"{stage}_N{N}_k{k}_b{idx:05d}_{h}", N, k, cur))

        for c in cs:  # z-sorted, so the current cell has the batch's largest edge count
            e = c["z"] * N / 2 * 1.05 + 64
            for r in range(c["M"]):
                if cur and (len(cur) >= bmax or (len(cur) + 1) * e > edge_budget):
                    close(cur, idx)
                    cur, idx = [], idx + 1
                cur.append((c["cell_id"], r))
        if cur:
            close(cur, idx)
    return batches


def batch_seed(base: int, bid: str) -> int:
    return int(hashlib.sha256(f"{base}:{bid}".encode()).hexdigest()[:15], 16)


def run_batch(bid, N, k, items, cell_by_id, dev, base_seed, dtc_cache):
    B = len(items)
    seed = batch_seed(base_seed, bid)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    zs = np.array([cell_by_id[c]["z"] for c, _ in items])
    pairs = N * (N - 1) // 2
    m = rng.binomial(pairs, zs / (N - 1))
    g = engine.sample_gnm_batch(N, torch.as_tensor(m, device=dev), dev, gen)
    inputs = engine.sample_inputs(N, k, B, dev, gen)
    # per-node thresholds (vectorised): homogeneous phi per graph, or U[0,1] per node
    labs = [cell_by_id[c]["phi"] for c, _ in items]
    is_u = torch.tensor([lab == "U01" for lab in labs], device=dev)
    phi_g = torch.tensor([0.0 if lab == "U01" else float(lab) for lab in labs],
                         dtype=torch.float64, device=dev)
    phi_node = phi_g.repeat_interleave(N)
    if bool(is_u.any()):
        u = torch.rand(B * N, device=dev, generator=gen, dtype=torch.float64).clamp_(min=PHI_MIN)
        phi_node = torch.where(is_u.repeat_interleave(N), u, phi_node)
    tau = engine.thresholds(g.deg, phi_node)
    t0 = time.time()
    L, rounds = engine.cascade_frontier(g, inputs, k, phi_node)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t_casc = time.time() - t0
    st, in_gcc = structure_stats(g, inputs, tau, N, B)
    cell_of_graph = torch.as_tensor([c for c, _ in items], device=dev)
    fs, ens = function_stats(L, inputs, in_gcc, N, B, k, dtc_cache, cell_of_graph,
                             k_in_gcc=st["k_in_gcc"])
    del L
    t_all = time.time() - t0

    rows = {
        "cell_id": [c for c, _ in items], "rep": [r for _, r in items],
        "N": N, "k": k,
        "z": zs, "phi": [cell_by_id[c]["phi"] for c, _ in items],
        "m_edges": m, "batch": bid, "seed": seed, "rounds_batch": rounds,
    }
    for d in (st, fs):
        for key, v in d.items():
            a = v.cpu().numpy()
            rows[key] = list(a) if a.ndim == 2 else a
    df = pd.DataFrame(rows)
    return df, ens, dict(t_casc=t_casc, t_all=t_all, rounds=rounds, B=B)


def atomic_parquet(df, path: Path):
    tmp = path.with_name(path.name + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrs):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="C", choices=list(grid.STAGES))
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--k", type=int, nargs="*")
    ap.add_argument("--phi", nargs="*", help="phi labels, e.g. 0.1 U01")
    ap.add_argument("--z", type=float, nargs="*", help="restrict to these z (smoke tests)")
    ap.add_argument("--M", type=int, help="override realizations per cell")
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--entry-budget", type=float, default=3e8, help="max B*N*2^k per batch")
    ap.add_argument("--edge-budget", type=float, default=6e7, help="max undirected edges/batch")
    ap.add_argument("--node-budget", type=float, default=2.5e7, help="max B*N per batch")
    ap.add_argument("--gather-budget", type=float, default=2.5e8, help="max chunk*2^k entries")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fit-only", action="store_true",
                    help="gpufit smoke per (N,k) group (run inside gq debug), then exit")
    ap.add_argument("--use-fit", action="store_true",
                    help="batch sizes from the saved gpufit reports (edge budget still applies)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)

    allc = grid.cells(args.stage)                       # cell ids fixed by the stage grid
    sel = [c for c in allc
           if (not args.N or c["N"] in args.N) and (not args.k or c["k"] in args.k)
           and (not args.phi or c["phi"] in args.phi)
           and (not args.z or any(abs(c["z"] - z) < 1e-9 for z in args.z))]
    if args.M:
        sel = [dict(c, M=args.M) for c in sel]
    cell_by_id = {c["cell_id"]: c for c in sel}
    if args.fit_only:
        run_fits(args.stage, sel, cell_by_id, dev, args)
        return
    bmax_of = None
    if args.use_fit:
        if str(TOOLS) not in sys.path:
            sys.path.insert(0, str(TOOLS))
        import gpufit

        def bmax_of(N, k):
            rep = gpufit.load(fit_name(args.stage, N, k))
            if not rep:
                sys.exit(f"no gpufit report for {fit_name(args.stage, N, k)}; run --fit-only first")
            return int(rep["best_batch"])
    batches = plan_batches(sel, args.entry_budget, args.edge_budget, bmax_of,
                           args.node_budget, args.stage)
    n_real = sum(len(b[3]) for b in batches)
    print(f"cells {len(sel)}  realizations {n_real}  batches {len(batches)}  device {dev}",
          flush=True)
    if args.dry_run:
        return

    meta = dict(
        args=vars(args), engine_sha256=sha256(Path(engine.__file__)),
        closed_forms_sha256=sha256(Path(engine.__file__).with_name("closed_forms.py")),
        code_sha256={p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out / f"meta_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(meta, indent=1))
    pd.DataFrame(sel).to_parquet(out / f"cells_{args.stage}.parquet", index=False)

    caches: dict[int, DTCCache] = {}
    done = 0
    t_start = time.time()
    for bi, (bid, N, k, items) in enumerate(batches):
        if args.max_batches is not None and done >= args.max_batches:
            break
        pq = out / "batches" / f"{bid}.parquet"
        if pq.exists():
            continue
        if should_stop():
            print("STOPPED: stop file seen; exiting cleanly (resume by re-running)", flush=True)
            sys.exit(3)
        engine.EDGE_CHUNK = max(1 << 14, int(args.gather_budget // (1 << k)))
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        cache = caches.setdefault(k, DTCCache(k))
        df, ens, info = run_batch(bid, N, k, items, cell_by_id, dev, args.seed, cache)
        if ens is not None:
            ens["z"] = np.array([cell_by_id[int(c)]["z"] for c in ens["cell"]])
            ens["phi"] = np.array([cell_by_id[int(c)]["phi"] for c in ens["cell"]])
            ens["N"], ens["k"] = np.array(N), np.array(k)
            atomic_npz(out / "batches" / f"{bid}_ens.npz", **ens)
        atomic_parquet(df, pq)
        done += 1
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        el = time.time() - t_start
        print(f"[{bi+1}/{len(batches)}] {bid} B={info['B']} rounds={info['rounds']} "
              f"casc={info['t_casc']:.2f}s all={info['t_all']:.2f}s peak={peak:.1f}GB "
              f"elapsed={el/60:.1f}min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
