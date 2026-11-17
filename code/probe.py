"""E2-probe: complexity, anisotropy and diversity for growing k (outline §4.4-4.5, §5.3, §6.2-6.3,
§10.5), where exact 2^k truth tables are infeasible.

Probe patterns on a random subset R of r = min(k, R_MAX) inputs (inputs are placed uniformly,
so the first r are a uniform subset):
    col 0          all inputs on                   -> f(1)    (non-constant iff f(1) = 1)
    cols 1..r      only input i on (e_i)           -> f(e_i)  ("OR-type" relevance)
    cols r+1..2r   all on except input i (1 - e_i) -> f(1-e_i) ("AND-type" relevance)
Input i (in R) is detected relevant iff f(e_i) = 1  or  (f(1) = 1 and f(1-e_i) = 0).
This is exact in the OR regime and a lower bound in general (validated vs exact at k <= 8).
Estimates: Psi (exact), Lam/k = mean_{i in R} P(i relevant), anisotropy Var_{i in R} m_i,
D_sig = distinct probe signatures among responsive nodes (a lower bound on diversity).

  python probe.py --out data/e2p [--validate] [--fit-only]
"""
from __future__ import annotations

import argparse
import contextlib
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
from observables import DTCCache, components, engine, function_stats, pack_words, structure_stats  # noqa: E402,F401
from sweep import TOOLS, atomic_parquet, batch_seed, sha256, should_stop  # noqa: E402

torch.set_grad_enabled(False)

R_MAX = 32
NS = [1000, 10000, 100000]
PHIS = [0.1, 0.2]
LAMBDA = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
CONTEXT_Z = [0.5, 0.75, 1.5, 2.0, 3.0]
M_OF_N = {1000: 1000, 10000: 300, 100000: 100}


def k_laws(N):
    out = [("fixed", k, k) for k in (2, 5, 8)]
    out += [("scaled", c, max(1, round(c * N ** (1 / 3)))) for c in (0.3, 1.0, 3.0, 10.0)]
    out += [("rho", r, max(1, round(r * N))) for r in (1e-3, 1e-2)]
    return out


def cells():
    out = []
    for N in NS:
        zs = sorted({round(z, 6) for z in CONTEXT_Z + [1 + l * N ** (-1 / 3) for l in LAMBDA]})
        for phi in PHIS:
            for law, par, k in k_laws(N):
                for z in zs:
                    out.append(dict(N=N, z=z, phi=phi, law=law, par=par, k=k, sel="random",
                                    M=M_OF_N[N]))
            if N == 10000:  # input-selection robustness (§10.5): inputs on the highest-degree nodes
                for law, par, k in [("fixed", 5, 5), ("scaled", 1.0, round(N ** (1 / 3)))]:
                    for z in zs:
                        out.append(dict(N=N, z=z, phi=phi, law=law, par=par, k=k, sel="hub",
                                        M=M_OF_N[N]))
    for i, c in enumerate(out):
        c["cell_id"] = i
    return out


def probe_patterns(k, r, device, dtype=torch.float32):
    """[k, 2r+1] input values per probe column (rows = inputs)."""
    X = torch.zeros(k, 2 * r + 1, dtype=dtype, device=device)
    X[:, 0] = 1
    for i in range(r):
        X[i, 1 + i] = 1
        X[:, r + 1 + i] = 1
        X[i, r + 1 + i] = 0
    return X


@contextlib.contextmanager
def pattern_setup(X):
    """Run engine.cascade_frontier on arbitrary input patterns X [k, P] (instead of all 2^k)."""
    orig = engine._setup

    def setup(g, inputs, k, phi, dtype):
        tau = engine.thresholds(g.deg, phi).unsqueeze(1)
        flat = inputs.reshape(-1)
        L = torch.zeros(g.n_nodes, X.shape[1], dtype=dtype, device=g.src.device)
        L[flat] = X.to(dtype).repeat(inputs.shape[0], 1)
        upd = torch.ones(g.n_nodes, 1, dtype=torch.bool, device=g.src.device)
        upd[flat] = False
        return tau, L, upd

    engine._setup = setup
    try:
        yield
    finally:
        engine._setup = orig


def hub_inputs(g, N, k, B, gen):
    """k highest-degree nodes per graph (ties broken at random)."""
    deg = g.deg.view(B, N).double()
    noise = torch.rand(B, N, device=deg.device, generator=gen, dtype=torch.float64) * 0.5
    idx = (deg + noise).topk(k, dim=1).indices
    return idx + torch.arange(B, device=deg.device).unsqueeze(1) * N


def probe_stats(L, inputs, N, B, k, r):
    dev = L.device
    n = L.shape[0]
    noninput = torch.ones(n, dtype=torch.bool, device=dev)
    noninput[inputs.reshape(-1)] = False
    T = L[noninput] > 0
    gid = (torch.arange(n, device=dev) // N)[noninput]
    f1 = T[:, 0]
    fe = T[:, 1:r + 1]
    fne = T[:, r + 1:2 * r + 1]
    rel_or = fe
    rel = fe | (f1.unsqueeze(1) & ~fne)
    n_ni = float(N - k)

    def gsum(x):
        return torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, gid, x.double())

    m = torch.zeros(B, r, dtype=torch.float64, device=dev).index_add_(0, gid, rel.double()) / n_ni
    nresp = gsum(f1)
    out = dict(psi=nresp / n_ni, lamk=m.mean(1), lamk_or=gsum(rel_or.sum(1)) / (n_ni * r),
               aniso=m.var(1, unbiased=False) if r > 1 else torch.zeros(B, dtype=torch.float64, device=dev),
               n_resp=nresp, m=m)
    out["clam"] = torch.where(nresp > 0, out["lamk"] * k * n_ni / nresp.clamp(min=1),
                              torch.full_like(nresp, float("nan")))   # E|Rel| among responsive
    resp = f1
    nb, rb = 25, r + 1
    zB = torch.zeros(B, dtype=torch.float64, device=dev)
    if bool(resp.any()):
        # functional classes = nodes sharing a probe signature (= the function, in the OR regime)
        w = pack_words(T[resp])
        u, inv, cnt = torch.unique(torch.cat([gid[resp].unsqueeze(1), w], 1), dim=0,
                                   return_inverse=True, return_counts=True)
        g0 = u[:, 0]
        out["D_sig"] = torch.bincount(g0, minlength=B)
        cplx = torch.zeros(u.shape[0], dtype=torch.long, device=dev).scatter_reduce_(
            0, inv, rel[resp].sum(1), reduce="amax")          # |Rel ∩ R| of the class
        cmax = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(0, g0, cnt, reduce="amax")
        pos = torch.arange(cnt.numel(), device=dev)
        first = torch.full((B,), cnt.numel(), dtype=torch.long, device=dev).scatter_reduce_(
            0, g0, torch.where(cnt == cmax[g0], pos, torch.full_like(pos, cnt.numel())), reduce="amin")
        rest = pos != first[g0]
        out["mass_max"] = cmax.double() / n_ni
        out["mass_2nd"] = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(
            0, g0[rest], cnt[rest], reduce="amax").double() / n_ni
        out["chi_F"] = zB.clone().index_add_(0, g0[rest], cnt[rest].double() ** 2) / float(N)
        b = torch.log2(cnt[rest].double()).floor().long().clamp(max=nb - 1)
        out["supp_hist"] = torch.zeros(B, nb, dtype=torch.long, device=dev).index_put_(
            (g0[rest], b), torch.ones_like(b), accumulate=True)
        out["cplx_hist"] = torch.zeros(B, rb, dtype=torch.long, device=dev).index_put_(
            (g0[rest], cplx[rest]), torch.ones_like(b), accumulate=True)
        # mean log2 support per complexity level (support-complexity relation of finite classes)
        out["supp_by_cplx"] = torch.zeros(B, rb, dtype=torch.float64, device=dev).index_put_(
            (g0[rest], cplx[rest]), torch.log2(cnt[rest].double()), accumulate=True)
    else:
        out["D_sig"] = torch.zeros(B, dtype=torch.long, device=dev)
        out["mass_max"], out["mass_2nd"], out["chi_F"] = zB.clone(), zB.clone(), zB.clone()
        out["supp_hist"] = torch.zeros(B, nb, dtype=torch.long, device=dev)
        out["cplx_hist"] = torch.zeros(B, rb, dtype=torch.long, device=dev)
        out["supp_by_cplx"] = torch.zeros(B, rb, dtype=torch.float64, device=dev)
    return out, rel


def run_batch(bid, N, k, items, by, dev, base_seed, validate=False):
    B = len(items)
    r = min(k, R_MAX)
    seed = batch_seed(base_seed, bid)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cs = [by[c] for c, _ in items]
    zs = np.array([c["z"] for c in cs])
    m = rng.binomial(N * (N - 1) // 2, zs / (N - 1))
    g = engine.sample_gnm_batch(N, torch.as_tensor(m, device=dev), dev, gen)
    sel = cs[0]["sel"]
    inputs = hub_inputs(g, N, k, B, gen) if sel == "hub" else engine.sample_inputs(N, k, B, dev, gen)
    phi_node = torch.tensor([c["phi"] for c in cs], dtype=torch.float64,
                            device=dev).repeat_interleave(N)
    X = probe_patterns(k, r, dev)
    with pattern_setup(X):
        L, rounds = engine.cascade_frontier(g, inputs, k, phi_node)
    ps, rel = probe_stats(L, inputs, N, B, k, r)
    del L
    rows = dict(cell_id=[c for c, _ in items], rep=[q for _, q in items], N=N, k=k, r=r,
                z=zs, phi=[c["phi"] for c in cs], law=[c["law"] for c in cs],
                par=[c["par"] for c in cs], sel=sel, batch=bid, seed=seed, rounds_batch=rounds)
    if validate:  # exact truth tables on the SAME graphs/inputs (k <= 8)
        L2, _ = engine.cascade_frontier(g, inputs, k, phi_node)
        tau = engine.thresholds(g.deg, phi_node)
        st, in_gcc = structure_stats(g, inputs, tau, N, B)
        fs, _ = function_stats(L2, inputs, in_gcc, N, B, k, DTCCache(k), torch.zeros(B, dtype=torch.long, device=dev))
        rows["psi_exact"] = fs["psi"].cpu().numpy()
        rows["lamk_exact"] = (fs["lam"] / k).cpu().numpy()
        rows["aniso_exact"] = fs["m"].var(1, unbiased=False).cpu().numpy()
        rows["D_exact"] = fs["D_nc"].cpu().numpy()
        rows["rho_or"] = fs["rho_or"].cpu().numpy()
        rows["mass_max_exact"] = fs["mass_max"].cpu().numpy()
        rows["chi_F_exact"] = fs["chi_F"].cpu().numpy()
        rows["mass_2nd_exact"] = fs["mass_2nd"].cpu().numpy()
    for key, v in ps.items():
        a = v.cpu().numpy()
        rows[key] = list(a) if a.ndim == 2 else a
    return pd.DataFrame(rows)


def plan(cs, node_budget, entry_budget, edge_budget):
    batches = []
    groups = {}
    for c in cs:
        groups.setdefault((c["N"], c["k"], c["sel"]), []).append(c)
    for (N, k, sel), sub in sorted(groups.items()):
        sub = sorted(sub, key=lambda c: (c["z"], c["phi"]))
        r = min(k, R_MAX)
        bmax = max(1, min(int(node_budget // N), int(entry_budget // (N * (2 * r + 1)))))
        cur, idx = [], 0

        def close(cur, idx):
            h = hashlib.sha256(json.dumps(cur).encode()).hexdigest()[:10]
            batches.append((f"P_N{N}_k{k}_{sel}_b{idx:05d}_{h}", N, k, cur))

        for c in sub:
            e = c["z"] * N / 2 * 1.05 + 64
            for q in range(c["M"]):
                if cur and (len(cur) >= bmax or (len(cur) + 1) * e > edge_budget):
                    close(cur, idx)
                    cur, idx = [], idx + 1
                cur.append((c["cell_id"], q))
        if cur:
            close(cur, idx)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--M", type=int)
    ap.add_argument("--validate", action="store_true", help="fixed k<=8 cells only, with exact")
    ap.add_argument("--node-budget", type=float, default=2.5e7)
    ap.add_argument("--entry-budget", type=float, default=3e8)
    ap.add_argument("--edge-budget", type=float, default=6e7)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    cs = [c for c in cells() if not a.N or c["N"] in a.N]
    if a.validate:
        cs = [c for c in cs if c["law"] == "fixed" and c["sel"] == "random"]
    if a.M:
        cs = [dict(c, M=a.M) for c in cs]
    by = {c["cell_id"]: c for c in cs}
    batches = plan(cs, a.node_budget, a.entry_budget, a.edge_budget)
    print(f"cells {len(cs)} realizations {sum(c['M'] for c in cs)} batches {len(batches)}", flush=True)
    if a.fit_only:
        sys.path.insert(0, str(TOOLS))
        import gpufit
        N = max(c["N"] for c in cs)
        kk = max(c["k"] for c in cs if c["N"] == N)
        sub = [b for b in batches if b[1] == N and b[2] == kk]
        bmax = max(len(b[3]) for b in sub)
        ids = [c["cell_id"] for c in cs if c["N"] == N and c["k"] == kk and c["sel"] == "random"]
        rng = np.random.default_rng(0)

        def step(b):
            items = sorted([(int(rng.choice(ids)), 0) for _ in range(b)], key=lambda it: by[it[0]]["z"])
            run_batch("fit", N, kk, items, by, dev, a.seed)

        rep = gpufit.fit(step, name=f"FP-basic: probe N{N} k{kk}", project="RES-28",
                         units_total=sum(c["M"] for c in cs),
                         key_files=[__file__, str(HERE / "observables.py"), engine.__file__],
                         start=max(1, bmax // 4), seconds_per_size=1.0, max_batch=bmax)
        print("FIT", rep["best_batch"], rep["units_per_s"], rep["peak_vram_gb"], rep["util_mean"],
              rep["projected_min"], rep["verdict"], flush=True)
        return
    meta = dict(args=vars(a), R_MAX=R_MAX, engine_sha256=sha256(Path(engine.__file__)),
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
        engine.EDGE_CHUNK = max(1 << 14, int(2.5e8 // (2 * min(k, R_MAX) + 1)))
        atomic_parquet(run_batch(bid, N, k, items, by, dev, a.seed, a.validate), pq)
        if bi % 20 == 0:
            print(f"[{bi+1}/{len(batches)}] {bid} B={len(items)} elapsed={(time.time()-t0)/60:.1f}min",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
