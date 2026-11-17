"""Correctness gates for the E1 sweep code.  Run: python -m pytest -q tests  (from Code/)."""
import itertools
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dtc import dtc_bruteforce, dtc_exact  # noqa: E402
from observables import (DTCCache, components, engine, function_stats, pack_words,  # noqa: E402
                         structure_stats, unpack_words)
from ltm_learning.reference import cascade_all_patterns  # noqa: E402

DEV = torch.device("cpu")


# ---------------------------------------------------------------- DTC
def test_dtc_all_functions_k3():
    T = np.array([[(f >> p) & 1 for p in range(8)] for f in range(256)])
    fast = dtc_exact(T, 3)
    slow = [dtc_bruteforce(t, 3) for t in T]
    assert list(fast) == slow


def test_dtc_random_k4_k5():
    rng = np.random.default_rng(0)
    for k in (4, 5):
        T = rng.integers(0, 2, size=(40, 1 << k))
        assert list(dtc_exact(T, k)) == [dtc_bruteforce(t, k) for t in T]


def test_dtc_known():
    k = 4
    p = np.arange(16)
    bit = lambda i: (p >> i) & 1  # noqa: E731
    OR3 = (bit(0) | bit(1) | bit(2))
    AND2 = bit(0) & bit(1)
    MAJ3 = ((bit(0) + bit(1) + bit(2)) >= 2).astype(int)
    X1 = bit(3)
    C = np.zeros(16, int)
    assert list(dtc_exact(np.stack([OR3, AND2, MAJ3, X1, C]), k)) == [3, 2, 3, 1, 0]


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    for P in (2, 32, 64, 256):
        T = rng.integers(0, 2, size=(17, P)).astype(bool)
        w = pack_words(torch.from_numpy(T)).numpy()
        assert np.array_equal(unpack_words(w, P).astype(bool), T)


# ---------------------------------------------------------------- components
def test_components_vs_networkx():
    gen = torch.Generator().manual_seed(3)
    N, B = 80, 6
    m = torch.tensor([20, 40, 60, 80, 120, 200])
    g = engine.sample_gnm_batch(N, m, DEV, gen)
    lab = components(g.n_nodes, g.row, g.col).numpy()
    G = nx.Graph()
    G.add_nodes_from(range(g.n_nodes))
    G.add_edges_from(zip(g.src.tolist(), g.dst.tolist()))
    for comp in nx.connected_components(G):
        comp = list(comp)
        assert len(set(lab[comp])) == 1 and lab[comp[0]] == min(comp)


# ---------------------------------------------------------------- full observables vs brute force
def brute_force_graph(N, src, dst, inputs, k, phi):
    """Independent recomputation of every observable for one graph."""
    S_all = cascade_all_patterns(N, src, dst, inputs, k, phi)    # [N, 2^k]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(src, dst))
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    gcc = comps[0]
    deg = np.array([G.degree(v) for v in range(N)])
    phis = np.full(N, phi) if np.isscalar(phi) else np.asarray(phi)
    tau = np.array([math.ceil(p * d - 1e-9) if d > 0 else np.inf for p, d in zip(phis, deg)])
    vul = [v for v in range(N) if deg[v] >= 1 and tau[v] <= 1]
    Gv = G.subgraph(vul)
    s_vul = max((len(c) for c in nx.connected_components(Gv)), default=0)
    P = 1 << k
    ni = [v for v in range(N) if v not in set(inputs)]
    funcs, resp, rels, gccfun, finfun = [], 0, [], set(), set()
    m = np.zeros(k)
    dtcs = []
    blocked = 0
    for v in ni:
        t = tuple(S_all[v])
        funcs.append(t)
        r = [i for i in range(k) if any(t[p] != t[p ^ (1 << i)] for p in range(P))]
        rels.append(len(r))
        for i in r:
            m[i] += 1
        nonconst = 0 < sum(t) < P
        resp += nonconst
        dtcs.append(dtc_bruteforce(t, k))
        if v in gcc and len(r) < sum(i in gcc for i in inputs):
            blocked += 1
        if nonconst:
            (gccfun if v in gcc else finfun).add(t)
    n_ni = len(ni)
    vals, cnts = np.unique(np.array(funcs), axis=0, return_counts=True)
    pr = cnts / n_ni
    nc = sorted([c for v, c in zip(vals, cnts) if 0 < v.sum() < P], reverse=True)
    mass_max = nc[0] / n_ni if nc else 0.0
    mass_2nd = nc[1] / n_ni if len(nc) > 1 else 0.0
    chi_F = sum(c * c for c in nc[1:]) / N if nc else 0.0
    supp_hist = np.zeros(25, int)
    for c in nc[1:]:
        supp_hist[min(int(np.floor(np.log2(c))), 24)] += 1
    return dict(
        S=len(gcc) / N, S_vul=s_vul / N, k_in_gcc=sum(i in gcc for i in inputs),
        psi=resp / n_ni, lam=np.mean(rels), dtc=np.mean(dtcs), m=m / n_ni,
        D_all=len(vals), H=float(-(pr * np.log(pr)).sum()),
        D_nc=len({f for f in funcs if 0 < sum(f) < P}),
        D_nc_gcc=len(gccfun), D_nc_fin=len(finfun),
        rel_hist=np.bincount(rels, minlength=k + 1),
        n_gcc_blocked=blocked, mass_max=mass_max, mass_2nd=mass_2nd, chi_F=chi_F,
        supp_hist=supp_hist,
    )


@pytest.mark.parametrize("k,phis", [(2, [0.1, 0.3, 0.5, "U"]), (3, [0.05, 0.2, 0.34, "U"]),
                                    (4, [0.15, 0.25, 0.1, 0.5])])
def test_observables_vs_bruteforce(k, phis):
    N = 70
    gen = torch.Generator().manual_seed(10 + k)
    rng = np.random.default_rng(k)
    zs = [0.8, 1.5, 2.5, 4.0, 6.0]
    items = [(z, ph) for z in zs for ph in phis]
    B = len(items)
    m = torch.tensor([int(rng.binomial(N * (N - 1) // 2, z / (N - 1))) for z, _ in items])
    g = engine.sample_gnm_batch(N, m, DEV, gen)
    inputs = engine.sample_inputs(N, k, B, DEV, gen)
    phi_node = torch.empty(B * N, dtype=torch.float64)
    for b, (_, ph) in enumerate(items):
        phi_node[b * N:(b + 1) * N] = (torch.rand(N, generator=gen, dtype=torch.float64)
                                       .clamp(min=1e-6) if ph == "U" else ph)
    tau = engine.thresholds(g.deg, phi_node)
    L, _ = engine.cascade_frontier(g, inputs, k, phi_node)
    st, in_gcc = structure_stats(g, inputs, tau, N, B)
    fs, ens = function_stats(L, inputs, in_gcc, N, B, k, DTCCache(k),
                             torch.arange(B), k_in_gcc=st["k_in_gcc"])
    src, dst = g.src.numpy(), g.dst.numpy()
    for b in range(B):
        sel = (src // N) == b
        bf = brute_force_graph(N, src[sel] - b * N, dst[sel] - b * N,
                               (inputs[b] - b * N).tolist(), k,
                               phi_node[b * N:(b + 1) * N].numpy())
        for key in ("S", "S_vul", "k_in_gcc"):
            assert float(st[key][b]) == pytest.approx(bf[key]), (key, b, items[b])
        for key in ("psi", "lam", "dtc", "D_all", "H", "D_nc", "D_nc_gcc", "D_nc_fin",
                    "n_gcc_blocked", "mass_max", "mass_2nd", "chi_F"):
            assert float(fs[key][b]) == pytest.approx(bf[key]), (key, b, items[b])
        assert np.allclose(fs["m"][b].numpy(), bf["m"])
        assert np.array_equal(fs["rel_hist"][b].numpy(), bf["rel_hist"])
        assert np.array_equal(fs["supp_hist"][b].numpy(), bf["supp_hist"])
        assert int(fs["n_nonmono"][b]) == 0 and int(fs["n_const1"][b]) == 0


def test_or_regime_closed_form():
    """phi small, N large: Psi and Lambda match S[1-(1-S)^k] and k S^2 (statistical)."""
    from ltm_learning.closed_forms import giant_fraction
    N, B, k, z, phi = 20000, 40, 3, 2.0, 0.05
    gen = torch.Generator().manual_seed(7)
    g = engine.sample_gnp_batch(N, z, B, DEV, gen)
    inputs = engine.sample_inputs(N, k, B, DEV, gen)
    tau = engine.thresholds(g.deg, phi)
    L, _ = engine.cascade_frontier(g, inputs, k, phi)
    st, in_gcc = structure_stats(g, inputs, tau, N, B)
    fs, _ = function_stats(L, inputs, in_gcc, N, B, k, DTCCache(k), torch.zeros(B, dtype=torch.long))
    S = giant_fraction(z)
    assert float(fs["psi"].mean()) == pytest.approx(S * (1 - (1 - S) ** k), rel=0.05)
    assert float(fs["lam"].mean()) == pytest.approx(k * S * S, rel=0.06)
    assert float(np.nanmean(fs["rho_or"].numpy())) > 0.99
