"""Batched linear-threshold cascade engine (PyTorch).

One cascade in matrix form, for adjacency A and labelling vector L (1 = active):

    d = A·1,   tau = ceil(phi * d),   L_{t+1} = L_t  OR  [A·L_t >= tau],   inputs clamped.

Vectorisation:
  * L is [nodes, 2^k]: every input pattern is a column, so a node's final row is its truth table.
  * B graphs are stacked block-diagonally (node ids offset by b*N), so one pass advances B * 2^k
    cascades.  A is never dense: A·L is an index_add over the directed edge list.

Two solvers that produce identical iterates:
  * cascade_dense:    recompute C = A·L from all edges every iteration (as in fp_gpu.py).
  * cascade_frontier: keep C = A·L and add A·dL, where dL marks nodes that just switched on.
                      By linearity this is the same C; each node enters dL at most once per
                      pattern, so the work is a BFS from the seeds.  Small frontiers gather only
                      their own out-edges (push); large ones do a full-edge pass of A·dL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

torch.set_grad_enabled(False)

_EPS = 1e-9


@dataclass
class GraphBatch:
    """B undirected simple graphs on N nodes each, stacked block-diagonally."""

    N: int
    B: int
    src: torch.Tensor     # [E] long, each undirected edge once (global ids)
    dst: torch.Tensor     # [E] long
    row: torch.Tensor     # [2E] long, directed edges sorted by row (CSR order)
    col: torch.Tensor     # [2E] long
    rowptr: torch.Tensor  # [B*N + 1] long
    deg: torch.Tensor     # [B*N] long

    @property
    def n_nodes(self) -> int:
        return self.N * self.B

    @property
    def n_edges(self) -> int:
        return self.src.numel()


def graph_from_edges(N: int, B: int, src: torch.Tensor, dst: torch.Tensor) -> GraphBatch:
    """Build a GraphBatch from undirected edges given once each, as global node ids."""
    device = src.device
    n_nodes = N * B
    u = torch.cat([src, dst])
    v = torch.cat([dst, src])
    order = torch.argsort(u)
    row, col = u[order], v[order]
    deg = torch.bincount(row, minlength=n_nodes)
    rowptr = torch.zeros(n_nodes + 1, dtype=torch.long, device=device)
    rowptr[1:] = torch.cumsum(deg, 0)
    return GraphBatch(N, B, src, dst, row, col, rowptr, deg)


def ordered_pair_keys(N: int, B: int, m_max: int, device: torch.device,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """[B, m_max] distinct unordered node pairs per graph, as keys a*N + b (a < b), in
    uniformly random order.  Any prefix of length m is a uniform G(N, m) edge set, so the
    rows are Newman-Ziff edge-addition sequences."""
    m = torch.full((B,), m_max, dtype=torch.long, device=device)
    g = sample_gnm_batch(N, m, device, generator, _return_keys=True)
    return g


def graph_from_keys(N: int, keys: torch.Tensor) -> GraphBatch:
    """GraphBatch from [B, m] pair keys (graph b gets the pairs in row b)."""
    B = keys.shape[0]
    off = (torch.arange(B, device=keys.device) * N).unsqueeze(1)
    src = (torch.div(keys, N, rounding_mode="floor") + off).reshape(-1)
    dst = (keys % N + off).reshape(-1)
    return graph_from_edges(N, B, src, dst)


def sample_gnm_batch(
    N: int,
    m: torch.Tensor,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    _return_keys: bool = False,
):
    """B independent uniform G(N, m_b) graphs; m is a LongTensor [B] of edge counts.

    Candidate pairs are drawn uniformly over unordered pairs (no self-loops), duplicates are
    dropped keeping first occurrences, and the first m_b survivors are kept, so graph b is a
    uniformly random m_b-edge simple graph.
    """
    B = m.numel()
    m = m.to(device=device, dtype=torch.long)
    m_max = int(m.max().item()) if B else 0
    if m_max == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        if _return_keys:
            return torch.empty(B, 0, dtype=torch.long, device=device)
        return graph_from_edges(N, B, empty, empty)
    if m_max > N * (N - 1) // 2:
        raise ValueError("more edges requested than node pairs")

    extra = 64
    while True:
        # duplicates are rare when m << N^2/2; oversample and retry if any graph falls short
        n_cand = int(m_max * 1.05) + extra
        a = torch.randint(0, N, (B, n_cand), device=device, generator=generator)
        b = torch.randint(0, N - 1, (B, n_cand), device=device, generator=generator)
        b = b + (b >= a).long()
        key = torch.minimum(a, b) * N + torch.maximum(a, b)  # [B, n_cand]
        gkey = key + torch.arange(B, device=device).unsqueeze(1) * (N * N)
        flat = gkey.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        pos = torch.arange(flat.numel(), device=device)
        first = torch.full((uniq.numel(),), flat.numel(), dtype=torch.long, device=device)
        first.scatter_reduce_(0, inv, pos, reduce="amin")
        keep = (first[inv] == pos).reshape(B, n_cand)
        rank = torch.cumsum(keep.long(), dim=1) - 1
        keep &= rank < m.unsqueeze(1)
        if bool((keep.sum(dim=1) == m).all()):
            break
        extra *= 4

    if _return_keys:  # all rows have m_max survivors; keep[b] selects them in drawn order
        return key[keep].reshape(B, m_max)
    gid = torch.arange(B, device=device).unsqueeze(1).expand(B, n_cand)[keep]
    k = key[keep]
    off = gid * N
    src = torch.div(k, N, rounding_mode="floor") + off
    dst = k % N + off
    return graph_from_edges(N, B, src, dst)


def sample_gnp_batch(
    N: int,
    z: float,
    B: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> GraphBatch:
    """B independent G(N, p) graphs with p = z/(N-1), so the mean degree is z."""
    pairs = N * (N - 1) // 2
    p = z / (N - 1)
    m = torch.binomial(
        torch.full((B,), float(pairs), device=device),
        torch.full((B,), p, device=device),
        generator=generator,
    ).long()
    return sample_gnm_batch(N, m, device, generator)


def sample_phi_uniform(n_nodes: int, phi_mean: float, width: float, device: torch.device,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Per-node thresholds phi_i ~ U[phi_mean (1 - width), phi_mean (1 + width)]."""
    u = torch.rand(n_nodes, device=device, generator=generator, dtype=torch.float64)
    return phi_mean * (1.0 + width * (2.0 * u - 1.0))


def sample_inputs(N: int, k: int, B: int, device: torch.device,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """k distinct uniformly random input nodes per graph, as global ids [B, k]."""
    r = torch.rand(B, N, device=device, generator=generator)
    idx = r.topk(k, dim=1).indices
    return idx + torch.arange(B, device=device).unsqueeze(1) * N


def input_patterns(k: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """[k, 2^k]: entry (i, p) is bit i of pattern p, the value of input i under pattern p."""
    p = torch.arange(1 << k, device=device)
    i = torch.arange(k, device=device).unsqueeze(1)
    return ((p >> i) & 1).to(dtype)


def thresholds(deg: torch.Tensor, phi) -> torch.Tensor:
    """tau = ceil(phi * d) in float32; isolated nodes get an unreachable threshold.

    phi is a scalar (homogeneous) or a per-node tensor shaped like deg (threshold disorder).
    """
    if torch.is_tensor(phi):
        phi = phi.to(device=deg.device, dtype=torch.float64)
    tau = torch.ceil(phi * deg.double() - _EPS).float()
    tau[deg == 0] = float("inf")
    return tau


EDGE_CHUNK = 1 << 24  # directed edges per gather; bounds the [edges, 2^k] temporary


def _gather_add(C: torch.Tensor, tgt: torch.Tensor, X: torch.Tensor, src: torch.Tensor):
    """C[tgt[e]] += X[src[e]] for all e, in chunks so the gathered rows fit in memory."""
    for s in range(0, tgt.numel(), EDGE_CHUNK):
        C.index_add_(0, tgt[s:s + EDGE_CHUNK], X.index_select(0, src[s:s + EDGE_CHUNK]))


def _setup(g: GraphBatch, inputs: torch.Tensor, k: int, phi: float, dtype):
    device = g.src.device
    P = 1 << k
    tau = thresholds(g.deg, phi).unsqueeze(1)
    inputs_flat = inputs.reshape(-1)
    x = input_patterns(k, device, dtype)
    L = torch.zeros(g.n_nodes, P, dtype=dtype, device=device)
    L[inputs_flat] = x.repeat(inputs.shape[0], 1)
    updatable = torch.ones(g.n_nodes, 1, dtype=torch.bool, device=device)
    updatable[inputs_flat] = False
    return tau, L, updatable


def cascade_dense(g: GraphBatch, inputs: torch.Tensor, k: int, phi: float,
                  max_iter: int = 1_000_000, dtype=torch.float32):
    """Least fixed point by full recomputation of A·L each iteration. Returns (L, n_iter)."""
    tau, L, updatable = _setup(g, inputs, k, phi, dtype)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        C = torch.zeros_like(L)
        _gather_add(C, g.row, L, g.col)
        new = ((C >= tau) & updatable & (L == 0)).to(dtype)
        if not bool(new.any()):
            break
        L += new
    return L, n_iter


def cascade_frontier(g: GraphBatch, inputs: torch.Tensor, k: int, phi: float,
                     push_frac: float = 0.05, max_iter: int = 1_000_000,
                     dtype=torch.float32):
    """Least fixed point by synchronous rounds with incremental counts C += A·dL, where dL is
    the set of nodes newly activated in the previous round. This is not a single-visit BFS: a
    node below threshold is rechecked whenever a neighbour switches on later. Same rounds and
    same result as cascade_dense (repeat L <- [A L >= tau] until nothing changes); see
    tests/test_engine.py.

    push_frac: if the frontier's out-edges exceed this fraction of all directed edges, the
    update is a full-edge pass of A·dL instead of a gather of the frontier's own edges.
    Returns (L, n_iter).
    """
    tau, L, updatable = _setup(g, inputs, k, phi, dtype)
    device = L.device
    C = torch.zeros_like(L)
    n_dir = g.col.numel()
    all_rows = torch.arange(g.n_nodes, device=device)
    # sparse frontier: node ids `rows` and their newly active patterns `Fv` [len(rows), 2^k]
    rows = torch.unique(inputs.reshape(-1))
    Fv = L[rows]
    n_iter = 0
    while rows.numel() > 0 and n_iter < max_iter:
        n_iter += 1  # update rounds, counted as in cascade_dense (incl. the final no-op)
        n_out = g.rowptr[rows + 1] - g.rowptr[rows]
        total = int(n_out.sum().item())
        if total > push_frac * n_dir:
            F = torch.zeros_like(L)
            F[rows] = Fv
            _gather_add(C, g.row, F, g.col)
            del F
            cand = all_rows
        elif total > 0:
            starts = torch.repeat_interleave(g.rowptr[rows], n_out, output_size=total)
            before = torch.repeat_interleave(torch.cumsum(n_out, 0) - n_out, n_out, output_size=total)
            eidx = starts + torch.arange(total, device=device) - before
            local = torch.repeat_interleave(torch.arange(rows.numel(), device=device), n_out,
                                            output_size=total)
            tgt = g.col[eidx]
            _gather_add(C, tgt, Fv, local)
            cand = torch.unique(tgt)  # only nodes whose counts changed can switch on
        else:
            break
        new = ((C[cand] >= tau[cand]) & updatable[cand] & (L[cand] == 0)).to(dtype)
        keep = new.any(dim=1)
        rows, Fv = cand[keep], new[keep]
        L[rows] += Fv
    return L, n_iter


def truth_table_ids(L: torch.Tensor) -> torch.Tensor:
    """Pack each row of a 0/1 state matrix [n, 2^k] (k <= 6) into an int64 truth-table id.

    Bit p of the id is the node's output on pattern p.  For k = 6 the top bit lands in the
    sign bit; ids stay unique because the bits are distinct powers of two.
    """
    P = L.shape[1]
    if P > 64:
        raise ValueError("truth-table ids support k <= 6")
    w = torch.ones(P, dtype=torch.long, device=L.device) << torch.arange(P, device=L.device)
    return (L.long() * w).sum(dim=1)
