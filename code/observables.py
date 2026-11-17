"""Per-realization observables for one batch of cascades (structure + function + diversity).

Everything stays on the GPU except the exact-DTC step, which runs on the (small) set of
distinct non-constant truth tables in the batch, with a persistent cache.

Conventions (see RESEARCH_PROPOSAL.md, section "Definitions"):
  * non-input nodes only; n_ni = N - k per graph.
  * function key = exact truth table packed into W = ceil(2^k / 64) int64 words.
  * GCC = largest connected component (ties: smallest min-node-id label).
  * vulnerable node: deg >= 1 and tau = ceil(phi*deg) <= 1 (fires on one active neighbour).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ENGINE_SRC = Path(__file__).resolve().parent
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))
from ltm_learning import engine  # noqa: E402

from dtc import dtc_exact  # noqa: E402

torch.set_grad_enabled(False)


# ----------------------------------------------------------------------------- structure
def components(n_nodes: int, row: torch.Tensor, col: torch.Tensor,
               node_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Connected-component labels (= smallest node id in the component) by min-label
    propagation with pointer jumping.  If node_mask is given, only edges with both ends in
    the mask are used (nodes outside the mask become singletons)."""
    dev = row.device
    if node_mask is not None:
        keep = node_mask[row] & node_mask[col]
        row, col = row[keep], col[keep]
    lab = torch.arange(n_nodes, device=dev)
    while True:
        new = lab.clone()
        if row.numel():
            new.scatter_reduce_(0, row, lab[col], reduce="amin")
        new = new[new]
        new = new[new]
        if torch.equal(new, lab):
            return lab
        lab = new


def _per_graph_argmax_label(lab, size_of, N, B, node_ok=None):
    """For each graph, the label of its largest component (ties -> smallest label) and size."""
    n = lab.numel()
    dev = lab.device
    key = size_of * n + (n - 1 - lab)
    if node_ok is not None:
        key = torch.where(node_ok, key, torch.full_like(key, -1))
    best = torch.full((B,), -1, dtype=torch.long, device=dev)
    best.scatter_reduce_(0, torch.arange(n, device=dev) // N, key, reduce="amax")
    size = torch.where(best >= 0, best // n, torch.zeros_like(best))
    label = torch.where(best >= 0, n - 1 - best % n, torch.full_like(best, -1))
    return label, size


def structure_stats(g, inputs, tau, N, B):
    """Returns dict of [B] tensors + in_gcc [B*N] bool."""
    n = g.n_nodes
    dev = g.row.device
    gid = torch.arange(n, device=dev) // N
    lab = components(n, g.row, g.col)
    size = torch.bincount(lab, minlength=n)
    size_of = size[lab]
    gcc_label, gcc_size = _per_graph_argmax_label(lab, size_of, N, B)
    in_gcc = lab == gcc_label[gid]
    # second largest component and mean finite-cluster size (cluster of a random non-GCC node)
    _, s2 = _per_graph_argmax_label(lab, size_of, N, B, node_ok=~in_gcc)
    nfin = torch.zeros(B, device=dev, dtype=torch.float64).index_add_(
        0, gid, (~in_gcc).double())
    sfin = torch.zeros(B, device=dev, dtype=torch.float64).index_add_(
        0, gid, torch.where(in_gcc, 0, size_of).double())
    chi_fin = sfin / nfin.clamp(min=1)
    # inputs
    inp = inputs.reshape(-1)
    k = inputs.shape[1]
    k_in_gcc = in_gcc[inp].view(B, k).sum(1)
    pairs = torch.stack([torch.arange(B, device=dev).repeat_interleave(k), lab[inp]], 1)
    n_input_comps = torch.bincount(torch.unique(pairs, dim=0)[:, 0], minlength=B)
    # vulnerable subgraph
    vul = (tau <= 1) & (g.deg >= 1)
    vlab = components(n, g.row, g.col, node_mask=vul)
    vsize_of = torch.bincount(vlab, minlength=n)[vlab]
    _, s_vul = _per_graph_argmax_label(vlab, vsize_of, N, B, node_ok=vul)
    frac_vul = torch.zeros(B, device=dev, dtype=torch.float64).index_add_(0, gid, vul.double()) / N
    return dict(
        S=gcc_size.double() / N, S2=s2.double() / N, chi_fin=chi_fin,
        k_in_gcc=k_in_gcc, n_input_comps=n_input_comps,
        S_vul=s_vul.double() / N, frac_vul=frac_vul,
        # per-input placement (for conditioning m_{b,i} on placement; component counting)
        in_deg=g.deg[inp].view(B, k), in_comp=size_of[inp].view(B, k),
        in_gccbit=in_gcc[inp].view(B, k).long(),
    ), in_gcc


# ----------------------------------------------------------------------------- function
def pack_words(T: torch.Tensor) -> torch.Tensor:
    """[n, P] bool -> [n, W] int64 exact packing (bit p of word w = output on pattern 64w+p)."""
    n, P = T.shape
    W = max(1, math.ceil(P / 64))
    out = torch.empty(n, W, dtype=torch.long, device=T.device)
    rows = 1 << 21  # row chunks bound the int64 temporary to rows*64*8 B = 1 GiB
    for w in range(W):
        lo, hi = 64 * w, min(P, 64 * (w + 1))
        sh = torch.arange(hi - lo, device=T.device)
        for r0 in range(0, n, rows):
            out[r0:r0 + rows, w] = (T[r0:r0 + rows, lo:hi].long() << sh).sum(1)
    return out


def unpack_words(words: np.ndarray, P: int) -> np.ndarray:
    """[F, W] int64 -> [F, P] uint8."""
    u = words.astype(np.uint64)
    bits = ((u[:, :, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
    return bits.reshape(words.shape[0], -1)[:, :P]


def or_tables(k: int, device) -> torch.Tensor:
    """[2^k, 2^k] bool: row = relevant-input mask A, column p: OR_{i in A} x_i."""
    P = 1 << k
    p = torch.arange(P, device=device)
    return (p.unsqueeze(0) & p.unsqueeze(1)) != 0


class DTCCache:
    """Exact DTC per distinct truth table, cached across batches (key = packed words bytes)."""

    def __init__(self, k: int):
        self.k = k
        self.d: dict[bytes, int] = {}

    def lookup(self, words_np: np.ndarray) -> np.ndarray:
        keys = [w.tobytes() for w in words_np]
        miss = [i for i, kk in enumerate(keys) if kk not in self.d]
        if miss:
            T = unpack_words(words_np[miss], 1 << self.k)
            vals = dtc_exact(T, self.k)
            for i, v in zip(miss, vals):
                self.d[keys[i]] = int(v)
        return np.array([self.d[kk] for kk in keys], dtype=np.int64)


def _group_unique(keys2d: torch.Tensor):
    """torch.unique over rows; returns (uniq, inverse, counts).

    Fast path: a [n, 2] key whose columns fit in 31 and 32 bits (graph id, packed truth table
    for k <= 5) is combined into one int64 and uniqued in 1-D (~50x faster than dim=0)."""
    if keys2d.shape[1] == 2 and keys2d.numel():
        a, b = keys2d[:, 0], keys2d[:, 1]
        if int(a.min()) >= 0 and int(a.max()) < (1 << 31) and int(b.min()) >= 0 \
                and int(b.max()) < (1 << 32):
            u, inv, c = torch.unique(a * (1 << 32) + b, return_inverse=True, return_counts=True)
            return torch.stack([u >> 32, u & ((1 << 32) - 1)], 1), inv, c
    return torch.unique(keys2d, dim=0, return_inverse=True, return_counts=True)


def function_stats(L, inputs, in_gcc, N, B, k, dtc_cache: DTCCache, cell_of_graph,
                   k_in_gcc=None):
    """Per-graph functional observables. Returns (dict of [B] tensors / [B,x] tensors,
    ensemble dict for per-cell function counts)."""
    dev = L.device
    P = 1 << k
    n = L.shape[0]
    noninput = torch.ones(n, dtype=torch.bool, device=dev)
    noninput[inputs.reshape(-1)] = False
    T = L[noninput] > 0                          # [n_ni_tot, P]
    gid = (torch.arange(n, device=dev) // N)[noninput]
    gcc = in_gcc[noninput]
    n_ni = float(N - k)

    ones = T.sum(1)
    resp = (ones > 0) & (ones < P)
    const1 = ones == P
    rel = torch.zeros(T.shape[0], k, dtype=torch.bool, device=dev)
    nonmono = torch.zeros(T.shape[0], dtype=torch.long, device=dev)
    p = torch.arange(P, device=dev)
    for i in range(k):
        flip = p ^ (1 << i)
        rel[:, i] = (T != T[:, flip]).any(1)
        lo_pat = p[((p >> i) & 1) == 0]
        nonmono += (T[:, lo_pat] & ~T[:, lo_pat | (1 << i)]).sum(1)
    relcount = rel.sum(1)
    relmask = (rel.long() << torch.arange(k, device=dev)).sum(1)
    is_disj = (T == or_tables(k, dev)[relmask]).all(1) & resp

    words = pack_words(T)                        # [n_ni_tot, W]
    # exact DTC via distinct non-constant tables in this batch
    dtc = torch.zeros(T.shape[0], dtype=torch.long, device=dev)
    if bool(resp.any()):
        uw, inv = torch.unique(words[resp], dim=0, return_inverse=True)
        dvals = torch.from_numpy(dtc_cache.lookup(uw.cpu().numpy())).to(dev)
        dtc[resp] = dvals[inv]

    def gsum(x):
        return torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, gid, x.double())

    n_resp = gsum(resp)
    out = dict(
        psi=n_resp / n_ni,
        lam=gsum(relcount) / n_ni,
        dtc=gsum(dtc) / n_ni,
        cdtc=torch.where(n_resp > 0, gsum(dtc) / n_resp.clamp(min=1),
                         torch.full_like(n_resp, float("nan"))),
        act=gsum(ones) / (n_ni * P),            # mean active fraction over all patterns
        rho_or=torch.where(n_resp > 0, gsum(is_disj) / n_resp.clamp(min=1),
                           torch.full_like(n_resp, float("nan"))),
        n_resp=n_resp,
        n_dtc_lt_rel=gsum(dtc < relcount),
        n_nonmono=gsum(nonmono > 0),
        n_const1=gsum(const1),
        psi_gcc=gsum(resp & gcc) / n_ni,
    )
    if k_in_gcc is not None:
        # GCC nodes ignoring some GCC input (clamped-off inputs acting as cut vertices, or
        # threshold blocking): the within-GCC source of functional diversity (RQ3)
        out["n_gcc_blocked"] = gsum(gcc & (relcount < k_in_gcc[gid]))
        out["n_gcc"] = gsum(gcc)
    # per-input relevance fractions m_{b,i}  (Sym_k anisotropy)
    m = torch.zeros(B, k, dtype=torch.float64, device=dev).index_add_(0, gid, rel.double())
    out["m"] = m / n_ni
    # histograms of |Rel| and DTC (non-input nodes, constants in bin 0)
    out["rel_hist"] = torch.zeros(B, k + 1, dtype=torch.long, device=dev).index_put_(
        (gid, relcount), torch.ones_like(gid), accumulate=True)
    out["dtc_hist"] = torch.zeros(B, k + 1, dtype=torch.long, device=dev).index_put_(
        (gid, dtc.clamp(max=k)), torch.ones_like(gid), accumulate=True)

    # ---- within-graph diversity
    key_all = torch.cat([gid.unsqueeze(1), words], 1)
    u, _, c = _group_unique(key_all)
    ub = u[:, 0]
    out["D_all"] = torch.bincount(ub, minlength=B)
    pr = c.double() / n_ni
    out["H"] = torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, ub, -pr * pr.log())

    def distinct(mask, extra=None):
        if not bool(mask.any()):
            return torch.zeros(B, dtype=torch.long, device=dev), None, None
        cols = [gid[mask].unsqueeze(1)]
        if extra is not None:
            cols.append(extra[mask].unsqueeze(1))
        cols.append(words[mask])
        uu, _, cc = _group_unique(torch.cat(cols, 1))
        return torch.bincount(uu[:, 0], minlength=B), uu, cc

    out["simpson"] = torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, ub, pr * pr)
    out["D_nc"], u_nc, c_nc = distinct(resp)
    nanB = torch.full((B,), float("nan"), dtype=torch.float64, device=dev)
    zB = torch.zeros(B, dtype=torch.float64, device=dev)
    if u_nc is not None:
        g0 = u_nc[:, 0]
        q = c_nc.double() / n_resp[g0]
        out["H_nc"] = torch.where(n_resp > 0, zB.clone().index_add_(0, g0, -q * q.log()), nanB)
        out["simpson_nc"] = torch.where(n_resp > 0, zB.clone().index_add_(0, g0, q * q), nanB)
        out["f1_nc"] = torch.bincount(g0[c_nc == 1], minlength=B)   # functions seen once
        out["f2_nc"] = torch.bincount(g0[c_nc == 2], minlength=B)   # ... twice (coverage)
        # function-space condensate: the most common non-constant function per graph has
        # measure mass_max; the rest are "finite functional clusters" whose squared supports
        # give the functional susceptibility chi_F = sum_{f != condensate} |supp f|^2 / N
        cmax = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(
            0, g0, c_nc, reduce="amax")
        c2 = torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, g0, c_nc.double() ** 2)
        out["mass_max"] = cmax.double() / n_ni
        out["chi_F"] = (c2 - cmax.double() ** 2) / float(N)
        # the condensate = first occurrence of the max count in each graph; the rest are the
        # finite functional classes: second-largest class (cf. S2) and log2 size histogram (cf. n_s)
        pos = torch.arange(c_nc.numel(), device=dev)
        first = torch.full((B,), c_nc.numel(), dtype=torch.long, device=dev).scatter_reduce_(
            0, g0, torch.where(c_nc == cmax[g0], pos, torch.full_like(pos, c_nc.numel())),
            reduce="amin")
        rest = pos != first[g0]
        c2nd = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(
            0, g0[rest], c_nc[rest], reduce="amax")
        out["mass_2nd"] = c2nd.double() / n_ni
        nb = 25
        b = torch.log2(c_nc[rest].double()).floor().long().clamp(max=nb - 1)
        out["supp_hist"] = torch.zeros(B, nb, dtype=torch.long, device=dev).index_put_(
            (g0[rest], b), torch.ones_like(b), accumulate=True)
    else:
        out["H_nc"], out["simpson_nc"] = nanB, nanB
        out["f1_nc"] = torch.zeros(B, dtype=torch.long, device=dev)
        out["f2_nc"] = torch.zeros(B, dtype=torch.long, device=dev)
        out["mass_max"] = zB.clone()
        out["chi_F"] = zB.clone()
        out["mass_2nd"] = zB.clone()
        out["supp_hist"] = torch.zeros(B, 25, dtype=torch.long, device=dev)
    out["D_nc_gcc"], _, _ = distinct(resp & gcc)
    out["D_nc_fin"], _, _ = distinct(resp & ~gcc)
    if bool(resp.any()):
        ur = torch.unique(torch.stack([gid[resp], relmask[resp]], 1), dim=0)
        out["D_rel"] = torch.bincount(ur[:, 0], minlength=B)
    else:
        out["D_rel"] = torch.zeros(B, dtype=torch.long, device=dev)

    # ---- ensemble function counts per cell (non-constant functions only)
    ens = None
    if bool(resp.any()):
        cell = cell_of_graph[gid[resp]]
        ue, _, ce = _group_unique(torch.cat([cell.unsqueeze(1), words[resp]], 1))
        # words stored as uint64 (bit-exact; int64 would show negatives for k >= 6)
        ens = dict(cell=ue[:, 0].cpu().numpy(),
                   words=ue[:, 1:].cpu().numpy().view(np.uint64),
                   count=ce.cpu().numpy())
    return out, ens
