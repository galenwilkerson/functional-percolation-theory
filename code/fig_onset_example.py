"""Fig. 1 (v16): connectivity vs function on one small graph at phi = 0.25, plus the input-level onset.

(a) connectivity: giant component vs the rest; (b) function of k = 3 clamped inputs at the cascade fixed point:
OR of the triggering inputs, non-OR (needs two inputs), finite-cluster function, zero; (c) probability that an input
controls a macroscopic part of the network (function) vs lies in the giant component (connectivity), against z.
"""
import itertools
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import watts  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "figures"))
CONN = Path(os.environ.get("CONN_CSV", HERE.parent / "summaries" / "ssb" / "f1_func_vs_conn.csv"))
PHI = 0.25

INK, MUTED = "#2b2a28", "#9a978f"
C_GCC, C_REST = "#e07b39", "#b9c6d6"
C_OR, C_NON, C_FIN, C_ZERO = "#2f6db5", "#c0392b", "#8fb3de", "#e4e2dc"


def fixed_point(G, inputs, x):
    """Least fixed point of the LTM with inputs clamped to x (clamped-off inputs never fire)."""
    on = {u for u, xi in zip(inputs, x) if xi}
    off = set(inputs) - on
    act = set(on)
    changed = True
    while changed:
        changed = False
        for u in G.nodes:
            if u in act or u in off:
                continue
            d = G.degree(u)
            if d == 0:
                continue
            need = math.ceil(PHI * d - 1e-9)
            if sum(1 for v in G.neighbors(u) if v in act) >= need:
                act.add(u)
                changed = True
    return act


def functions(G, inputs):
    k = len(inputs)
    pats = list(itertools.product((0, 1), repeat=k))
    acts = [fixed_point(G, inputs, x) for x in pats]
    f = {u: tuple(int(u in a) for a in acts) for u in G.nodes if u not in inputs}
    return pats, f


def classify(tt, pats):
    """'zero', ('or', S) or 'nonor' for a monotone truth table."""
    if not any(tt):
        return "zero", ()
    k = len(pats[0])
    S = tuple(i for i in range(k) if tt[pats.index(tuple(int(j == i) for j in range(k)))])
    if all(tt[n] == int(any(p[i] for i in S)) for n, p in enumerate(pats)):
        return "or", S
    return "nonor", ()


def find_example(N=90, z=2.2, tries=4000):
    for seed in range(tries):
        rng = np.random.default_rng(seed)
        G = nx.gnp_random_graph(N, z / N, seed=int(rng.integers(1 << 30)))
        gcc = max(nx.connected_components(G), key=len)
        if len(gcc) < 0.6 * N:
            continue
        vul = {u for u in G if 1 <= G.degree(u) <= int(1 / PHI)}
        # input B: in the GCC, every neighbour non-vulnerable
        Bs = [u for u in gcc if G.degree(u) >= 1 and all(v not in vul for v in G.neighbors(u))]
        if not Bs:
            continue
        vg = G.subgraph(vul & gcc)
        vcl = max(nx.connected_components(vg), key=len)
        if len(vcl) < 0.4 * N:
            continue
        rest = [c for c in nx.connected_components(G) if c is not gcc and 3 <= len(c) <= 8]
        if not rest:
            continue
        for B in Bs:
            As = [u for u in vcl if u != B and G.degree(u) <= 2 and u not in set(G.neighbors(B))]
            for A in As[:5]:
                C = min(rest[0])
                inputs = [A, B, C]
                pats, f = functions(G, inputs)
                cls = {u: classify(t, pats) for u, t in f.items()}
                n_or = sum(1 for u, c in cls.items() if c[0] == "or" and c[1] == (0,))
                n_non = sum(1 for c in cls.values() if c[0] == "nonor")
                b_alone = sum(1 for c in cls.values() if c[0] == "or" and 1 in c[1])
                if n_or >= 0.4 * N and 2 <= n_non <= 12 and b_alone == 0:
                    return G, inputs, cls, gcc, seed
    raise RuntimeError("no example found")


def draw_graph(ax, G, pos, inputs, colors, title, note=None):
    G = G.subgraph([u for u in G if u in pos])
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color="#c9c6bf")
    others = [u for u in G if u not in inputs]
    nx.draw_networkx_nodes(G, pos, nodelist=others, node_color=[colors[u] for u in others], node_size=26,
                           linewidths=0.4, edgecolors="white", ax=ax)
    big = [u for u in others if colors[u] == C_NON]
    nx.draw_networkx_nodes(G, pos, nodelist=big, node_color=C_NON, node_size=60, linewidths=0.5,
                           edgecolors="white", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=inputs, node_color=INK, node_shape="s", node_size=48, ax=ax)
    for lab, u in zip("ABC", inputs):
        x, y = pos[u]
        ax.text(x, y + 0.055, lab, ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK)
    if note:
        u, txt = note
        x, y = pos[u]
        ax.annotate(txt, (x, y), xytext=(x - 0.42, y + 0.2), fontsize=6.5, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_axis_off()


def layout(G, gcc):
    """Giant component on the left (Kamada-Kawai); finite components of size >= 2 stacked in a column on the
    right; isolated nodes are not drawn."""
    pos = nx.kamada_kawai_layout(G.subgraph(gcc))
    xs = np.array(list(pos.values()))
    lo, hi = xs.min(0), xs.max(0)
    pos = {u: (p - lo) / (hi - lo) * np.array([0.74, 1.0]) for u, p in pos.items()}
    comps = sorted((c for c in nx.connected_components(G) if c is not gcc and len(c) >= 2), key=len, reverse=True)
    y = 1.0
    for c in comps:
        h = 0.05 * math.sqrt(len(c))
        sub = nx.spring_layout(G.subgraph(c), seed=1, scale=h)
        for u, p in sub.items():
            pos[u] = np.array([0.93, y - h]) + p
        y -= 2 * h + 0.07
    return pos


def main():
    G, inputs, cls, gcc, seed = find_example()
    pos = layout(G, gcc)
    conn_col = {u: (C_GCC if u in gcc else C_REST) for u in G}
    fun_col = {}
    for u in G:
        if u in inputs:
            continue
        c = cls[u]
        if c[0] == "zero":
            fun_col[u] = C_ZERO
        elif c[0] == "nonor":
            fun_col[u] = C_NON
        elif u in gcc and c[1] == (0,):
            fun_col[u] = C_OR
        else:
            fun_col[u] = C_FIN
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.12)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    draw_graph(a, G, pos, inputs, conn_col, "(a) connectivity")
    draw_graph(b, G, pos, inputs, fun_col, "(b) function", note=(inputs[1], "B: connected,\nno control"))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_GCC, label="giant component"),
         Line2D([], [], marker="o", ls="", color=C_REST, label="other components")]
    a.legend(handles=h, loc="lower left", fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0)
    h = [Line2D([], [], marker="o", ls="", color=C_OR, label=r"$x_A$ (OR of triggers)"),
         Line2D([], [], marker="o", ls="", color=C_NON, label=r"non-OR ($x_A\wedge x_B$, ...)"),
         Line2D([], [], marker="o", ls="", color=C_FIN, label="finite-cluster function"),
         Line2D([], [], marker="o", ls="", color=C_ZERO, label="zero")]
    b.legend(handles=h, loc="lower left", fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0)

    # (c) input-level onset
    d = pd.read_csv(CONN)
    d = d[d.N == d.N.max()].copy()
    d["wf"], d["wc"] = d.q_func * d.n, d.q_conn * d.n
    zz = np.linspace(0.6, 3.6, 241)
    S = np.array([0.0 if z <= 1 else float(_S(z)) for z in zz])
    Pt = np.array([watts.trigger_prob(z, PHI) for z in zz])
    c.plot(zz, S, "-", color=C_GCC, lw=1.4, label="in giant component, $S$")
    c.plot(zz, Pt, "-", color=C_OR, lw=1.4, label=r"controls the cascade, $P_t$")
    g = d[np.isclose(d.phi, PHI)].groupby("z")[["wf", "wc", "n"]].sum()
    g["qf"], g["qc"] = g.wf / g.n, g.wc / g.n
    c.plot(g.index, g.qc, "o", color=C_GCC, ms=3.5, mfc="white")
    c.plot(g.index, g.qf, "s", color=C_OR, ms=3.5)
    zlo = 1.114
    c.axvline(1.0, color=MUTED, lw=0.7, ls=":")
    c.axvline(zlo, color=MUTED, lw=0.7, ls="--")
    c.text(0.97, 0.97, "$z_c=1$", ha="right", fontsize=7, color=MUTED, transform=c.get_xaxis_transform())
    c.text(zlo + 0.04, 0.86, r"$z_{\rm lo}=1.114$", ha="left", fontsize=7, color=MUTED,
           transform=c.get_xaxis_transform())
    c.set_xlabel("mean degree $z$", fontsize=8)
    c.set_ylabel("probability per input", fontsize=8)
    c.set_title(r"(c) $\varphi=0.25$: function starts at $z_{\rm lo}$", fontsize=9, loc="left")
    c.set_xlim(0.6, 3.6)
    c.set_ylim(0, 1.06)
    c.tick_params(labelsize=7)
    c.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    for s in ("top", "right"):
        c.spines[s].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_onset_example.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_onset_example.png", dpi=170, bbox_inches="tight")
    n_or = sum(1 for u in G if fun_col.get(u) == C_OR)
    n_non = sum(1 for u in G if fun_col.get(u) == C_NON)
    print(f"seed={seed} N={G.number_of_nodes()} gcc={len(gcc)} OR={n_or} nonOR={n_non} inputs={inputs} "
          f"deg={[G.degree(u) for u in inputs]}")


def _S(z):
    s = 0.5
    for _ in range(2000):
        s = 1 - math.exp(-z * s)
    return s


if __name__ == "__main__":
    main()
