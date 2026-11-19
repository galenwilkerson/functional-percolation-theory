"""Fig. 1 (v16): connectivity is necessary, not sufficient, for function.

Top: one small G(N, z/N) (z = 2.2) with k = 3 clamped inputs (squares), drawn twice with the same layout:
(a) simple contagion (phi = 0.1; every node here has degree <= 10), (b) phi = 0.25. Node colour = function at the
cascade fixed point; orange edges = giant component. Bottom: probability that an input lies in the giant component (S)
or controls the global cascade (Watts' P_t) at phi = 0.25, against z. The graph was chosen by seed search so that two
inputs lie in the giant component and only one controls at phi = 0.25 (typical: about half of samples).
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

INK, MUTED = "#2b2a28", "#5f5c56"
C_GCC, C_EDGE = "#e07b39", "#8a877f"
C_OR, C_NON, C_FIN, C_ZERO = "#2f6db5", "#c0392b", "#6aa84f", "#e4e2dc"
C_AB = "#8e5bb5"
C_RING = "#d4a017"


def fixed_point(G, inputs, x, phi=None):
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
            need = math.ceil((PHI if phi is None else phi) * d - 1e-9)
            if sum(1 for v in G.neighbors(u) if v in act) >= need:
                act.add(u)
                changed = True
    return act


def functions(G, inputs, phi=None):
    k = len(inputs)
    pats = list(itertools.product((0, 1), repeat=k))
    acts = [fixed_point(G, inputs, x, phi) for x in pats]
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



def macro_classes(G, inputs, cls):
    """Colour key per node: 'or' (an OR function shared by >= 25% of nodes), 'non', 'fin', 'zero'."""
    from collections import Counter
    cnt = Counter(c for c in cls.values() if c[0] == "or")
    out = {}
    for u, c in cls.items():
        if c[0] == "zero":
            out[u] = "zero"
        elif c[0] == "nonor":
            out[u] = "non"
        else:
            out[u] = "or" if cnt[c] >= 0.25 * G.number_of_nodes() else "fin"
    return out


def find_low(N=90, z=0.8, tries=4000):
    for seed in range(tries):
        rng = np.random.default_rng(10_000 + seed)
        G = nx.gnp_random_graph(N, z / N, seed=int(rng.integers(1 << 30)))
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps[0]) > 0.25 * N or len(comps[0]) < 8:
            continue
        cand = [c for c in comps if len(c) >= 4][:3]
        if len(cand) < 3:
            continue
        inputs = [min(c, key=G.degree) for c in cand]
        pats, f = functions(G, inputs)
        return G, inputs, {u: classify(t, pats) for u, t in f.items()}, seed
    raise RuntimeError("no low example")


def find_high(N=90, z=3.5, tries=4000):
    for seed in range(tries):
        rng = np.random.default_rng(20_000 + seed)
        G = nx.gnp_random_graph(N, z / N, seed=int(rng.integers(1 << 30)))
        gcc = max(nx.connected_components(G), key=len)
        if len(gcc) < 0.9 * N:
            continue
        inputs = list(rng.choice(sorted(gcc), 3, replace=False))
        pats, f = functions(G, inputs)
        cls = {u: classify(t, pats) for u, t in f.items()}
        mc = macro_classes(G, inputs, cls)
        nz = sum(v != "zero" for v in mc.values())
        if 3 <= nz <= 0.12 * N and not any(v == "non" for v in mc.values()):
            return G, inputs, cls, seed
    raise RuntimeError("no high example")


def layout(G):
    """Graphviz neato positions (components packed), isolated nodes not drawn; scaled to the unit square."""
    import subprocess
    H = G.subgraph([u for u in G if G.degree(u) > 0])
    lines = ["graph G {", "graph [pack=true, overlap=false, splines=false, sep=0.4, start=3];",
             "node [shape=point, width=0.06];"]
    lines += [f"{u};" for u in H.nodes()] + [f"{u} -- {v};" for u, v in H.edges()] + ["}"]
    out = subprocess.run(["neato", "-Tplain"], input="\n".join(lines), capture_output=True, text=True,
                         check=True).stdout
    pos = {int(f[1]): np.array([float(f[2]), float(f[3])]) for f in (ln.split() for ln in out.splitlines())
           if f and f[0] == "node"}
    xs = np.array(list(pos.values()))
    lo, hi = xs.min(0), xs.max(0)
    span = (hi - lo).max()
    return {u: (q - lo) / span for u, q in pos.items()}


FILL = {"A": C_OR, "AB": C_AB, "non": C_NON, "C": C_FIN, "zero": C_ZERO, "other": "#b9b6ae"}


def by_inputs(cls):
    """Colour key from which inputs a node responds to: A, A or B, A and B (non-OR), C, none."""
    key = {(0,): "A", (0, 1): "AB", (2,): "C"}
    return {u: ("zero" if c[0] == "zero" else "non" if c[0] == "nonor" else key.get(c[1], "other"))
            for u, c in cls.items()}


def draw_graph(ax, G, inputs, mc, title, notes=(), ring=(), sym=None):
    pos = layout(G)
    H = G.subgraph([u for u in G if u in pos])
    gcc = max(nx.connected_components(G), key=len)
    big = len(gcc) >= 0.4 * G.number_of_nodes()
    eg = [e for e in H.edges if big and e[0] in gcc]
    eo = [e for e in H.edges if e not in set(eg)]
    nx.draw_networkx_edges(H, pos, edgelist=eo, ax=ax, width=0.8, edge_color=C_EDGE)
    nx.draw_networkx_edges(H, pos, edgelist=eg, ax=ax, width=0.9, edge_color=C_GCC)
    others = [u for u in H if u not in inputs]
    small = [u for u in others if mc[u] != "non"]
    nx.draw_networkx_nodes(H, pos, nodelist=small, node_color=[FILL[mc[u]] for u in small], node_size=30,
                           linewidths=0.5, edgecolors=INK, ax=ax)
    non = [u for u in others if mc[u] == "non"]
    nx.draw_networkx_nodes(H, pos, nodelist=non, node_color=C_NON, node_size=52, linewidths=0.5,
                           edgecolors=INK, ax=ax)
    if ring:
        nx.draw_networkx_nodes(H, pos, nodelist=list(ring), node_color="none", node_shape="s", node_size=190,
                               edgecolors=C_RING, linewidths=1.8, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=inputs, node_color=INK, node_shape="s", node_size=55, ax=ax)
    if sym:
        ax.text(0.0, -0.02, sym, transform=ax.transAxes, fontsize=7.4, color=INK, ha="left", va="top")
    for lab, u in zip("ABC", inputs):
        x, y = pos[u]
        others_xy = np.array([pos[v] for v in H if v != u])
        best, bd = None, -1
        for dx, dy in ((0, .13), (0, -.13), (.12, 0), (-.12, 0), (.1, .1), (-.1, .1), (.1, -.1),
                       (-.1, -.1)):
            d = np.min(np.hypot(others_xy[:, 0] - (x + dx), others_xy[:, 1] - (y + dy)))
            if d > bd:
                best, bd = (dx, dy), d
        ax.text(x + best[0], y + best[1], lab, ha="center", va="center", fontsize=9, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=7)
    for u, txt, dx, dy in notes:
        x, y = pos[u]
        ax.annotate(txt, (x, y), xytext=(x + dx, y + dy), fontsize=6.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_axis_off()


def _S(z):
    s = 0.5
    for _ in range(4000):
        s = 1 - math.exp(-z * s)
    return s if z > 1 else 0.0


def main():
    from matplotlib.lines import Line2D
    Z = 2.2
    G, inputs, _, gcc, seed = find_example(N=50, z=Z, tries=20000)
    inputs = inputs[:2]                       # A (triggers) and B (connected, does not trigger)
    pats, fb = functions(G, inputs, phi=PHI)
    cls_b = {u: classify(t, pats) for u, t in fb.items()}
    pats, fa = functions(G, inputs, phi=0.1)
    assert max(dict(G.degree).values()) <= 10
    cls_a = {u: classify(t, pats) for u, t in fa.items()}
    mc_a, mc_b = by_inputs(cls_a), by_inputs(cls_b)

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.26, 1.0], hspace=0.42, wspace=0.12)
    ga, gb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    draw_graph(ga, G, inputs, mc_a, f"(a) simple contagion, $z={Z}$\nfunction = connectivity",
               ring=inputs[:2],
               sym="$A$ and $B$ both control the giant cascade (gold boxes)\n"
                   "swapping $A\\leftrightarrow B$ changes nothing: symmetric")
    draw_graph(gb, G, inputs, mc_b, f"(b) $\\varphi=0.25$, same graph\nconnectivity is not sufficient",
               ring=inputs[:1],
               sym="only $A$ controls the giant cascade (gold box); $B$ is connected\n"
                   "swapping $A\\leftrightarrow B$ changes the function: broken")

    def key(ax, items):
        ax.set_axis_off()
        h = []
        for kind, c_, l in items:
            if kind == "node":
                h.append(Line2D([], [], marker="o", ls="", mfc=c_, mec=INK, mew=0.5, ms=5.5, label=l))
            elif kind == "input":
                h.append(Line2D([], [], marker="s", ls="", color=INK, ms=5, label=l))
            elif kind == "box":
                h.append(Line2D([], [], marker="s", ls="", mfc="none", mec=C_RING, mew=1.6, ms=8, label=l))
            else:
                h.append(Line2D([], [], ls="-", lw=1.2, color=c_, label=l))
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.4, frameon=False, handletextpad=0.3,
                  columnspacing=0.7, borderaxespad=0, handlelength=1.2)
    key(fig.add_subplot(gs[1, :]), [
        ("node", C_AB, r"responds to $A$ or $B$ (OR, $x_A\vee x_B$)"),
        ("node", C_OR, r"responds to $A$ only ($x_A$)"),
        ("node", C_NON, r"needs $A$ and $B$ (non-OR, $x_A\wedge x_B$)"),
        ("node", C_ZERO, "never responds"),
        ("input", INK, "input"),
        ("box", C_RING, "controls the giant cascade"),
        ("edge", C_GCC, "giant component")])

    c = fig.add_subplot(gs[2, :])
    d = pd.read_csv(CONN)
    d = d[d.N == d.N.max()].copy()
    d["wf"], d["wc"] = d.q_func * d.n, d.q_conn * d.n
    g = d[np.isclose(d.phi, PHI)].groupby("z")[["wf", "wc", "n"]].sum()
    g["qf"], g["qc"] = g.wf / g.n, g.wc / g.n
    zz = np.linspace(0.2, 3.2, 400)
    c.plot(zz, [_S(z) for z in zz], "-", color=C_GCC, lw=1.5,
           label=r"connected: input in the giant component ($S$); in simple contagion also controlling")
    c.plot(zz, [watts.trigger_prob(z, PHI) for z in zz], "-", color=C_OR, lw=1.5,
           label=r"controlling: input triggers a global cascade at $\varphi=0.25$ ($P_t$)")
    for zv in (1.0, 1.114):
        c.axvline(zv, color=MUTED, lw=0.9, ls=":")
    c.text(0.98, 1.03, "$z_c$", ha="right", fontsize=8, color=MUTED)
    c.text(1.14, 1.03, r"$z_{\rm lo}$", ha="left", fontsize=8, color=MUTED)
    c.axvline(Z, color=INK, lw=0.9, ls="--", zorder=0)
    c.plot([Z, Z], [-0.1, 0], transform=c.get_xaxis_transform(), color=INK, lw=0.9, ls="--", clip_on=False)
    for yv, lab, col in ((_S(Z), "a", C_GCC), (watts.trigger_prob(Z, PHI), "b", C_OR)):
        c.plot([Z], [yv], "o", ms=9, mfc="white", mec=col, mew=1.3, zorder=5)
        c.text(Z, yv, lab, ha="center", va="center", fontsize=7.5, fontweight="bold", color=col, zorder=6)
    c.text(Z, -0.11, f"$z={Z}$", ha="center", va="top", fontsize=8, color=INK, fontweight="bold",
           transform=c.get_xaxis_transform(), clip_on=False)
    c.set_xlim(0.2, 3.2)
    c.set_ylim(0, 1.27)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.set_xlabel("mean degree $z$ (average number of links per node)", fontsize=9, labelpad=14)
    c.set_ylabel("probability that an input is\nconnected / controlling", fontsize=9)
    c.set_title(r"(c) inputs are connected from $z_c$ on, but control only from $z_{\rm lo}$ on, and not all of them",
                fontsize=9, loc="left", pad=14)
    c.tick_params(labelsize=8)
    for sp in ("top", "right"):
        c.spines[sp].set_visible(False)
    c.text(1.3, 1.26, "connected: in the giant\ncomponent, $S$ (= controlling\nin simple contagion)",
           fontsize=7.6, color=C_GCC, ha="left", va="top")
    c.text(2.3, 0.45, "controlling: input triggers\na global cascade at\n$\\varphi=0.25$, $P_t$", fontsize=7.6,
           color=C_OR, ha="left", va="top")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_onset_example.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_onset_example.png", dpi=170, bbox_inches="tight")
    print("seed", seed, "a:", {k: list(mc_a.values()).count(k) for k in set(mc_a.values())},
          "b:", {k: list(mc_b.values()).count(k) for k in set(mc_b.values())})


if __name__ == "__main__":
    main()
