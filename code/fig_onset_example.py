"""Fig. 1 (v16): function vs connectivity at phi = 0.25.

Top: one small G(N, z/N) at each of three mean degrees, k = 3 clamped inputs (squares). Node colour = function at the
cascade fixed point (OR of the triggering inputs on a macroscopic set, non-OR, finite-cluster function, zero); orange
edges = giant component. Bottom: probability that an input lies in the giant component (S) or controls the global
cascade (Watts' P_t) against z, with the three example z marked. Examples are random graphs, chosen by seed search to
show the effects named in the caption.
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
C_GCC, C_EDGE = "#e07b39", "#8a877f"
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
    """Largest component on the left (Kamada-Kawai) if it is big; the other components of size >= 2 in a column on
    the right, or in a grid when there is no giant component; isolated nodes are not drawn."""
    comps = sorted((c for c in nx.connected_components(G) if len(c) >= 2), key=len, reverse=True)
    pos = {}
    if len(comps[0]) >= 0.4 * G.number_of_nodes():
        p = nx.kamada_kawai_layout(G.subgraph(comps[0]))
        xs = np.array(list(p.values()))
        lo, hi = xs.min(0), xs.max(0)
        pos.update({u: (q - lo) / (hi - lo) * np.array([0.78, 1.0]) for u, q in p.items()})
        y = 1.0
        for c in comps[1:]:
            h = 0.045 * math.sqrt(len(c))
            sub = nx.spring_layout(G.subgraph(c), seed=1, scale=h)
            pos.update({u: np.array([0.95, y - h]) + q for u, q in sub.items()})
            y -= 2 * h + 0.06
    else:
        ncol = 4
        for i, c in enumerate(comps):
            h = 0.06 * math.sqrt(len(c))
            sub = nx.kamada_kawai_layout(G.subgraph(c), scale=h) if len(c) > 2 else nx.spring_layout(
                G.subgraph(c), seed=1, scale=h)
            cx, cy = (i % ncol) / (ncol - 1), 1 - (i // ncol) * 0.26
            pos.update({u: np.array([cx, cy]) + q for u, q in sub.items()})
    return pos


FILL = {"or": C_OR, "non": C_NON, "fin": C_FIN, "zero": C_ZERO}


def draw_graph(ax, G, inputs, mc, title, notes=()):
    pos = layout(G)
    H = G.subgraph([u for u in G if u in pos])
    gcc = max(nx.connected_components(G), key=len)
    big = len(gcc) >= 0.4 * G.number_of_nodes()
    eg = [e for e in H.edges if big and e[0] in gcc]
    eo = [e for e in H.edges if e not in set(eg)]
    nx.draw_networkx_edges(H, pos, edgelist=eo, ax=ax, width=0.9, edge_color=C_EDGE)
    nx.draw_networkx_edges(H, pos, edgelist=eg, ax=ax, width=1.0, edge_color=C_GCC)
    others = [u for u in H if u not in inputs]
    small = [u for u in others if mc[u] != "non"]
    nx.draw_networkx_nodes(H, pos, nodelist=small, node_color=[FILL[mc[u]] for u in small], node_size=24,
                           linewidths=0.5, edgecolors=INK, ax=ax)
    non = [u for u in others if mc[u] == "non"]
    nx.draw_networkx_nodes(H, pos, nodelist=non, node_color=C_NON, node_size=50, linewidths=0.5,
                           edgecolors=INK, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=inputs, node_color=INK, node_shape="s", node_size=46, ax=ax)
    for lab, u in zip("ABC", inputs):
        x, y = pos[u]
        ax.text(x, y + 0.05, lab, ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK)
    for u, txt, dx, dy in notes:
        x, y = pos[u]
        ax.annotate(txt, (x, y), xytext=(x + dx, y + dy), fontsize=6.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_axis_off()


def _S(z):
    s = 0.5
    for _ in range(4000):
        s = 1 - math.exp(-z * s)
    return s if z > 1 else 0.0


def main():
    from matplotlib.lines import Line2D
    ZS = (0.8, 2.2, 3.5)
    Glo, ilo, clo, slo = find_low(z=ZS[0])
    Gmid, imid, cmid, _, smid = find_example(z=ZS[1])
    Ghi, ihi, chi, shi = find_high(z=ZS[2])

    fig = plt.figure(figsize=(7.0, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.08)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_graph(ax[0], Glo, ilo, macro_classes(Glo, ilo, clo),
               f"(a) $z={ZS[0]}$\nno giant component")
    draw_graph(ax[1], Gmid, imid, macro_classes(Gmid, imid, cmid),
               f"(b) $z={ZS[1]}$\n$A$ controls; $B$ only connected",
)
    draw_graph(ax[2], Ghi, ihi, macro_classes(Ghi, ihi, chi),
               f"(c) $z={ZS[2]}$\nconnected; no input controls")

    c = fig.add_subplot(gs[1, :2])
    d = pd.read_csv(CONN)
    d = d[d.N == d.N.max()].copy()
    d["wf"], d["wc"] = d.q_func * d.n, d.q_conn * d.n
    g = d[np.isclose(d.phi, PHI)].groupby("z")[["wf", "wc", "n"]].sum()
    g["qf"], g["qc"] = g.wf / g.n, g.wc / g.n
    zz = np.linspace(0.4, 4.3, 300)
    c.plot(zz, [_S(z) for z in zz], "-", color=C_GCC, lw=1.5, label="input in giant component ($S$)")
    c.plot(zz, [watts.trigger_prob(z, PHI) for z in zz], "-", color=C_OR, lw=1.5,
           label="input controls the cascade ($P_t$)")
    c.plot(g.index, g.qc, "o", color=C_GCC, ms=3.5, mfc="white")
    c.plot(g.index, g.qf, "s", color=C_OR, ms=3.5)
    for zv, lab in ((1.0, "$z_c$"), (1.114, r"$z_{\rm lo}$"), (3.863, r"$z_{\rm hi}$")):
        c.axvline(zv, color=MUTED, lw=0.7, ls="--")
    c.text(0.98, 1.0, "$z_c$", ha="right", fontsize=7, color=MUTED)
    c.text(1.14, 1.0, r"$z_{\rm lo}$", ha="left", fontsize=7, color=MUTED)
    c.text(3.89, 1.0, r"$z_{\rm hi}$", ha="left", fontsize=7, color=MUTED)
    for zv, lab in zip(ZS, "abc"):
        c.plot([zv], [1.1], "v", color=INK, ms=4)
        c.text(zv, 1.14, f"({lab})", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    c.set_xlim(0.4, 4.3)
    c.set_ylim(0, 1.2)
    c.set_xlabel("mean degree $z$", fontsize=8)
    c.set_ylabel("probability per input", fontsize=8)
    c.set_title(r"(d) $\varphi=0.25$: connectivity starts at $z_c$, function at $z_{\rm lo}$", fontsize=8.5, loc="left", pad=14)
    c.tick_params(labelsize=7)
    for sp in ("top", "right"):
        c.spines[sp].set_visible(False)
    c.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.6, 0.02))

    lg = fig.add_subplot(gs[1, 2])
    lg.set_axis_off()
    h = [Line2D([], [], marker="o", ls="", mfc=C_OR, mec=INK, mew=0.5, label=r"macroscopic OR of the triggers"),
         Line2D([], [], marker="o", ls="", mfc=C_NON, mec=INK, mew=0.5, ms=8, label=r"non-OR ($x_A\wedge x_B$)"),
         Line2D([], [], marker="o", ls="", mfc=C_FIN, mec=INK, mew=0.5, label="finite-cluster function"),
         Line2D([], [], marker="o", ls="", mfc=C_ZERO, mec=INK, mew=0.5, label="zero function"),
         Line2D([], [], marker="s", ls="", color=INK, label="input"),
         Line2D([], [], ls="-", lw=1.2, color=C_GCC, label="giant-component edge"),
         Line2D([], [], ls="-", lw=1.0, color=C_EDGE, label="other edge")]
    lg.legend(handles=h, loc="center left", fontsize=7, frameon=False, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_onset_example.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_onset_example.png", dpi=170, bbox_inches="tight")
    print("seeds", slo, smid, shi)


if __name__ == "__main__":
    main()
