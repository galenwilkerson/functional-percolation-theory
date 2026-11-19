# Functional percolation: theory

Code and summary data for G. J. Wilkerson, *Statistical Mechanics of Functional Percolation: Input-Driven Response in Network Cascades*.

Linear-threshold cascades on Erdős–Rényi graphs and the 2D square lattice. With k input nodes clamped,
each node's final state is a Boolean function of the inputs; the code computes these functions exactly
(truth tables, decision-tree complexity, diversity, information) and compares them with percolation
theory.

## Layout

| Path | Contents |
|---|---|
| `code/` | simulation, analysis and figure scripts |
| `code/ltm_learning/` | vendored GPU cascade engine (`engine.py`), closed forms (`closed_forms.py`) and a reference cascade for the tests (`reference.py`) |
| `code/tests/` | unit tests (`pytest code/tests`) |
| `summaries/` | the summary tables every figure is drawn from (about 11 MB) |

## Install

```
pip install -r requirements.txt
```

The figures and the CPU scripts run on any machine. The random-graph truth-table simulations
(`sweep.py`, `field.py`, `e3.py`, `probe.py`) need a CUDA GPU with PyTorch.

## Regenerate the figures

```
PAPER=1 FIG_OUT=figures python code/paper_figures.py
FIG_OUT=figures python code/fig_synergy.py
```

This makes exactly the 15 figures of the paper from `summaries/` (override with `FP_DATA=<dir>`) in about a
minute on a CPU. Each figure can also be made on its own by calling its function (table below); the other
`V*` flags in `paper_figures.py` regenerate earlier drafts' figures and are not needed.

| Figure | Function in `paper_figures.py` | Summary table | Produced by |
|---|---|---|---|
| Fig. 1 (schematic) | `fig_schematic3_v9` (`V9THREE=1`) | `schematic_weights/curve.csv` | `schematic_weights.py --curve` (CPU); example graphs drawn with graphviz `neato` |
| Fig. 2 (equation of state) | `fig_field_v3` | `e2f/analysis/field_summary.csv` | `field.py`, `analyze_field.py` (GPU) |
| SM: functional observables | `fig_composites3_v7` | `e1c/analysis/cells_summary.csv`, `e1w/analysis/window_summary.csv` | `sweep.py` with `grid.py`, then `analyze.py`, `analyze_window.py` (GPU) |
| SM: 2D lattice | `fig_lattice_v4` | `lattice2d/summary.csv` | `lattice2d.py` (CPU) |
| SM: finite-size scaling | `fig_fss` | `extra/O_psi_window.csv`, `aldous/aldous_vs_e1w.csv` | `open_items.py`; Aldous prediction `aldous.py` (and `aldous.py --analyze`) |
| SM: crossover in $kS$ | `fig_kS` | `extra/B_kS_crossover.csv` | `analyze_extra.py` |
| SM: synergy at finite input density | `fig_synergy.py` | `individual/FIXHI.csv`, `synergy_theory/synergy_theory.csv` | `individual_inputs.py --stage FIXHI` (CPU; exact classification with $k+\#\{R_u\}$ cascades), `synergy_theory.py` (tree recursion) |
| SM: growing $k$ | `fig_growing_k` | `growing_k/` | `growing_k.py` (CPU) |
| SM: phase map and onsets | `fig_phase_onset_v6` | e1c and e1w summaries | as above |
| SM: finite input density | `fig_finiteh_v6` | `finite_h/summary.csv`, `lattice_h/summary.csv` | `finite_h.py`, `lattice_h.py` (CPU) |
| SM: information about the inputs | `fig_info_si_v9` | `er_info/summary.csv` | `er_info.py` (theory in `info_check.py`) |
| Further SM figures | `fig_symmetry`, `fig_selection`, `fig_fluct`, `fig_dtc_v4` | `e3/`, `input_selection/`, `extra/` | `e3.py`, `input_selection.py`, `analyze_extra.py` |

Theory checks: `verify_theory.py`, `watts.py` (vulnerable-cluster quantities), `open_items.py`,
`open_items2.py`, `er_checks.py`, `class_dilution.py` (exact class construction for large $k$),
`exact_cluster_size.py` (exact finite-$N$ mean cluster size, for the finite-size-scaling check).
`lattice_info.py` (2D information runs) is kept for completeness; its results are not used in the paper.

## Symmetry breaking of the input symmetry (for the next version of the paper)

With k inputs, relabelling them (Sym_k) is a symmetry of the ensemble of networks. These scripts measure how
functional percolation breaks it: the relevance of each input (fraction of nodes whose function depends on it),
the residual symmetry group of the macroscopic function, and their finite-size scaling. All exact, CPU only;
summary tables in `summaries/ssb/`.

| Script | What it does |
|---|---|
| `ssb.py traj` / `ssb.py tilt` | grown networks (edges added one at a time) and independent networks: relevance vector, residual group, one-edge response |
| `ssb_func.py conn` / `density` / `finiteh` | functional relevance against connectivity reach; beyond-OR functions at finite input density; trigger set at finite input density |
| `ssb_extra.py upper` / `lattice` | upper cascade edge with individual inputs (OR of a trigger set vs a quorum on the count); the square lattice |
| `ssb_fss.py` | finite-size scaling at the onset, $z=z_c+\lambda N^{-1/3}$ |
| `ssb_theory.py` | closed forms (mixture of broken states, $P(\text{broken})$, residual groups) |
| `ssb_analyze.py`, `ssb_func_analyze.py`, `ssb_gud_analyze.py`, `ssb_fss_analyze.py` | analysis and figures (`SSB_DATA`, `SSB_OUT` override the paths) |
| `grouped_inputs.py`, `nonor_theory.py` | grouped inputs and the tree recursion for non-OR functions |

Run a simulation with `--out data/ssb/<name>.parquet` from the repository root's `code/` folder, or set `SSB_DATA`.

## Rerunning the simulations

Each simulation script documents its command line in its header and writes to `data/` (not
tracked). The `--fit-only` / `--use-fit` options of the GPU scripts use an optional batch-size
helper (`GPUFIT_PATH`) and can be ignored; the default batch sizes work.

## How to cite

The paper (in preparation):

```bibtex
@unpublished{Wilkerson2026FunctionalPercolation,
  author = {Wilkerson, Galen J.},
  title  = {Statistical Mechanics of Functional Percolation: Input-Driven Response in Network Cascades},
  year   = {2026},
  note   = {Paper in preparation}
}
```

The software:

```bibtex
@software{Wilkerson2026FunctionalPercolationCode,
  author  = {Wilkerson, Galen J.},
  title   = {Functional percolation: theory (code and summary data)},
  year    = {2026},
  url     = {https://github.com/galenwilkerson/functional-percolation-theory},
  version = {main, 2026-11-18}
}
```

GitHub's "Cite this repository" button reads the same information from `CITATION.cff`.

## License

Code: MIT (see `LICENSE`).
