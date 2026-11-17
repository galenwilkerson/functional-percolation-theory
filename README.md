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
V3=1 V4=1 V5=1 V6=1 V7=1 V9=1 FIG_OUT=figures python code/paper_figures.py
```

This reads `summaries/` (override with `FP_DATA=<dir>`) and takes about a minute on a CPU.

| Figure | Function in `paper_figures.py` | Summary table | Produced by |
|---|---|---|---|
| Fig. 1 (schematic) | `fig_schematic3_v9` (`V9THREE=1`) | `schematic_weights/curve.csv` | `schematic_weights.py --curve` (CPU); example graphs drawn with graphviz `neato` |
| Fig. 2 (equation of state) | `fig_field_v3` | `e2f/analysis/field_summary.csv` | `field.py`, `analyze_field.py` (GPU) |
| SM: functional observables | `fig_composites3_v7` | `e1c/analysis/cells_summary.csv`, `e1w/analysis/window_summary.csv` | `sweep.py` with `grid.py`, then `analyze.py`, `analyze_window.py` (GPU) |
| SM: 2D lattice | `fig_lattice_v4` | `lattice2d/summary.csv` | `lattice2d.py` (CPU) |
| SM: finite-size scaling | `fig_fss_v3` | `extra/O_psi_window.csv` | `open_items.py`; Aldous prediction `aldous.py` |
| SM: crossover in $kS$ | `fig_kS` | `extra/B_kS_crossover.csv` | `analyze_extra.py` |
| SM: growing $k$ | `fig_growing_k` | `growing_k/` | `growing_k.py` (CPU) |
| SM: phase map and onsets | `fig_phase_onset_v6` | e1c and e1w summaries | as above |
| SM: finite input density | `fig_finiteh_v6` | `finite_h/summary.csv`, `lattice_h/summary.csv` | `finite_h.py`, `lattice_h.py` (CPU) |
| SM: information about the inputs | `fig_info_si_v9` | `er_info/summary.csv` | `er_info.py` (theory in `info_check.py`) |
| Further SM figures | `fig_symmetry`, `fig_selection`, `fig_fluct`, `fig_dtc_v4` | `e3/`, `input_selection/`, `extra/` | `e3.py`, `input_selection.py`, `analyze_extra.py` |

Theory checks: `verify_theory.py`, `watts.py` (vulnerable-cluster quantities), `open_items.py`,
`open_items2.py`, `er_checks.py`, `class_dilution.py` (exact class construction for large $k$).
`lattice_info.py` (2D information runs) is kept for completeness; its results are not used in the paper.

## Rerunning the simulations

Each simulation script documents its command line in its header and writes to `data/` (not
tracked). The `--fit-only` / `--use-fit` options of the GPU scripts use an optional batch-size
helper (`GPUFIT_PATH`) and can be ignored; the default batch sizes work.

## License

Code: MIT (see `LICENSE`).
