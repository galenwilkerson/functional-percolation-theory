"""Information about the inputs on Erdos-Renyi graphs, simple-contagion limit, uniform inputs.
Per-node I and joint J (see info_check.py for the definitions and the exact theory).

  python er_info.py --out data/er_info
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from info_check import sample, theory  # noqa: E402


def one(task):
    N, z, k, seed = task
    I, J = sample(N, z, k, np.random.default_rng(seed))
    return N, z, k, I, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/er_info")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    zs = np.round(np.concatenate([np.arange(0.2, 0.9, 0.1), np.arange(0.9, 1.3, 0.025),
                                  np.arange(1.3, 2.0, 0.1), np.arange(2.0, 5.01, 0.5)]), 3)
    tasks = []
    for N, R in ((10**4, 400), (10**5, 200), (10**6, 40)):
        for k in (4, 8, 12):
            for i, z in enumerate(zs):
                for r in range(R):
                    tasks.append((N, float(z), k, 7 * N + 1009 * k + 100003 * i + r))
    ncpu = int(os.environ.get("GQ_CPUS", os.cpu_count() or 4))
    with Pool(ncpu) as pool:
        rows = pool.map(one, tasks, chunksize=4)
    df = pd.DataFrame(rows, columns=["N", "z", "k", "I", "J"])
    df.to_csv(out / "samples.csv", index=False)
    s = df.groupby(["N", "k", "z"]).agg(I=("I", "mean"), I_se=("I", "sem"), J=("J", "mean"),
                                        J_se=("J", "sem"), n=("J", "size")).reset_index()
    th = np.array([theory(z, k) for z, k in zip(s.z, s.k)])
    s["I_th"], s["J_th"] = th[:, 0], th[:, 1]
    s.to_csv(out / "summary.csv", index=False)
    for (N, k), g in s.groupby(["N", "k"]):
        jp = g.loc[g.J.idxmax()]
        print(f"N={N:8d} k={k:2d}  J peak z={jp.z:.3f}  J/J_th at z>=1.5: "
              f"{(g[g.z >= 1.5].J / g[g.z >= 1.5].J_th).median():.3f}")


if __name__ == "__main__":
    main()
