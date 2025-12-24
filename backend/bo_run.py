#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from remote_manager import RemoteManager
from bo_core import (
    BayesianOptimizer,
    Bounds,
    MotionCandidate,
    ResultsStore,
    is_valid_candidate,
    vector_to_candidate,
)


def evaluate_one(rm: RemoteManager, x: np.ndarray) -> float:
    cand = vector_to_candidate(x)
    # run_remote expects list of (Hx,Hy,t) and center_z
    return rm.run_remote(points=cand.to_points(), center_z=cand.cz)


def main():
    base = Path(__file__).resolve().parent

    rm = RemoteManager(
        config_path=base / "machines.json",
        generator_script=base / "make_motion_tables.py",
        case_src_dir=(base / ".." / "case").resolve(),
        temp_dir=(base / ".." / "temp").resolve(),
    )

    bounds = Bounds()
    bo = BayesianOptimizer(bounds=bounds, max_step_deg=170.0, seed=0)
    store = ResultsStore(base / "results.csv")

    # Resume if exists
    df = store.load()
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            x = np.array([row[f"hx{i}"] for i in range(8)] + [row[f"hy{i}"] for i in range(8)] + [row["cz"]])
            y = float(row["score"])
            bo.tell(x, y)

    batch = min(rm.cfg.max_parallel, len(rm.cfg.machines))

    # Budget: you control this
    total_evals_target = 80  # example
    while len(bo.y) < total_evals_target:
        # Ask new batch
        xs = bo.ask_batch(batch_size=batch, pool_size=6000, xi=0.01, min_dist=10.0)

        # Evaluate in parallel; RemoteManager internally hands out machines
        scores: List[float] = [None] * len(xs)
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futs = {ex.submit(evaluate_one, rm, x): i for i, x in enumerate(xs)}
            for fut in as_completed(futs):
                i = futs[fut]
                y = float(fut.result())
                scores[i] = y

        # Tell + persist
        for x, y in zip(xs, scores):
            bo.tell(x, y)
            store.append(x, y)
            print(f"[OK] score={y:.6f} cz={x[16]:.4f}")

        best = max(bo.y)
        print(f"==> evals={len(bo.y)} best={best:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()

