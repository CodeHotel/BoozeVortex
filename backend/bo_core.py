#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


DT = 0.1  # 8 points, 0.1s apart


@dataclass(frozen=True)
class Bounds:
    hx: Tuple[float, float] = (0.0, 315.0)
    hy: Tuple[float, float] = (85.0, 285.0)
    cz: Tuple[float, float] = (0.03, 0.15)


@dataclass(frozen=True)
class MotionCandidate:
    hx: np.ndarray  # shape (8,)
    hy: np.ndarray  # shape (8,)
    cz: float

    def to_points(self) -> List[Tuple[float, float, float]]:
        return [(float(self.hx[i]), float(self.hy[i]), float(i * DT)) for i in range(8)]


def is_valid_candidate(c: MotionCandidate, b: Bounds, max_step_deg: float = 170.0) -> bool:
    if np.any(c.hx < b.hx[0]) or np.any(c.hx > b.hx[1]):
        return False
    if np.any(c.hy < b.hy[0]) or np.any(c.hy > b.hy[1]):
        return False
    if not (b.cz[0] <= c.cz <= b.cz[1]):
        return False
    # critical: avoid unwrap "reverse"
    if np.any(np.abs(np.diff(c.hx)) > max_step_deg):
        return False
    if np.any(np.abs(np.diff(c.hy)) > max_step_deg):
        return False
    return True


def candidate_to_vector(c: MotionCandidate) -> np.ndarray:
    return np.concatenate([c.hx, c.hy, np.array([c.cz], dtype=float)])


def vector_to_candidate(x: np.ndarray) -> MotionCandidate:
    hx = np.array(x[:8], dtype=float)
    hy = np.array(x[8:16], dtype=float)
    cz = float(x[16])
    return MotionCandidate(hx=hx, hy=hy, cz=cz)


def random_valid_candidates(
    n: int, bounds: Bounds, max_step_deg: float, rng: np.random.Generator
) -> List[MotionCandidate]:
    out: List[MotionCandidate] = []
    # rejection sample (ok because constraints are mild)
    while len(out) < n:
        hx = rng.uniform(bounds.hx[0], bounds.hx[1], size=8)
        hy = rng.uniform(bounds.hy[0], bounds.hy[1], size=8)
        cz = float(rng.uniform(bounds.cz[0], bounds.cz[1]))
        c = MotionCandidate(hx=hx, hy=hy, cz=cz)
        if is_valid_candidate(c, bounds, max_step_deg=max_step_deg):
            out.append(c)
    return out


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    # EI for maximization
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_y - xi
    Z = imp / sigma
    # normal pdf/cdf
    pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z * Z)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(Z / np.sqrt(2)))
    return imp * cdf + sigma * pdf


class BayesianOptimizer:
    def __init__(
        self,
        bounds: Bounds,
        max_step_deg: float = 170.0,
        seed: int = 0,
    ):
        self.bounds = bounds
        self.max_step_deg = max_step_deg
        self.rng = np.random.default_rng(seed)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(17), nu=2.5) + WhiteKernel(
            noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1)
        )
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)

        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def tell(self, x: np.ndarray, y: float) -> None:
        self.X.append(np.array(x, dtype=float))
        self.y.append(float(y))

    def fit(self) -> None:
        X = np.vstack(self.X)
        y = np.array(self.y, dtype=float)
        self.gp.fit(X, y)

    def ask_batch(
        self,
        batch_size: int,
        pool_size: int = 5000,
        xi: float = 0.01,
        min_dist: float = 5.0,
    ) -> List[np.ndarray]:
        """
        Greedy batch: sample a random pool of valid candidates, score by EI, pick diverse top-K.
        min_dist is L2 distance in parameter space (deg units + cz).
        """
        if len(self.X) < 5:
            # early: just random valid
            cands = random_valid_candidates(batch_size, self.bounds, self.max_step_deg, self.rng)
            return [candidate_to_vector(c) for c in cands]

        self.fit()
        best_y = float(np.max(self.y))

        pool = random_valid_candidates(pool_size, self.bounds, self.max_step_deg, self.rng)
        Xpool = np.vstack([candidate_to_vector(c) for c in pool])

        mu, std = self.gp.predict(Xpool, return_std=True)
        ei = expected_improvement(mu, std, best_y=best_y, xi=xi)

        order = np.argsort(-ei)  # descending EI
        chosen: List[np.ndarray] = []
        for idx in order:
            x = Xpool[idx]
            if not chosen:
                chosen.append(x)
            else:
                d = np.min([np.linalg.norm(x - c) for c in chosen])
                if d >= min_dist:
                    chosen.append(x)
            if len(chosen) >= batch_size:
                break

        # fallback if diversity constraint too strict
        while len(chosen) < batch_size:
            chosen.append(Xpool[order[len(chosen)]])

        return chosen


class ResultsStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, x: np.ndarray, y: float) -> None:
        header = [f"hx{i}" for i in range(8)] + [f"hy{i}" for i in range(8)] + ["cz", "score"]
        row = list(map(float, x[:16])) + [float(x[16]), float(y)]
        exists = self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)

    def load(self) -> Optional[pd.DataFrame]:
        if not self.path.exists():
            return None
        return pd.read_csv(self.path)

