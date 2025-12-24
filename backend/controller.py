#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bo_core import BayesianOptimizer, Bounds, vector_to_candidate
from remote_manager import RemoteManager, MachineSpec


@dataclass
class RunRecord:
    case_id: int
    machine: str
    status: str              # "queued"|"running"|"done"|"error"|"killed"
    score: Optional[float] = None
    error: Optional[str] = None


class Controller:
    def __init__(self, base_dir: Path):
        self.base = base_dir
        self.rm = RemoteManager(
            config_path=self.base / "machines.json",
            generator_script=self.base / "make_motion_tables.py",
            case_src_dir=(self.base / ".." / "case").resolve(),
            temp_dir=(self.base / ".." / "temp").resolve(),
        )

        self.db_path = self.base / "bo_state.sqlite"
        self._init_db()

        self.bounds = Bounds()
        self.bo = BayesianOptimizer(bounds=self.bounds, max_step_deg=170.0, seed=0)
        self._load_results_into_bo()

        self._log_lock = threading.Lock()
        self._logs: List[str] = []

        self._state_lock = threading.Lock()
        self._paused = True
        self._stopping = False

        # in-flight: case_id -> (machineSpec, containerName)
        self._running_lock = threading.Lock()
        self._running: Dict[int, MachineSpec] = {}

        self._thread: Optional[threading.Thread] = None

    # ---------- DB ----------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
              CREATE TABLE IF NOT EXISTS runs (
                case_id INTEGER PRIMARY KEY,
                x_json TEXT NOT NULL,
                machine TEXT,
                status TEXT NOT NULL,
                started_at REAL,
                finished_at REAL,
                score REAL,
                error TEXT
              )
            """)
            con.execute("""
              CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
              )
            """)
            con.commit()

    def _db_set(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT INTO system_state(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            con.commit()

    def _db_get(self, key: str, default: str) -> str:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT value FROM system_state WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else default

    def _db_insert_run(self, case_id: int, x: np.ndarray, status: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO runs(case_id, x_json, status) VALUES(?,?,?)",
                (case_id, json.dumps(x.tolist()), status),
            )
            con.commit()

    def _db_update_run(self, case_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join([f"{k}=?" for k in fields.keys()])
        vals = list(fields.values()) + [case_id]
        with sqlite3.connect(self.db_path) as con:
            con.execute(f"UPDATE runs SET {cols} WHERE case_id=?", vals)
            con.commit()

    def _load_results_into_bo(self) -> None:
        results_csv = self.base / "results.csv"
        if not results_csv.exists():
            return
        import pandas as pd
        df = pd.read_csv(results_csv)
        for _, row in df.iterrows():
            x = np.array([row[f"hx{i}"] for i in range(8)] + [row[f"hy{i}"] for i in range(8)] + [row["cz"]])
            y = float(row["score"])
            self.bo.tell(x, y)

    # ---------- Logs ----------
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._log_lock:
            self._logs.append(line)
            self._logs = self._logs[-500:]  # keep last 500

    def get_logs(self) -> List[str]:
        with self._log_lock:
            return list(self._logs)

    # ---------- Control ----------
    def start(self) -> None:
        with self._state_lock:
            self._paused = False
            self._stopping = False
            self._db_set("paused", "0")
            self._db_set("stopping", "0")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self.log("Started / resumed scheduling.")

    def pause(self) -> None:
        with self._state_lock:
            self._paused = True
            self._db_set("paused", "1")
        self.log("Paused scheduling (running jobs continue).")

    def resume(self) -> None:
        self.start()

    def force_stop(self) -> None:
        # stop scheduling + kill all containers we know about
        with self._state_lock:
            self._paused = True
            self._stopping = True
            self._db_set("paused", "1")
            self._db_set("stopping", "1")
        self.log("Force stop requested: killing running containers...")

        with self._running_lock:
            running_items = list(self._running.items())

        for case_id, m in running_items:
            try:
                self.rm.force_stop_case(m, case_id)  # you add this to RemoteManager
                self._db_update_run(case_id, status="killed", error="force_stopped", finished_at=time.time())
                self.log(f"Killed case{case_id} on {m.name}")
            except Exception as e:
                self.log(f"Failed to kill case{case_id} on {m.name}: {e}")

        with self._running_lock:
            self._running.clear()

    # ---------- Status ----------
    def status(self) -> Dict:
        with self._state_lock:
            paused = self._paused
            stopping = self._stopping

        best = max(self.bo.y) if self.bo.y else None
        return {
            "paused": paused,
            "stopping": stopping,
            "n_evals": len(self.bo.y),
            "best": best,
            "machines": [m.name for m in self.rm.cfg.machines],
            "running": self._running_snapshot(),
        }

    def _running_snapshot(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        with self._running_lock:
            for cid, m in self._running.items():
                out.setdefault(m.name, []).append(cid)
        return out

    # ---------- BO Loop ----------
    def _loop(self) -> None:
        # batch size = min(max_parallel, num machines)
        batch = min(self.rm.cfg.max_parallel, len(self.rm.cfg.machines))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        while True:
            with self._state_lock:
                if self._stopping:
                    time.sleep(0.5)
                    continue
                if self._paused:
                    time.sleep(0.5)
                    continue

            # Ask candidates
            xs = self.bo.ask_batch(batch_size=batch, pool_size=6000, xi=0.01, min_dist=10.0)

            # Evaluate in parallel
            def _eval(x: np.ndarray) -> Tuple[np.ndarray, float]:
                # NOTE: RemoteManager already manages machine checkout,
                # but for force-stop we want to know where it ran.
                # Easiest: add a callback in RemoteManager later.
                y = self._evaluate_one(x)
                return x, y

            self.log(f"Dispatching batch of {len(xs)} runs...")
            with ThreadPoolExecutor(max_workers=batch) as ex:
                futs = [ex.submit(_eval, x) for x in xs]
                for fut in as_completed(futs):
                    try:
                        x, y = fut.result()
                        self.bo.tell(x, y)
                        self._append_result_csv(x, y)
                        self.log(f"OK score={y:.6f} cz={x[16]:.4f}")
                    except Exception as e:
                        self.log(f"ERROR during evaluation: {e}")

    def _append_result_csv(self, x: np.ndarray, y: float) -> None:
        # reuse your ResultsStore if you want; keeping minimal here
        import csv
        path = self.base / "results.csv"
        header = [f"hx{i}" for i in range(8)] + [f"hy{i}" for i in range(8)] + ["cz", "score"]
        row = list(map(float, x[:16])) + [float(x[16]), float(y)]
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)

    def _evaluate_one(self, x: np.ndarray) -> float:
        cand = vector_to_candidate(x)
        # case_id allocation is internal; RemoteManager returns score
        return self.rm.run_remote(points=cand.to_points(), center_z=cand.cz)

