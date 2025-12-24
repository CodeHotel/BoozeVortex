#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import paramiko


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _read_float(path: Path) -> float:
    return float(path.read_text().strip())


# ---------------------------
# Machine config
# ---------------------------

@dataclass(frozen=True)
class MachineSpec:
    name: str
    host: str
    user: str
    port: int = 22
    password: Optional[str] = None
    key_path: Optional[str] = None
    key_passphrase: Optional[str] = None


@dataclass(frozen=True)
class RemoteConfig:
    backup_dir: Path
    remote_root: str              # desired: "/root"
    max_parallel: int
    machines: List[MachineSpec]


def load_config(path: Path) -> RemoteConfig:
    obj = json.loads(path.read_text())

    backup_dir = Path(obj["backup_dir"]).expanduser().resolve()
    remote_root = obj.get("remote_root", "/root")
    max_parallel = int(obj.get("max_parallel", 4))

    machines: List[MachineSpec] = []
    for m in obj["machines"]:
        machines.append(
            MachineSpec(
                name=m["name"],
                host=m["host"],
                user=m["user"],
                port=int(m.get("port", 22)),
                password=m.get("password"),
                key_path=m.get("key_path"),
                key_passphrase=m.get("key_passphrase"),
            )
        )

    return RemoteConfig(
        backup_dir=backup_dir,
        remote_root=remote_root,
        max_parallel=max_parallel,
        machines=machines,
    )


# ---------------------------
# SSH client wrapper
# ---------------------------

class SSHSession:
    def __init__(self, spec: MachineSpec):
        self.spec = spec
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect(self, timeout: int = 10) -> None:
        kwargs: Dict[str, Any] = dict(
            hostname=self.spec.host,
            port=self.spec.port,
            username=self.spec.user,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            look_for_keys=False,
            allow_agent=True,
        )

        if self.spec.key_path:
            kwargs["key_filename"] = os.path.expanduser(self.spec.key_path)
            if self.spec.key_passphrase:
                kwargs["passphrase"] = self.spec.key_passphrase

        if self.spec.password:
            kwargs["password"] = self.spec.password

        self.client.connect(**kwargs)

    def close(self) -> None:
        self.client.close()

    def run(self, cmd: str, get_pty: bool = False, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        stdin, stdout, stderr = self.client.exec_command(cmd, get_pty=get_pty, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def sftp(self) -> paramiko.SFTPClient:
        return self.client.open_sftp()


# ---------------------------
# Remote Manager
# ---------------------------

class RemoteManager:
    """
    - Reads machines.json
    - Tests SSH on startup
    - run_remote(points, center_z) does:
        (critical) copy ../case -> ../temp/case{N} and run make_motion_tables.py into constant/
        upload -> remote_root/case{N}  (you want /root/case{N})
        docker run with -v remote_root:/case so inside container /case/case{N} exists
        pull back remote_root/case{N} -> local backup_dir/case{N}
        parse E_water.txt and return float

    Force-stop support:
      containers are named cfd_case{N}; force_stop_all() kills all cfd_case* on every machine.
    """

    def __init__(
        self,
        config_path: Path,
        generator_script: Path,
        case_src_dir: Path,
        temp_dir: Path,
        runs_jsonl: Optional[Path] = None,
    ):
        self.cfg = load_config(config_path)
        self.generator_script = generator_script.resolve()
        self.case_src_dir = case_src_dir.resolve()
        self.temp_dir = temp_dir.resolve()
        self.runs_jsonl = runs_jsonl  # optional append-only run log

        _ensure_dir(self.cfg.backup_dir)
        _ensure_dir(self.temp_dir)

        self._lock = threading.Lock()
        self._next_case_file = self.temp_dir / ".next_case_id"

        self._machine_lock = threading.Lock()
        self._available: List[MachineSpec] = list(self.cfg.machines)

        self._startup_test()

    # ---------- logging ----------
    def _append_jsonl(self, obj: dict) -> None:
        if self.runs_jsonl is None:
            return
        self.runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.runs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    # ---------- startup ----------
    def _startup_test(self) -> None:
        bad: List[str] = []
        for m in self.cfg.machines:
            try:
                s = SSHSession(m)
                s.connect(timeout=8)
                rc, out, err = s.run("echo OK && uname -a | head -n 1")
                s.close()
                if rc != 0 or "OK" not in out:
                    bad.append(f"{m.name} (rc={rc} err={err.strip()})")
            except Exception as e:
                bad.append(f"{m.name} ({type(e).__name__}: {e})")
        if bad:
            raise RuntimeError("SSH startup test failed for: " + ", ".join(bad))

    # ---------- case id ----------
    def _alloc_case_id(self) -> int:
        with self._lock:
            n = int(self._next_case_file.read_text().strip()) if self._next_case_file.exists() else 0
            n += 1
            _atomic_write_text(self._next_case_file, str(n))
            return n

    # ---------- critical local prep ----------
    def _copy_case_and_generate(
        self,
        case_id: int,
        points: List[Tuple[float, float, float]],
        origin_xyz: Tuple[float, float, float],
        n_points: int = 200,
    ) -> Path:
        """
        CRITICAL SECTION:
          - copy ../case -> ../temp/case{N}
          - run make_motion_tables.py into ../temp/case{N}/constant
        """
        with self._lock:
            dst = self.temp_dir / f"case{case_id}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(self.case_src_dir, dst)

            pts_list = [f"({hx},{hy},{t})" for hx, hy, t in points]
            cmd = [
                "python",
                str(self.generator_script),
                "--points",
                *pts_list,
                "--origin",
                f"({origin_xyz[0]},{origin_xyz[1]},{origin_xyz[2]})",
                "--output_dir",
                str(dst / "constant"),
                "--n_points",
                str(n_points),
            ]

            proc = subprocess.run(
                cmd,
                cwd=str(self.generator_script.parent),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Generator failed (case{case_id}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
        return dst

    # ---------- machine pool ----------
    def _checkout_machine(self) -> MachineSpec:
        while True:
            with self._machine_lock:
                if self._available:
                    return self._available.pop()
            time.sleep(0.2)

    def _return_machine(self, m: MachineSpec) -> None:
        with self._machine_lock:
            self._available.append(m)

    # ---------- sftp helpers ----------
    def _sftp_put_dir(self, sess: SSHSession, local_dir: Path, remote_dir: str) -> None:
        sftp = sess.sftp()
        try:
            sess.run(f'mkdir -p "{remote_dir}"')
            for root, _, files in os.walk(local_dir):
                rel = os.path.relpath(root, str(local_dir))
                rdir = remote_dir if rel == "." else f"{remote_dir}/{rel}"
                sess.run(f'mkdir -p "{rdir}"')
                for fn in files:
                    sftp.put(os.path.join(root, fn), f"{rdir}/{fn}")
        finally:
            sftp.close()

    def _sftp_get_dir(self, sess: SSHSession, remote_dir: str, local_dir: Path) -> None:
        sftp = sess.sftp()
        try:
            _ensure_dir(local_dir)
            for entry in sftp.listdir_attr(remote_dir):
                rpath = f"{remote_dir}/{entry.filename}"
                lpath = local_dir / entry.filename
                if stat.S_ISDIR(entry.st_mode):
                    self._sftp_get_dir(sess, rpath, lpath)
                else:
                    sftp.get(rpath, str(lpath))
        finally:
            sftp.close()

    # ---------- force stop ----------
    def force_stop_all(self) -> None:
        """
        Best-effort: kill any containers named cfd_case* on all machines.
        """
        for m in self.cfg.machines:
            sess = SSHSession(m)
            try:
                sess.connect(timeout=10)
                cmd = r"docker ps --format '{{.Names}}' | grep '^cfd_case' | xargs -r docker rm -f"
                sess.run(cmd)
            except Exception:
                pass
            finally:
                try:
                    sess.close()
                except Exception:
                    pass

    def force_stop_case(self, machine: MachineSpec, case_id: int) -> None:
        sess = SSHSession(machine)
        try:
            sess.connect(timeout=10)
            sess.run(f"docker rm -f cfd_case{case_id} || true")
        finally:
            sess.close()

    # ---------- main API ----------
    def run_remote(
        self,
        points: List[Tuple[float, float, float]],
        center_z: float,
        center_xy: Tuple[float, float] = (0.0, 0.0),
        n_points: int = 200,
        docker_image: str = "cfd:latest",
        run_cores: int = 32,
    ) -> Tuple[float, int, str]:
        """
        Returns: (score, case_id, machine_name)

        Remote layout:
          remote_root=/root
          remote_case=/root/case{N}

        Docker mount:
          -v /root:/case, then cd /case/case{N}
        """
        if not (0.03 <= center_z <= 0.15):
            raise ValueError(f"center_z out of range: {center_z}")

        case_id = self._alloc_case_id()
        origin = (center_xy[0], center_xy[1], center_z)

        self._append_jsonl({
            "event": "allocated",
            "ts": time.time(),
            "case_id": case_id,
            "center": origin,
            "points": points,
        })

        local_case_dir = self._copy_case_and_generate(
            case_id=case_id,
            points=points,
            origin_xyz=origin,
            n_points=n_points,
        )

        m = self._checkout_machine()
        sess: Optional[SSHSession] = None
        started = time.time()
        try:
            sess = SSHSession(m)
            sess.connect(timeout=15)

            remote_root = self.cfg.remote_root  # should be "/root"
            remote_case = f"{remote_root}/case{case_id}"

            # Ensure remote_root exists
            rc, out, err = sess.run(f"mkdir -p {remote_root}")
            if rc != 0:
                raise RuntimeError(f"[{m.name}] mkdir {remote_root} failed: {err}")

            # Upload case directory to remote_root/case{N}
            sess.run(f"rm -rf {remote_case} && mkdir -p {remote_case}")
            self._sftp_put_dir(sess, local_case_dir, remote_case)

            self._append_jsonl({
                "event": "uploaded",
                "ts": time.time(),
                "case_id": case_id,
                "machine": m.name,
                "remote_case": remote_case,
            })

            # Run docker non-interactive, named container for force-stop
            container_name = f"cfd_case{case_id}"
            docker_cmd = (
                f'docker run --name {container_name} --rm '
                f'-v "{remote_root}:/case:rw" '
                f'{docker_image} '
                f'bash -lc "cd /case/case{case_id} && ./run {run_cores}"'
            )
            rc, out, err = sess.run(docker_cmd, get_pty=False, timeout=None)
            if rc != 0:
                self._append_jsonl({
                    "event": "docker_failed",
                    "ts": time.time(),
                    "case_id": case_id,
                    "machine": m.name,
                    "rc": rc,
                    "stderr": err[-4000:],
                })
                raise RuntimeError(
                    f"[{m.name}] docker/run failed rc={rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )

            # Download remote_case back to local backup_dir/case{N}
            backup_case_dir = self.cfg.backup_dir / f"case{case_id}"
            if backup_case_dir.exists():
                shutil.rmtree(backup_case_dir)
            _ensure_dir(backup_case_dir.parent)

            self._sftp_get_dir(sess, remote_case, backup_case_dir)

            metric_path = backup_case_dir / "E_water.txt"
            if not metric_path.exists():
                self._append_jsonl({
                    "event": "missing_metric",
                    "ts": time.time(),
                    "case_id": case_id,
                    "machine": m.name,
                })
                raise RuntimeError(f"E_water.txt not found in backup: {metric_path}")

            score = _read_float(metric_path)
            self._append_jsonl({
                "event": "done",
                "ts": time.time(),
                "case_id": case_id,
                "machine": m.name,
                "score": score,
                "elapsed_s": time.time() - started,
            })
            return score, case_id, m.name

        finally:
            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass
            self._return_machine(m)

