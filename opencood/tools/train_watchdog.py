#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from typing import Optional, Tuple


def _read_tail(path: str, max_bytes: int = 200_000) -> str:
    if not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _read_since(path: str, offset: int, max_bytes: int = 200_000) -> str:
    if not os.path.exists(path):
        return ""
    try:
        size = os.path.getsize(path)
    except Exception:
        return ""
    if offset >= size:
        return ""
    with open(path, "rb") as f:
        try:
            f.seek(int(offset), os.SEEK_SET)
        except Exception:
            return ""
        data = f.read()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _parse_last_val_loss(log_text: str) -> Optional[Tuple[int, float]]:
    """
    Matches: At epoch 31, the validation loss is 0.123456
    Returns: (epoch, loss)
    """
    matches = list(
        re.finditer(r"At epoch\s+(\d+),\s+the validation loss is\s+([0-9.eE+-]+)", log_text)
    )
    if not matches:
        return None
    m = matches[-1]
    try:
        epoch = int(m.group(1))
        loss = float(m.group(2))
    except Exception:
        return None
    return epoch, loss


def _has_finished(log_text: str) -> bool:
    return "Training Finished" in log_text


def _has_error(log_text: str) -> bool:
    patterns = [
        r"\bRuntimeError\b",
        r"\bValueError\b",
        r"\bKeyError\b",
        r"\bIndexError\b",
        r"\bAssertionError\b",
        r"\bFileNotFoundError\b",
        r"\bEOFError\b",
        r"\bOSError\b",
        r"CUDA out of memory",
        r"\bNCCL\b.*\b(ERROR|WARN)\b",
        r"\bSegmentation fault\b",
        r"\bBus error\b",
        r"\bKilled\b",
    ]
    return any(re.search(p, log_text) for p in patterns)


def _find_existing_train_pgid(model_dir: str) -> Optional[int]:
    """
    Return PGID for an existing torchrun/train_ddp process using the same model_dir,
    or None if not found.
    """
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,pgid,args"], universal_newlines=True)
    except Exception:
        return None

    token_model = f"--model_dir {model_dir}"
    candidates = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "opencood/tools/train_ddp.py" not in line:
            continue
        if token_model not in line:
            continue
        # Prefer the torchrun wrapper (group leader often has pid==pgid).
        if "torch.distributed.run" in line or "torchrun" in line:
            candidates.append(line)
    if not candidates:
        # Fall back to any matching process (e.g., rank processes); they still share PGID.
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if "opencood/tools/train_ddp.py" not in line:
                continue
            if token_model not in line:
                continue
            candidates.append(line)

    best_pgid = None
    for line in candidates:
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except Exception:
            continue
        if pid == pgid:
            return pgid
        best_pgid = pgid
    return best_pgid


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_pgid(pgid: int, timeout_sec: int = 30) -> None:
    if not _pgid_alive(pgid):
        return
    try:
        os.killpg(int(pgid), signal.SIGINT)
    except Exception:
        pass

    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if not _pgid_alive(pgid):
            return
        time.sleep(0.5)

    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except Exception:
        pass


def _spawn_detached(cmd: str, log_path: str, cwd: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    # Detach into a new process group so the watchdog can terminate all ranks.
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _terminate_process_group(proc: subprocess.Popen, timeout_sec: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _build_train_cmd(opt: argparse.Namespace) -> str:
    args = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(opt.nproc)}",
        "opencood/tools/train_ddp.py",
        "-y",
        opt.yaml,
        "--model_dir",
        opt.model_dir,
        "--fusion_method",
        opt.fusion_method,
        "--num_workers",
        str(int(opt.num_workers)),
    ]
    if opt.init_model_dir:
        args += ["--init_model_dir", opt.init_model_dir]
    if opt.half:
        args.append("--half")
    if opt.no_test:
        args.append("--no_test")
    if opt.max_epochs > 0:
        args += ["--max_epochs", str(int(opt.max_epochs))]
    if opt.max_train_steps > 0:
        args += ["--max_train_steps", str(int(opt.max_train_steps))]
    if opt.max_val_steps > 0:
        args += ["--max_val_steps", str(int(opt.max_val_steps))]

    prefix = []
    if opt.cuda_visible_devices:
        prefix.append(f"CUDA_VISIBLE_DEVICES={opt.cuda_visible_devices}")
    if opt.omp_num_threads:
        prefix.append(f"OMP_NUM_THREADS={int(opt.omp_num_threads)}")
    if opt.mkl_num_threads:
        prefix.append(f"MKL_NUM_THREADS={int(opt.mkl_num_threads)}")
    if opt.mamba_root_prefix:
        prefix.append(f"MAMBA_ROOT_PREFIX={opt.mamba_root_prefix}")

    # Use micromamba env if provided.
    if opt.micromamba_bin and opt.env_name:
        cmd = " ".join(prefix + [opt.micromamba_bin, "run", "-n", opt.env_name] + args)
    else:
        cmd = " ".join(prefix + args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog for OpenCOOD DDP training (auto-restart + early-stop).")
    parser.add_argument("--yaml", required=True, help="Training YAML (relative to HEAL root or absolute).")
    parser.add_argument("--model_dir", required=True, help="Run directory (relative to HEAL root or absolute).")
    parser.add_argument(
        "--init_model_dir",
        default="",
        help="Optional: initialize weights from another checkpoint directory, but start from epoch 0.",
    )
    parser.add_argument("--fusion_method", default="intermediate")
    parser.add_argument("--cuda_visible_devices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--nproc", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no_test", action="store_true", default=True)
    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_val_steps", type=int, default=0)
    parser.add_argument("--check_interval", type=int, default=30)
    parser.add_argument("--patience", type=int, default=0, help="Early-stop after N val evals without improvement. 0 disables.")
    parser.add_argument("--min_delta", type=float, default=1e-6, help="Minimum improvement in val loss to reset patience.")
    parser.add_argument("--max_restarts", type=int, default=20)
    parser.add_argument(
        "--stall_timeout",
        type=int,
        default=900,
        help="Restart if train log does not change for this many seconds while the process is still alive.",
    )
    parser.add_argument("--run_sweep_on_finish", action="store_true", help="Run run_noise_sweep.sh after Training Finished.")

    # Environment controls (defaults match this repo's setup).
    parser.add_argument("--micromamba_bin", default=os.path.expanduser("~/.local/micromamba/bin/micromamba"))
    parser.add_argument("--env_name", default="heal")
    parser.add_argument("--mamba_root_prefix", default=os.path.expanduser("~/.micromamba"))
    parser.add_argument("--omp_num_threads", type=int, default=1)
    parser.add_argument("--mkl_num_threads", type=int, default=1)
    opt = parser.parse_args()

    heal_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    yaml_path = opt.yaml
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(heal_root, yaml_path)
    model_dir = opt.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(heal_root, model_dir)
    init_model_dir = opt.init_model_dir
    if init_model_dir:
        if not os.path.isabs(init_model_dir):
            init_model_dir = os.path.join(heal_root, init_model_dir)
    opt.yaml = os.path.relpath(yaml_path, heal_root)
    opt.model_dir = os.path.relpath(model_dir, heal_root)
    if init_model_dir:
        opt.init_model_dir = os.path.relpath(init_model_dir, heal_root)

    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "train_stdout.log")
    cmd_path = os.path.join(model_dir, "train_cmd_watchdog.txt")
    pid_path = os.path.join(model_dir, "train_watchdog.pid")

    best_val = float("inf")
    bad_count = 0
    restarts = 0

    while True:
        existing_pgid = _find_existing_train_pgid(opt.model_dir)
        proc = None
        pgid = None
        if existing_pgid is not None:
            pgid = int(existing_pgid)
            print(f"[watchdog] found existing training pgid={pgid}; attach mode.", flush=True)
        else:
            train_cmd = _build_train_cmd(opt)
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write(train_cmd + "\n")

            print(f"[watchdog] launch: {train_cmd}", flush=True)
            proc = _spawn_detached(train_cmd, log_path=log_path, cwd=heal_root)
            with open(pid_path, "w", encoding="utf-8") as f:
                f.write(str(proc.pid) + "\n")
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = proc.pid

        last_log_change = time.time()
        last_log_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        error_cursor = last_log_size

        while True:
            time.sleep(max(1, int(opt.check_interval)))
            try:
                cur_size = os.path.getsize(log_path)
            except Exception:
                cur_size = last_log_size
            if cur_size != last_log_size:
                last_log_size = cur_size
                last_log_change = time.time()

            tail_text = _read_tail(log_path)
            if _has_finished(tail_text):
                print("[watchdog] detected Training Finished", flush=True)
                if pgid is not None:
                    _terminate_pgid(pgid, timeout_sec=30)
                if opt.run_sweep_on_finish:
                    sweep_cmd = f"bash opencood/tools/run_noise_sweep.sh {opt.model_dir}"
                    print(f"[watchdog] running sweep: {sweep_cmd}", flush=True)
                    subprocess.call(sweep_cmd, shell=True, cwd=heal_root)
                return

            new_text = _read_since(log_path, error_cursor)
            error_cursor = last_log_size
            if new_text and _has_error(new_text):
                print("[watchdog] detected error pattern in logs; restarting.", flush=True)
                if pgid is not None:
                    _terminate_pgid(pgid, timeout_sec=60)
                break

            parsed = _parse_last_val_loss(tail_text)
            if parsed is not None:
                epoch, val_loss = parsed
                improved = (best_val - val_loss) > float(opt.min_delta)
                if improved:
                    best_val = val_loss
                    bad_count = 0
                else:
                    if opt.patience > 0:
                        bad_count += 1
                print(
                    f"[watchdog] val_loss epoch={epoch} loss={val_loss:.6f} "
                    f"best={best_val:.6f} bad_count={bad_count}/{opt.patience or 0}"
                    ,
                    flush=True,
                )
                if opt.patience > 0 and bad_count >= int(opt.patience):
                    print("[watchdog] early-stop triggered (val loss plateau).", flush=True)
                    _terminate_process_group(proc, timeout_sec=60)
                    return

            if int(opt.stall_timeout) > 0 and (time.time() - last_log_change) > float(opt.stall_timeout):
                print(
                    f"[watchdog] stalled: no log update for {int(time.time() - last_log_change)}s; restarting.",
                    flush=True,
                )
                if pgid is not None:
                    _terminate_pgid(pgid, timeout_sec=60)
                break

            if pgid is not None and not _pgid_alive(pgid):
                rc = proc.returncode if proc is not None else "unknown"
                print(f"[watchdog] train process group exited (code {rc})", flush=True)
                break

        # Restart logic
        restarts += 1
        if restarts > int(opt.max_restarts):
            print("[watchdog] max_restarts exceeded; giving up.", flush=True)
            sys.exit(2)
        print(
            f"[watchdog] restarting in 10s... (restarts={restarts}/{opt.max_restarts})",
            flush=True,
        )
        time.sleep(10)


if __name__ == "__main__":
    main()
