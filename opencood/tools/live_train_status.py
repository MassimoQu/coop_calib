#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


def _read_tail(path: str, max_bytes: int = 200_000) -> str:
    if not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _parse_train_progress(log_text: str) -> Optional[Tuple[int, int, int]]:
    """
    Matches: [epoch 8][170/401]
    Returns: (epoch, step, total_steps)
    """
    matches = list(re.finditer(r"\[epoch\s+(\d+)\]\[(\d+)\s*/\s*(\d+)\]", log_text))
    if not matches:
        return None
    m = matches[-1]
    try:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except Exception:
        return None


def _parse_last_val_loss(log_text: str) -> Optional[Tuple[int, float]]:
    matches = list(re.finditer(r"At epoch\s+(\d+),\s+the validation loss is\s+([0-9.eE+-]+)", log_text))
    if not matches:
        return None
    m = matches[-1]
    try:
        return int(m.group(1)), float(m.group(2))
    except Exception:
        return None


def _has_finished(log_text: str) -> bool:
    return "Training Finished" in log_text


def _find_train_pgid(model_dir: str) -> Optional[int]:
    """
    Best-effort: find an existing torchrun/train_ddp process group using the same model_dir.
    """
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,pgid,args"], universal_newlines=True)
    except Exception:
        return None

    token = f"--model_dir {model_dir}"
    candidates = []
    for line in out.splitlines():
        line = line.strip()
        if "opencood/tools/train_ddp.py" not in line:
            continue
        if token not in line:
            continue
        if "torch.distributed.run" in line or "torchrun" in line:
            candidates.append(line)
    if not candidates:
        for line in out.splitlines():
            line = line.strip()
            if "opencood/tools/train_ddp.py" not in line:
                continue
            if token not in line:
                continue
            candidates.append(line)

    best = None
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
        best = pgid
    return best


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _gpu_snapshot() -> Optional[str]:
    """
    Returns a compact nvidia-smi snapshot string, or None if unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            universal_newlines=True,
        )
    except Exception:
        return None
    lines = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx, util, mem_used, mem_total = parts[:4]
            lines.append(f"{idx}:{util}% {mem_used}/{mem_total}MB")
    return " | ".join(lines) if lines else None


def _build_status(model_dir: str, log_text: str, *, include_gpu: bool) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "ts": int(time.time()),
        "finished": _has_finished(log_text),
    }

    prog = _parse_train_progress(log_text)
    if prog is not None:
        epoch, step, total = prog
        status.update({"epoch": epoch, "step": step, "steps_total": total})

    val = _parse_last_val_loss(log_text)
    if val is not None:
        val_epoch, val_loss = val
        status.update({"val_epoch": val_epoch, "val_loss": float(val_loss)})

    pgid = _find_train_pgid(model_dir)
    status["train_pgid"] = int(pgid) if pgid is not None else None
    status["train_alive"] = bool(_pgid_alive(int(pgid))) if pgid is not None else False

    if include_gpu:
        status["gpu"] = _gpu_snapshot()

    return status


def main() -> None:
    p = argparse.ArgumentParser(description="Lightweight status logger for HEAL/OpenCOOD training runs.")
    p.add_argument("--model_dir", required=True, help="Run directory (relative to HEAL root or absolute).")
    p.add_argument("--interval", type=int, default=30, help="Seconds between updates.")
    p.add_argument("--out", default="", help="Output log path (default: <model_dir>/live_status.jsonl).")
    p.add_argument("--include_gpu", action="store_true", help="Include nvidia-smi snapshot.")
    args = p.parse_args()

    heal_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(heal_root, model_dir)
    os.makedirs(model_dir, exist_ok=True)

    out_path = args.out.strip() or os.path.join(model_dir, "live_status.jsonl")
    log_path = os.path.join(model_dir, "train_stdout.log")

    interval = max(1, int(args.interval))

    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            log_text = _read_tail(log_path)
            status = _build_status(args.model_dir, log_text, include_gpu=bool(args.include_gpu))
            f.write(json.dumps(status, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()

