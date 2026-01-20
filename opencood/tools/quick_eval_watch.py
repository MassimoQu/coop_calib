#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight evaluator that watches a training run directory and periodically
runs a small `inference_w_noise.py` job when a new bestval checkpoint appears.

This is useful when training runs for many hours and you want an automated,
low-cost signal that AP is improving (without waiting for the full sweep).
"""

import argparse
import json
import os
import subprocess
import time
from typing import Optional, Tuple


def _find_latest_bestval_epoch(model_dir_abs: str) -> Optional[int]:
    best = -1
    if not os.path.isdir(model_dir_abs):
        return None
    for name in os.listdir(model_dir_abs):
        if not name.startswith("net_epoch_bestval_at") or not name.endswith(".pth"):
            continue
        raw = name[len("net_epoch_bestval_at") : -len(".pth")]
        try:
            epoch = int(raw)
        except Exception:
            continue
        best = max(best, epoch)
    return None if best < 0 else best


def _read_tail(path: str, max_bytes: int = 200_000) -> str:
    if not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _is_training_finished(train_log: str) -> bool:
    return "Training Finished" in _read_tail(train_log)


def _wait_file_stable(path: str, *, checks: int = 3, sleep_sec: float = 2.0) -> bool:
    """Best-effort: avoid reading a checkpoint while it's still being written."""
    if not os.path.exists(path):
        return False
    last = None
    for _ in range(max(1, int(checks))):
        try:
            size = os.path.getsize(path)
        except Exception:
            return False
        if last is not None and size == last:
            return True
        last = size
        time.sleep(float(sleep_sec))
    return True


def _parse_ap_yaml(path: str) -> Tuple[float, float, float]:
    # Keep this dependency-free-ish: parse only the simple list format we emit.
    ap30 = ap50 = ap70 = 0.0
    if not os.path.exists(path):
        return ap30, ap50, ap70
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        ap30 = float((d.get("ap30") or [0.0])[0])
        ap50 = float((d.get("ap50") or [0.0])[0])
        ap70 = float((d.get("ap70") or [0.0])[0])
    except Exception:
        # If parsing fails, just return zeros and let the log capture details.
        pass
    return ap30, ap50, ap70


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a run dir and run quick AP eval when bestval updates.")
    p.add_argument("--model_dir", required=True, help="Run dir (relative to HEAL root, e.g. opencood/logs/xxx).")
    p.add_argument("--fusion_method", default="intermediate")
    p.add_argument("--pos_std", type=float, default=1.0)
    p.add_argument("--rot_std", type=float, default=1.0)
    p.add_argument("--noise_target", choices=["all", "ego", "non-ego"], default="all")
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--interval", type=int, default=600, help="Polling interval seconds.")
    p.add_argument("--gpu", default="8", help="CUDA_VISIBLE_DEVICES for the eval job.")
    p.add_argument("--stop_when_finished", action="store_true")
    p.add_argument("--out", default="", help="JSONL output path (default: <model_dir>/quick_eval_history.jsonl).")
    args = p.parse_args()

    heal_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = args.model_dir
    model_dir_abs = model_dir if os.path.isabs(model_dir) else os.path.join(heal_root, model_dir)

    train_log = os.path.join(model_dir_abs, "train_stdout.log")
    out_path = args.out.strip() or os.path.join(model_dir_abs, "quick_eval_history.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    micromamba_bin = os.environ.get("MICROMAMBA_BIN", os.path.join(os.path.expanduser("~"), ".local/micromamba/bin/micromamba"))
    mamba_root_prefix = os.environ.get("MAMBA_ROOT_PREFIX", os.path.join(os.path.expanduser("~"), ".micromamba"))
    env_name = os.environ.get("ENV_NAME", "heal")

    last_epoch = -1
    # Resume the last evaluated epoch (if any) to avoid duplicate runs.
    if os.path.exists(out_path):
        try:
            with open(out_path, "rb") as f:
                lines = f.read().splitlines()
            if lines:
                last = json.loads(lines[-1].decode("utf-8", errors="replace"))
                last_epoch = int(last.get("bestval_epoch", -1))
        except Exception:
            last_epoch = -1
    while True:
        epoch = _find_latest_bestval_epoch(model_dir_abs)
        if epoch is not None and epoch > last_epoch:
            ckpt = os.path.join(model_dir_abs, f"net_epoch_bestval_at{epoch}.pth")
            _wait_file_stable(ckpt)

            note = f"_quick{int(args.max_eval_samples)}_bestval{epoch}"
            log_dir = os.path.join(model_dir_abs, "quick_eval_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"quick_eval_bestval{epoch}.log")

            cmd = [
                micromamba_bin,
                "run",
                "-n",
                env_name,
                "python",
                "opencood/tools/inference_w_noise.py",
                "--model_dir",
                model_dir,
                "--fusion_method",
                args.fusion_method,
                "--pos-std-list",
                str(float(args.pos_std)),
                "--rot-std-list",
                str(float(args.rot_std)),
                "--sweep-mode",
                "paired",
                "--noise-target",
                args.noise_target,
                "--max-eval-samples",
                str(int(args.max_eval_samples)),
                "--log-interval",
                "50",
                "--note",
                note,
            ]

            with open(log_path, "a", encoding="utf-8", buffering=1) as f:
                f.write(f"[quick_eval_watch] ts={int(time.time())} bestval_epoch={epoch} cmd={' '.join(cmd)}\n")
                f.flush()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "MAMBA_ROOT_PREFIX": mamba_root_prefix}
                rc = subprocess.call(cmd, cwd=heal_root, stdout=f, stderr=subprocess.STDOUT, env=env)
                f.write(f"[quick_eval_watch] rc={rc}\n")

            ap_path = os.path.join(model_dir_abs, f"AP030507_none{note}.yaml")
            ap30, ap50, ap70 = _parse_ap_yaml(ap_path)

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": int(time.time()),
                            "bestval_epoch": int(epoch),
                            "pos_std": float(args.pos_std),
                            "rot_std": float(args.rot_std),
                            "noise_target": str(args.noise_target),
                            "max_eval_samples": int(args.max_eval_samples),
                            "ap30": float(ap30),
                            "ap50": float(ap50),
                            "ap70": float(ap70),
                            "ap_yaml": os.path.relpath(ap_path, model_dir_abs) if os.path.exists(ap_path) else "",
                            "log": os.path.relpath(log_path, model_dir_abs),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            last_epoch = epoch

        if args.stop_when_finished and _is_training_finished(train_log):
            break

        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
