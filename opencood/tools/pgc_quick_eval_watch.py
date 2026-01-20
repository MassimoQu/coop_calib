#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Watch a PGC training run (checkpoints written by train_v2vloc_pgc.py) and
periodically run pose inference + pose error evaluation on a small subset.

This is meant to keep long PGC runs "supervised" without manual babysitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from typing import Dict, Optional, Tuple


_EPOCH_RE = re.compile(r"_epoch(\d+)\.pth$")


def _find_latest_epoch_ckpt(ckpt_path: str) -> Optional[Tuple[int, str]]:
    ckpt_path = os.path.abspath(ckpt_path)
    d = os.path.dirname(ckpt_path)
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    suffix = os.path.splitext(ckpt_path)[1]
    if not os.path.isdir(d):
        return None

    best_epoch = -1
    best_path = ""
    for name in os.listdir(d):
        if not name.startswith(stem + "_epoch") or not name.endswith(suffix):
            continue
        m = _EPOCH_RE.search(name)
        if not m:
            continue
        try:
            epoch = int(m.group(1))
        except Exception:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(d, name)

    if best_epoch < 0:
        return None
    return best_epoch, best_path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid(pid_file: str) -> Optional[int]:
    if not pid_file:
        return None
    if not os.path.exists(pid_file):
        return None
    try:
        raw = open(pid_file, "r", encoding="utf-8").read().strip()
        return int(raw)
    except Exception:
        return None


def _run(cmd: list, *, cwd: str, env: Dict[str, str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        f.write(f"[pgc_quick_eval_watch] ts={int(time.time())} cmd={' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch PGC checkpoints and run quick pose evaluation.")
    ap.add_argument("--hypes_yaml", "-y", required=True, help="HEAL/OpenCOOD yaml used for the dataset.")
    ap.add_argument("--ckpt_path", required=True, help="Final ckpt path passed to train_v2vloc_pgc.py.")
    ap.add_argument("--pid_file", default="", help="Optional pid file of the training process. If provided, stop when it exits.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--interval", type=int, default=600, help="Polling interval seconds.")
    ap.add_argument("--gpu", default="8", help="CUDA_VISIBLE_DEVICES for the eval job.")
    ap.add_argument("--out", default="", help="JSONL output path (default: next to ckpt_path).")
    ap.add_argument("--note", default="", help="Optional suffix to disambiguate multiple watchers.")
    args = ap.parse_args()

    # __file__ lives at HEAL/opencood/tools/*.py; we want HEAL/ as the working
    # directory because the inner scripts are invoked as `python opencood/tools/...`.
    heal_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    micromamba_bin = os.environ.get("MICROMAMBA_BIN", os.path.join(os.path.expanduser("~"), ".local/micromamba/bin/micromamba"))
    mamba_root_prefix = os.environ.get("MAMBA_ROOT_PREFIX", os.path.join(os.path.expanduser("~"), ".micromamba"))
    env_name = os.environ.get("ENV_NAME", "heal")

    ckpt_path = args.ckpt_path
    # Make the yaml path absolute so the inner subprocess (cwd=HEAL/) can always
    # locate it, regardless of where the watcher was started.
    hypes_yaml = os.path.abspath(args.hypes_yaml)
    out_path = args.out.strip()
    if not out_path:
        d = os.path.dirname(os.path.abspath(ckpt_path))
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        note = (args.note or "").strip().replace("/", "_").replace(" ", "")
        if note:
            note = "_" + note
        out_path = os.path.join(d, f"{stem}_quick_pose_eval{note}.jsonl")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "quick_pose_eval_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Resume the last evaluated epoch.
    last_epoch = -1
    if os.path.exists(out_path):
        try:
            with open(out_path, "rb") as f:
                lines = f.read().splitlines()
            if lines:
                last = json.loads(lines[-1].decode("utf-8", errors="replace"))
                last_epoch = int(last.get("epoch", -1))
        except Exception:
            last_epoch = -1

    while True:
        pid = _read_pid(args.pid_file)
        if pid is not None and not _pid_alive(pid):
            break

        found = _find_latest_epoch_ckpt(ckpt_path)
        if found is not None:
            epoch, ckpt = found
            if epoch > last_epoch:
                stem = os.path.splitext(os.path.basename(ckpt_path))[0]
                suffix = f"_epoch{epoch}_{args.split}{int(args.max_samples)}"
                pose_json = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), f"{stem}_pose{suffix}.json")
                metrics_json = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), f"{stem}_pose{suffix}_metrics.json")
                log_path = os.path.join(log_dir, f"epoch{epoch}_{args.split}{int(args.max_samples)}.log")

                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(args.gpu),
                    "MAMBA_ROOT_PREFIX": mamba_root_prefix,
                    "PYTHONUNBUFFERED": "1",
                }

                infer_cmd = [
                    micromamba_bin,
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "opencood/tools/infer_v2vloc_pgc_pose.py",
                    "-y",
                    hypes_yaml,
                    "--pgc_ckpt",
                    ckpt,
                    "--out_json",
                    pose_json,
                    "--split",
                    args.split,
                    "--max_samples",
                    str(int(args.max_samples)),
                ]
                rc = _run(infer_cmd, cwd=heal_root, env=env, log_path=log_path)
                if rc != 0:
                    time.sleep(max(10, int(args.interval)))
                    continue

                eval_cmd = [
                    micromamba_bin,
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "opencood/tools/eval_pgc_pose_json.py",
                    "-y",
                    hypes_yaml,
                    "--pgc_json",
                    pose_json,
                    "--split",
                    args.split,
                    "--max_samples",
                    str(int(args.max_samples)),
                    "--out",
                    metrics_json,
                ]
                rc = _run(eval_cmd, cwd=heal_root, env=env, log_path=log_path)
                if rc != 0:
                    time.sleep(max(10, int(args.interval)))
                    continue

                rep = _load_json(metrics_json)
                rel = (rep.get("rel_pose") or {})
                te = (rel.get("te_m") or {})
                re_deg = (rel.get("re_deg") or {})
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": int(time.time()),
                                "epoch": int(epoch),
                                "ckpt": ckpt,
                                "split": str(args.split),
                                "max_samples": int(args.max_samples),
                                "te_mean_m": te.get("mean"),
                                "te_median_m": te.get("median"),
                                "re_mean_deg": re_deg.get("mean"),
                                "re_median_deg": re_deg.get("median"),
                                "metrics_json": metrics_json,
                                "log": log_path,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                last_epoch = int(epoch)

        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
