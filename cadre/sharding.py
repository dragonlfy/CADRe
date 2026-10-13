"""Run a script on several GPUs: one child process per GPU, each on its own shard of the records."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def launch(script, gpus, args):
    """Start `script --shard k --num-shards n <args>` with CUDA_VISIBLE_DEVICES set to the k-th GPU; wait for all."""
    gpus = [g.strip() for g in str(gpus).split(",") if g.strip()]
    children = [subprocess.Popen([sys.executable, script, "--shard", str(k), "--num-shards", str(len(gpus)), *args],
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": g}) for k, g in enumerate(gpus)]
    codes = [c.wait() for c in children]
    if any(codes):
        raise SystemExit(f"a shard failed (exit codes {codes}); rerun the same command to resume")


def shard(items, k, n):
    return [x for j, x in enumerate(items) if j % n == k]


def done_ids(path):
    """Ids already written to a JSON-lines output, so an interrupted run resumes where it stopped."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return out
