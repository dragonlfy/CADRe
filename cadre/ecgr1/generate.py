"""Initial reports of the frozen model, produced with ECG-R1's official inference entry point (swift infer, PyTorch
back-end): official image and raw signal, greedy decoding, at most 2,048 new tokens."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cadre import config
from cadre.ecgr1.model import ENV, model_overlay


def generate_reports(records, batch_size=32, max_new_tokens=2048):
    """records: dicts with image, ecg and prompt. Returns the reports in the same order."""
    code = config.path("ecgr1_code")
    with tempfile.TemporaryDirectory() as tmp:
        requests, result = Path(tmp) / "requests.jsonl", Path(tmp) / "responses.jsonl"
        with requests.open("w") as handle:
            for r in records:
                handle.write(json.dumps({"messages": [{"role": "user", "content": "<ecg><image>\n" + r["prompt"]}],
                                         "images": [r["image"]], "objects": {"ecg": [r["ecg"]]}}) + "\n")
        command = [sys.executable, "swift/cli/infer.py", "--model", str(model_overlay()), "--model_type", "ecg_r1",
                   "--template", "ecg_r1", "--torch_dtype", "bfloat16", "--custom_register_path", "ecg_r1/register.py",
                   "--infer_backend", "pt", "--val_dataset", str(requests), "--result_path", str(result),
                   "--max_batch_size", str(batch_size), "--task_type", "causal_lm", "--max_new_tokens",
                   str(max_new_tokens), "--temperature", "0.0", "--repetition_penalty", "1.0", "--use_hf", "true"]
        env = {**os.environ, **ENV, "ECG_TOWER_PATH": str(config.path("ecg_tower")),
               "PYTHONPATH": str(code) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        done = subprocess.run(command, cwd=code, env=env, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"ECG-R1 inference failed: {done.stderr[-4000:]}")
        rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
    if len(rows) != len(records):
        raise RuntimeError(f"{len(rows)} responses for {len(records)} records")
    return [str(r["response"]) for r in rows]
