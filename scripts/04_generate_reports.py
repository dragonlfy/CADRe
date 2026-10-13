"""Step 4: initial reports of the frozen ECG-R1 on the memory and test ECGs (GPU, resumable).

  python scripts/04_generate_reports.py --split test --gpus 0,1,2,3
  python scripts/04_generate_reports.py --split memory --gpus 0,1,2,3

Uses ECG-R1's official inference entry point (greedy decoding, at most 2,048 new tokens). Writes
work/reports/<split>.jsonl with {"id", "prediction_raw"} per ECG.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadre import config  # noqa: E402
from cadre.data import records  # noqa: E402
from cadre.sharding import done_ids, launch, shard  # noqa: E402


def run_shard(split, k, n, batch_size, chunk):
    from cadre.ecgr1.generate import generate_reports
    out = config.work("reports", f"{split}_shard{k}of{n}.jsonl")
    done = done_ids(out)
    todo = [r for r in shard(list(records(split)), k, n) if r["id"] not in done]
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        reports = generate_reports(part, batch_size=batch_size)
        with open(out, "a") as handle:
            for r, text in zip(part, reports):
                handle.write(json.dumps({"id": r["id"], "prediction_raw": text}) + "\n")
        print(f"[shard {k}] {start + len(part)}/{len(todo)}", flush=True)


def merge(split, n):
    rows = {}
    for k in range(n):
        for line in config.work("reports", f"{split}_shard{k}of{n}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = r
    ids = [r["id"] for r in records(split)]
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} reports missing; rerun the same command")
    with open(config.work("reports", f"{split}.jsonl"), "w") as handle:
        for i in ids:
            handle.write(json.dumps(rows[i]) + "\n")
    print(f"{split}: {len(ids)} reports -> {config.work('reports', f'{split}.jsonl')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["memory", "test"], required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000, help="records per inference call (the resume granularity)")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--num-shards", type=int)
    a = ap.parse_args()
    if a.shard is not None:
        run_shard(a.split, a.shard, a.num_shards, a.batch_size, a.chunk)
        return
    launch(__file__, a.gpus, ["--split", a.split, "--batch-size", str(a.batch_size), "--chunk", str(a.chunk)])
    merge(a.split, len(a.gpus.split(",")))


if __name__ == "__main__":
    main()
