"""Step 5: diagnosis-level beliefs of the frozen ECG-R1 on the memory and test ECGs (GPU, resumable).

  python scripts/05_elicit_beliefs.py --split test --gpus 0,1,2,3
  python scripts/05_elicit_beliefs.py --split memory --gpus 0,1,2,3

For each ECG the image and signal are encoded once, and the 28 diagnosis questions are scored on the cached prefix.
Writes work/beliefs/<split>.npz with ids, concepts (the queried diagnoses) and belief (one row per ECG).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadre import config  # noqa: E402
from cadre.data import records  # noqa: E402
from cadre.sharding import done_ids, launch, shard  # noqa: E402
from cadre.vocabulary import QUERIED  # noqa: E402


def run_shard(split, k, n):
    from cadre.ecgr1.beliefs import BeliefExtractor
    from cadre.ecgr1.model import ECGR1
    out = config.work("beliefs", f"{split}_shard{k}of{n}.jsonl")
    done = done_ids(out)
    todo = [r for r in shard(list(records(split)), k, n) if r["id"] not in done]
    extract = BeliefExtractor(ECGR1(device="cuda:0"))
    with open(out, "a") as handle:
        for j, r in enumerate(todo, 1):
            try:
                belief = [float(x) for x in extract(r)]
            except Exception as error:  # an unreadable input is recorded and left out of the memory
                print(f"[shard {k}] {r['id']}: {type(error).__name__}: {error}", flush=True)
                belief = [float("nan")] * len(QUERIED)
            handle.write(json.dumps({"id": r["id"], "belief": belief}) + "\n")
            if j % 200 == 0:
                handle.flush()
                print(f"[shard {k}] {j}/{len(todo)}", flush=True)


def merge(split, n):
    rows = {}
    for k in range(n):
        for line in config.work("beliefs", f"{split}_shard{k}of{n}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = r["belief"]
    ids = [r["id"] for r in records(split) if r["id"] in rows]
    B = np.array([rows[i] for i in ids], dtype=np.float32)
    np.savez(config.work("beliefs", f"{split}.npz"), ids=np.array(ids), concepts=np.array(QUERIED), belief=B)
    print(f"{split}: beliefs of {len(ids)} ECGs over {len(QUERIED)} diagnoses -> {config.work('beliefs', f'{split}.npz')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["memory", "test"], required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--num-shards", type=int)
    a = ap.parse_args()
    if a.shard is not None:
        run_shard(a.split, a.shard, a.num_shards)
        return
    launch(__file__, a.gpus, ["--split", a.split])
    merge(a.split, len(a.gpus.split(",")))


if __name__ == "__main__":
    main()
