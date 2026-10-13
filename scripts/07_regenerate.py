"""Step 7: evidence-grounded regeneration of the revised test reports (GPU, resumable).

  python scripts/07_regenerate.py --gpus 0,1,2,3

The evidence review of each revised record opens the assistant turn, and the frozen ECG-R1 writes the report from
there. Records with an empty revision set keep their initial report. Writes work/reports/test_cadre.jsonl with
{"id", "prediction_raw", "revised", "evidence"} per test record.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadre import config  # noqa: E402
from cadre.data import records  # noqa: E402
from cadre.sharding import done_ids, launch, shard  # noqa: E402


def _decisions():
    d = json.loads(config.work("revision", "decisions.json").read_text())
    return {i: x for i, x in d.items() if x["add"] or x["replaced_by"]}


def run_shard(k, n, batch_size, max_new_tokens):
    from cadre.anchors import load_measurements
    from cadre.ecgr1.model import ECGR1
    from cadre.ecgr1.regenerate import regenerate
    decisions = _decisions()
    out = config.work("reports", f"test_cadre_shard{k}of{n}.jsonl")
    done = done_ids(out)
    todo = [r for r in shard([r for r in records("test") if r["id"] in decisions], k, n) if r["id"] not in done]
    model, measurements = ECGR1(device="cuda:0"), load_measurements("test")
    for start in range(0, len(todo), batch_size):
        part = todo[start:start + batch_size]
        results = regenerate(model, part, decisions, measurements, batch_size=batch_size, max_new_tokens=max_new_tokens)
        with open(out, "a") as handle:
            for r, (text, evidence) in zip(part, results):
                handle.write(json.dumps({"id": r["id"], "prediction_raw": text, "evidence": evidence}) + "\n")
        print(f"[shard {k}] {start + len(part)}/{len(todo)}", flush=True)


def merge(n):
    from cadre.revision import load_reports
    initial, decisions, regenerated = load_reports("test"), _decisions(), {}
    for k in range(n):
        for line in config.work("reports", f"test_cadre_shard{k}of{n}.jsonl").read_text().splitlines():
            r = json.loads(line)
            regenerated[r["id"]] = r
    missing = [i for i in decisions if i not in regenerated]
    if missing:
        raise SystemExit(f"{len(missing)} regenerations missing; rerun the same command")
    with open(config.work("reports", "test_cadre.jsonl"), "w") as handle:
        for r in records("test"):
            i = r["id"]
            row = regenerated.get(i)
            handle.write(json.dumps({"id": i, "prediction_raw": row["prediction_raw"] if row else initial[i],
                                     "revised": row is not None, "evidence": row["evidence"] if row else []}) + "\n")
    print(f"{len(regenerated)} reports regenerated -> {config.work('reports', 'test_cadre.jsonl')}")


def main():
    s = config.settings()["regeneration"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch-size", type=int, default=s["batch_size"])
    ap.add_argument("--max-new-tokens", type=int, default=s["max_new_tokens"])
    ap.add_argument("--shard", type=int)
    ap.add_argument("--num-shards", type=int)
    a = ap.parse_args()
    if a.shard is not None:
        run_shard(a.shard, a.num_shards, a.batch_size, a.max_new_tokens)
        return
    launch(__file__, a.gpus, ["--batch-size", str(a.batch_size), "--max-new-tokens", str(a.max_new_tokens)])
    merge(len(a.gpus.split(",")))


if __name__ == "__main__":
    main()
