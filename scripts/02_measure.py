"""Step 2: criterion measurements of every memory and test ECG (CPU, resumable).

  python scripts/02_measure.py --split memory --procs 48
  python scripts/02_measure.py --split test --procs 48

Writes work/measurements/<split>.jsonl: one row per ECG with intervals, rhythm regularity, P-wave detection, frontal
axis, QRS voltage, morphology and the pacing-spike index.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadre import config  # noqa: E402
from cadre.data import records  # noqa: E402


def one(r):
    from cadre.measure.instrument import measure, pacing_spike_index
    row = measure(r["id"], r["ecg"])
    spike = pacing_spike_index(r["ecg"])
    if spike is not None:
        row["pace_spike"] = {"median": spike[0], "max": spike[1]}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["memory", "test"], required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()

    out = config.work("measurements", f"{a.split}.jsonl")
    done = {str(i) for i in _done(out)}
    todo = [r for r in records(a.split) if r["id"] not in done]
    print(f"{a.split}: {len(todo)} ECGs to measure ({len(done)} already done)", flush=True)
    with Pool(a.procs) as pool, open(out, "a") as handle:
        for n, row in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
            handle.write(json.dumps(row) + "\n")
            if n % 1000 == 0:
                handle.flush()
                print(f"  {n}/{len(todo)}", flush=True)


def _done(path):
    if not path.exists():
        return set()
    return {json.loads(line)["record_id"] for line in path.read_text().splitlines() if line.strip()}


if __name__ == "__main__":
    main()
