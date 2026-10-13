"""Step 1: labels of the experience memory and inputs of the expert model (CPU).

  python scripts/01_prepare_data.py                 # memory labels, expert signals and statement labels
  python scripts/01_prepare_data.py --labels-only   # memory labels only

Writes
  work/labels/machine_memory.npz             diagnoses named by the machine statements of every memory ECG
  work/labels/interpretation_findings.json   findings stated by the clinical interpretation of every memory ECG
  work/expert/{pool,memory,test}_*           signals, index files and statement labels for the expert model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadre import config  # noqa: E402
from cadre.data import records  # noqa: E402
from cadre.labels import interpretation_findings, machine_concepts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--procs", type=int, default=32, help="processes for reading WFDB records")
    a = ap.parse_args()

    ids = [r["id"] for r in records("memory")]
    Y = machine_concepts(ids)
    np.savez(config.work("labels", "machine_memory.npz"), ids=np.array(ids), Y=Y)
    interp = interpretation_findings(ids)
    config.work("labels", "interpretation_findings.json").write_text(json.dumps(interp))
    print(f"memory: {len(ids)} ECGs, {len(interp)} with a clinical interpretation", flush=True)
    if a.labels_only:
        return

    from cadre.expert.data import convert, splits, statement_labels
    sets = splits()
    for name, recs in sets.items():
        convert(name, recs, procs=a.procs)
    vocab = statement_labels(sets)
    print(f"expert vocabulary: {len(vocab)} statements; pool {len(sets['pool'])} ECGs", flush=True)


if __name__ == "__main__":
    main()
