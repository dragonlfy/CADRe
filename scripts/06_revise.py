"""Step 6: revision sets of the test records (CPU).

  python scripts/06_revise.py            # method settings from configs/cadre.yaml
  python scripts/06_revise.py --select   # first re-run the cross-validated selection within the memory

Writes work/revision/decisions.json: per test record, the diagnoses its report states, the diagnoses CADRe adds, the
values it replaces ({"old": "new"}), and the calibrated probabilities.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from cadre import config  # noqa: E402
from cadre.revision import Memory, Target, revision_set, select_axis, select_findings, select_rhythm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="re-select the settings of findings, rhythm and axis")
    a = ap.parse_args()

    s = copy.deepcopy(config.settings())
    mem, tgt = Memory(), Target("test")
    print(f"experience memory: {len(mem.ids)} ECGs; test: {len(tgt.ids)} records", flush=True)
    if a.select:
        f = select_findings(mem, s)
        s["findings"].update({k: f[k] for k in ("labels", "w", "k", "beta")})
        s["rhythm"].update(select_rhythm(mem, s))
        s["axis"].update(select_axis(mem, s))
        config.work("revision", "selected.yaml").write_text(
            yaml.safe_dump({k: s[k] for k in ("findings", "rhythm", "axis")}, sort_keys=False))
        print("selected settings -> work/revision/selected.yaml", flush=True)

    decisions = revision_set(mem, tgt, s)
    config.work("revision", "decisions.json").write_text(json.dumps(decisions, indent=1))
    revised = [i for i, d in decisions.items() if d["add"] or d["replaced_by"]]
    adds = sum(len(decisions[i]["add"]) for i in revised)
    print(f"{len(revised)} of {len(decisions)} records revised, {adds} diagnoses asserted "
          f"-> {config.work('revision', 'decisions.json')}")


if __name__ == "__main__":
    main()
