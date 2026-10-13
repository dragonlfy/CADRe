"""Step 8: evaluate the initial and the revised reports on the test set.

  export JUDGE_API_KEY=...          # only if the endpoint requires a key
  python scripts/08_evaluate.py --base-url http://localhost:8000/v1 --model deepseek-v4-flash
  python scripts/08_evaluate.py --judge-free-only

LLM judge: the ECG-R1 rubric through an OpenAI-compatible endpoint; eight criteria per report, paired bootstrap of
CADRe against the base model. Judge-free: micro precision, recall and F1 of the diagnoses in each report's answer
list against those the reference states, overall and per diagnostic category. Writes work/evaluation/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadre import config  # noqa: E402
from cadre.evaluation import judge_free  # noqa: E402
from cadre.evaluation.bootstrap import paired  # noqa: E402
from cadre.labels import test_references  # noqa: E402
from cadre.revision import load_reports  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint of the judge, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", help="judge model name served by the endpoint")
    ap.add_argument("--api-key-env", default="JUDGE_API_KEY", help="environment variable that holds the API key")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge-free-only", action="store_true")
    a = ap.parse_args()

    refs = test_references()
    base, cadre = load_reports("test"), load_reports("test_cadre")
    ids = [i for i in base if i in refs]
    refs = {i: refs[i] for i in ids}
    summary = {"records": len(ids)}

    jf = judge_free.evaluate(base, cadre, refs)
    summary["judge_free"] = jf
    print(f"\nJudge-free diagnosis matching ({len(ids)} records), micro P / R / F1")
    print(f"{'category':<22}{'ECG-R1':>22}{'ECG-R1 + CADRe':>24}")
    for name, v in jf.items():
        b, c = v["base"], v["cadre"]
        print(f"{name:<22}{b['precision']:>8.2f}{b['recall']:>7.2f}{b['f1']:>7.2f}"
              f"{c['precision']:>10.2f}{c['recall']:>7.2f}{c['f1']:>7.2f}")
    lo, hi = jf["Overall"]["delta_f1_ci95"]
    print(f"Overall F1 gain, 95% CI: [{lo:+.2f}, {hi:+.2f}]")

    if not a.judge_free_only:
        if not (a.base_url and a.model):
            raise SystemExit("--base-url and --model are required for the LLM judge (or pass --judge-free-only)")
        from cadre.evaluation.judge import Client, judge_reports
        client = Client(a.base_url, a.model, a.api_key_env, cache_dir=config.work("evaluation", "cache", "x").parent)
        scored = {}
        for name, reports in (("base", base), ("cadre", cadre)):
            scored[name] = judge_reports({i: reports[i] for i in ids}, refs, client,
                                         config.work("evaluation", f"judge_{name}.jsonl"), workers=a.workers)
            print(f"judged {name}: {len(scored[name])} of {len(ids)} reports", flush=True)
        result = paired(scored["base"], scored["cadre"])
        summary["judge"] = {"model": a.model, **result}
        print(f"\nLLM judge {a.model} ({result['records']} records)")
        print(f"{'metric':<8}{'ECG-R1':>10}{'+ CADRe':>10}{'delta':>9}   95% CI")
        for name, v in result.items():
            if name == "records":
                continue
            print(f"{name:<8}{v['base']:>10.2f}{v['cadre']:>10.2f}{v['delta']:>+9.2f}   "
                  f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")

    config.work("evaluation", "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsummary -> {config.work('evaluation', 'summary.json')}")


if __name__ == "__main__":
    main()
