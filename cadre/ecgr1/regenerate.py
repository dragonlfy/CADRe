"""Evidence-grounded report regeneration: the evidence review of a record's revision set opens the assistant turn
as "Step 0" in ECG-R1's step format, and the same frozen model continues from the header of Step 1."""

from __future__ import annotations

from cadre.evidence import evidence_lines, evidence_review


def regenerate(model, records, decisions, measurements, batch_size=12, max_new_tokens=2048):
    """records: dicts with image, ecg and prompt, all with a non-empty revision set. Returns (report, evidence)
    pairs; each report starts with its evidence review."""
    tok = model.tokenizer
    out = []
    for b in range(0, len(records), batch_size):
        chunk = records[b:b + batch_size]
        prefixes, encs, lines = [], [], []
        for r in chunk:
            ev = evidence_lines(decisions[r["id"]], measurements.get(r["id"], {}))
            prefix = evidence_review(ev)
            encs.append(model.with_prefix(model.encode(r, r["prompt"]), tok.encode(prefix, add_special_tokens=False)))
            prefixes.append(prefix)
            lines.append(ev)
        for prefix, text, ev in zip(prefixes, model.generate(encs, max_new_tokens=max_new_tokens), lines):
            out.append((prefix + text, ev))
    return out
