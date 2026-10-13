"""Labels of the experience memory and references of the test set.

Machine statements   the MIMIC-IV-ECG machine report of every record (machine_measurements.csv, report_0..report_17),
                     cleaned like the ECG-Grounding references and read as diagnoses.
Interpretations      the clinical interpretation of every training record in ECG-Grounding, read as diagnoses.
Test references      the ECG-Grounding reference of every test record: its statement list and its interpretation.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache

import numpy as np

from cadre import config
from cadre.vocabulary import FINDINGS, NAMES, report_concepts, statement_concepts


def _normalise(text):
    return " ".join(text.strip().lower().split())


def clean_statement(statement):
    """The ECG-Grounding reference convention: no '***' markers, no ' - age undetermined' suffix."""
    text = _normalise(statement.replace("*", " "))
    for suffix in (" - age undetermined", "- age undetermined"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


@lru_cache(maxsize=1)
def machine_statements():
    """study id -> frozenset of cleaned machine-report statements."""
    out = {}
    with (config.path("mimic_ecg") / "machine_measurements.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels = {clean_statement(row[f"report_{i}"]) for i in range(18)
                      if row.get(f"report_{i}") and row[f"report_{i}"].strip()}
            out[row["study_id"]] = frozenset(labels)
    return out


def machine_concepts(ids):
    """(len(ids), len(NAMES)) uint8: the diagnoses each record's machine statements name."""
    statements, cache = machine_statements(), {}
    Y = np.zeros((len(ids), len(NAMES)), dtype=np.uint8)
    for k, i in enumerate(ids):
        for s in statements.get(i, ()):
            if s not in cache:
                cache[s] = [NAMES.index(c) for c in statement_concepts(s, machine=True)]
            Y[k, cache[s]] = 1
    return Y


def interpretation_findings(ids):
    """id -> findings that the ECG-Grounding training interpretation of the record states."""
    refs = {str(r["id"]): r["conversations"][-1]["value"] for r in json.load(open(config.path("train_references")))}
    return {i: sorted(set(report_concepts(refs[i])) & set(FINDINGS)) for i in ids if i in refs}


@lru_cache(maxsize=1)
def train_references():
    """id -> (statement list, interpretation) of every ECG-Grounding training record."""
    return {str(r["id"]): (r.get("report") or "", r["conversations"][-1]["value"])
            for r in json.load(open(config.path("train_references")))}


@lru_cache(maxsize=1)
def test_references():
    """id -> {"reference_report", "reference_interpretation"} of every test record (ECG-Grounding)."""
    out = {}
    for r in json.load(open(config.path("test_references"))):
        out[str(r["id"])] = {"reference_report": r["report"], "reference_interpretation": r["conversations"][-1]["value"]}
    return out


def reference_concepts(report, interpretation):
    """Diagnoses a reference states, in its statement list or its interpretation."""
    return set(report_concepts(report)) | set(report_concepts(interpretation))
