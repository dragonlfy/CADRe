"""Judge-free evaluation: the diagnoses written in each report's answer list are matched with those its reference
states (statement list or clinical interpretation), over the 27 diagnoses of CADRe; precision, recall and F1 are
micro-averaged, overall and per diagnostic category."""

from __future__ import annotations

import re

import numpy as np

from cadre.vocabulary import FINDINGS, report_concepts

CATEGORIES = {"Overall": FINDINGS + ["AFIB", "AFLT", "PACE", "LAD", "RAD"],
              "Rhythm": ["AFIB", "AFLT", "PACE", "PVC", "PAC"],
              "Conduction": ["CRBBB", "IRBBB", "CLBBB", "ILBBB", "LAFB", "LPFB", "IVCD", "1AVB", "WPW"],
              "Axis": ["LAD", "RAD"],
              "Chamber and voltage": ["LVH", "RVH", "LAE", "RAE", "LVOLT"],
              "Infarction": ["MI_inferior", "MI_anterior", "MI_lateral"],
              "Repolarization": ["STT", "EREPOL", "LNGQT"]}


def _answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return m.group(1) if m else text


def counts(reports, references, concepts):
    """Per record: (true positives, false positives, false negatives) over `concepts`."""
    cs, out = set(concepts), {}
    for i, ref in references.items():
        r = (set(report_concepts(ref["reference_report"])) | set(report_concepts(ref["reference_interpretation"]))) & cs
        g = set(report_concepts(_answer(reports[i]))) & cs
        out[i] = (len(r & g), len(g - r), len(r - g))
    return out


def scores(c):
    tp, fp, fn = (sum(v[k] for v in c.values()) for k in range(3))
    return {"precision": 100 * tp / max(1, tp + fp), "recall": 100 * tp / max(1, tp + fn),
            "f1": 200 * tp / max(1, 2 * tp + fp + fn)}


def evaluate(base_reports, cadre_reports, references, resamples=10000, seed=27):
    out = {}
    for name, concepts in CATEGORIES.items():
        a, b = counts(base_reports, references, concepts), counts(cadre_reports, references, concepts)
        out[name] = {"base": scores(a), "cadre": scores(b)}
    a, b = counts(base_reports, references, CATEGORIES["Overall"]), counts(cadre_reports, references, CATEGORIES["Overall"])
    ids = sorted(references)
    A, B = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))

    def f1(X):
        s = X[draws].sum(1)
        return 200 * s[:, 0] / np.maximum(1, 2 * s[:, 0] + s[:, 1] + s[:, 2])
    d = f1(B) - f1(A)
    out["Overall"]["delta_f1_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out
