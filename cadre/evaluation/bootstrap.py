"""Paired bootstrap over records: 10,000 resamples (seed 27); the 2.5th and 97.5th percentiles of the difference of
means form the 95% confidence interval."""

from __future__ import annotations

import numpy as np

METRICS = [("diagnosis_accuracy_binary", "DA"), ("diagnosis_accuracy_credit", "Credit"), ("AnalysisCompleteness", "AC"),
           ("AnalysisRelevance", "AR"), ("LeadEvidenceValidity", "LEV"), ("GroundedECGUnderstanding", "GEU"),
           ("EvidenceBasedReasoning", "EBR"), ("RealisticDiagnosticProcess", "RDP")]


def paired(a: dict, b: dict, resamples=10000, seed=27):
    """a, b: {record id: {metric: value}} of the base model and CADRe, scored by the same judge."""
    ids = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
    out = {"records": len(ids)}
    for key, name in METRICS:
        x = np.array([a[i][key] for i in ids])
        y = np.array([b[i][key] for i in ids])
        d = y - x
        boot = d[draws].mean(1)
        out[name] = {"base": float(x.mean()), "cadre": float(y.mean()), "delta": float(d.mean()),
                     "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out
