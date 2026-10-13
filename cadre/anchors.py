"""Criterion measurements: for each diagnosis, the waveform measurements in which its clinical criteria are written.

Atrial fibrillation is defined by an irregular RR series without P waves, first-degree AV block by PR > 200 ms, axis
deviation by the frontal QRS axis, LVH by voltage. These measurements form the criterion part m_c of the state of
diagnosis c, so that retrieved cases are similar on the evidence that defines c.
"""

from __future__ import annotations

import json
import math

import numpy as np

from cadre import config

INFERIOR, ANTERIOR, LATERAL = ("II", "III", "aVF"), ("V1", "V2", "V3", "V4"), ("I", "aVL", "V5", "V6")


def _get(m, *path):
    for k in path:
        if not isinstance(m, dict) or k not in m:
            return math.nan
        m = m[k]
    if m is None or isinstance(m, (dict, list, str)):
        return math.nan
    return float(m)


def _max_q(m, leads):
    per = (m.get("morphology") or {}).get("per_lead") or {}
    vals = [(per.get(lead) or {}).get("q_ms") for lead in leads]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else math.nan


def _st(m):
    return {lead: v for lead, v in ((m.get("st_mv") or {}).get("value") or {}).items() if v is not None}


def _t(m):
    return {lead: v for lead, v in ((m.get("t_amplitude_mv") or {}).get("value") or {}).items() if v is not None}


_RHYTHM = lambda m: [_get(m, "rr_cv", "value"), _get(m, "p_wave", "value"), _get(m, "scalar", "heart_rate_bpm", "value")]  # noqa: E731
_QRS = lambda m: [_get(m, "scalar", "qrs_ms", "value")]  # noqa: E731
_AXIS = lambda m: [_get(m, "frontal_axis", "value_degrees")]  # noqa: E731

ANCHORS = {
    "SR": _RHYTHM, "AFIB": _RHYTHM, "AFLT": _RHYTHM,
    "PACE": lambda m: [_get(m, "pace_spike", "median"), _get(m, "pace_spike", "max"), _get(m, "scalar", "qrs_ms", "value")],
    "LAD": _AXIS, "RAD": _AXIS,
    "PVC": lambda m: [_get(m, "rr_cv", "value")], "PAC": lambda m: [_get(m, "rr_cv", "value")],
    "CRBBB": _QRS, "IRBBB": _QRS, "CLBBB": _QRS, "ILBBB": _QRS, "IVCD": _QRS,
    "WPW": lambda m: [_get(m, "scalar", "qrs_ms", "value"), _get(m, "scalar", "pr_ms", "value")],
    "LAFB": lambda m: [_get(m, "frontal_axis", "value_degrees"), _get(m, "scalar", "qrs_ms", "value")],
    "LPFB": lambda m: [_get(m, "frontal_axis", "value_degrees"), _get(m, "scalar", "qrs_ms", "value")],
    "1AVB": lambda m: [_get(m, "scalar", "pr_ms", "value")],
    "LNGQT": lambda m: [_get(m, "scalar", "qtc_ms", "value")],
    "LVH": lambda m: [_get(m, "voltage", "sokolow_lyon_mv"), _get(m, "voltage", "r_avl_mv")],
    "RVH": lambda m: [_get(m, "frontal_axis", "value_degrees"), _get(m, "morphology", "per_lead", "V1", "r_mv")],
    "LVOLT": lambda m: [_get(m, "morphology", "max_limb_pp_mv"), _get(m, "morphology", "max_precordial_pp_mv")],
    "LAE": lambda m: [_get(m, "morphology", "p_duration_ms"), _get(m, "morphology", "p_terminal_force_mvs")],
    "RAE": lambda m: [_get(m, "morphology", "p_duration_ms")],
    "MI_inferior": lambda m: [_max_q(m, INFERIOR)],
    "MI_anterior": lambda m: [_max_q(m, ANTERIOR)],
    "MI_lateral": lambda m: [_max_q(m, LATERAL)],
    "STT": lambda m: [max((abs(v) for v in _st(m).values()), default=math.nan), min(_t(m).values(), default=math.nan),
                      float(len((m.get("morphology") or {}).get("t_inversion_leads") or []))],
    "EREPOL": lambda m: [max(_st(m).values(), default=math.nan)],
}


def load_measurements(split):
    """record id -> measurement row (scripts/02_measure.py), including the pacing-spike index under "pace_spike"."""
    out = {}
    for line in open(config.work("measurements", f"{split}.jsonl")):
        r = json.loads(line)
        out[str(r["record_id"])] = r
    return out


def anchor_matrix(measurements, ids, concept):
    """(len(ids), n) array of the criterion measurements of `concept`; NaN where unavailable."""
    f = ANCHORS[concept]
    width = len(f({}))
    return np.array([f(measurements[i]) if i in measurements else [math.nan] * width for i in ids], dtype=float)
