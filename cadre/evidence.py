"""Evidence statements and the evidence review ("Step 0") that opens the regenerated reasoning trace.

Each accepted revision becomes one statement: an addition names the diagnosis, its calibrated probability, and the
criterion measurement of the record; a replacement names the stated value and the value that replaces it.
"""

from __future__ import annotations

import math

NAMES = {"PVC": "premature ventricular complexes", "PAC": "premature atrial complexes", "CRBBB": "right bundle branch block",
         "IRBBB": "incomplete right bundle branch block", "CLBBB": "left bundle branch block",
         "ILBBB": "incomplete left bundle branch block", "LAFB": "left anterior fascicular block",
         "LPFB": "left posterior fascicular block", "IVCD": "nonspecific intraventricular conduction delay",
         "1AVB": "first-degree AV block", "WPW": "Wolff-Parkinson-White pattern", "LVH": "left ventricular hypertrophy",
         "RVH": "right ventricular hypertrophy", "LAE": "left atrial enlargement", "RAE": "right atrial enlargement",
         "LNGQT": "prolonged QT interval", "LVOLT": "low QRS voltages", "MI_inferior": "inferior infarct",
         "MI_anterior": "anterior infarct", "MI_lateral": "lateral infarct", "STT": "ST-T abnormality",
         "EREPOL": "early repolarization",
         "PACE": "paced rhythm (artificial pacemaker)", "AFIB": "atrial fibrillation", "AFLT": "atrial flutter",
         "SR": "sinus rhythm", "SBRAD": "sinus bradycardia", "STACH": "sinus tachycardia",
         "LAD": "left axis deviation", "RAD": "right axis deviation"}
TERRITORY = {"MI_inferior": ("inferior", ["II", "III", "aVF"]), "MI_anterior": ("anterior", ["V1", "V2", "V3", "V4"]),
             "MI_lateral": ("lateral", ["I", "aVL", "V5", "V6"])}
STEP0_HEAD = "<think>\n**Step 0: Evidence review**\nAutomated analysis of the ECG signal and of confirmed similar cases: "
STEP1_HEAD = "\n\n**Step 1: Technical, Rate & Rhythm**\n"


def _get(m, *path):
    for k in path:
        if not isinstance(m, dict) or k not in m:
            return None
        m = m[k]
    return m if not (isinstance(m, float) and math.isnan(m)) else None


def measurement(c, m):
    """The criterion measurement of diagnosis c in record m, as a phrase; None when unavailable."""
    mo = m.get("morphology") or {}
    if c == "1AVB" and _get(m, "scalar", "pr_ms", "value"):
        return f"PR interval {_get(m, 'scalar', 'pr_ms', 'value'):.0f} ms"
    if c == "LNGQT" and _get(m, "scalar", "qtc_ms", "value"):
        return f"QTc {_get(m, 'scalar', 'qtc_ms', 'value'):.0f} ms"
    if c in ("CRBBB", "IRBBB", "CLBBB", "ILBBB", "IVCD", "WPW") and _get(m, "scalar", "qrs_ms", "value"):
        return f"QRS duration {_get(m, 'scalar', 'qrs_ms', 'value'):.0f} ms"
    if c in ("LAFB", "LPFB") and _get(m, "frontal_axis", "value_degrees") is not None:
        return f"frontal QRS axis {_get(m, 'frontal_axis', 'value_degrees'):.0f} degrees"
    if c in ("LVH", "RVH") and _get(m, "voltage", "sokolow_lyon_mv"):
        return (f"Sokolow-Lyon voltage {_get(m, 'voltage', 'sokolow_lyon_mv'):.2f} mV, R wave in aVL "
                f"{_get(m, 'voltage', 'r_avl_mv') or 0:.2f} mV")
    if c == "LVOLT" and mo.get("max_limb_pp_mv") is not None:
        return (f"largest limb-lead QRS amplitude {mo['max_limb_pp_mv']:.2f} mV, largest precordial QRS amplitude "
                f"{mo.get('max_precordial_pp_mv') or 0:.2f} mV")
    if c in TERRITORY:
        name, leads = TERRITORY[c]
        q = [lead for lead in (mo.get("pathologic_q_leads") or []) if lead in leads]
        if q:
            width = max((mo.get("per_lead", {}).get(lead, {}) or {}).get("q_ms") or 0 for lead in q)
            return f"Q waves in {', '.join(q)} (up to {width:.0f} ms)"
        return f"no pathologic Q waves in the {name} leads ({', '.join(leads)})"
    if c in ("STT", "EREPOL"):
        st = _get(m, "st_mv", "value") or {}
        if c == "EREPOL":
            up = [lead for lead, v in st.items() if v is not None and v >= 0.1]
            return f"ST elevation of at least 0.1 mV in {', '.join(up)}" if up else "no ST elevation of 0.1 mV or more"
        dev = [lead for lead, v in st.items() if v is not None and abs(v) >= 0.05]
        inv = mo.get("t_inversion_leads") or []
        parts = []
        if dev:
            parts.append(f"ST deviation of at least 0.05 mV in {', '.join(dev)}")
        if inv:
            parts.append(f"T-wave inversion in {', '.join(inv)}")
        return "; ".join(parts) if parts else "no ST deviation of 0.05 mV or more and no T-wave inversion"
    if c in ("LAE", "RAE") and mo.get("p_duration_ms") is not None:
        return f"P-wave duration {mo['p_duration_ms']:.0f} ms, P terminal force in V1 {mo.get('p_terminal_force_mvs') or 0:.3f} mV*s"
    if c in ("PVC", "PAC") and _get(m, "rr_cv", "value") is not None:
        return f"RR-interval coefficient of variation {_get(m, 'rr_cv', 'value'):.2f}"
    if c == "PACE" and _get(m, "pace_spike", "median") is not None:
        return (f"sharp pacing-spike index {_get(m, 'pace_spike', 'median'):.1f} (ECGs without pacing are usually below 2.2)"
                + (f"; QRS duration {_get(m, 'scalar', 'qrs_ms', 'value'):.0f} ms" if _get(m, "scalar", "qrs_ms", "value") else ""))
    if c in ("AFIB", "AFLT", "SR", "SBRAD", "STACH") and _get(m, "rr_cv", "value") is not None:
        parts = [f"RR-interval coefficient of variation {_get(m, 'rr_cv', 'value'):.2f}"]
        if _get(m, "p_wave", "value") is not None:
            parts.append(f"P waves detected before {100 * _get(m, 'p_wave', 'value'):.0f}% of QRS complexes")
        if _get(m, "scalar", "heart_rate_bpm", "value"):
            parts.append(f"heart rate {_get(m, 'scalar', 'heart_rate_bpm', 'value'):.0f} bpm")
        return ", ".join(parts)
    if c in ("LAD", "RAD") and _get(m, "frontal_axis", "value_degrees") is not None:
        return f"frontal QRS axis {_get(m, 'frontal_axis', 'value_degrees'):.0f} degrees"
    return None


def evidence_lines(decision, m):
    """The evidence statements of one record's revision set."""
    lines = []
    for c in decision["add"]:
        meas = measurement(c, m)
        lines.append(f"{NAMES[c]}: likely present (probability {decision['p_present'][c]:.2f})" + (f"; {meas}" if meas else ""))
    for old, new in decision.get("replaced_by", {}).items():
        lines.append(f"{NAMES[old]}: replaced by {NAMES[new]}")
    return lines


def evidence_review(lines):
    """The assistant prefix from which the frozen model regenerates its report."""
    return STEP0_HEAD + "; ".join(lines) + "." + STEP1_HEAD
