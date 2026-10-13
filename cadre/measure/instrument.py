"""Measure one ECG record: intervals by the consensus of two delineation back-ends (NeuroKit2 and WFDB), rhythm
regularity, P-wave detection, frontal QRS axis, QRS voltage, morphology, and the pacing-spike index.

Every measurement that CADRe uses as a criterion measurement or in an evidence statement comes from this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadre.measure.consensus import ConsensusOracle
from cadre.measure.frontal_axis import frontal_qrs_axis_evidence
from cadre.measure.morphology import morphology_evidence
from cadre.measure.neurokit_oracle import NeuroKitOracle
from cadre.measure.voltage import qrs_voltage_evidence
from cadre.measure.wfdb_loader import load_wfdb_record
from cadre.measure.wfdb_oracle import WFDBOracle

#: Agreement tolerances of the two delineation back-ends, per measurement.
TOLERANCES = {"heart_rate_bpm": 8.0, "rr_ms": 40.0, "pr_ms": 30.0, "qrs_ms": 20.0, "qt_ms": 35.0, "qtc_ms": 40.0,
              "st_mv": 0.08, "t_amplitude_mv": 0.12}
#: Coefficient of variation of the RR series above which the rhythm is irregular.
RR_IRREGULAR_CV = 0.12

_ORACLE: ConsensusOracle | None = None


def oracle() -> ConsensusOracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = ConsensusOracle([NeuroKitOracle(), WFDBOracle()], dict(TOLERANCES))
    return _ORACLE


def _rr_irregularity(rr_values: Any) -> dict[str, Any]:
    """Coefficient of variation of the RR series, per back-end and averaged."""
    per_backend: dict[str, float] = {}
    for backend, series in (rr_values or {}).items():
        array = np.asarray(series, dtype=np.float64)
        if array.size < 3 or float(np.mean(array)) <= 0.0:
            continue
        per_backend[backend] = float(np.std(array) / np.mean(array))
    if not per_backend:
        return {"value": None, "agree": False, "backend_values": {}}
    categories = {value > RR_IRREGULAR_CV for value in per_backend.values()}
    return {"value": float(np.mean(list(per_backend.values()))), "agree": len(per_backend) >= 2 and len(categories) == 1,
            "backend_values": per_backend}


def _p_wave_fraction(values: Any) -> dict[str, Any]:
    """Share of beats with a detected P wave, per back-end and averaged."""
    per_backend: dict[str, float] = {}
    for backend, series in (values or {}).items():
        array = np.asarray(series, dtype=np.float64)
        if array.size == 0:
            continue
        per_backend[backend] = float(np.mean(array))
    if not per_backend:
        return {"value": None, "agree": False, "backend_values": {}}
    categories = {value >= 0.5 for value in per_backend.values()}
    return {"value": float(np.mean(list(per_backend.values()))), "agree": len(per_backend) >= 2 and len(categories) == 1,
            "backend_values": per_backend}


def pacing_spike_index(signal_path: str) -> list[float] | None:
    """Pacing stimuli are deflections far narrower than any QRS complex, so they appear as a few samples with an
    extreme second difference. Per lead, the index is the ratio between the 99.95th and the 99th percentile of the
    absolute second difference (about 1-3 for native beats, much larger with pacing spikes). Returns the median and
    the maximum over the 12 leads."""
    import wfdb
    try:
        x = np.nan_to_num(wfdb.rdrecord(signal_path).p_signal)
    except Exception:
        return None
    d2 = np.abs(np.diff(x, 2, axis=0))
    tail = np.percentile(d2, 99.95, 0) / (np.percentile(d2, 99, 0) + 1e-6)
    return [float(np.median(tail)), float(tail.max())]


def measure(record_id: str, signal_path: str) -> dict[str, Any]:
    """All measurements of one record, as one JSON-serialisable row."""
    row: dict[str, Any] = {"record_id": record_id, "signal_path": signal_path, "instrument_valid": False}
    try:
        record = load_wfdb_record(signal_path, canonical_fs=500)
        consensus = oracle().measure(record)
    except Exception as error:  # an unreadable record is reported, not fatal
        row["failure_reason"] = f"{type(error).__name__}: {error}"
        return row
    if consensus is None:
        row["failure_reason"] = "no delineation back-end succeeded"
        return row
    scalar: dict[str, Any] = {}
    for name in ("heart_rate_bpm", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms"):
        item = consensus.get(name)
        scalar[name] = {"value": item.value if item else None, "agree": bool(item.agree) if item else False,
                        "spread": item.spread if item else None}
    st_item, t_item = consensus.get("st_mv"), consensus.get("t_amplitude_mv")
    rr, pw = consensus.get("rr_intervals_ms"), consensus.get("p_wave_presence")
    row.update({
        "instrument_valid": True,
        "oracle_agreement_rate": consensus.agreement_rate,
        "oracle_backends": consensus.oracle_backends,
        "scalar": scalar,
        "st_mv": {"value": st_item.value if st_item else None, "agree": bool(st_item.agree) if st_item else False,
                  "spread": st_item.spread if st_item else None},
        "t_amplitude_mv": {"value": t_item.value if t_item else None, "agree": bool(t_item.agree) if t_item else False},
        "rr_cv": _rr_irregularity(rr.oracle_values if rr else {}),
        "p_wave": _p_wave_fraction(pw.oracle_values if pw else {}),
        "frontal_axis": frontal_qrs_axis_evidence(record),
        "voltage": qrs_voltage_evidence(record),
        "morphology": morphology_evidence(record),
    })
    return row
