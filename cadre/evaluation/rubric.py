"""The evaluation rubric of ECG-R1 and the reduction of judge items to the eight reported metrics.

The user message is the ECG-R1 evaluation prompt, verbatim, with the generated report and the reference filled in. The
system message restates the rubric's own counting rule (one entry per diagnosis, feature, explanation, and lead) and
the JSON format. Metrics per record: DA = share of diagnosis items scored above 0; Credit = sum of diagnosis scores /
(2 x items); AC, AR, LEV = sums of their item scores; GEU, EBR, RDP = the 0-100 score.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

CRITERIA: tuple[str, ...] = (
    "DiagnosisAccuracy",
    "AnalysisCompleteness",
    "AnalysisRelevance",
    "LeadEvidenceValidity",
    "GroundedECGUnderstanding",
    "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess",
)

PER_ITEM_CRITERIA: tuple[str, ...] = (
    "DiagnosisAccuracy",
    "AnalysisCompleteness",
    "AnalysisRelevance",
    "LeadEvidenceValidity",
)

SUMMED_CRITERIA: tuple[str, ...] = (
    "AnalysisCompleteness",
    "AnalysisRelevance",
    "LeadEvidenceValidity",
)

GLOBAL_CRITERIA: tuple[str, ...] = (
    "GroundedECGUnderstanding",
    "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess",
)

TABLE1_PROMPT = """# Your task: Evaluate the alignment and quality of a generated ECG interpretation by comparing it to a ground truth clinician's interpretation.

## Evaluation Criteria:

    1. DiagnosisAccuracy: Evaluates whether the generated diagnosis is correct, specific, and supported by ECG findings.
       - Scoring
          +2 per diagnosis: Each correctly identified key diagnosis with supporting ECG features.
          +1 per diagnosis: Each mostly correct diagnosis but lacking key supporting details.
          +0 per diagnosis: Each incorrect or vague diagnosis not supported by ECG features.

    2. AnalysisCompleteness: Checks if all key ECG components (rhythm, intervals, waveforms, and lead-specific findings) are discussed.
       - Scoring
          +1 per feature: For each correctly addressed key ECG feature (e.g., rhythm, PR interval, QRS duration, ST segment, T wave morphology).
          +0 per missing feature: For each key feature omitted or inaccurately described.

    3. AnalysisRelevance: Assesses whether each provided explanation directly supports the diagnosis.
       - Scoring
          +2 per feature or per lead: Each point that strongly supports the diagnosis with clear ECG evidence.
          +1 per feature or per lead: Some points are relevant but not fully justified.
          +0: Includes unrelated or misleading explanations.

    4. LeadEvidenceValidity: Evaluates whether the lead-related statements are diagnostically necessary, correctly grounded, and free of unsupported lead-wise claims, rather than maximizing the number of mentioned leads.
       - Scoring
          +2 per key lead/region: For each diagnosis-critical lead (or contiguous lead group / territory) correctly referenced with explicit and matching ECG evidence (e.g., ST elevation/depression, T-wave inversion, pathologic Q waves, bundle-branch morphology) that supports the stated diagnosis.
          +1 per key lead/region: For each key lead/region mentioned with partially correct evidence, but missing important details (e.g., direction without magnitude/morphology) or showing minor inconsistencies.
          +0 per key lead/region: Key lead/region is omitted or the described finding is incorrect / non-grounded and does not support the diagnosis.

    5. GroundedECGUnderstanding: Determines if the interpretation references actual ECG features (e.g., QRS amplitude, PR interval) instead of generic terms.
       - Scoring (0-100)
          100: ECG findings are comprehensively cited, linked to diagnoses, and cover all relevant ECG features.
          80: ECG findings are explicitly cited and linked to diagnoses.
          50: Some ECG references exist but are incomplete.
          0: Lacks specific waveform references.

    6. EvidenceBasedReasoning: Evaluates whether the diagnosis follows logical, evidence-supported steps.
       - Scoring (0-100)
          100: Findings logically progress to diagnosis with thorough and clear justifications covering all necessary steps.
          80: Findings logically progress to diagnosis with clear justifications.
          50: Some reasoning exists but lacks complete step-by-step analysis.
          0: Reasoning is unclear or not derived from ECG findings.

    7. RealisticDiagnosticProcess: Assesses if the model mimics how a clinician interprets an ECG, considering all relevant factors.
       - Scoring (0-100)
          100: The analysis follows a structured clinical approach and considers all relevant clinical factors.
          80: The analysis follows a structured clinical approach.
          50: Some clinical reasoning is present but incomplete.
          0: The approach lacks structured clinical reasoning.

    NOTE: Each score must be calculated based on strict criteria to ensure objective evaluation.

## Generated ECG Interpretation:
{generated}

## Ground Truth Clinician's Interpretation:
{groundtruth}

**Response:** [Please organize your output in a JSON format for each criterion, including a brief explanation for each aspect. Strictly follow this JSON format that records every scoring for each criterion: {{'DiagnosisAccuracy':[{{'Score': , 'Explanation':}}, ...], 'AnalysisCompleteness':[{{'Score': , 'Explanation':}}, ...], 'AnalysisRelevance':[{{'Score': , 'Explanation':}}, ...], 'LeadEvidenceValidity':[{{'Score': , 'Explanation':}}, ...], 'GroundedECGUnderstanding':[{{'Score': , 'Explanation':}}, ...], 'EvidenceBasedReasoning':[{{'Score': , 'Explanation':}}, ...], 'RealisticDiagnosticProcess':[{{'Score': , 'Explanation':}}, ...]}}]"""

ENUMERATION_SYSTEM = (
    "You are grading with the rubric in the user message. Follow its counting "
    "rule exactly: DiagnosisAccuracy is scored '+2 per diagnosis', '+1 per "
    "diagnosis', '+0 per diagnosis', and AnalysisCompleteness, AnalysisRelevance "
    "and LeadEvidenceValidity are likewise scored per feature, per explanation "
    "and per lead. Emit one JSON array entry for each individual diagnosis, "
    "feature, explanation and lead you score -- never a single pooled entry "
    "summarising several of them. Score every diagnosis the generated "
    "interpretation asserts, including ones the ground truth does not mention, "
    "which take +0 when they are unsupported. Keep each Explanation brief."
    " The top-level response must be exactly one JSON object with the seven "
    "criterion names from the requested schema as its keys. Put arrays only "
    "inside those keys; never return a top-level array and never split the "
    "seven criteria across separate objects."
)

STRICT_JSON_RETRY_SYSTEM = (
    " This is a format-only retry after the previous response could not be parsed. "
    "Return one valid JSON object and nothing else. Use exactly the seven requested "
    "criterion keys. Every key must map to a non-empty array of objects with numeric "
    "Score and string Explanation fields. Check brackets and commas before replying."
)

def _extract_json(raw: str) -> dict[str, Any]:
    """Extract one scoring object from a judge reply without changing any score."""
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    stripped = raw.strip()
    try:
        candidates.append(json.loads(stripped))
    except json.JSONDecodeError:
        for index, character in enumerate(raw):
            if character not in '[{':
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            candidates.append(value)
            break
    if not candidates:
        raise ValueError('judge response did not contain JSON')
    value = candidates[0]
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for part in value:
            if not isinstance(part, dict) or not part:
                raise ValueError('top-level judge array lost criterion boundaries')
            if any((key not in CRITERIA or key in merged for key in part)):
                raise ValueError('top-level judge array is not a unique criterion map')
            merged.update(part)
        value = merged
    if not isinstance(value, dict):
        raise ValueError('judge response JSON must be an object')
    return value

_ENTRY_KEYS = ("Score", "Explanation")

def _score_value(value: Any) -> float:
    """Parse a numeric score or an unambiguous rubric-labelled score."""
    if isinstance(value, str):
        match = re.fullmatch('\\s*\\+?(-?\\d+(?:\\.\\d+)?)\\s*(?:per\\s+[A-Za-z /-]+)?\\s*', value)
        if match is None:
            raise ValueError(f'judge score is not numeric: {value!r}')
        value = match.group(1)
    return float(value)

def _unflatten(value: Any) -> Any:
    """Rebuild an entry written as a flat key/value array."""
    if not isinstance(value, list) or len(value) < 2 or len(value) % 2:
        return value
    keys = value[0::2]
    if not all((isinstance(key, str) and key in _ENTRY_KEYS for key in keys)):
        return value
    if len(set(keys)) != len(keys):
        return value
    return [dict(zip(keys, value[1::2], strict=True))]

def _normalize(payload: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Per-criterion score lists from a judge reply; out-of-range scores are rejected."""
    criteria: dict[str, list[dict[str, Any]]] = {}
    coerced = 0
    for name in CRITERIA:
        original = payload.get(name)
        raw_items = _unflatten(original)
        if raw_items is not original:
            coerced += 1
        if isinstance(raw_items, (int, float)) and (not isinstance(raw_items, bool)):
            raw_items = [{'Score': float(raw_items), 'Explanation': ''}]
            coerced += 1
        elif isinstance(raw_items, dict):
            raw_items = [raw_items]
            coerced += 1
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f'judge criterion is absent or empty: {name}')
        items = []
        for raw in raw_items:
            if isinstance(raw, (int, float)) and (not isinstance(raw, bool)):
                raw = {'Score': float(raw), 'Explanation': ''}
                coerced += 1
            if not isinstance(raw, dict):
                raise ValueError(f'judge criterion item is not an object: {name}')
            score = _score_value(raw.get('Score', raw.get('score')))
            if name in SUMMED_CRITERIA:
                cap = 1.0 if name == 'AnalysisCompleteness' else 2.0
                if score < 0.0:
                    raise ValueError(f'{name} score out of range: {score}')
                if score > cap:
                    coerced += 1
            elif name in PER_ITEM_CRITERIA:
                if not 0.0 <= score <= 2.0:
                    raise ValueError(f'{name} score out of range: {score}')
            elif not 0.0 <= score <= 100.0:
                raise ValueError(f'{name} score out of range: {score}')
            items.append({'Score': score, 'Explanation': str(raw.get('Explanation', raw.get('explanation', '')))})
        criteria[name] = items
    return (criteria, coerced)

def table1_metrics(criteria: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Record-level values of the eight metrics from the judge items of one record."""
    metrics: dict[str, float] = {}
    diagnosis = [item['Score'] for item in criteria['DiagnosisAccuracy']]
    metrics['diagnosis_accuracy_binary'] = 100.0 * sum((score > 0.0 for score in diagnosis)) / len(diagnosis)
    metrics['diagnosis_accuracy_credit'] = 100.0 * sum(diagnosis) / (2.0 * len(diagnosis))
    for name in ('AnalysisCompleteness', 'AnalysisRelevance', 'LeadEvidenceValidity'):
        metrics[name] = float(sum((item['Score'] for item in criteria[name])))
    for name in GLOBAL_CRITERIA:
        scores = [item['Score'] for item in criteria[name]]
        metrics[name] = float(sum(scores) / len(scores))
    return metrics
