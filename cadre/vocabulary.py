"""Diagnostic vocabulary of CADRe and the rules that read diagnoses from text.

The same rules read MIMIC-IV-ECG machine statements (memory labels), reference interpretations, and generated reports
(the diagnoses a report already states). Machine-specific phrasings collapse into one diagnosis, e.g. "probable left
atrial enlargement" and "possible left atrial abnormality" both read as LAE.

CADRe revises 27 diagnoses: 22 coexisting findings, which can be added to a report, and the values of two mutually
exclusive groups, the dominant rhythm (AFIB, AFLT, PACE) and the frontal axis (LAD, RAD), which replace the value a
report states. Sinus rhythm (SR) is queried as the default value of the rhythm group.
"""

import re

# name: canonical wording
CONCEPTS = {
    "SR": "Sinus rhythm", "SBRAD": "Sinus bradycardia", "STACH": "Sinus tachycardia", "SARRH": "Sinus arrhythmia",
    "AFIB": "Atrial fibrillation", "AFLT": "Atrial flutter", "PACE": "Paced rhythm", "SVT": "Supraventricular tachycardia",
    "LAD": "Left axis deviation", "RAD": "Right axis deviation", "NORM": "Normal ECG",
    "PVC": "Premature ventricular complexes", "PAC": "Premature atrial complexes",
    "CRBBB": "Right bundle branch block", "IRBBB": "Incomplete right bundle branch block",
    "CLBBB": "Left bundle branch block", "ILBBB": "Incomplete left bundle branch block",
    "LAFB": "Left anterior fascicular block", "LPFB": "Left posterior fascicular block",
    "IVCD": "Nonspecific intraventricular conduction delay", "1AVB": "First degree AV block",
    "WPW": "Wolff-Parkinson-White pattern",
    "LVH": "Left ventricular hypertrophy", "RVH": "Right ventricular hypertrophy",
    "LAE": "Left atrial enlargement", "RAE": "Right atrial enlargement",
    "LNGQT": "Prolonged QT interval", "LVOLT": "Low QRS voltages",
    "MI_inferior": "Inferior infarct", "MI_anterior": "Anterior infarct", "MI_lateral": "Lateral infarct",
    "STT": "Nonspecific ST-T abnormality", "EREPOL": "Early repolarization",
}
NAMES = list(CONCEPTS)

#: Coexisting findings, revised by addition.
FINDINGS = ["PVC", "PAC", "CRBBB", "IRBBB", "CLBBB", "ILBBB", "LAFB", "LPFB", "IVCD", "1AVB", "WPW", "LVH", "RVH",
            "LAE", "RAE", "LNGQT", "LVOLT", "MI_inferior", "MI_anterior", "MI_lateral", "STT", "EREPOL"]
#: Rhythm group: queried values (SR is the default value) and the sinus statements a replacement removes.
RHYTHM = ["SR", "AFIB", "AFLT", "PACE"]
SINUS = {"SR", "SBRAD", "STACH"}
#: Axis group.
AXIS = ["LAD", "RAD"]

#: Wording of each diagnosis in the belief query "Does this ECG show <wording>? Answer with one word: Present or Absent."
QUERY_WORDING = {
    "PVC": "premature ventricular complexes", "PAC": "premature atrial complexes", "CRBBB": "right bundle branch block",
    "IRBBB": "incomplete right bundle branch block", "CLBBB": "left bundle branch block",
    "ILBBB": "incomplete left bundle branch block", "LAFB": "left anterior fascicular block",
    "LPFB": "left posterior fascicular block", "IVCD": "a nonspecific intraventricular conduction delay",
    "1AVB": "first-degree atrioventricular block", "WPW": "a Wolff-Parkinson-White pattern",
    "LVH": "left ventricular hypertrophy", "RVH": "right ventricular hypertrophy", "LAE": "left atrial enlargement",
    "RAE": "right atrial enlargement", "LNGQT": "a prolonged QT interval", "LVOLT": "low QRS voltages",
    "MI_inferior": "an inferior myocardial infarction", "MI_anterior": "an anterior myocardial infarction",
    "MI_lateral": "a lateral myocardial infarction", "STT": "ST-T wave abnormalities", "EREPOL": "early repolarization",
    "SR": "sinus rhythm", "AFIB": "atrial fibrillation", "AFLT": "atrial flutter",
    "PACE": "a paced rhythm from an artificial pacemaker", "LAD": "left axis deviation", "RAD": "right axis deviation",
}
QUERY_TEMPLATE = "Does this ECG show {}? Answer with one word: Present or Absent."
#: Order of the queried diagnoses in every belief file.
QUERIED = FINDINGS + RHYTHM + AXIS

_STT = (r"st-t (wave )?(changes|abnormalit)|\bt[- ]wave (changes|abnormalit|inversion|flattening)|\bt (changes|abnormalit)|"
        r"inverted t[- ]waves?|t[- ]wave inversions?|\bst (segment )?(depression|elevation|changes|abnormalit)|st junctional|"
        r"nonspecific st|repolarization (abnormalit|changes)|ischa?emi")
#: Machine statements that flag ST elevation as possible acute injury are not read as a nonspecific ST-T abnormality.
_STT_MACHINE = r"st-t|\bt[- ]wave|\bt (changes|abnormalit)|st depression|st junctional|repolarization abnormality|ischa?emi|\bst changes"

RULES = [
    (r"^(probable )?sinus rhythm\b|\bnormal sinus rhythm\b", ["SR"]),
    (r"^(probable )?sinus bradycardia\b|\bsinus bradycardia\b", ["SBRAD"]),
    (r"^(probable )?sinus tachycardia\b|\bsinus tachycardia\b", ["STACH"]), (r"\bsinus arrhythmia\b", ["SARRH"]),
    (r"\batrial fibrillation\b", ["AFIB"]), (r"\batrial flutter\b", ["AFLT"]),
    (r"(?<!natural )(?<!natural cardiac )pacemaker (rhythm|spikes?|capture)|artificial pacemaker|electronic pacemaker|"
     r"sequential pacemaker|\b(atrial|ventricular|demand|av dual|dual chamber|sequential) pacing\b|"
     r"\b(atrial|ventricular|av|dual)[- ]?paced\b|\bpaced (rhythm|beats?|complex)", ["PACE"]),
    (r"supraventricular tachycardia|\bsvt\b", ["SVT"]),
    (r"\b(left axis deviation|leftward axis)\b", ["LAD"]), (r"\b(right axis deviation|rightward axis)\b", ["RAD"]),
    (r"\bpvcs?\b|\bpvc\(s\)|(?<!supra)ventricular (premature|ectopic|extrasystole)|premature ventricular", ["PVC"]),
    (r"\bpacs?\b|\bpac\(s\)|atrial premature|premature atrial|supraventricular (ectopic|extrasystole|premature)", ["PAC"]),
    (r"(?<!incomplete )(\brbbb\b|right bundle branch block)", ["CRBBB"]),
    (r"incomplete (rbbb|right bundle branch block)", ["IRBBB"]),
    (r"(?<!incomplete )(\blbbb\b|left bundle branch block)", ["CLBBB"]),
    (r"incomplete (lbbb|left bundle branch block)", ["ILBBB"]),
    (r"left anterior fascicular block|\blafb\b", ["LAFB"]), (r"left posterior fascicular block|\blpfb\b", ["LPFB"]),
    (r"\biv conduction (defect|delay)\b|intraventricular conduction (defect|delay|disturbance)", ["IVCD"]),
    (r"(1st|first)[- ]degree a-?v block|prolonged pr\b", ["1AVB"]), (r"wolff?-parkinson-white|\bwpw\b", ["WPW"]),
    (r"left ventricular hypertrophy|\blvh\b", ["LVH"]), (r"right ventricular hypertrophy|\brvh\b", ["RVH"]),
    (r"left atrial (enlargement|abnormality|overload)", ["LAE"]),
    (r"right atrial (enlargement|abnormality|overload)", ["RAE"]),
    (r"prolonged qtc?\b|long qtc?\b|qtc? prolongation", ["LNGQT"]), (r"low (qrs )?voltage|low .*lead voltage", ["LVOLT"]),
    (r"\binferior\b.*\binfarct", ["MI_inferior"]),
    (r"\b(anterior|anteroseptal|septal|anterolateral)\b.*\binfarct", ["MI_anterior"]),
    (r"\b(lateral|anterolateral|inferolateral)\b.*\binfarct", ["MI_lateral"]),
    (_STT, ["STT"]),
    (r"early repol", ["EREPOL"]),
    (r"^normal ecg$", ["NORM"]),
]
NOT_A_FINDING = re.compile(r"cannot rule out|probable normal variant")
NEGATION = re.compile(r"\b(no|not|without|absence of|absent|negative for|rule out|ruled out|rules out|ruling out|exclud\w*|"
                      r"free of|neither|nor|unremarkable|within normal limits|normal limits)\b")
ANSWER = re.compile(r"<answer>(?P<body>.*?)</answer>", re.S | re.I)


def norm(label):
    return " ".join(label.strip().lower().rstrip(".").split())


def answer_items(text):
    """(normalised item, original item) pairs of a report's <answer> list, in order."""
    m = ANSWER.search(text or "")
    if not m:
        return []
    items = [p.strip() for p in re.split(r"[;\n]+", m.group("body")) if p.strip()]
    return [(norm(p), p) for p in items]


def statement_concepts(statement, machine=False):
    """Diagnoses named by one statement (a machine statement, an answer item, or a sentence)."""
    s = norm(statement)
    if NOT_A_FINDING.search(s) and "early repol" not in s:
        return []
    out = []
    for pattern, names in RULES:
        if machine and names == ["STT"]:
            pattern = _STT_MACHINE
        if re.search(pattern, s):
            out += [n for n in names if n not in out]
    return out


def report_concepts(text):
    """Diagnoses a report asserts: from its <answer> list when it has one, otherwise from the sentences of the free
    text, skipping any clause with a negation cue ("no evidence of bundle branch block")."""
    if ANSWER.search(text or ""):
        return {c for s, _ in answer_items(text) for c in statement_concepts(s)}
    body = (text or "").split("</think>")[-1]
    found = set()
    for clause in re.split(r"[.;:\n]|,\s*(?:and|but|with)\s+|\bbut\b", body):
        clause = clause.strip().lower()
        if not clause or NEGATION.search(clause):
            continue
        for pattern, names in RULES:
            if pattern.startswith("^"):  # anchored rules are for statements; free text names the finding anywhere
                pattern = pattern.replace("^", r"\b")
            if re.search(pattern, clause):
                found.update(names)
    return found
