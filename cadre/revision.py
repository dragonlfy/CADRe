"""Criterion-anchored diagnostic revision and risk-calibrated update.

For a diagnosis c of an ECG x, the state is s_c(x) = [b_c, e_c, w_c m_c]: the belief elicited from the base model, the
expert judgment (largest standardized expert logit among the statements that name c), and the criterion measurements
of c, each standardized on the experience memory. The k memory ECGs nearest to s_c(x) give the empirical support
p_hat, and the initial belief l_c (logistic calibration of b_c on the memory) is revised to

    l'_c = l_c + [(1 + kappa) p_hat - kappa] / beta.

Coexisting findings are calibrated by an isotonic map fitted on out-of-fold memory scores and added when the
calibrated probability passes the findings threshold. Rhythm and axis values form exclusive groups: a value is
asserted when sigmoid(l'_c) passes the stricter replacement threshold, and it replaces the value the report states.
All hyperparameters are selected by patient-grouped cross-validation within the memory (`select_*`); the selected
values are stored in configs/cadre.yaml.
"""

from __future__ import annotations

import json
import zlib

import numpy as np

from cadre import config
from cadre.anchors import anchor_matrix, load_measurements
from cadre.data import by_id, patient_slice
from cadre.vocabulary import AXIS, FINDINGS, NAMES, QUERIED, RHYTHM, SINUS, report_concepts, statement_concepts


# ---------------------------------------------------------------------------------------------------- inputs
def load_beliefs(split):
    """id -> beliefs over QUERIED (scripts/05_elicit_beliefs.py)."""
    d = np.load(config.work("beliefs", f"{split}.npz"))
    assert list(d["concepts"]) == QUERIED
    return dict(zip([str(x) for x in d["ids"]], d["belief"].astype(float)))


def load_expert(split, ids):
    order = json.loads(config.work("expert", f"ids_{split}.json").read_text())
    row = {i: k for k, i in enumerate(order)}
    return np.load(config.work("expert", f"logits_{split}.npy"))[[row[i] for i in ids]]


def expert_vocabulary():
    return json.loads(config.work("expert", "vocab.json").read_text())


def load_reports(split):
    out = {}
    for line in open(config.work("reports", f"{split}.jsonl")):
        r = json.loads(line)
        out[r["id"]] = r["prediction_raw"]
    return out


def load_machine_labels(split, ids):
    d = np.load(config.work("labels", f"machine_{split}.npz"))
    row = {str(i): k for k, i in enumerate(d["ids"])}
    return d["Y"][[row[i] for i in ids]]


def statements_naming(concepts):
    """Per diagnosis, the expert outputs (machine statements) that name it."""
    vocab = expert_vocabulary()
    return {c: [x for x, v in enumerate(vocab) if c in statement_concepts(v)] for c in concepts}


def logistic_logit(z_train, y_train, z):
    """Initial belief l_c: logistic calibration of the elicited belief on the memory."""
    from sklearn.linear_model import LogisticRegression
    if y_train.min() == y_train.max():
        return np.full(len(z), -10.0)
    return LogisticRegression(max_iter=1000).fit(z_train[:, None], y_train).decision_function(z[:, None])


def revise(l0, p_hat, kappa, beta):
    return l0 + (p_hat * (1 + kappa) - kappa) / beta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Memory:
    """The experience memory: one row per memory ECG, ordered by id."""

    def __init__(self):
        beliefs = load_beliefs("memory")
        records = by_id("memory")
        self.ids = sorted(i for i in beliefs if i in records and not np.isnan(beliefs[i]).any())
        B = np.stack([beliefs[i] for i in self.ids])
        self.Z = {c: B[:, k] for k, c in enumerate(QUERIED)}
        self.E = load_expert("memory", self.ids)
        self.machine = load_machine_labels("memory", self.ids)
        interp = json.loads(config.work("labels", "interpretation_findings.json").read_text())
        self.interp = np.array([[c in set(interp.get(i, [])) for c in FINDINGS] for i in self.ids]).astype(int)
        reports = load_reports("memory")
        self.stated = [set(report_concepts(reports[i])) for i in self.ids]
        self.patient = [records[i]["patient"] for i in self.ids]
        self.fold = np.array([zlib.crc32(p.encode()) % 5 for p in self.patient])
        self.measurements = load_measurements("memory")

    def label(self, c):
        """Machine-statement label of c; for SR, any sinus rhythm."""
        if c == "SR":
            return self.machine[:, [NAMES.index(x) for x in sorted(SINUS)]].max(1).astype(int)
        return self.machine[:, NAMES.index(c)].astype(int)


class Target:
    """The records CADRe revises (the test set)."""

    def __init__(self, split="test"):
        beliefs = load_beliefs(split)
        self.ids = sorted(beliefs)
        B = np.stack([beliefs[i] for i in self.ids])
        self.Z = {c: B[:, k] for k, c in enumerate(QUERIED)}
        self.E = load_expert(split, self.ids)
        reports = load_reports(split)
        self.stated = [set(report_concepts(reports[i])) for i in self.ids]
        self.measurements = load_measurements(split)


class StateSpace:
    """Per-diagnosis states [b_c, e_c, w m_c], standardized on the memory."""

    def __init__(self, mem, concepts):
        from sklearn.preprocessing import StandardScaler
        self.concepts = list(concepts)
        Zm = np.stack([mem.Z[c] for c in self.concepts], 1)
        self.zs, self.es = StandardScaler().fit(Zm), StandardScaler().fit(mem.E)
        self.rel = statements_naming(self.concepts)
        self.med, self.asc = {}, {}
        for c in self.concepts:
            A = anchor_matrix(mem.measurements, mem.ids, c)
            self.med[c] = np.nanmedian(A, 0)
            self.asc[c] = StandardScaler().fit(np.where(np.isnan(A), self.med[c], A))

    def __call__(self, Z, E, measurements, ids, c, w):
        j = self.concepts.index(c)
        Zs = self.zs.transform(np.stack([Z[x] for x in self.concepts], 1))[:, [j]]
        e = self.es.transform(E)[:, self.rel[c]].max(1, keepdims=True) if self.rel[c] else np.zeros((len(ids), 1))
        parts = [Zs, e]
        if w > 0:
            A = anchor_matrix(measurements, ids, c)
            parts.append(w * self.asc[c].transform(np.where(np.isnan(A), self.med[c], A)))
        return np.hstack(parts)


def neighbours(X_memory, X, k):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k).fit(X_memory).kneighbors(X)[1]


# ---------------------------------------------------------------------------------------- coexisting findings
def _findings_labels(mem, source):
    machine = mem.machine[:, [NAMES.index(c) for c in FINDINGS]].astype(int)
    return machine if source == "machine" else np.maximum(machine, mem.interp)


def _findings_oof(mem, space, labels, w, k, beta, kappa):
    """Out-of-fold revised scores of every memory ECG and finding (patient-grouped folds)."""
    n = len(mem.ids)
    oof = np.zeros((n, len(FINDINGS)))
    X = {c: space(mem.Z, mem.E, mem.measurements, mem.ids, c, w) for c in FINDINGS}
    for f in range(5):
        tr, te = mem.fold != f, mem.fold == f
        for j, c in enumerate(FINDINGS):
            nb = neighbours(X[c][tr], X[c][te], k)
            y = labels[:, j]
            oof[te, j] = revise(logistic_logit(mem.Z[c][tr], y[tr], mem.Z[c][te]), y[tr][nb].mean(1), kappa, beta)
    return oof


def findings_decisions(mem, tgt, s):
    """Additions of coexisting findings: (per-record added findings, calibrated probabilities)."""
    from sklearn.isotonic import IsotonicRegression
    cfg = s["findings"]
    space = StateSpace(mem, FINDINGS)
    labels = _findings_labels(mem, cfg["labels"])
    union = _findings_labels(mem, "machine+interpretation")
    candidates = np.array([[c not in st for c in FINDINGS] for st in mem.stated])
    oof = _findings_oof(mem, space, labels, cfg["w"], cfg["k"], cfg["beta"], cfg["kappa"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[candidates], union[candidates])
    scores = np.zeros((len(tgt.ids), len(FINDINGS)))
    for j, c in enumerate(FINDINGS):
        Xm = space(mem.Z, mem.E, mem.measurements, mem.ids, c, cfg["w"])
        nb = neighbours(Xm, space(tgt.Z, tgt.E, tgt.measurements, tgt.ids, c, cfg["w"]), cfg["k"])
        scores[:, j] = revise(logistic_logit(mem.Z[c], labels[:, j], tgt.Z[c]), labels[:, j][nb].mean(1),
                              cfg["kappa"], cfg["beta"])
    q = iso.predict(scores.ravel()).reshape(scores.shape)
    out = {}
    for r, i in enumerate(tgt.ids):
        out[i] = {"add": [c for j, c in enumerate(FINDINGS) if q[r, j] >= cfg["threshold"] and c not in tgt.stated[r]],
                  "p_present": {c: float(q[r, j]) for j, c in enumerate(FINDINGS)}}
    return out


def select_findings(mem, s):
    """Cross-validated choice of (labels, w, k, beta) for coexisting findings: net utility of the additions on held-out
    folds, +1 for an addition the reference confirms and -u_minus otherwise, with u_minus = tau / (1 - tau)."""
    from sklearn.isotonic import IsotonicRegression
    cfg, grid = s["findings"], s["grids"]["findings"]
    tau = cfg["threshold"]
    u_minus = tau / (1 - tau)
    space = StateSpace(mem, FINDINGS)
    union = _findings_labels(mem, "machine+interpretation")
    labels = {source: _findings_labels(mem, source) for source in grid["labels"]}
    candidates = np.array([[c not in st for c in FINDINGS] for st in mem.stated])
    n, kmax = len(mem.ids), max(grid["k"])
    oof = {}
    for w in grid["w"]:
        X = {c: space(mem.Z, mem.E, mem.measurements, mem.ids, c, w) for c in FINDINGS}
        for f in range(5):
            tr, te = mem.fold != f, mem.fold == f
            for j, c in enumerate(FINDINGS):
                nb = neighbours(X[c][tr], X[c][te], kmax)
                for source, Y in labels.items():
                    y = Y[:, j]
                    l0 = logistic_logit(mem.Z[c][tr], y[tr], mem.Z[c][te])
                    for k in grid["k"]:
                        p = y[tr][nb[:, :k]].mean(1)
                        for beta in grid["beta"]:
                            key = (source, w, k, beta)
                            if key not in oof:
                                oof[key] = np.zeros((n, len(FINDINGS)))
                            oof[key][te, j] = revise(l0, p, cfg["kappa"], beta)
    table = {}
    for key, Zp in oof.items():
        net = 0.0
        for f in range(5):
            tr, te = (mem.fold != f)[:, None] & candidates, (mem.fold == f)[:, None] & candidates
            a = IsotonicRegression(out_of_bounds="clip").fit(Zp[tr], union[tr]).predict(Zp[te]) >= tau
            y = union[te]
            net += a[y == 1].sum() - u_minus * a[y == 0].sum()
        table[key] = 100 * net / n
    best = max(table, key=table.get)
    return {"labels": best[0], "w": best[1], "k": best[2], "beta": best[3], "net_per_100": table[best]}


# ----------------------------------------------------------------------------------- nonspecific ST-T abnormality
def stt_decisions(mem, tgt, s):
    """Nonspecific ST-T abnormality: machine-statement labels, state [b, e] (w = 0); the isotonic map is fitted on a
    patient-disjoint slice of the memory, scored against the reference of each held-out ECG."""
    from sklearn.isotonic import IsotonicRegression
    from cadre.labels import reference_concepts, train_references
    cfg = s["stt"]
    machine = mem.machine[:, [NAMES.index(c) for c in FINDINGS]].astype(int)

    def scores(rows, Z, E, ids):
        space = _SubSpace(mem, rows)
        out = np.zeros((len(ids), len(FINDINGS)))
        for j, c in enumerate(FINDINGS):
            nb = neighbours(space(mem.Z, mem.E, rows, c), space(Z, E, None, c), cfg["k"])
            y = machine[rows, j]
            out[:, j] = revise(logistic_logit(mem.Z[c][rows], y, Z[c]), y[nb].mean(1), cfg["kappa"], cfg["beta"])
        return out

    # calibration on the held-out slice, with a memory that excludes it
    held = np.array([patient_slice(p, cfg["calibration_slice_modulus"]) == cfg["calibration_slice"] for p in mem.patient])
    refs = train_references()
    hold = [k for k in np.where(held)[0] if mem.ids[k] in refs]
    Zh = {c: mem.Z[c][hold] for c in FINDINGS}
    ph = sigmoid(scores(np.where(~held)[0], Zh, mem.E[hold], [mem.ids[k] for k in hold]))
    x, y = [], []
    for r, k in enumerate(hold):
        ref = reference_concepts(*refs[mem.ids[k]])
        for j, c in enumerate(FINDINGS):
            if c not in mem.stated[k]:
                x.append(ph[r, j])
                y.append(c in ref)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.array(x), np.array(y, dtype=float))
    # the target records, with the whole memory
    j = FINDINGS.index("STT")
    q = iso.predict(sigmoid(scores(np.arange(len(mem.ids)), tgt.Z, tgt.E, tgt.ids)[:, j]))
    return {i: {"add": q[r] >= cfg["threshold"] and "STT" not in tgt.stated[r], "p": float(q[r])}
            for r, i in enumerate(tgt.ids)}


class _SubSpace:
    """[b_c, e_c] states standardized on a subset of the memory."""

    def __init__(self, mem, rows):
        from sklearn.preprocessing import StandardScaler
        Zm = np.stack([mem.Z[c][rows] for c in FINDINGS], 1)
        self.zs, self.es = StandardScaler().fit(Zm), StandardScaler().fit(mem.E[rows])
        self.rel = statements_naming(FINDINGS)
        self.rows = rows

    def __call__(self, Z, E, rows, c):
        if rows is not None:
            Z = {x: Z[x][rows] for x in FINDINGS}
            E = E[rows]
        j = FINDINGS.index(c)
        Zs = self.zs.transform(np.stack([Z[x] for x in FINDINGS], 1))[:, [j]]
        e = self.es.transform(E)[:, self.rel[c]].max(1, keepdims=True) if self.rel[c] else np.zeros((len(Zs), 1))
        return np.hstack([Zs, e])


# ------------------------------------------------------------------------------------------- exclusive groups
def _group_space(mem, concepts, anchor_of):
    from sklearn.preprocessing import StandardScaler
    Zm = np.stack([mem.Z[c] for c in concepts], 1)
    zs, es = StandardScaler().fit(Zm), StandardScaler().fit(mem.E)
    rel = statements_naming(concepts)
    med, asc = {}, {}
    for c in concepts:
        A = anchor_matrix(mem.measurements, mem.ids, anchor_of(c))
        med[c] = np.nanmedian(A, 0)
        asc[c] = StandardScaler().fit(np.where(np.isnan(A), med[c], A))

    def state(Z, E, measurements, ids, c, w):
        j = concepts.index(c)
        parts = [zs.transform(np.stack([Z[x] for x in concepts], 1))[:, [j]],
                 es.transform(E)[:, rel[c]].max(1, keepdims=True) if rel[c] else np.zeros((len(ids), 1))]
        if w > 0:
            A = anchor_matrix(measurements, ids, anchor_of(c))
            parts.append(w * asc[c].transform(np.where(np.isnan(A), med[c], A)))
        return np.hstack(parts)
    return state


def _rhythm_applicable(stated, c):
    if c == "SR":
        return np.array([bool(st & {"AFIB", "AFLT"}) and not (st & SINUS) for st in stated])
    return np.array([bool(st & SINUS) and c not in st for st in stated])


def _axis_applicable(stated, c):
    return np.array([c not in st for st in stated])


def _select_group(mem, concepts, anchor_of, applicable, kappa, grid):
    """Per value of an exclusive group: (k, beta, w) maximizing the cross-validated net utility of its assertions on
    the memory ECGs where the assertion could act; a value whose best utility is not positive is never asserted."""
    state = _group_space(mem, concepts, anchor_of)
    tau_logit = np.log(kappa)  # sigmoid(l') > kappa / (1 + kappa)
    chosen, enabled = {}, []
    for c in concepts:
        y, app = mem.label(c), applicable(mem.stated, c)
        cv = {(k, b, w): 0.0 for k in grid["k"] for b in grid["beta"] for w in grid["w"]}
        for f in range(5):
            tr, te = mem.fold != f, (mem.fold == f) & app
            Ztr, Zte = {x: mem.Z[x][tr] for x in concepts}, {x: mem.Z[x][te] for x in concepts}
            ids_tr, ids_te = [mem.ids[i] for i in np.where(tr)[0]], [mem.ids[i] for i in np.where(te)[0]]
            base = logistic_logit(mem.Z[c][tr], y[tr], mem.Z[c][te])
            for w in grid["w"]:
                for k in grid["k"]:
                    nb = neighbours(state(Ztr, mem.E[tr], mem.measurements, ids_tr, c, w),
                                    state(Zte, mem.E[te], mem.measurements, ids_te, c, w), k)
                    p = y[tr][nb].mean(1)
                    for b in grid["beta"]:
                        a = revise(base, p, kappa, b) > tau_logit
                        cv[(k, b, w)] += (a & (y[te] == 1)).sum() - kappa * (a & (y[te] == 0)).sum()
        best = max(cv, key=cv.get)
        chosen[c] = {"k": best[0], "beta": best[1], "w": best[2]}
        if cv[best] > 0:
            enabled.append(c)
    return {"values": chosen, "enabled": enabled}


def _group_probabilities(mem, tgt, concepts, anchor_of, cfg):
    state = _group_space(mem, concepts, anchor_of)
    P = {}
    for c in concepts:
        y, v = mem.label(c), cfg["values"][c]
        nb = neighbours(state(mem.Z, mem.E, mem.measurements, mem.ids, c, v["w"]),
                        state(tgt.Z, tgt.E, tgt.measurements, tgt.ids, c, v["w"]), v["k"])
        P[c] = sigmoid(revise(logistic_logit(mem.Z[c], y, tgt.Z[c]), y[nb].mean(1), cfg["kappa"], v["beta"]))
    return P


def rhythm_decisions(mem, tgt, s):
    """Rhythm replacements: a sinus rhythm replaced by atrial flutter or a paced rhythm, atrial fibrillation replaced
    by atrial flutter."""
    cfg = s["rhythm"]
    tau = cfg["kappa"] / (1 + cfg["kappa"])
    P = _group_probabilities(mem, tgt, RHYTHM, lambda c: c, cfg)
    asserted = {c: (P[c] > tau) if c in cfg["enabled"] else np.zeros(len(tgt.ids), bool) for c in RHYTHM}
    out = {}
    for r, i in enumerate(tgt.ids):
        stated = tgt.stated[r]
        sinus = stated & SINUS
        add, replaced = [], []
        if sinus:
            for c in ("AFIB", "AFLT", "PACE"):
                if c not in stated and asserted[c][r]:
                    add.append(c)
            if add:
                replaced += sorted(sinus)
                add = add[:1] if len(add) > 1 and "PACE" not in add else add
        if "AFIB" in stated and P["AFIB"][r] < 1 - tau and asserted["AFLT"][r]:
            add.append("AFLT")
            replaced.append("AFIB")
        elif ({"AFIB", "AFLT"} & stated) and not sinus and asserted["SR"][r] and \
                all(P[c][r] < 1 - tau for c in ({"AFIB", "AFLT"} & stated)):
            add.append("SR")
            replaced += sorted({"AFIB", "AFLT"} & stated)
        if add or replaced:
            out[i] = {"add": add, "replaced": replaced, "p_present": {c: float(P[c][r]) for c in RHYTHM}}
    return out


def axis_decisions(mem, tgt, s):
    """Axis deviations: asserted when the report states none, or replacing the opposite deviation."""
    cfg = s["axis"]
    tau = cfg["kappa"] / (1 + cfg["kappa"])
    P = _group_probabilities(mem, tgt, AXIS, lambda c: "LAD", cfg)  # both read the measured frontal axis
    out = {}
    for r, i in enumerate(tgt.ids):
        stated = tgt.stated[r]
        add, replaced = [], []
        for c in AXIS:
            other = "RAD" if c == "LAD" else "LAD"
            if c in cfg["enabled"] and c not in stated and P[c][r] > tau:
                if other in stated:
                    if P[other][r] < 1 - tau:
                        add.append(c)
                        replaced.append(other)
                else:
                    add.append(c)
        if len(add) == 2:  # both deviations asserted: keep the more probable one
            keep = max(add, key=lambda c: P[c][r])
            add, replaced = [keep], [x for x in replaced if x != keep]
        if add:
            out[i] = {"add": add, "replaced": replaced, "p_present": {c: float(P[c][r]) for c in AXIS}}
    return out


def select_rhythm(mem, s):
    return _select_group(mem, RHYTHM, lambda c: c, _rhythm_applicable, s["rhythm"]["kappa"], s["grids"]["rhythm"])


def select_axis(mem, s):
    return _select_group(mem, AXIS, lambda c: "LAD", _axis_applicable, s["axis"]["kappa"], s["grids"]["axis"])


# ------------------------------------------------------------------------------------------------- revision set
def revision_set(mem, tgt, s):
    """Per record: the stated diagnoses, the additions, the replacements, and the calibrated probabilities."""
    findings = findings_decisions(mem, tgt, s)
    stt = stt_decisions(mem, tgt, s)
    groups = [rhythm_decisions(mem, tgt, s), axis_decisions(mem, tgt, s)]
    out = {}
    for r, i in enumerate(tgt.ids):
        d = findings[i]
        rec = {"stated": sorted(tgt.stated[r]), "add": [c for c in d["add"] if c != "STT"], "replaced_by": {},
               "p_present": dict(d["p_present"])}
        for g in groups:
            if i in g:
                rec["add"] += g[i]["add"]
                rec["p_present"].update(g[i]["p_present"])
                rec["replaced_by"].update({c: g[i]["add"][0] for c in g[i]["replaced"]})
        if stt[i]["add"]:  # the ST-T statement follows the rhythm and axis statements
            rec["add"].append("STT")
            rec["p_present"]["STT"] = stt[i]["p"]
        out[i] = rec
    return out
