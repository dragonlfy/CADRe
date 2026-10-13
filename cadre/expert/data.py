"""Signals and statement labels for the expert model.

pool    every MIMIC-IV-ECG record of a patient outside the test set that is not in the ECG-R1 training corpus, so
        that the expert's outputs are out of sample on both the experience memory and the test set
memory  the memory records; test  the test records
Signals: raw WFDB digital values (200 adu/mV), native MIMIC lead order, 12 x 5000 samples (10 s at 500 Hz), cut or
zero-padded. Labels: multi-hot over the 150 most frequent pool statements.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from cadre import config
from cadre.data import _records, records
from cadre.labels import machine_statements

NATIVE = ("I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6")
TOP = 150


def _read(path):
    import wfdb
    rec = wfdb.rdrecord(str(config.path("mimic_ecg") / path), physical=False)
    names = list(rec.sig_name)
    sig = np.asarray(rec.d_signal, dtype=np.int64)
    if names != list(NATIVE):
        sig = sig[:, [names.index(n) for n in NATIVE]]
    sig[(sig < -32767) | (sig > 32767)] = 0
    out = np.zeros((12, 5000), dtype=np.int16)
    n = min(5000, sig.shape[0])
    out[:, :n] = sig[:n].T.astype(np.int16)
    return out


def _work(args):
    array_path, n_total, chunk = args
    mm = np.memmap(array_path, dtype=np.int16, mode="r+", shape=(n_total, 12, 5000))
    flags = []
    for idx, path in chunk:
        try:
            mm[idx] = _read(path)
            flags.append((idx, 1))
        except Exception:
            flags.append((idx, 0))
    mm.flush()
    return flags


def splits():
    rows = list(csv.DictReader(open(config.path("mimic_ecg") / "record_list.csv")))
    by_id = {r["study_id"]: r for r in rows}
    test_patients = {r["patient"].lstrip("p") for r in records("test")}
    corpus = {r["id"] for r in _records("train_corpus")}
    pool = [r for r in rows if r["subject_id"] not in test_patients and r["study_id"] not in corpus]
    return {"pool": pool, "memory": [by_id[r["id"]] for r in records("memory")],
            "test": [by_id[r["id"]] for r in records("test")]}


def convert(name, recs, procs=32):
    """Write the signals of `recs` to work/expert/<name>_signals.int16 and an index file."""
    path = config.work("expert", f"{name}_signals.int16")
    n = len(recs)
    np.memmap(path, dtype=np.int16, mode="w+", shape=(n, 12, 5000)).flush()
    items = [(i, r["path"]) for i, r in enumerate(recs)]
    chunks = [items[k:k + 500] for k in range(0, n, 500)]
    ok = np.zeros(n, dtype=np.uint8)
    tick = time.time()
    with Pool(procs) as pool:
        for flags in pool.imap_unordered(_work, [(str(path), n, c) for c in chunks]):
            for idx, good in flags:
                ok[idx] = good
    print(f"{name}: {n} records, readable {int(ok.sum())}, {n / (time.time() - tick):.0f} records/s", flush=True)
    with open(config.work("expert", f"{name}_index.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["row", "study_id", "subject_id", "ok"])
        for i, r in enumerate(recs):
            w.writerow([i, r["study_id"], r["subject_id"], int(ok[i])])


def statement_labels(sets):
    """The 150 most frequent pool statements, and multi-hot labels over them for every set."""
    ml = machine_statements()

    def clean(study):
        return {s.rstrip(".").strip() for s in ml.get(study, ()) if not s.startswith("---")}

    counts = Counter(s for r in sets["pool"] for s in clean(r["study_id"]))
    vocab = [s for s, _ in counts.most_common(TOP)]
    config.work("expert", "vocab.json").write_text(json.dumps(vocab, indent=1))
    col = {v: k for k, v in enumerate(vocab)}
    for name, recs in sets.items():
        Y = np.zeros((len(recs), len(vocab)), dtype=np.uint8)
        for i, r in enumerate(recs):
            for s in clean(r["study_id"]):
                if s in col:
                    Y[i, col[s]] = 1
        np.save(config.work("expert", f"{name}_labels.npy"), Y)
    return vocab
