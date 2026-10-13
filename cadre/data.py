"""Records of the two splits CADRe uses.

memory  the ECG-R1 training corpus (ECG-Protocol-Guided-Grounding-CoT-30k), from which the experience memory is built.
        A 4% patient-level slice (CRC32 of the patient id modulo 25 equal to 1) is held out and not used.
test    the ECG-Grounding MIMIC-IV test set (2,381 records).

Each record carries its official ECG-Grounding image, the raw 12-lead MIMIC-IV-ECG WFDB record, and the
interpretation prompt.
"""

from __future__ import annotations

import json
import zlib
from functools import lru_cache

from cadre import config


def patient_slice(patient: str, modulus: int = 25) -> int:
    return zlib.crc32(patient.encode()) % modulus


def _records(source_key: str) -> list[dict]:
    images, mimic = config.path("images"), config.path("mimic_ecg")
    rows = []
    for line in config.path(source_key).read_text().splitlines():
        r = json.loads(line)
        roles = {m["role"]: m["content"] for m in r["messages"]}
        ecg = r["objects"]["ecg"][0]
        rows.append({"id": str(r["id"]), "patient": ecg.split("/")[3],
                     "image": str(images / r["images"][0].replace("gen_images/", "", 1)),
                     "ecg": str(mimic / ecg.replace("mimic-iv/", "", 1)),
                     "prompt": roles["user"].replace("<ecg><image>\n", "", 1)})
    return rows


@lru_cache(maxsize=None)
def records(split: str) -> tuple[dict, ...]:
    """split: "memory" or "test"."""
    if split == "test":
        return tuple(_records("test_set"))
    s = config.settings()["memory"]
    rows = _records("train_corpus")
    return tuple(r for r in rows if patient_slice(r["patient"], s["slice_modulus"]) not in s["excluded_slices"])


def by_id(split: str) -> dict[str, dict]:
    return {r["id"]: r for r in records(split)}
