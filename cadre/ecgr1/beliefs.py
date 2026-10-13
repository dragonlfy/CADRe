"""Diagnosis-level belief extraction.

For every diagnosis c the frozen model is asked "Does this ECG show <c>? Answer with one word: Present or Absent.",
with its reasoning closed and its answer opened ("<think>\n\n</think>\n\n<answer>"), and the belief is

    b_c = log p(Present | x, q_c) - log p(Absent | x, q_c)

for the next token. The image and the signal are encoded once per record and the 28 questions continue from the
cached prefix.
"""

from __future__ import annotations

import numpy as np

from cadre.vocabulary import QUERIED, QUERY_TEMPLATE, QUERY_WORDING

ANSWER_PREFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<answer>"


class BeliefExtractor:
    def __init__(self, model):
        self.m = model
        tok = model.tokenizer
        self.present = tok.encode("Present", add_special_tokens=False)[0]
        self.absent = tok.encode("Absent", add_special_tokens=False)[0]
        tail = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
        self.questions = [tok.encode("\n" + QUERY_TEMPLATE.format(QUERY_WORDING[c]), add_special_tokens=False) + tail
                          for c in QUERIED]
        self.marks = {tok.convert_tokens_to_ids(x) for x in ("<|vision_end|>", "<|ecg_end|>")}

    def __call__(self, record):
        """Beliefs of one record over QUERIED."""
        base = self.m.encode(record, record["prompt"])
        full = list(base["input_ids"])
        cut = max(j for j, x in enumerate(full) if x in self.marks) + 1  # the shared prefix ends after the ECG and image
        shared = {k: (v[:cut] if isinstance(v, list) and len(v) == len(full) else v) for k, v in base.items()}
        if "labels" in shared:
            shared["labels"] = [-100] * cut
        cache, first, mask = self.m.prefill(shared)
        nxt = self.m.continue_from(cache, first, mask, self.questions)
        return np.array([float(lp[self.present] - lp[self.absent]) for lp in nxt], dtype=np.float32)
