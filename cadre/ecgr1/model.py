"""Load the frozen ECG-R1-8B-RL with its official template and inputs (ECG image and raw 12-lead signal), score
continuations of a conversation, and generate greedily. No parameter of the model is updated.

The ECG encoder path is set through a symlinked copy of the checkpoint directory (only config.json is rewritten), so
the downloaded checkpoint itself is never modified.
"""

from __future__ import annotations

import copy
import json
import os
import sys

from cadre import config

ENV = {"IMAGE_MAX_TOKEN_NUM": "768", "ECG_SEQ_LENGTH": "5000", "ECG_PATCH_SIZE": "50", "ECG_PROJECTOR_TYPE": "mlp2x_gelu",
       "ECG_MODEL_CONFIG": "coca_ViT-B-32", "FREEZE_ECG_TOWER": "True", "FREEZE_ECG_PROJECTOR": "True"}


def model_overlay():
    """A copy of the checkpoint directory (symlinks) whose config.json points at the ECG encoder and enables the
    KV cache for multimodal generation."""
    out = config.work("ecgr1_overlay", "config.json").parent
    model = config.path("ecgr1_model")
    if not (out / "config.json").exists():
        for child in model.iterdir():
            if child.is_file() and child.name != "config.json" and not (out / child.name).exists():
                (out / child.name).symlink_to(child.resolve())
        cfg = json.loads((model / "config.json").read_text())
        cfg["ecg_tower_path"] = str(config.path("ecg_tower"))
        cfg["use_cache"] = True
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["use_cache"] = True
        (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return out


class ECGR1:
    def __init__(self, device="cuda:0"):
        os.environ.update({**ENV, "ECG_TOWER_PATH": str(config.path("ecg_tower"))})
        code = config.path("ecgr1_code")
        sys.path.insert(0, str(code))
        cwd = os.getcwd()
        os.chdir(code)
        import torch
        import ecg_r1.register  # noqa: F401  registers the ecg_r1 model type and template
        from swift.llm import get_model_tokenizer, get_template

        self.torch = torch
        self.model, self.processor = get_model_tokenizer(str(model_overlay()), torch.bfloat16, model_type="ecg_r1",
                                                         device_map=device)
        self.model.eval()
        self.template = get_template("ecg_r1", self.processor)
        self.template.set_mode("pt")
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.device = device
        os.chdir(cwd)

    def encode(self, record, prompt):
        """The official encoding of user(<ecg><image> prompt) up to the start of the assistant turn."""
        from swift.llm import InferRequest
        request = InferRequest(messages=[{"role": "user", "content": "<ecg><image>\n" + prompt}],
                               images=[record["image"]], objects={"ecg": [record["ecg"]]})
        return self.template.encode(request)

    def with_prefix(self, enc, prefix_ids):
        """A copy of an encoding with `prefix_ids` appended to the assistant turn."""
        ids = list(enc["input_ids"]) + list(prefix_ids)
        out = {k: v for k, v in enc.items() if not (isinstance(v, list) and len(v) == len(enc["input_ids"]))}
        out["input_ids"] = ids
        if "labels" in enc:
            out["labels"] = [-100] * len(ids)
        return out

    def _batch(self, encs):
        self.template.padding_side = "left"
        batch = self.template.data_collator(encs)
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        batch.pop("labels", None)
        return batch

    # ---- shared-prefix scoring: one prefill per record, then short continuations on the cached prefix ---------------
    def prefill(self, enc):
        batch = self._batch([enc])
        with self.torch.no_grad():
            out = self.model(**batch, use_cache=True, logits_to_keep=1)
        return out.past_key_values, self.torch.log_softmax(out.logits[0, -1].float(), -1), batch["attention_mask"]

    def continue_from(self, cache, first_lp, prefix_mask, conts, chunk=11):
        """For each continuation (a token list) appended to the cached prefix: the log-probabilities of the next token."""
        torch = self.torch
        results = []
        for start in range(0, len(conts), chunk):
            part = conts[start:start + chunk]
            n, L = len(part), max(len(c) for c in part)
            pad = self.tokenizer.pad_token_id or 0
            ids = torch.full((n, L), pad, dtype=torch.long, device=self.device)
            mask = torch.zeros((n, L), dtype=torch.long, device=self.device)
            for b, c in enumerate(part):
                ids[b, :len(c)] = torch.tensor(c, device=self.device)
                mask[b, :len(c)] = 1
            past = copy.deepcopy(cache)
            past.batch_repeat_interleave(n)
            full_mask = torch.cat([prefix_mask.expand(n, -1), mask], 1)
            P = prefix_mask.shape[1]
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=full_mask, past_key_values=past, use_cache=True,
                                 cache_position=torch.arange(P, P + L, device=self.device))
            for b, c in enumerate(part):
                results.append(torch.log_softmax(out.logits[b, len(c) - 1].float(), -1).cpu())
            del past, out
        return results

    # ---- greedy generation: temperature 0, repetition penalty 1.0 -------------------------------------------------
    def generate(self, encs, max_new_tokens=2048):
        batch = self._batch(encs)
        n_in = batch["input_ids"].shape[1]
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                                      eos_token_id=im_end, pad_token_id=self.tokenizer.pad_token_id or im_end)
        texts = []
        for row in out[:, n_in:].tolist():
            if im_end in row:
                row = row[:row.index(im_end)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=False).replace("<|endoftext|>", ""))
        return texts
