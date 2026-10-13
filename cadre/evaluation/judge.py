"""LLM judge with the ECG-R1 rubric, through any OpenAI-compatible chat-completions endpoint.

Requests use temperature 0, the JSON response format, no thinking, and at most 3,000 output tokens. Responses are
cached by a hash of the request, so a report receives the same scores in every system that contains it. A reply that
cannot be parsed is requested once more with a format-only follow-up message.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from cadre.evaluation.rubric import (ENUMERATION_SYSTEM, STRICT_JSON_RETRY_SYSTEM, TABLE1_PROMPT, _extract_json, _JSON_OBJECT,
                                     _normalize, table1_metrics)

RETRY_MESSAGE = ("The preceding response was not valid JSON. Re-evaluate the same report under the unchanged rubric and "
                 "return exactly one valid JSON object, with no commentary.")


class Client:
    """Cached, retrying OpenAI-compatible chat-completions client. The API key is read from an environment variable."""

    def __init__(self, base_url, model, api_key_env, cache_dir, timeout=600, retries=4):
        self.base_url, self.model, self.api_key_env = base_url.rstrip("/"), model, api_key_env
        self.cache_dir, self.timeout, self.retries = Path(cache_dir), timeout, retries

    def chat(self, messages, **parameters):
        key = hashlib.sha256(json.dumps({"model": self.model, "messages": messages, "parameters": parameters},
                                        sort_keys=True).encode()).hexdigest()
        cached = self.cache_dir / self.model / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        error = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers, timeout=self.timeout,
                                  data=json.dumps({"model": self.model, "messages": messages, **parameters}))
                r.raise_for_status()
                payload = r.json()
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(payload, indent=2, sort_keys=True))
                return payload
            except (requests.RequestException, json.JSONDecodeError) as e:
                error = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"endpoint {self.base_url} unavailable: {error}")


def _content(response):
    choice = response["choices"][0]
    message = choice.get("message") or {}
    raw = str(message.get("content") or "")
    if not _JSON_OBJECT.search(raw):  # some servers return a reasoning field before the answer
        for field in ("reasoning_content", "reasoning"):
            extra = str(message.get(field) or "")
            if _JSON_OBJECT.search(extra):
                return extra
    return raw


def score(client, generated, reference, max_tokens=3000):
    """Judge one report against its reference. Returns {"criteria", "metrics"} or {"failure"}."""
    messages = [{"role": "system", "content": ENUMERATION_SYSTEM + STRICT_JSON_RETRY_SYSTEM},
                {"role": "user", "content": TABLE1_PROMPT.format(generated=generated, groundtruth=reference)}]
    params = dict(temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"},
                  chat_template_kwargs={"enable_thinking": False})
    try:
        try:
            criteria, _ = _normalize(_extract_json(_content(client.chat(messages, **params))))
        except ValueError:
            retry = messages + [{"role": "user", "content": RETRY_MESSAGE}]
            criteria, _ = _normalize(_extract_json(_content(client.chat(retry, **params))))
        return {"criteria": criteria, "metrics": table1_metrics(criteria)}
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        return {"failure": f"{type(error).__name__}: {error}"}


def reference_text(ref):
    """The reference given to the judge: the clinical interpretation of the ECG-Grounding reference."""
    return ref["reference_interpretation"] or ref["reference_report"]


def judge_reports(reports: dict[str, str], references: dict[str, dict], client: Client, out_path: Path, workers=16):
    """Score every report; writes one JSON line per record to out_path and returns {id: metrics}."""
    ids = sorted(reports)

    def one(i):
        return i, score(client, reports[i], reference_text(references[i]))

    results = {}
    with ThreadPoolExecutor(workers) as pool, open(out_path, "w") as handle:
        for i, r in pool.map(one, ids):
            handle.write(json.dumps({"id": i, **r}) + "\n")
            if "metrics" in r:
                results[i] = r["metrics"]
    return results
