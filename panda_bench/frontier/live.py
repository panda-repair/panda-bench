"""The live backend: a fixed paired sample through Panda's public API, to check the replay.

Panda's API runs at most two requests at a time per workspace and charges every answer, so
this backend answers a sample, not the whole benchmark.  The sample is fixed before any call:
the same seeded item ids for every setup, sent item by item, so no setup is cut short.  Rows
go to ``<live run>/answers/<benchmark>.jsonl`` under setup names ``live:<setup>`` in the
replay's row shape, so the same judge and scorer grade them, and ``compare`` pairs each live
answer with the replay's answer to the same item.

Each request carries an idempotency key, so a resent request is never charged twice; a lost
answer whose request may have run is recorded as lost and never sent again blindly.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import iter_jsonl


def _cents(panda: Optional[Dict[str, Any]]) -> int:
    """``panda.charge.minor_units`` (a decimal string in USD cents), else 0."""
    try:
        return int(((panda or {}).get("charge") or {}).get("minor_units") or 0)
    except (TypeError, ValueError):
        return 0

BASE_URL = os.environ.get("PANDA_BASE_URL", "https://app.panda.repair/v1")
KEY_FILE = os.environ.get("PANDA_API_KEY_FILE", "")  # or PANDA_API_KEY
CONCURRENCY = 2  # Panda's per-workspace limit (docs/panda-api.md)
# Replay setups with a live selector.  The custom trio and the self-merge control have none
# (a custom Panel needs a saved preset; production refuses a Panel with a repeated member).
SELECTORS = {
    "deepseek-v4.1-flash": "deepseek-v4.1-flash", "glm-5.3": "glm-5.3", "glm-5.3-flash": "glm-5.3-flash",
    "gemini-3.8-flash": "gemini-3.8-flash", "gemini-4-argon": "gemini-4-argon",
    "panel:panda": "panel:panda", "panel:panda-quality": "panel:panda-quality",
    "auto": "auto", "auto:quality": "auto:quality",
}
KEEP_PANDA_FIELDS = ("status", "mode", "route", "panel", "charge", "agents", "synthesis")


def sample_items(items, n: int, seed: str) -> list:
    """``n`` items chosen by a seeded hash of their ids, the same for every setup."""
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest())
    return ranked[:n]


def _request(body: Dict[str, Any], key: str, idempotency_key: str, timeout: float = 900.0):
    request = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Idempotency-Key": idempotency_key, "User-Agent": "panda-bench/frontier"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def answer_live(selector: str, messages, max_tokens: int, key: str, idempotency_key: str) -> Dict[str, Any]:
    body = {"model": selector, "max_tokens": max_tokens,
            "messages": [{"role": role, "content": text} for role, text in messages]}
    began = time.perf_counter()
    attempts: List[str] = []
    for attempt in range(8):
        try:
            payload = _request(body, key, idempotency_key)
            attempts.append("ok")
            break
        except urllib.error.HTTPError as error:
            raw = error.read()[:2000].decode("utf-8", "replace")
            try:
                detail = json.loads(raw)
            except ValueError:
                detail = {"raw": raw[:300]}
            err = detail.get("error") if isinstance(detail, dict) else None
            code = (err or {}).get("code") if isinstance(err, dict) else None
            attempts.append(f"http_{error.code}:{code}")
            if error.code == 429 and code in ("throttle_mixture_concurrency_exhausted", "throttle_workspace_rate_limited"):
                time.sleep(float(error.headers.get("Retry-After") or 1) + attempt)  # refused before running: safe
                continue
            panda = detail.get("panda") if isinstance(detail, dict) else None
            return {"status": "failed", "text": None, "attempts": attempts, "error_code": code,
                    "panda": {k: (panda or {}).get(k) for k in KEEP_PANDA_FIELDS if panda and k in panda},
                    "wall_seconds": round(time.perf_counter() - began, 3)}
        except Exception as error:  # the request may have run and been charged: never resent blindly
            attempts.append(type(error).__name__)
            return {"status": "lost", "text": None, "attempts": attempts,
                    "wall_seconds": round(time.perf_counter() - began, 3)}
    else:
        return {"status": "failed", "text": None, "attempts": attempts, "error_code": "concurrency_exhausted",
                "wall_seconds": round(time.perf_counter() - began, 3)}
    choice = (payload.get("choices") or [{}])[0]
    panda = payload.get("panda") or {}
    text = (choice.get("message") or {}).get("content")
    return {"status": panda.get("status") or "completed", "text": text if text else None,
            "finish": choice.get("finish_reason"), "model": payload.get("model"),
            "panda": {k: panda.get(k) for k in KEEP_PANDA_FIELDS if k in panda},
            "usage": payload.get("usage"), "attempts": attempts,
            "wall_seconds": round(time.perf_counter() - began, 3)}


def run_sample(live_dir: Path, benchmark: str, bench, items, setups: Sequence[str], n: int,
               seed: str = "panda-bench.frontier.live.v1", max_charge_usd: float = 4.0) -> Dict[str, int]:
    """Answer the fixed sample of ``n`` items with every live selector, item by item,
    resumably; stops scheduling once the recorded charges reach ``max_charge_usd``."""
    key = os.environ.get("PANDA_API_KEY") or (Path(KEY_FILE).expanduser().read_text().strip() if KEY_FILE else "")
    if not key:
        raise RuntimeError("set PANDA_API_KEY or PANDA_API_KEY_FILE")
    path = live_dir / "answers" / (benchmark + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(row["setup"], row["item"]) for row in iter_jsonl(path)}
    spent = [sum(_cents(row.get("panda")) for row in iter_jsonl(path)) / 100.0]
    chosen = sample_items(items, n, seed)
    tasks = [(setup, item) for item in chosen for setup in setups if ("live:" + setup, item.id) not in done]
    lock = threading.Lock()
    counts = {"sent": 0, "skipped_budget": 0}

    def work(task):
        setup, item = task
        with lock:
            if spent[0] >= max_charge_usd:
                counts["skipped_budget"] += 1
                return
        ikey = hashlib.sha256(f"{seed}|{benchmark}|{setup}|{item.id}".encode()).hexdigest()[:48]
        result = answer_live(SELECTORS[setup], item.messages, bench.max_output_tokens, key, ikey)
        charge = _cents(result.get("panda"))
        row = {"benchmark": benchmark, "setup": "live:" + setup, "item": item.id, "backend": "live",
               "idempotency_key": ikey, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "route": (result.get("panda") or {}).get("route"), "charge_cents": charge, **result}
        with lock:
            spent[0] += charge / 100.0
            counts["sent"] += 1
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(work, tasks))
    counts["charge_usd"] = round(spent[0], 2)
    return counts
