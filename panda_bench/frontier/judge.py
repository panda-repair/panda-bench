"""Judge calls through Anthropic's Message Batches API, resumably and within a spend cap.

Every benchmark that needs a judge returns ``JudgeRequest`` objects per answer; this module
sends each once to the stand-in judge (Claude Sonnet 5, since the official judges are
retired or need keys this project does not hold; each benchmark's doc says which) and keeps
every reply in ``<run>/judge/<benchmark>.jsonl``.

A request's id binds the judge settings, the benchmark, setup, item and request name, and a
digest of the exact answer judged, so a replaced answer is never graded by an older reply.
A request already submitted is not sent again while its batch is open; once its batch has
ended without a usable reply, ``retry_failed`` may send it again (at most ``MAX_ATTEMPTS``
times in all).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .core import iter_jsonl, sha256_text

JUDGE_MODEL = os.environ.get("PANDA_BENCH_JUDGE", "claude-sonnet-5")
# The judge's settings, part of every request id, so changing them re-judges everything.
# Thinking is off, as the official graders (GPT-4o, GPT-4.1, GPT-5.4 low) answer directly; with
# Sonnet 5's default, thinking sometimes used a request's whole output bound before any text.
JUDGE_SETTINGS = "sonnet5-thinking-disabled-v1"
# Batch prices in USD per million tokens (half the list price of $2 / $10), 2026-11-18.
BATCH_PRICE = {"claude-sonnet-5": (1.0, 5.0)}
KEY_FILE = os.environ.get("PANDA_BENCH_JUDGE_KEY_FILE", "")  # or ANTHROPIC_API_KEY
CHUNK = 2_000  # requests per batch: keeps a batch far under the API's size limit
EXPECTED_OUTPUT_TOKENS = 300  # for the preflight estimate; spend is tracked from real usage
MAX_ATTEMPTS = 3


def custom_id(benchmark: str, setup: str, item: str, name: str, answer_sha256: str) -> str:
    key = json.dumps([JUDGE_SETTINGS, benchmark, setup, item, name, answer_sha256], separators=(",", ":"))
    return "j" + hashlib.sha256(key.encode()).hexdigest()[:40]


def read_judged(path: Path) -> Dict[str, Dict[str, Any]]:
    """The newest reply per request id."""
    return {row["custom_id"]: row for row in iter_jsonl(path)}


def _attempts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in iter_jsonl(path):
        counts[row["custom_id"]] = counts.get(row["custom_id"], 0) + 1
    return counts


def _open_requests(run_dir: Path) -> Set[str]:
    """Ids submitted in a batch that has not been collected yet."""
    judge_dir = run_dir / "judge"
    collected_path = judge_dir / "collected.json"
    collected = set(json.loads(collected_path.read_text())) if collected_path.is_file() else set()
    open_ids: Set[str] = set()
    for record in iter_jsonl(judge_dir / "batches.jsonl"):
        if record["batch"] not in collected:
            open_ids.update(r["custom_id"] for r in record["requests"])
    return open_ids


def judged_cost_usd(rows: Sequence[Dict[str, Any]]) -> float:
    price_in, price_out = BATCH_PRICE[JUDGE_MODEL]
    total = 0.0
    for row in rows:
        usage = row.get("usage") or {}
        total += (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)) * price_in / 1e6
        total += usage.get("output_tokens", 0) * price_out / 1e6
    return total


def replies_for(judged: Dict[str, Dict[str, Any]], benchmark: str, setup: str, item, text: str, bench):
    """``{request name: reply text or None}`` for one answer, or None while any is unjudged."""
    digest = sha256_text(text)
    replies = {}
    for request in bench.judge_requests(item, text):
        row = judged.get(custom_id(benchmark, setup, item.id, request.name, digest))
        if row is None:
            return None
        replies[request.name] = row.get("text")
    return replies


def pending_requests(run_dir: Path, benchmark: str, bench, items_by_id, answers, retry_failed: bool,
                     failed_items: Optional[Set[Tuple[str, str]]] = None):
    """The judge requests still to send: (custom id, row metadata, request).

    A request is sent when it has no reply and is not in an open batch.  With
    ``retry_failed``, a request whose reply is missing or unusable is sent again, up to
    ``MAX_ATTEMPTS`` in all; ``failed_items`` names (setup, item) pairs whose grade found an
    unusable reply, so every request of those answers is sent again."""
    path = run_dir / "judge" / (benchmark + ".jsonl")
    judged = read_judged(path)
    attempts = _attempts(path)
    open_ids = _open_requests(run_dir)
    out = []
    for (setup, item_id), row in answers.items():
        text = row.get("text")
        if text is None or item_id not in items_by_id:
            continue
        digest = sha256_text(text)
        redo_item = retry_failed and failed_items is not None and (setup, item_id) in failed_items
        for request in bench.judge_requests(items_by_id[item_id], text):
            cid = custom_id(benchmark, setup, item_id, request.name, digest)
            if cid in open_ids:
                continue
            seen = judged.get(cid)
            if seen is not None:
                usable = seen.get("text") is not None and not redo_item
                if usable or not retry_failed or attempts.get(cid, 0) >= MAX_ATTEMPTS:
                    continue
            out.append((cid, {"benchmark": benchmark, "setup": setup, "item": item_id, "name": request.name,
                              "answer_sha256": digest}, request))
    return out


def estimate_usd(requests) -> Tuple[float, float]:
    """(expected, worst case) in USD: input from characters (3 per token, conservative),
    output at the expected length or at every request's full bound."""
    price_in, price_out = BATCH_PRICE[JUDGE_MODEL]
    tokens_in = sum((len(r.user) + len(r.system or "")) / 3.0 for _, _, r in requests)
    expected = tokens_in * price_in / 1e6 + len(requests) * EXPECTED_OUTPUT_TOKENS * price_out / 1e6
    worst = tokens_in * price_in / 1e6 + sum(r.max_tokens for _, _, r in requests) * price_out / 1e6
    return expected, worst


def _client():
    import anthropic

    # No SDK retries on create: a retried create after an ambiguous failure could buy a batch
    # twice.  A failed create stops the run for a person to reconcile.
    key = os.environ.get("ANTHROPIC_API_KEY") or (Path(KEY_FILE).expanduser().read_text().strip() if KEY_FILE else "")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY or PANDA_BENCH_JUDGE_KEY_FILE")
    return anthropic.Anthropic(api_key=key, max_retries=0)


def submit(run_dir: Path, benchmark: str, requests, max_new_usd: float) -> List[str]:
    """Send ``requests`` as batches; returns the batch ids, recorded in
    ``<run>/judge/batches.jsonl`` with their request metadata so collection can resume."""
    expected, worst = estimate_usd(requests)
    if expected > max_new_usd:
        raise SystemExit(f"refused: {len(requests)} judge requests expected ${expected:.2f} "
                         f"(worst ${worst:.2f}) > cap ${max_new_usd:.2f}")
    client = _client()
    judge_dir = run_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for start in range(0, len(requests), CHUNK):
        chunk = requests[start:start + CHUNK]
        body = []
        for cid, _, request in chunk:
            # No temperature: Claude Sonnet 5 refuses the parameter ("deprecated for this
            # model"), so the judge samples at its default.
            params: Dict[str, Any] = {"model": JUDGE_MODEL, "max_tokens": request.max_tokens,
                                      "thinking": {"type": "disabled"},
                                      "messages": [{"role": "user", "content": request.user}]}
            if request.system:
                params["system"] = request.system
            body.append({"custom_id": cid, "params": params})
        batch = client.messages.batches.create(requests=body)
        ids.append(batch.id)
        with (judge_dir / "batches.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"batch": batch.id, "benchmark": benchmark, "count": len(chunk),
                                     "judge": JUDGE_MODEL, "judge_settings": JUDGE_SETTINGS,
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "requests": [{"custom_id": cid, **meta} for cid, meta, _ in chunk]}) + "\n")
    return ids


def collect(run_dir: Path, wait: bool = True, poll_seconds: float = 60.0) -> Dict[str, int]:
    """Collect every recorded batch that has ended; with ``wait``, wait for the rest.  Only
    the batch's own request ids are kept, each once; a request the batch never answered is
    left to ``retry_failed``."""
    judge_dir = run_dir / "judge"
    batches = list(iter_jsonl(judge_dir / "batches.jsonl"))
    if not batches:
        return {}
    client = _client()
    collected_path = judge_dir / "collected.json"
    collected = set(json.loads(collected_path.read_text())) if collected_path.is_file() else set()
    counts: Dict[str, int] = {}
    for record in batches:
        if record["batch"] in collected:
            continue
        while True:
            batch = client.messages.batches.retrieve(record["batch"])
            if batch.processing_status == "ended" or not wait:
                break
            time.sleep(poll_seconds)
        if batch.processing_status != "ended":
            continue
        meta = {r["custom_id"]: r for r in record["requests"]}
        rows: Dict[str, Dict[str, Any]] = {}
        for result in client.messages.batches.results(record["batch"]):
            if result.custom_id not in meta or result.custom_id in rows:
                continue
            row: Dict[str, Any] = {**meta[result.custom_id], "batch": record["batch"],
                                   "judge": record.get("judge", JUDGE_MODEL),
                                   "judge_settings": record.get("judge_settings", JUDGE_SETTINGS),
                                   "result": result.result.type, "text": None, "usage": None}
            if result.result.type == "succeeded":
                message = result.result.message
                row["text"] = "".join(block.text for block in message.content if block.type == "text") or None
                row["stop_reason"] = message.stop_reason
                usage = message.usage
                row["usage"] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0}
            else:
                error = getattr(result.result, "error", None)
                row["error"] = getattr(getattr(error, "error", None), "type", None) or str(result.result.type)
            rows[result.custom_id] = row
        # Written whole, then the batch is marked collected: a crash in between re-collects it,
        # and readers keep the newest row per id.
        with (judge_dir / (record["benchmark"] + ".jsonl")).open("a", encoding="utf-8") as out:
            for row in rows.values():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[record["benchmark"]] = counts.get(record["benchmark"], 0) + len(rows)
        if len(rows) != len(meta):
            counts[record["benchmark"] + ":unanswered"] = len(meta) - len(rows)
        collected.add(record["batch"])
        collected_path.write_text(json.dumps(sorted(collected)))
    return counts
