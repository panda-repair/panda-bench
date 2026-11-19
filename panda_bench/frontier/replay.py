"""The replay backend: Panda's own product code, calling the providers directly.

Panda's API runs at most two requests at a time per workspace, which would stretch a
benchmark of thousands of Panel and router runs over days.  The replay backend runs the
same product logic without that admission step: each call's provider body is built by
Panda's own ``product_text_inputs`` from its review-core menu, each answer is classified by
Panda's own parser, Panels use Panda's own draft order, combiner conversation, output bound
and fallback rules, and ``auto`` uses Panda's own router, profile and route table.  Only
the transport differs: Workers AI over its REST API and Gemini over Vertex AI's global
endpoint, instead of Control's bindings.  A sample of live Panda API runs checks that the
two agree (``live.py``).

This module needs a checkout of Panda's (private) platform source, named by
``PANDA_ZERO_SRC`` (its ``apps/platform/src``) and ``PANDA_CONTRACTS_SRC``; the public
benchmark modules, graders and releases do not.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _path in (os.environ.get("PANDA_ZERO_SRC"), os.environ.get("PANDA_CONTRACTS_SRC")):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

from panda.execution.adapters.hop_task_embedder import load_task_profile, parse_embedding  # noqa: E402
from panda.execution.application.auto_router import AutoRouter  # noqa: E402
from panda.execution.domain.auto_router import routing_text  # noqa: E402
from panda.execution.domain.conversation import TextConversation, TextMessage  # noqa: E402
from panda.execution.application.product_runs import STAND_IN_CODES  # noqa: E402
from panda.execution.domain.mixtures import (  # noqa: E402
    CALL_DEADLINE_SECONDS,
    MAX_OUTPUT_TOKENS,
    SLOW_CALL_DEADLINE_SECONDS,
    CallUsage,
    conversation_request_bytes,
    panel_draft_order,
    panel_synthesis_budget_bytes,
    panel_synthesis_conversation,
    panel_synthesis_output_tokens,
    panel_synthesis_temperature,
)
from panda.reviews.adapters.mixture_models import ReviewCoreMixtureCatalog  # noqa: E402
from panda.reviews.adapters.review_model import (  # noqa: E402
    ProductTextPrompt,
    parse_product_text_body,
    product_text_inputs,
)
from panda.reviews.application.review_cores import CORES_BY_MODEL_ID, PANDA_PANELS  # noqa: E402

CATALOG = ReviewCoreMixtureCatalog()
# Provider settings come from the environment (docs/frontier/README.md), never from this file.
ACCOUNT = os.environ.get("PANDA_CF_ACCOUNT", "")
GCP_PROJECT = os.environ.get("PANDA_GCP_PROJECT", "")
CF_TOKEN_FILE = os.environ.get("PANDA_CF_TOKEN_FILE", "")
# A provider error worth sending again: rate limits and capacity (Workers AI 3021 and
# 3040 arrive as 429), timeouts and server faults.  Our own load causes most of them, so
# they say nothing about the product; each first attempt's outcome is still recorded.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}
MAX_ATTEMPTS = 4
# A call's verdict in the product's own codes (``classify`` in execution/domain/mixtures.py):
# "completed", or a failure code.  Only a combiner that fails with one of the product's
# ``STAND_IN_CODES`` is replaced by its strongest member's answer (ADR 0069 OD10); a served
# refusal ("answer_refused") or an unknown outcome ("outcome_unknown") fails the Panel, as in
# production.  Our transport's last attempt stands in for Control's: a definite refusal (an HTTP
# 4xx, a 429 included) was not served ("model_refused"); a timeout, dropped connection or 5xx
# may have been ("outcome_unknown").
_ANSWER_VERDICTS = {"text": "completed", "refused": "answer_refused", "incomplete": "answer_incomplete",
                    "invalid": "answer_invalid"}


def _transport_verdict(attempts: List[str]) -> str:
    last = attempts[-1] if attempts else "transport"
    if last.startswith("http_4"):
        return "model_refused"
    return "outcome_unknown"


@dataclass
class CallRecord:
    """One model call as the release keeps it: no prompt or answer text."""

    model: str
    role: str  # "single", "member", "synthesis" or "router_embedding"
    outcome: str  # "text", "refused", "invalid", "incomplete", "error"
    reason: str
    finish: Optional[str]
    verdict: str = "outcome_unknown"  # the product's call verdict code, or "completed"
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    list_cost_nanos: int = 0
    seconds: float = 0.0  # the last attempt's wall time
    wall_seconds: float = 0.0  # every attempt and back-off
    attempts: List[str] = field(default_factory=list)


@dataclass
class Answer:
    """A setup's answer to one item."""

    text: Optional[str]
    status: str  # "completed", "partial", "single_answer", "failed"
    cut: bool
    calls: List[CallRecord]
    route: Optional[Dict[str, Any]] = None
    panel: Optional[Dict[str, Any]] = None
    wall_seconds: float = 0.0

    def list_cost_nanos(self) -> int:
        return sum(call.list_cost_nanos for call in self.calls)

    def to_json(self) -> Dict[str, Any]:
        value = asdict(self)
        value["list_cost_nanos"] = self.list_cost_nanos()
        return value


class RateLimiter:
    """At most ``rpm`` call starts per rolling minute for one model, shared by every thread,
    and a pause of the whole model after a rate-limit refusal."""

    def __init__(self, rpm: float) -> None:
        self.rpm = rpm
        self._starts: deque = deque()
        self._paused_until = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= 60.0:
                    self._starts.popleft()
                wait = max(0.0, self._paused_until - now)
                if not wait and len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                if not wait:
                    wait = 60.0 - (now - self._starts[0]) + 0.05
            time.sleep(min(wait, 5.0) + random.random() * 0.2)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)


class Providers:
    """Product text calls over the providers' own APIs, rate limited per model."""

    def __init__(self, rpm: Dict[str, float], default_rpm: float = 30.0) -> None:
        self._limiters: Dict[str, RateLimiter] = {}
        self._rpm = dict(rpm)
        self._default_rpm = default_rpm
        self._lock = threading.Lock()
        missing = [name for name, value in (("PANDA_CF_ACCOUNT", ACCOUNT), ("PANDA_GCP_PROJECT", GCP_PROJECT),
                                            ("PANDA_CF_TOKEN_FILE", CF_TOKEN_FILE)) if not value]
        if missing:
            raise RuntimeError("set " + ", ".join(missing) + " (see docs/frontier/README.md)")
        self._cf_token = Path(CF_TOKEN_FILE).expanduser().read_text().strip()
        self._gcp = {"token": None, "at": 0.0}

    def limiter(self, model: str) -> RateLimiter:
        with self._lock:
            if model not in self._limiters:
                self._limiters[model] = RateLimiter(self._rpm.get(model, self._default_rpm))
            return self._limiters[model]

    def _gcp_token(self, refresh: bool = False) -> str:
        with self._lock:
            if refresh or self._gcp["token"] is None or time.time() - self._gcp["at"] > 300:
                self._gcp["token"] = subprocess.run(
                    ["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True
                ).stdout.strip()
                self._gcp["at"] = time.time()
            return self._gcp["token"]

    def _endpoint(self, model: str) -> Tuple[str, Dict[str, str], str]:
        if model.startswith("@cf/"):  # every Workers AI model, the router's embedding model too
            url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}"
            return url, {"Authorization": "Bearer " + self._cf_token, "Content-Type": "application/json"}, "cf"
        name = model.split("/", 1)[1]
        url = (f"https://aiplatform.googleapis.com/v1/projects/{GCP_PROJECT}/locations/global"
               f"/publishers/google/models/{name}:generateContent")
        headers = {"Authorization": "Bearer " + self._gcp_token(), "Content-Type": "application/json",
                   "x-goog-user-project": GCP_PROJECT}
        return url, headers, "vertex"

    def post(self, model: str, body: Dict[str, Any], timeout: float) -> Tuple[Optional[Dict[str, Any]], List[str], float, float]:
        """The provider's JSON answer (unwrapped), each attempt's outcome, the last
        attempt's seconds and the total wall seconds; None after the last attempt fails."""
        attempts: List[str] = []
        began = time.perf_counter()
        last = 0.0
        data = json.dumps(body).encode()
        for attempt in range(MAX_ATTEMPTS):
            self.limiter(model).acquire()
            url, headers, kind = self._endpoint(model)
            start = time.perf_counter()
            try:
                request = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)
                last = time.perf_counter() - start
                attempts.append("ok")
                if kind == "cf":
                    payload = payload.get("result", payload) if isinstance(payload, dict) else payload
                return payload, attempts, last, time.perf_counter() - began
            except urllib.error.HTTPError as error:
                last = time.perf_counter() - start
                detail = error.read()[:300].decode("utf-8", "replace")
                code = error.code
                tag = f"http_{code}"
                if kind == "cf" and code == 429:
                    tag += ":3040" if "3040" in detail else ":3021" if "3021" in detail else ""
                attempts.append(tag)
                if code == 401 and kind == "vertex":
                    self._gcp_token(refresh=True)
                    continue
                if code in RETRY_STATUS:
                    # A per-minute limit refusal (Workers AI 3021, or a Vertex quota 429) pauses
                    # the whole model, since every call would be refused.  A capacity refusal
                    # (3040) is Cloudflare-wide and passes in seconds: only this call waits.
                    if code == 429 and not tag.endswith(":3040"):
                        self.limiter(model).pause(30.0 * (attempt + 1))
                    time.sleep(5.0 * (attempt + 1) + random.random())
                    continue
                return None, attempts, last, time.perf_counter() - began
            except Exception as error:  # timeouts and dropped connections
                last = time.perf_counter() - start
                attempts.append("timeout" if "timed out" in str(error).lower() else "transport")
                time.sleep(5.0 * (attempt + 1) + random.random())
        return None, attempts, last, time.perf_counter() - began

    def product_call(self, model: str, conversation: TextConversation, max_output_tokens: int,
                     role: str, temperature: Optional[float] = None) -> Tuple[Optional[str], Optional[str], CallRecord]:
        """One product text call: (shown text, finish reason, record)."""
        output = min(max_output_tokens, CATALOG.output_bound(model))
        slow = CATALOG.slow(model) or output > MAX_OUTPUT_TOKENS
        deadline = SLOW_CALL_DEADLINE_SECONDS if slow else CALL_DEADLINE_SECONDS
        prompt = ProductTextPrompt(
            model_id=model,
            messages=tuple((m.role, m.text) for m in conversation.messages),
            max_output_tokens=output,
            deadline_seconds=deadline,
            temperature=temperature,
        )
        body = product_text_inputs(prompt)
        payload, attempts, seconds, wall = self.post(model, body, deadline)
        record = CallRecord(model=model, role=role, outcome="error", reason="transport", finish=None,
                            seconds=round(seconds, 3), wall_seconds=round(wall, 3), attempts=attempts,
                            verdict=_transport_verdict(attempts))
        if payload is None or not isinstance(payload, dict):
            return None, None, record
        core = CORES_BY_MODEL_ID[model]
        answer = parse_product_text_body(payload, model_id=model, request_format=core.request_format)
        record.outcome, record.reason, record.finish = answer.outcome, answer.reason, answer.finish_reason
        usage = answer.usage
        if usage is None:
            # The product settles no answer without usage: its outcome is unknown.
            record.verdict = "outcome_unknown"
            return None, None, record
        record.verdict = _ANSWER_VERDICTS.get(answer.outcome, "answer_invalid")
        if usage is not None:
            record.input_tokens = usage.input_tokens
            record.cached_input_tokens = usage.cached_input_tokens
            record.output_tokens = usage.output_tokens
            try:
                record.list_cost_nanos = CATALOG.list_cost_nanos(model, CallUsage(
                    usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens, usage.output_tokens))
            except Exception:
                record.list_cost_nanos = 0
        return (answer.text if record.verdict == "completed" else None), answer.finish_reason, record

    def embed(self, model: str, text: str, dimensions: int) -> Tuple[Optional[Tuple[float, ...]], CallRecord]:
        payload, attempts, seconds, wall = self.post(model, {"input": [text]}, 30.0)
        record = CallRecord(model=model, role="router_embedding", outcome="error", reason="transport", finish=None,
                            seconds=round(seconds, 3), wall_seconds=round(wall, 3), attempts=attempts)
        if payload is None:
            return None, record
        vector = parse_embedding(json.dumps(payload).encode(), dimensions)
        record.outcome, record.reason = ("text", "embedding") if vector is not None else ("invalid", "embedding_shape")
        return vector, record


def conversation_of(messages: Sequence[Tuple[str, str]]) -> TextConversation:
    return TextConversation(tuple(TextMessage(role, text) for role, text in messages))


def resolve(alias: str) -> str:
    model = CATALOG.resolve(alias)
    if model is None:
        raise ValueError("unknown model: " + alias)
    return model


# -- setups -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Setup:
    """One way of answering: a model, a Panel (members and combiner) or an ``auto`` mode."""

    name: str
    kind: str  # "single", "panel" or "auto"
    models: Tuple[str, ...] = ()
    synthesizer: Optional[str] = None
    selector: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.models:
            value["models"] = [CATALOG.alias(m) for m in self.models]
        if self.synthesizer:
            value["synthesizer"] = CATALOG.alias(self.synthesizer)
        if self.selector:
            value["selector"] = self.selector
        return value


def single(alias: str) -> Setup:
    return Setup(alias, "single", (resolve(alias),))


def panel(name: str, members: Sequence[str], synthesizer: Optional[str] = None) -> Setup:
    models = tuple(resolve(m) for m in members)
    combiner = resolve(synthesizer) if synthesizer else min(models, key=CATALOG.referee_rank)
    return Setup(name, "panel", models, combiner)


def panda_panel(name: str) -> Setup:
    spec = next(p for p in PANDA_PANELS if p.name == name)
    return panel("panel:" + name, spec.members, spec.combiner)


def auto(selector: str) -> Setup:
    return Setup(selector, "auto", selector=selector)


class Runner:
    """Answers items with setups, through ``Providers``."""

    def __init__(self, providers: Providers, router_deadline: float = 5.0) -> None:
        self.providers = providers
        self.profile = load_task_profile()
        self.router_deadline = router_deadline
        self.allowed = set(CATALOG.models()) | {"panel:" + p.name for p in PANDA_PANELS}

    def answer(self, setup: Setup, messages: Sequence[Tuple[str, str]], max_output_tokens: int, run_id: str) -> Answer:
        began = time.perf_counter()
        conversation = conversation_of(messages)
        if setup.kind == "single":
            result = self._single(setup.models[0], conversation, max_output_tokens)
        elif setup.kind == "panel":
            result = self._panel(setup, conversation, max_output_tokens, run_id)
        else:
            result = self._auto(setup, conversation, max_output_tokens, run_id)
        result.wall_seconds = round(time.perf_counter() - began, 3)
        return result

    def _single(self, model: str, conversation: TextConversation, max_output_tokens: int, role: str = "single") -> Answer:
        text, finish, record = self.providers.product_call(model, conversation, max_output_tokens, role)
        if text is None:
            return Answer(None, "failed", False, [record])
        return Answer(text, "completed", finish == "length", [record])

    def _panel(self, setup: Setup, conversation: TextConversation, max_output_tokens: int, run_id: str) -> Answer:
        members = setup.models
        with ThreadPoolExecutor(max_workers=len(members)) as pool:
            futures = [pool.submit(self.providers.product_call, m, conversation, max_output_tokens, "member")
                       for m in members]
            results = [f.result() for f in futures]
        calls = [record for _, _, record in results]
        answers = tuple((index, text, finish == "length")
                        for index, (text, finish, record) in enumerate(results, start=1)
                        if text is not None and record.verdict == "completed")
        info: Dict[str, Any] = {"members": [CATALOG.alias(m) for m in members],
                                "synthesizer": CATALOG.alias(setup.synthesizer),
                                "answered": [index for index, _, _ in answers]}
        if not answers:
            info["mode"] = "no_answers"
            return Answer(None, "failed", False, calls, panel=info)
        if len(answers) == 1:
            # One answer has nothing to combine: it is the Panel's (OD8), and the run is partial.
            index, text, cut = answers[0]
            info["mode"] = "single_answer"
            return Answer(text, "partial", cut, calls, panel=info)
        everyone = len(answers) == len(members)
        order = panel_draft_order(run_id, tuple(index for index, _, _ in answers))
        drafts = sorted(answers, key=lambda a: order.index(a[0]))
        info["draft_order"] = [index for index, _, _ in drafts]
        bound = CATALOG.output_bound(setup.synthesizer)
        reply = min(max_output_tokens, bound)
        combiner_conversation = panel_synthesis_conversation(
            conversation, tuple((place, text, cut) for place, (_, text, cut) in enumerate(drafts, start=1)), reply)
        budget = panel_synthesis_budget_bytes(
            conversation, tuple((index, min(max_output_tokens, CATALOG.output_bound(members[index - 1])))
                                for index in range(1, len(members) + 1)),
            CATALOG.request_bound_bytes(), reply)
        if conversation_request_bytes(combiner_conversation) > budget:
            verdict = "input_too_large"  # never sent, as in production
            info["synthesis_verdict"] = verdict
        else:
            output = panel_synthesis_output_tokens(reply, bound, CATALOG.thinks(setup.synthesizer))
            text, finish, record = self.providers.product_call(
                setup.synthesizer, combiner_conversation, output, "synthesis", panel_synthesis_temperature(None))
            calls.append(record)
            verdict = record.verdict
            info["synthesis_verdict"] = verdict
            if text is not None:
                info["mode"] = "synthesis"
                return Answer(text, "completed" if everyone else "partial", finish == "length", calls, panel=info)
        if verdict in STAND_IN_CODES:
            # ADR 0069 OD10: the strongest member that answered stands in, a complete answer
            # before a cut one, then the referee ranking.
            index, text, cut = min(answers, key=lambda a: (a[2], CATALOG.referee_rank(members[a[0] - 1])))
            info["mode"] = "stand_in"
            info["stand_in"] = index
            return Answer(text, "partial", cut, calls, panel=info)
        info["mode"] = "synthesis_failed"
        return Answer(None, "failed", False, calls, panel=info)

    def _auto(self, setup: Setup, conversation: TextConversation, max_output_tokens: int, run_id: str) -> Answer:
        # The embedding is made first, outside the router's deadline: this harness paces and
        # retries its calls from outside Cloudflare, so a late embedding here would be our
        # queue, not the product's hop.  The router then classifies with that embedding of the
        # same routing text, so ``auto`` runs as when production's embedding arrives in time
        # (production drops a late one and answers with the mode's default model).
        embed_records: List[CallRecord] = []
        expected = routing_text(conversation)
        vector, record = self.providers.embed(self.profile.model, self.profile.prefix + expected,
                                              self.profile.dimensions)
        embed_records.append(record)

        class _Embedder:
            async def embed(self, text: str, *, deadline_seconds: float):
                return vector if text == expected else None

        router = AutoRouter(_Embedder(), self.profile, deadline_seconds=self.router_deadline)
        route = asyncio.run(router.route(setup.selector, conversation, self.allowed))
        if route is None:
            return Answer(None, "failed", False, embed_records, route={"pick": None})
        route_info = {"pick": route.pick if route.pick.startswith("panel:") else CATALOG.alias(route.pick),
                      "task": route.task, "classification": route.classification, "mode": route.mode}
        if route.pick.startswith("panel:"):
            result = self._panel(panda_panel(route.pick.split(":", 1)[1]), conversation, max_output_tokens, run_id)
        else:
            result = self._single(route.pick, conversation, max_output_tokens)
        result.calls = embed_records + result.calls
        result.route = route_info
        return result
