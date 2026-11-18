"""Current-product runs with two workers and a durable per-request ledger.

This module is safe to import for plans: the OpenAI SDK and worker pool are lazy.
"""
import copy
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from .storage import ROOT, bundle_path, digest, read_json, update_latest, verify_files, write_json, write_rows

DEFAULT_MODELS = ("panel:panda", "panel:panda-quality", "gemini-4-argon")
PUBLIC_AGENT_MODELS = frozenset((*DEFAULT_MODELS, "deepseek-v4-pro", "gemini-4-flash", "glm-5.3", "glm-5.3-flash", "@cf/zai-org/glm-5.3-flash"))


def base_url(value):
    value = value.rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[:-len("/chat/completions")]
    # Keep credentials/query strings out of both requests and local manifests.
    if not re.fullmatch(r"https?://[A-Za-z0-9.:-]+(?:/[A-Za-z0-9_.-]+)*", value):
        raise ValueError("--base-url must be an HTTP(S) origin/path without credentials, query or fragment")
    if value.startswith("http://") and not re.match(r"http://(?:127\.0\.0\.1|localhost)(?::[0-9]+)?(?:/|$)", value):
        raise ValueError("Use HTTPS except for a local test server")
    return value


def model_names(value):
    names = value.split(",")
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_:./-]{0,199}", n) for n in names):
        raise ValueError("--models must contain unique comma-separated model selectors")
    if "gemini-4-argon" not in names:
        raise ValueError("Include gemini-4-argon: this report requires its paired baseline and cost share")
    return names


def plan_run(manifest, models, endpoint, budget):
    return {"stage": manifest["stage"], "models": models, "prompt_count": len(manifest["prompts"]),
            "request_count": len(manifest["prompts"]) * len(models), "base_url": endpoint,
            "max_completion_tokens": 4096, "concurrency": 2, "budget_usd": budget,
            "generation_estimate_usd": None,
            "budget_policy": "Stop scheduling at the settled charge threshold; up to two in-flight requests can overshoot. Unknown charges block resume.",
            "cost_basis": "Actual Panda USD charges. No live quote or hard spend guarantee; historical provider list prices are not customer charges.",
            "requests": [{"prompt_id": p["id"], "model": m, "max_completion_tokens": 4096}
                         for p in manifest["prompts"] for m in models]}


def request_body(model, prompt):
    return {"model": model, "messages": [{"role": "user", "content": prompt["prompt"]}], "max_completion_tokens": 4096}


def request_key(run_id, prompt_id, model, body):
    return digest({"run": run_id, "prompt": prompt_id, "model": model, "body": body})


def cents(value):
    if not isinstance(value, dict) or value.get("schema_version") != "panda.usd-amount.v1" or value.get("currency") != "USD":
        raise ValueError("Charge is missing or has an unsupported USD schema")
    units = value.get("minor_units")
    if not isinstance(units, str) or not re.fullmatch(r"0|[1-9][0-9]{0,17}", units):
        raise ValueError("Charge must be a finite nonnegative integer string of USD cents")
    return int(units)


def safe_usage(value):
    if value is None:
        return None
    keys = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")
    return {k: value[k] for k in keys if isinstance(value.get(k), int) and not isinstance(value[k], bool) and value[k] >= 0} if isinstance(value, dict) else None


def safe_agent(value):
    if not isinstance(value, dict):
        return {}
    out = {}
    if isinstance(value.get("index"), int) and not isinstance(value["index"], bool) and 1 <= value["index"] <= 3:
        out["index"] = value["index"]
    model = value.get("model")
    if isinstance(model, str) and model in PUBLIC_AGENT_MODELS:
        out["model"] = model
    for key, allowed in (("status", ("completed", "failed", "indeterminate", "not_run")),
                         ("finish_reason", ("stop", "length", "content_filter"))):
        if value.get(key) in allowed:
            out[key] = value[key]
    if isinstance(value.get("charged"), bool):
        out["charged"] = value["charged"]
    if isinstance(value.get("cost_microdollars"), int) and not isinstance(value["cost_microdollars"], bool) and value["cost_microdollars"] >= 0:
        out["cost_microdollars"] = value["cost_microdollars"]
    out["usage"] = safe_usage(value.get("usage"))
    # No service IDs, arbitrary error messages, headers, or internal output copies.
    return out


def adapt_response(body, http_status=200):
    """Normalize only documented fields; an uncertain cost is never zero."""
    out = {"state": "blocked", "reason": "unknown_response_or_charge", "charge_cents": None}
    if not isinstance(body, dict) or not isinstance(body.get("panda"), dict):
        return out
    panda = body["panda"]
    try:
        out["charge_cents"] = cents(panda.get("charge"))
    except ValueError:
        return out
    if panda.get("status") not in ("completed", "partial", "failed"):
        return out
    out["panda"] = {"status": panda["status"], "agents": [safe_agent(a) for a in panda.get("agents", [])] if isinstance(panda.get("agents"), list) else [],
                    "charge": {"schema_version": "panda.usd-amount.v1", "currency": "USD", "minor_units": str(out["charge_cents"])}}
    if isinstance(panda.get("usage_complete"), bool):
        out["panda"]["usage_complete"] = panda["usage_complete"]
    if isinstance(panda.get("synthesis"), dict):
        out["panda"]["synthesis"] = safe_agent(panda["synthesis"])
    if panda.get("replay") is True or http_status == 409:
        out["reason"] = "settled_replay_has_no_answer"
        return out
    if panda["status"] == "failed":
        out.update(state="completed", reason=None, status="failed", text="")
        return out
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError):
        text = None
    if http_status != 200 or not isinstance(text, str) or not text.strip():
        out["reason"] = "settled_response_has_no_answer"
        return out
    out.update(state="completed", reason=None, status="ok", text=text)
    return out


class PandaTransport:
    def __init__(self, endpoint):
        import os
        key = os.environ.get("PANDA_API_KEY")
        if not key:
            raise ValueError("PANDA_API_KEY is required for live generation")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=600.0)

    def __call__(self, body, key):
        try:
            response = self.client.chat.completions.create(**body, extra_headers={"Idempotency-Key": key})
            return adapt_response(response.model_dump(mode="json"))
        except Exception as exc:
            # APIStatusError.body may contain a charged failure/replay. Only the
            # explicit response adapter sees it; exception strings never persist.
            response = getattr(exc, "body", None)
            status = getattr(exc, "status_code", None)
            return adapt_response(response, status)


def fingerprint(manifest, models, endpoint):
    return digest({"prompts": [{"id": p["id"], "sha256": p["sha256"]} for p in manifest["prompts"]],
                   "models": models, "base_url": endpoint, "max_completion_tokens": 4096})


def initialize(source, models, endpoint, output, prompts):
    manifest = copy.deepcopy(source)
    manifest.pop("files", None)
    manifest["schema"] = "panda-bench.run/v1"
    manifest["purpose"] = "New current-product experiment; independent generation and a new candidate bundle."
    manifest["run_state"] = "pending"
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["options"] = [{"id": m, "display_id": m, "name": m, "kind": "current_product", "settings": {"max_completion_tokens": 4096}} for m in models]
    manifest["excluded_prompts"] = []
    manifest["price_table"] = {}
    manifest["accounting"] = "Actual Panda customer charges, not historical provider list costs."
    manifest["failure_policy"] = {"generation_failure_grade": 1, "fallback": "Current Panda partial answers are judged as returned; no historical inheritance.", "judge_error": "Exclude the entire prompt."}
    manifest["judge"]["bundle_before_shuffle"] = {}
    manifest["judge"]["shuffle_seed_template"] = "current-run:{prompt_id}"
    manifest["selection_note"] = "Selected stage prompts reused; new answers and listwise context can change scores."
    ledger = {"schema": "panda-bench.ledger/v1", "run_id": uuid.uuid4().hex,
              "fingerprint": fingerprint(source, models, endpoint), "base_url": endpoint, "models": models, "requests": []}
    for prompt in manifest["prompts"]:
        for model in models:
            body = request_body(model, prompts[prompt["id"]])
            ledger["requests"].append({"prompt_id": prompt["id"], "model": model, "body_sha256": digest(body),
                "key": request_key(ledger["run_id"], prompt["id"], model, body), "state": "pending", "charge_cents": None})
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "run-state.json", ledger)
    return manifest, ledger


def materialize(output, manifest, ledger):
    answers, usage = [], []
    for row in ledger["requests"]:
        if row["state"] != "completed":
            continue
        answers.append({"prompt_id": row["prompt_id"], "option": row["model"], "status": row["status"],
                        "fallback_option": None, "text": row["text"], "panda": row.get("panda", {})})
        usage.append({"prompt_id": row["prompt_id"], "option": row["model"], "calls": [],
                      "cost_usd": row["charge_cents"] / 100, "wall_s": row["wall_s"], "cost_basis": "panda_charge"})
    for prompt in manifest["prompts"]:
        manifest["judge"]["bundle_before_shuffle"][prompt["id"]] = sorted(a["option"] for a in answers if a["prompt_id"] == prompt["id"] and a["status"] == "ok")
    write_rows(output / "answers.jsonl", answers)
    write_rows(output / "usage.jsonl", usage)
    write_json(output / "manifest.json", manifest)


def execute(output, manifest, ledger, prompts, budget, transport):
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    unknown = [r for r in ledger["requests"] if r["state"] not in ("pending", "completed")]
    if unknown:
        raise ValueError("An earlier request has an unknown or missing-answer outcome. Reconcile it with Panda; no automatic replay or fresh key is safe.")
    seen = set()
    for row in ledger["requests"]:
        pair = (row["prompt_id"], row["model"])
        if pair in seen:
            raise ValueError("Duplicate request in resume ledger")
        seen.add(pair)
        charge = row.get("charge_cents")
        if row["state"] == "completed" and (not isinstance(charge, int) or isinstance(charge, bool) or charge < 0):
            raise ValueError("Completed request has an unknown or invalid charge")
        body = request_body(row["model"], prompts[row["prompt_id"]])
        if request_key(ledger["run_id"], row["prompt_id"], row["model"], body) != row["key"]:
            raise ValueError("Request content changed; resume refused")
        if digest(body) != row["body_sha256"]:
            raise ValueError("Request content changed; resume refused")
    limit = Decimal(str(budget)) * 100
    pending = iter(r for r in ledger["requests"] if r["state"] == "pending")
    futures, exhausted, blocked = {}, False, False

    def invoke(row):
        start = time.monotonic()
        try:
            result = transport(request_body(row["model"], prompts[row["prompt_id"]]), row["key"])
            # Transports return only adapter output. Tests inject the same shape.
        except Exception:
            result = {"state": "blocked", "reason": "transport_outcome_unknown", "charge_cents": None}
        return {**result, "wall_s": time.monotonic() - start}

    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            spent = sum(r["charge_cents"] for r in ledger["requests"] if r["charge_cents"] is not None)
            while not exhausted and not blocked and spent < limit and len(futures) < 2:
                row = next(pending, None)
                if row is None:
                    exhausted = True
                    break
                row["state"] = "submitting"
                write_json(output / "run-state.json", ledger)
                futures[pool.submit(invoke, row)] = row
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row = futures.pop(future)
                row.update(future.result())
                blocked = blocked or row["state"] != "completed"
            write_json(output / "run-state.json", ledger)
            materialize(output, manifest, ledger)
    complete = all(r["state"] == "completed" for r in ledger["requests"])
    manifest["run_state"] = "completed" if complete else "blocked" if blocked else "budget_stopped"
    write_json(output / "run-state.json", ledger)
    materialize(output, manifest, ledger)
    total = sum(r["charge_cents"] for r in ledger["requests"] if r["charge_cents"] is not None)
    print(f"Run {manifest['run_state']}; known settled charges ${total / 100:.2f}; {sum(r['state'] == 'completed' for r in ledger['requests'])}/{len(ledger['requests'])} completed pairs.")
    if not complete:
        print("Incomplete runs cannot be graded. Unknown outcomes need manual reconciliation; budget stops may resume with a higher cap.")
    return 0 if complete else 2


def command(args):
    from pathlib import Path
    source = bundle_path(args.source)
    original = read_json(source / "manifest.json")
    verify_files(source, original)
    models, endpoint = model_names(args.models), base_url(args.base_url)
    plan = plan_run(original, models, endpoint, args.max_cost_usd)
    print(json.dumps(plan, indent=2))
    if args.plan:
        return 0
    from .datasets import load_prompts
    prompts = load_prompts(original, allow_fetch=True)
    if args.resume:
        output = Path(args.resume).resolve()
        manifest, ledger = read_json(output / "manifest.json"), read_json(output / "run-state.json")
        if ledger["fingerprint"] != fingerprint(original, models, endpoint):
            raise ValueError("Resume models, dataset or endpoint differs from the original run")
        if (output / "judge-state.json").exists():
            raise ValueError("This run already has a grading transaction")
        if any(r["state"] not in ("pending", "completed") for r in ledger["requests"]):
            raise ValueError("Unknown request outcome blocks resume; reconcile it without issuing a new key")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path(ROOT) / "runs" / (stamp + "-run")
        manifest, ledger = initialize(original, models, endpoint, output, prompts)
    if all(r["state"] == "completed" for r in ledger["requests"]):
        print("All prompt/model pairs are already completed; no calls made.")
        return 0
    transport = PandaTransport(endpoint)
    update_latest(output)
    return execute(output, manifest, ledger, prompts, args.max_cost_usd, transport)
