"""Listwise judge plans are offline; SDK initialization is explicitly deferred."""
import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from .storage import ROOT, bundle_path, canonical, index_rows, read_json, read_rows, update_latest, verify_files, write_json, write_rows, finite_nonnegative


def judge_plan(manifest, answers, model):
    settings = manifest["judge"]
    if model != settings["model"]:
        raise ValueError("No verified rate table for this judge override; add its rates to a copied manifest first")
    for rate in settings["rates_usd_per_million"].values():
        finite_nonnegative(rate, "judge rate")
    if not isinstance(settings["max_tokens"], int) or isinstance(settings["max_tokens"], bool) or settings["max_tokens"] <= 0:
        raise ValueError("Invalid judge output allowance")
    index = index_rows(answers)
    requests, missing = [], []
    prefix = settings["shuffle_seed_template"]
    for prompt in sorted(manifest["prompts"], key=lambda p: p["id"]):
        pid = prompt["id"]
        entries = list(settings["bundle_before_shuffle"][pid])
        if not entries:
            continue
        rng = random.Random(prefix.format(prompt_id=pid))
        rng.shuffle(entries)
        text_bytes = 0
        for option in entries:
            row = index.get((option, pid))
            if row is None or row.get("text") is None:
                missing.append({"prompt_id": pid, "option": option})
            else:
                text_bytes += len(row["text"].encode("utf-8"))
        # UTF-8 byte count plus ample framing allowance is a conservative offline
        # token-count proxy, not an exact Anthropic tokenizer or a price quotation.
        upper = (text_bytes + prompt["prompt_utf8_bytes"] + prompt["checklist_utf8_bytes"]
                 + len(settings["system"].encode()) + len(canonical(settings["schema"])) + 4096)
        requests.append({"prompt_id": pid, "options": entries, "input_token_upper_estimate": upper,
                         "max_output_tokens": settings["max_tokens"]})
    rates = settings["rates_usd_per_million"]
    cost = sum(r["input_token_upper_estimate"] * rates["input"] + r["max_output_tokens"] * rates["output"] for r in requests) / 1e6
    return {"requests": requests, "request_count": len(requests), "answer_count": sum(len(r["options"]) for r in requests),
            "missing_answers": missing, "judge": model, "conservative_cost_usd": round(cost, 6),
            "historical_reference_usd": 10.07 if manifest.get("schema") == "panda-bench.release/v1" and manifest["stage"] == "stage-4" else None,
            "historical_reference_scope": "Original ten-arm stage-4 judge usage; not a forecast for new candidate bundles.",
            "output_allowance_usd": len(requests) * settings["max_tokens"] * rates["output"] / 1e6,
            "estimate_basis": "Historical batch rates; byte-based input upper estimate plus full output allowance. Not a current-price guarantee."}


def build_requests(manifest, answers, prompts, plan):
    index = index_rows(answers)
    settings = manifest["judge"]
    output, mappings = [], {}
    for request in plan["requests"]:
        pid = request["prompt_id"]
        prompt = prompts[pid]
        labels = [chr(ord("A") + i) for i in range(len(request["options"]))]
        parts = ["<user_request>", prompt["prompt"], "</user_request>"]
        if prompt.get("checklist"):
            parts += ["", "<checklist>", *["- " + item for item in prompt["checklist"]], "</checklist>"]
        for label, option in zip(labels, request["options"]):
            parts += ["", f'<answer label="{label}">', index[(option, pid)]["text"], "</answer>"]
        parts += ["", f"Grade all {len(labels)} answers ({', '.join(labels)})."]
        custom = "p" + hashlib.sha256(pid.encode()).hexdigest()[:40]
        mappings[custom] = {"prompt_id": pid, "labels": dict(zip(labels, request["options"]))}
        output.append({"custom_id": custom, "params": {
            "model": plan["judge"], "max_tokens": settings["max_tokens"], "system": settings["system"],
            "messages": [{"role": "user", "content": "\n".join(parts)}],
            "output_config": {"effort": settings["effort"], "format": {"type": "json_schema", "schema": settings["schema"]}},
        }})
    return output, mappings


def parse_grades(text, labels):
    # Historical parser: unknown labels ignored, last duplicate wins, int then
    # clamp to 1..10. A missing grade stays missing (never becomes a failure).
    grades = json.loads(text)["grades"]
    return {labels[item["label"]]: max(1, min(10, int(item["score"])))
            for item in grades if item["label"] in labels}


def collect_grades(results, mappings, template):
    scores, excluded, seen = {}, [], set()
    usage = {"input_tokens": 0, "output_tokens": 0}
    for result in results:
        custom = result["custom_id"]
        if custom not in mappings or custom in seen:
            raise ValueError("Unexpected or duplicate batch result")
        seen.add(custom)
        pid, labels = mappings[custom]["prompt_id"], mappings[custom]["labels"]
        item = result["result"]
        if item["type"] != "succeeded":
            excluded.append({"prompt_id": pid, "reason": "judge_request_failed"})
            continue
        message = item["message"]
        usage["input_tokens"] += message.get("usage", {}).get("input_tokens", 0)
        usage["output_tokens"] += message.get("usage", {}).get("output_tokens", 0)
        if message["stop_reason"] != "end_turn":
            excluded.append({"prompt_id": pid, "reason": "judge_did_not_finish"})
            continue
        try:
            text = next(b["text"] for b in message["content"] if b["type"] == "text")
            values = parse_grades(text, labels)
        except (ValueError, KeyError, StopIteration, TypeError):
            raise ValueError("Malformed judge result; collection was not committed") from None
        scores.update({(option, pid): value for option, value in values.items()})
    if seen != set(mappings):
        raise ValueError("Batch result stream is incomplete; no grades were committed")
    rows = [{**row, "grade": scores.get((row["option"], row["prompt_id"]))} for row in template]
    return rows, excluded, usage


def execute_batch(requests, mappings, manifest, template, output, client=None):
    """No implicit resubmission after an uncertain create/collection outcome."""
    if not requests:
        write_rows(output / "grades.jsonl", template)
        manifest["excluded_prompts"] = []
        write_json(output / "manifest.json", manifest)
        write_json(output / "judge-state.json", {"state": "completed", "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0})
        return
    if client is None:
        import os
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for live grading")
        import anthropic
        client = anthropic.Anthropic(api_key=key, max_retries=0)
    write_json(output / "judge-state.json", {"state": "submitting", "request_count": len(requests)})
    try:
        batch = client.messages.batches.create(requests=requests)
        while batch.processing_status != "ended":
            time.sleep(30)
            batch = client.messages.batches.retrieve(batch.id)
        results = (r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in client.messages.batches.results(batch.id))
        grades, excluded, usage = collect_grades(results, mappings, template)
    except Exception:
        # No remote identifier, raw exception or prompt payload is written to disk.
        write_json(output / "judge-state.json", {"state": "blocked", "reason": "submission_or_collection_outcome_uncertain", "request_count": len(requests)})
        raise ValueError("Judge submission/collection did not complete. Inspect your provider account; do not blindly resubmit.") from None
    write_rows(output / "grades.jsonl", grades)
    manifest["excluded_prompts"] = excluded
    write_json(output / "manifest.json", manifest)
    rates = manifest["judge"]["rates_usd_per_million"]
    write_json(output / "judge-state.json", {"state": "completed", **usage,
        "estimated_cost_usd": (usage["input_tokens"] * rates["input"] + usage["output_tokens"] * rates["output"]) / 1e6})


def command(args):
    from pathlib import Path
    source = bundle_path(args.source)
    manifest = read_json(source / "manifest.json")
    verify_files(source, manifest)
    answers = read_rows(source / "answers.jsonl")
    if manifest.get("run_state") not in (None, "completed"):
        raise ValueError("Run is incomplete or has unknown charges; finish or reconcile it before grading")
    plan = judge_plan(manifest, answers, args.judge)
    plan["budget_usd"] = args.max_cost_usd
    plan["within_budget"] = plan["conservative_cost_usd"] <= args.max_cost_usd
    print(json.dumps(plan, indent=2))
    if args.plan:
        return 0
    if plan["missing_answers"]:
        raise ValueError("An archived answer is withheld/missing; exact listwise regrading is unavailable")
    if not plan["within_budget"]:
        raise ValueError("Conservative judge exposure exceeds --max-cost-usd; no batch submitted")
    from .datasets import load_prompts
    prompts = load_prompts(manifest, allow_fetch=True)
    requests, mappings = build_requests(manifest, answers, prompts, plan)
    if args.command == "regrade":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path(ROOT) / "runs" / (stamp + "-regrade")
        output.mkdir(parents=True)
        write_rows(output / "answers.jsonl", answers)
        write_rows(output / "usage.jsonl", read_rows(source / "usage.jsonl"))
        template = read_rows(source / "grades.jsonl")
        manifest = copy.deepcopy(manifest)
        manifest.pop("files", None)
        manifest["schema"] = "panda-bench.run/v1"
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest["purpose"] = "Regrade original candidate bundle; new judge sample, not historical identity."
        write_json(output / "manifest.json", manifest)
        update_latest(output)
    else:
        output = Path(source)
        if (output / "judge-state.json").exists():
            raise ValueError("This run already has a judge transaction; copy the run for a deliberate new grading pass")
        template = [{"prompt_id": a["prompt_id"], "option": a["option"], "status": a["status"], "fallback_option": a.get("fallback_option"), "grade": None} for a in answers]
    execute_batch(requests, mappings, manifest, template, output)
    print("Graded results:", output)
    return 0
