"""Deterministic arithmetic for archived grades. No provider or network imports."""
import random
import statistics
from .storage import finite_nonnegative, index_rows, read_json, read_rows, verify_files


def resolved_grade(index, option, prompt, trail=()):
    if option in trail:
        raise ValueError("Cyclic fallback")
    row = index[(option, prompt)]
    if row["status"] == "failed":
        return 1
    if row["status"] == "fallback":
        return resolved_grade(index, row["fallback_option"], prompt, trail + (option,))
    if row["status"] != "ok":
        raise ValueError("Unresolved generation status")
    value = row.get("grade")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError("Missing or invalid judge grade")
    return value


def paired(values, baseline, settings):
    if not values or len(values) != len(baseline):
        raise ValueError("Paired sample is empty or misaligned")
    diffs = [a - b for a, b in zip(values, baseline)]
    rng = random.Random(settings["seed"])
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(settings["resamples"]))
    return statistics.mean(diffs), means[settings["lower_index"]], means[settings["upper_index"]]


def calculate(manifest, grades, usage):
    index, costs = index_rows(grades), index_rows(usage)
    excluded = {row["prompt_id"] for row in manifest.get("excluded_prompts", [])}
    ids = sorted(p["id"] for p in manifest["prompts"] if p["id"] not in excluded)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("No unique scored prompts")
    baseline = manifest["baseline"]
    unit = 1 if manifest["stage"] == "stage-2" else 10
    scale = 10 / unit
    base = {pid: resolved_grade(index, baseline, pid) for pid in ids}
    base_cost = statistics.mean(finite_nonnegative(costs[(baseline, pid)]["cost_usd"], "cost") for pid in ids)
    if base_cost <= 0:
        raise ValueError("Baseline cost must be positive")
    output = []
    for option in manifest["options"]:
        name = option["id"]
        raw = [resolved_grade(index, name, pid) for pid in ids]
        if any(value is None for value in raw):
            print("Missing judge grades; option omitted:", name)
            continue
        values = [unit * value for value in raw]
        paired_ids = [pid for pid in ids if base[pid] is not None]
        diff, low, high = paired([unit * resolved_grade(index, name, pid) for pid in paired_ids],
                                 [unit * base[pid] for pid in paired_ids], manifest["bootstrap"])
        cost = statistics.mean(finite_nonnegative(costs[(name, pid)]["cost_usd"], "cost") for pid in ids)
        waits = sorted(finite_nonnegative(costs[(name, pid)]["wall_s"], "wait") for pid in ids)
        median = waits[len(waits) // 2]
        intermediate = manifest["display"].get("median_intermediate_decimals")
        if intermediate is not None:
            median = round(median, intermediate)
        decimals = manifest["display"]["score_decimals"]
        output.append({"id": option.get("display_id", name), "score": round(statistics.mean(values) * scale, decimals),
                       "vs_argon": round(diff * scale, decimals), "vs_argon_low": round(low * scale, decimals),
                       "vs_argon_high": round(high * scale, decimals), "cost_1k": round(cost * 1000, 2),
                       "cost_pct": round(100 * cost / base_cost), "wait_p50_s": round(median), "prompts": len(paired_ids)})
    return output


def check(path):
    manifest = read_json(path / "manifest.json")
    verify_files(path, manifest)
    actual = calculate(manifest, read_rows(path / "grades.jsonl"), read_rows(path / "usage.jsonl"))
    expected_path = path / "expected.json"
    is_release = manifest.get("schema") == "panda-bench.release/v1"
    if is_release and not expected_path.is_file():
        raise ValueError("Release expected.json is missing")
    expected = {r["id"]: r for r in read_json(expected_path)} if expected_path.is_file() else None
    mismatches = []
    if expected is not None and set(expected) != {r["id"] for r in actual}:
        mismatches.append("Option set differs")
    print("option | score | difference [95% interval] | cost/1k | cost% | median s | n | expected")
    for row in actual:
        bad = [k for k, value in row.items() if k != "id" and expected is not None and expected.get(row["id"], {}).get(k) != value]
        mismatches.extend(row["id"] + ": " + k for k in bad)
        status = "new results" if expected is None else "MATCH" if not bad else "MISMATCH: " + ",".join(bad)
        print(f"{row['id']} | {row['score']} | {row['vs_argon']:+g} [{row['vs_argon_low']:+g}, {row['vs_argon_high']:+g}] | ${row['cost_1k']:.2f} | {row['cost_pct']}% | {row['wait_p50_s']} | {row['prompts']} | {status}")
        if bad:
            print("  expected:", {k: expected.get(row["id"], {}).get(k) for k in bad})
    if mismatches:
        print("CHECK FAILED:", "; ".join(mismatches))
        return 1
    print("All displayed arithmetic matches." if expected is not None else "New results; no historical equality claim.")
    return 0
