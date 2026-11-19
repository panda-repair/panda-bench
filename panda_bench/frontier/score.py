"""Scores: each setup's grades on each benchmark, with coverage, cost, latency and failures.

A setup's score is published only when it is complete: every item has a final answer row
(an answer, or a failure graded as the board grades one) and every judge request of every
answer has a usable reply.  Otherwise the result carries ``complete: false`` and the counts
that are missing, so a partial number is never mistaken for a board-comparable one.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .judge import read_judged, replies_for
from .run import CACHE, load_benchmark, read_rows


def _nearest_rank(values: List[float], q: float) -> Optional[float]:
    """The q-quantile by the nearest-rank method (the smallest value with at least q of the
    sample at or below it); None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 2)


def _status(row: Dict[str, Any]) -> str:
    """The product's run status: a Panel that lost a member, kept one answer or used a
    stand-in is partial (rows written before 2026-11-18 ~16:40Z said "completed" or
    "single_answer" for some of these)."""
    panel = row.get("panel") or {}
    status = row.get("status")
    if panel and row.get("text") is not None:
        everyone = len(panel.get("answered", [])) == len(panel.get("members", []))
        if panel.get("mode") != "synthesis" or not everyone:
            return "partial"
        return "completed"
    return status


def service_seconds(row: Dict[str, Any]) -> float:
    """The answer's time in model calls, without this harness's queueing: a single model's
    call; a Panel's slowest member, then its combiner; ``auto``'s embedding, then its pick.
    Each call counts its last attempt only (retries and back-off are our load's, and are
    reported apart as first-attempt trouble)."""
    calls = row.get("calls") or []
    embed = sum(c.get("seconds", 0.0) for c in calls if c.get("role") == "router_embedding")
    members = [c.get("seconds", 0.0) for c in calls if c.get("role") == "member"]
    synthesis = sum(c.get("seconds", 0.0) for c in calls if c.get("role") == "synthesis")
    single = sum(c.get("seconds", 0.0) for c in calls if c.get("role") == "single")
    return round(embed + (max(members) if members else 0.0) + synthesis + single, 3)


def retail_cents(list_cost_nanos: int, answered: bool) -> int:
    """Panda's charge for one run (docs/panda-api.md): list cost x 3/2, rounded once to the
    cent (half away from zero), at least 1 cent when anything answered; nothing otherwise."""
    if not answered:
        return 0
    cents = list_cost_nanos * 3 / 2 / 10_000_000
    return max(1, int(math.floor(cents + 0.5)))


def failed_judge_items(run_dir: Path, name: str) -> Set[Tuple[str, str]]:
    """(setup, item) pairs whose answer has a judge reply the benchmark could not use."""
    bench = load_benchmark(name)
    items = {item.id: item for item in bench.load(CACHE / name)}
    answers = read_rows(run_dir / "answers" / (name + ".jsonl"))
    judged = read_judged(run_dir / "judge" / (name + ".jsonl"))
    failed = set()
    for (setup, item_id), row in answers.items():
        text = row.get("text")
        if text is None or item_id not in items:
            continue
        replies = replies_for(judged, name, setup, items[item_id], text, bench)
        if replies is not None and bench.grade(items[item_id], text, replies).get("failed_judge_calls"):
            failed.add((setup, item_id))
    return failed


def score_benchmark(run_dir: Path, name: str, grades_out: Optional[Path] = None) -> Dict[str, Any]:
    bench = load_benchmark(name)
    items = bench.load(CACHE / name)
    answers = read_rows(run_dir / "answers" / (name + ".jsonl"))
    judged = read_judged(run_dir / "judge" / (name + ".jsonl"))
    setups = sorted({setup for setup, _ in answers})
    results: Dict[str, Any] = {}
    grade_rows: List[Dict[str, Any]] = []
    for setup in setups:
        grades, pending, missing, judge_failed = [], 0, 0, 0
        cost_nanos, retail, walls, first_trouble = 0, 0, [], 0
        statuses, routes, modes = Counter(), Counter(), Counter()
        for item in items:
            row = answers.get((setup, item.id))
            if row is None or row.get("status") == "harness_error":
                missing += 1
                continue
            text = row.get("text")
            # Spend and timing count every final answer row, graded or not.
            cost_nanos += row.get("list_cost_nanos", 0) or 0
            retail += retail_cents(row.get("list_cost_nanos", 0) or 0, text is not None)
            walls.append(service_seconds(row))
            statuses[_status(row)] += 1
            calls = [c for c in row.get("calls", []) if c.get("role") != "router_embedding"]
            first_trouble += any(c.get("attempts") and c["attempts"][0] != "ok" for c in calls)
            if row.get("route"):
                routes[str(row["route"].get("pick"))] += 1
            if row.get("panel"):
                modes[str(row["panel"].get("mode"))] += 1
            replies = replies_for(judged, name, setup, item, text, bench) if text is not None else {}
            if replies is None:
                pending += 1
                continue
            grade = dict(bench.grade(item, text, replies))
            grade["item"] = item.id
            judge_failed += bool(grade.get("failed_judge_calls"))
            grades.append(grade)
            grade_rows.append({"benchmark": name, "setup": setup, **grade})
        complete = not missing and not pending and not judge_failed
        summary = dict(bench.score(grades)) if grades else {"score": None}
        answered_rows = len(items) - missing
        summary.update({
            "complete": complete, "items": len(items), "graded": len(grades), "not_answered": missing,
            "pending_judge": pending, "items_with_failed_judge_calls": judge_failed,
            "list_usd_total": round(cost_nanos / 1e9, 4),
            "list_usd_per_item": round(cost_nanos / 1e9 / answered_rows, 6) if answered_rows else None,
            "retail_usd_total": round(retail / 100, 2),
            "retail_usd_per_item": round(retail / 100 / answered_rows, 4) if answered_rows else None,
            "service_seconds_median": round(statistics.median(walls), 2) if walls else None,
            "service_seconds_p90": _nearest_rank(walls, 0.9), "service_samples": len(walls),
            "statuses": dict(statuses), "first_attempt_trouble": first_trouble,
        })
        if routes:
            summary["routes"] = dict(routes)
        if modes:
            summary["panel_modes"] = dict(modes)
        results[setup] = summary
    if grades_out is not None:
        grades_out.parent.mkdir(parents=True, exist_ok=True)
        with grades_out.open("w", encoding="utf-8") as handle:
            for row in grade_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results


def score_run(run_dir: Path, benchmarks: Sequence[str]) -> Dict[str, Any]:
    out = {name: score_benchmark(run_dir, name, run_dir / "grades" / (name + ".jsonl")) for name in benchmarks}
    (run_dir / "results.json").write_text(json.dumps(out, indent=1))
    return out


def table(results: Dict[str, Any]) -> str:
    lines = []
    for name, by_setup in results.items():
        lines.append(f"== {name}")
        for setup, s in sorted(by_setup.items(), key=lambda kv: -(kv[1].get("score") or -1)):
            score = s.get("score")
            mark = "" if s["complete"] else "  (incomplete)"
            lines.append(f"  {setup:30s} score={'-' if score is None else round(score, 2):>6}  graded={s['graded']:>4}/{s['items']:<4}"
                         f" pending={s['pending_judge']:>4} judge_failed={s['items_with_failed_judge_calls']:>3}"
                         f"  retail$/item={s['retail_usd_per_item']}  median={s['service_seconds_median']}s{mark}")
    return "\n".join(lines)
