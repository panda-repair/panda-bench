"""Grader checks: how far the stand-in judge sits from the official one.

The stand-in judge grades answers whose official scores are published, and the difference
is the judge offset every benchmark result carries:

- MultiChallenge: three of the paper's stored 2024 responses (Mixtral-8x7B, Llama-3.3-70B,
  o1-preview), on the paper's own 273 conversations, against the paper's scores (GPT-4o
  judge).
- HealthBench Professional: the 525 physician-written answers in the dataset, against the
  paper's physician baseline of 43.7 (GPT-5.4 grader).

``python -m panda_bench.frontier.checks submit --run DIR`` sends the judge requests (about
1,950 at about $0.002 each); ``report --run DIR`` waits for them and prints the offsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from . import judge
from .run import CACHE

CHECKS = {
    "multichallenge-2025": "multichallenge",
    "healthbench-professional-physician": "healthbench-professional",
}


def _sets():
    from .benchmarks import healthbench_professional as hb
    from .benchmarks import multichallenge as mc

    responses = mc.reference_responses(CACHE / "multichallenge")
    mc_answers = {("ref:" + model, item_id): {"setup": "ref:" + model, "item": item_id, "status": "completed", "text": text}
                  for model in mc.REFERENCE_RECOMMENDED for item_id, text in responses[model].items()}
    physician = hb.reference_answers(CACHE / "healthbench-professional")
    hb_answers = {("ref:physician", item_id): {"setup": "ref:physician", "item": item_id, "status": "completed", "text": text}
                  for item_id, text in physician.items()}
    return {
        "multichallenge-2025": (mc.REFERENCE_BENCHMARK, CACHE / "multichallenge", mc_answers),
        "healthbench-professional-physician": (hb.BENCHMARK, CACHE / "healthbench-professional", hb_answers),
    }


def submit(run_dir: Path, max_new_usd: float) -> None:
    for name, (bench, cache, answers) in _sets().items():
        items = {item.id: item for item in bench.load(cache)}
        requests = judge.pending_requests(run_dir, name, bench, items, answers, retry_failed=True)
        if requests:
            expected, worst = judge.estimate_usd(requests)
            print(f"{name}: {len(requests)} judge requests, expected ${expected:.2f}, worst ${worst:.2f}")
            print("  batches:", judge.submit(run_dir, name, requests, max_new_usd))


def report(run_dir: Path) -> Dict[str, Any]:
    from .benchmarks import multichallenge as mc

    judge.collect(run_dir, wait=True)
    out: Dict[str, Any] = {}
    for name, (bench, cache, answers) in _sets().items():
        items = bench.load(cache)
        judged_rows = judge.read_judged(run_dir / "judge" / (name + ".jsonl"))
        # The grader check was judged at 2026-11-18 ~16:17Z, before judge ids bound the
        # answer's digest.  Its answers are fixed published texts, one per (setup, item), so a
        # reply is also found by setup, item, request and judge settings.
        by_request = {(r.get("setup"), r.get("item"), r.get("name")): r for r in judged_rows.values()
                      if r.get("judge_settings") == judge.JUDGE_SETTINGS}
        by_setup: Dict[str, Any] = {}
        for setup in sorted({s for s, _ in answers}):
            grades = []
            for item in items:
                row = answers.get((setup, item.id))
                text = row["text"] if row else None
                replies = judge.replies_for(judged_rows, name, setup, item, text, bench) if text else {}
                if replies is None:
                    found = {r.name: by_request.get((setup, item.id, r.name)) for r in bench.judge_requests(item, text)}
                    replies = {k: v.get("text") for k, v in found.items() if v is not None} if all(found.values()) else None
                grades.append(dict(bench.grade(item, text, replies if replies is not None else {})))
            by_setup[setup] = bench.score(grades)
        out[name] = by_setup
    # The published numbers these stand in for.
    out["published"] = {"multichallenge-2025": {m: mc.REFERENCE_SCORES.get(m) for m in mc.REFERENCE_RECOMMENDED},
                        "healthbench-professional-physician": {"ref:physician": {"score": 43.7, "source": "arXiv 2604.27470 §5.1"}}}
    (run_dir / "grader-check.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m panda_bench.frontier.checks")
    parser.add_argument("command", choices=("submit", "report"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--max-new-usd", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.command == "submit":
        submit(args.run, args.max_new_usd)
    else:
        print(json.dumps(report(args.run), indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
