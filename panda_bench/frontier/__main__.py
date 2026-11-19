"""``python -m panda_bench.frontier``: plan, generate, judge and score a frontier comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m panda_bench.frontier")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "judge", "score"):
        p = sub.add_parser(command)
        p.add_argument("--run", required=True, type=Path, help="the run folder (answers, judge, results)")
        p.add_argument("--benchmarks", default="gpqa-diamond,simpleqa-verified,multichallenge,healthbench-professional")
        if command == "generate":
            p.add_argument("--limit", type=int, default=None, help="first N items of each benchmark (a pilot)")
            p.add_argument("--workers", type=int, default=64)
            p.add_argument("--setups", default=None, help="comma-separated setup names (default: all)")
            p.add_argument("--max-list-usd", type=float, default=400.0)
            p.add_argument("--retry-failed", action="store_true")
        if command == "judge":
            p.add_argument("--max-new-usd", type=float, default=40.0, help="refuse a submission expected to cost more")
            p.add_argument("--collect-only", action="store_true")
            p.add_argument("--no-wait", action="store_true")
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--dry-run", action="store_true", help="count and estimate; send nothing")
    rel = sub.add_parser("release", help="write the public release files of a finished run")
    rel.add_argument("--run", required=True, type=Path, help="the run folder")
    rel.add_argument("--out", required=True, type=Path, help="the release folder, replaced whole")
    rel.add_argument("--benchmarks", default="gpqa-diamond,simpleqa-verified,multichallenge,healthbench-professional,routerarena")
    rel.add_argument("--live", type=Path, default=None, help="the live sample's run folder")
    rel.add_argument("--extra", type=Path, action="append", default=[], help="a JSON file to copy to the release root")
    rel.add_argument("--notes", type=Path, default=None, help="a JSON file of the publisher's notes")
    rel.add_argument("--allow-incomplete", action="store_true")
    chk = sub.add_parser("check-release", help="verify a release's files and recompute its scores from its grades")
    chk.add_argument("release", type=Path)
    live = sub.add_parser("live", help="answer a fixed paired sample through Panda's public API")
    live.add_argument("--run", required=True, type=Path, help="the live run folder")
    live.add_argument("--benchmarks", default="healthbench-professional,multichallenge,simpleqa-verified,gpqa-diamond")
    live.add_argument("--setups", default="panel:panda,panel:panda-quality,auto,auto:quality")
    live.add_argument("--n", type=int, default=8, help="items per benchmark")
    live.add_argument("--max-charge-usd", type=float, default=3.5)
    args = parser.parse_args(argv)
    if args.command == "check-release":
        from .release import check_release

        problems = check_release(args.release)
        print("\n".join(problems) if problems else "every file and score checks")
        return 1 if problems else 0
    benchmarks = [b for b in args.benchmarks.split(",") if b]

    if args.command == "live":
        from .live import run_sample
        from .run import CACHE, load_benchmark

        for name in [b for b in args.benchmarks.split(",") if b]:
            bench = load_benchmark(name)
            print(name, run_sample(args.run, name, bench, bench.load(CACHE / name), args.setups.split(","), args.n,
                                   max_charge_usd=args.max_charge_usd))
        return 0
    if args.command == "generate":
        from .run import generate

        generate(args.run, benchmarks, args.limit, args.workers,
                 args.setups.split(",") if args.setups else None, args.max_list_usd, args.retry_failed)
        return 0
    if args.command == "judge":
        from . import judge
        from .run import CACHE, load_benchmark, read_rows

        if not args.collect_only:
            for name in benchmarks:
                bench = load_benchmark(name)
                items = {item.id: item for item in bench.load(CACHE / name)}
                answers = read_rows(args.run / "answers" / (name + ".jsonl"))
                failed = None
                if args.retry_failed:
                    from .score import failed_judge_items

                    failed = failed_judge_items(args.run, name)
                    print(f"{name}: {len(failed)} answers with an unusable judge reply")
                requests = judge.pending_requests(args.run, name, bench, items, answers, args.retry_failed, failed)
                if not requests:
                    continue
                expected, worst = judge.estimate_usd(requests)
                print(f"{name}: {len(requests)} judge requests, expected ${expected:.2f}, worst ${worst:.2f}")
                if not args.dry_run:
                    print("  batches:", judge.submit(args.run, name, requests, args.max_new_usd))
        if not args.dry_run:
            print(json.dumps(judge.collect(args.run, wait=not args.no_wait)))
        return 0
    if args.command == "release":
        from .release import write_release

        manifest = write_release(
            args.run, args.out, benchmarks,
            notes=json.loads(args.notes.read_text()) if args.notes else None, live_dir=args.live,
            extra={path.name: json.loads(path.read_text()) for path in args.extra},
            allow_incomplete=args.allow_incomplete)
        print(json.dumps({name: {k: b[k] for k in ("answers", "judge_replies", "redone", "complete")}
                          for name, b in manifest["benchmarks"].items()}, indent=1))
        return 0
    if args.command == "score":
        from .score import score_run, table

        print(table(score_run(args.run, benchmarks)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
