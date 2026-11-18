"""CLI dispatch keeps network libraries out of check and plan paths."""
import argparse
import math
from .storage import bundle_path


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite amount")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panda-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("check", help="Recompute archived arithmetic offline")
    cmd.add_argument("source")
    for name in ("regrade", "grade"):
        cmd = sub.add_parser(name)
        cmd.add_argument("source")
        cmd.add_argument("--max-cost-usd", type=positive, required=True)
        cmd.add_argument("--plan", action="store_true")
        cmd.add_argument("--judge", default="claude-opus-5-5")
    cmd = sub.add_parser("fetch", help="Fetch pinned public WildBench files; no model calls")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            from .analysis import check
            return check(bundle_path(args.source))
        if args.command == "fetch":
            from .datasets import fetch
            fetch()
            return 0
        from .judging import command
        return command(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        # Messages originate in local validation; never interpolate provider exceptions.
        print("ERROR:", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
