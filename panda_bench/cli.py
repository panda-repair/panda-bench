"""Offline arithmetic entry point."""
import argparse
from .storage import bundle_path

def main(argv=None):
    parser=argparse.ArgumentParser(prog="panda-bench")
    sub=parser.add_subparsers(dest="command",required=True)
    cmd=sub.add_parser("check")
    cmd.add_argument("source")
    args=parser.parse_args(argv)
    try:
        from .analysis import check
        return check(bundle_path(args.source))
    except (ValueError,KeyError,FileNotFoundError) as exc:
        print("ERROR:",str(exc))
        return 2
