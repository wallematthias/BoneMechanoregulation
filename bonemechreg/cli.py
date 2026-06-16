"""Command-line interface for BoneMechanoregulation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from bonemechreg.results import format_workflow_summary
from bonemechreg.standalone import run_standalone_analysis
from bonemechreg.post_timelapse import run_post_timelapse_mechanoregulation


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mechanoregulation`` command parser."""
    parser = argparse.ArgumentParser(prog="mechanoregulation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("input_dir", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("dataset_root", type=Path)
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process-style exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "analyze":
        outputs = run_standalone_analysis(args.input_dir)
        print(f"csv={outputs['csv']}")
        print(f"png={outputs['png']}")
        return 0
    if args.command == "run":
        summary = run_post_timelapse_mechanoregulation(
            dataset_root=args.dataset_root,
            profile=args.profile,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
        )
        print(format_workflow_summary(summary))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
