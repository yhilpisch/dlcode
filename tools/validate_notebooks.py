#!/usr/bin/env python3
"""Validate notebooks for 5_dltex.

Runs check_notebooks.py and supports skipping execution.

Usage:
  python tools/validate_notebooks.py [--skip-execute] [options]
"""
from __future__ import annotations

import argparse
import subprocess as sp
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-execute", action="store_true", help="do not execute notebooks")
    ap.add_argument("--paths", nargs="+", default=["notebooks"], help="paths to scan")
    ap.add_argument("--pattern", default="*.ipynb", help="glob within directories")
    ap.add_argument("--exclude", action="append", default=[], help="regex to exclude (repeatable)")
    ap.add_argument("--timeout", type=int, default=180, help="per-cell timeout")
    ap.add_argument("--kernel", default="python3", help="kernel name")
    ap.add_argument("--allow-errors", action="store_true", help="do not fail on cell exceptions")
    ap.add_argument("--fail-fast", action="store_true", help="stop on first failure")
    ap.add_argument("--outdir", type=str, default=None, help="save executed notebooks under this dir")
    ap.add_argument("--list", action="store_true", help="list notebooks and exit")
    ap.add_argument("--verbose", action="store_true", help="verbose output")
    ap.add_argument("--normalize-ids", action="store_true", help="normalize missing/duplicate cell ids")
    ap.add_argument("--write-normalized", action="store_true", help="write normalized notebooks")
    args = ap.parse_args()

    cmd = [sys.executable, str(TOOLS / "check_notebooks.py")]
    cmd += ["--paths", *args.paths]
    cmd += ["--pattern", args.pattern]
    for rx in args.exclude:
        cmd += ["--exclude", rx]
    cmd += ["--timeout", str(args.timeout)]
    cmd += ["--kernel", args.kernel]
    if args.allow_errors:
        cmd.append("--allow-errors")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    if args.list:
        cmd.append("--list")
    if args.verbose:
        cmd.append("--verbose")
    if args.normalize_ids:
        cmd.append("--normalize-ids")
    if args.write_normalized:
        cmd.append("--write-normalized")
    if args.skip_execute:
        cmd.append("--skip-execute")

    return sp.run(cmd, cwd=ROOT, text=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
