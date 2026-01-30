#!/usr/bin/env python3
"""Validate Python code and chapter scripts for 5_dltex.

Runs two checks:
1) check_code.py: syntax + line-length scan across code/
2) check_chapter_scripts.py: import probe + execution for code/chNN scripts

Usage:
  python tools/validate_code.py [--code-only|--chapters-only] [options]
"""
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    proc = sp.run(cmd, cwd=ROOT, env=env, text=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true", help="run check_code only")
    ap.add_argument("--chapters-only", action="store_true", help="run chapter scripts only")
    ap.add_argument("--max-len", type=int, default=85, help="max line length for check_code")
    ap.add_argument("--quiet", action="store_true", help="suppress per-file OK lines")
    ap.add_argument("--dry-run", action="store_true", help="list checks and exit")
    ap.add_argument("--skip-execute", action="store_true", help="skip chapter script execution")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-script timeout (seconds)")
    ap.add_argument("--filter", type=str, default="", help="substring filter for chapter scripts")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "chapters"),
        help="output dir for chapter script figures",
    )
    args = ap.parse_args()

    run_code = not args.chapters_only
    run_chapters = not args.code_only and not args.skip_execute

    exit_codes: list[int] = []

    if run_code:
        cmd = [
            sys.executable,
            str(TOOLS / "check_code.py"),
            "--root",
            "code",
            "--max-len",
            str(args.max_len),
        ]
        if args.quiet:
            cmd.append("--quiet")
        if args.dry_run:
            cmd.append("--dry-run")
        exit_codes.append(run_cmd(cmd))

    if run_chapters:
        env = os.environ.copy()
        env["DL_TIMEOUT_SEC"] = str(args.timeout)
        if args.filter:
            env["DL_FILTER"] = args.filter
        if args.output_dir:
            env["DL_OUTPUT_DIR"] = args.output_dir
        cmd = [sys.executable, str(TOOLS / "check_chapter_scripts.py")]
        if args.dry_run:
            cmd.append("--dry-run")
        exit_codes.append(run_cmd(cmd, env=env))

    return 1 if any(code != 0 for code in exit_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
