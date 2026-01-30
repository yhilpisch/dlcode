#!/usr/bin/env python3
"""Validate figure-generator scripts for 5_dltex.

Runs two checks:
1) check_code.py on code/figures (syntax + line-length)
2) Execute all scripts under code/figures

Usage:
  python tools/validate_figures.py [--timeout 120] [--filter ch07] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "code" / "figures"
TOOLS = Path(__file__).resolve().parent


def iter_scripts() -> list[Path]:
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(SCRIPTS_DIR.glob("*.py"))


def fmt_duration(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000.0:
        return f"{ms:.0f}ms"
    return f"{seconds:.2f}s"


def run_script(path: Path, timeout: float) -> tuple[int, str]:
    cmd = [sys.executable, str(path)]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    mplcfg = ROOT / "outputs" / "_mpl"
    mplcfg.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mplcfg))
    try:
        proc = sp.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout
    except sp.TimeoutExpired as exc:
        return 124, f"TIMEOUT after {timeout}s\n{exc.output or ''}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true", help="run code checks only")
    ap.add_argument("--scripts-only", action="store_true", help="run scripts only")
    ap.add_argument("--skip-execute", action="store_true", help="skip figure script execution")
    ap.add_argument("--max-len", type=int, default=85, help="max line length for check_code")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-script timeout in seconds")
    ap.add_argument("--filter", type=str, default="", help="substring filter on filenames")
    ap.add_argument("--dry-run", action="store_true", help="list scripts and exit")
    args = ap.parse_args()

    paths = [p for p in iter_scripts() if args.filter in p.name]
    if not paths:
        print(f"No figure scripts found in {SCRIPTS_DIR}")
        return 0
    if args.dry_run:
        print(f"Dry run: {len(paths)} script(s) in {SCRIPTS_DIR}")
        for p in paths:
            print(p.relative_to(ROOT))
        return 0

    run_code = not args.scripts_only
    run_scripts = not args.code_only and not args.skip_execute
    exit_codes: list[int] = []

    if run_code:
        cmd = [
            sys.executable,
            str(TOOLS / "check_code.py"),
            "--root",
            str(SCRIPTS_DIR),
            "--max-len",
            str(args.max_len),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        exit_codes.append(sp.run(cmd, cwd=ROOT, text=True).returncode)

    if not run_scripts:
        return 1 if any(code != 0 for code in exit_codes) else 0

    print(f"Running {len(paths)} figure script(s) with timeout={args.timeout:.0f}s each\n")
    failures: list[tuple[Path, int]] = []

    for i, path in enumerate(paths, 1):
        rel = path.relative_to(ROOT)
        print(f"[{i}/{len(paths)}] {rel}")
        t0 = time.perf_counter()
        code, out = run_script(path, args.timeout)
        dt = time.perf_counter() - t0
        msg = "OK" if code == 0 else f"FAIL (code={code})"
        print(f"  • Execute: {msg}   ({fmt_duration(dt)})\n")
        if code != 0:
            failures.append((path, code))
            print("----- output (truncated to last 200 lines) -----")
            lines = out.splitlines()[-200:]
            print("\n".join(lines))
            print("----- end output -----\n")

    if failures:
        print("Summary: FAIL")
        for path, code in failures:
            print(f"  - {path.name}: exit {code}")
        return 1
    if any(code != 0 for code in exit_codes):
        return 1
    print("Summary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
