#!/usr/bin/env python3
"""Quick checks for Python code files in code/.

Runs light validation similar to the notebook checker but for .py files:
- Syntax check via compile()
- Line-length check (default 85 chars)
- UTF-8 readability

Outputs a per-file status and a concise summary at the end.

Usage:
  python tools/check_code.py [--root code] [--max-len 85] [--quiet] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXCLUDE_NAMES = {"__pycache__", ".ipynb_checkpoints", ".DS_Store", "outputs", "logs"}


def iter_code_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_NAMES for part in p.parts):
            continue
        files.append(p)
    return files


def is_utf8_readable(p: Path) -> bool:
    try:
        _ = p.read_text(encoding="utf-8")
        return True
    except Exception:
        return False


def check_file(p: Path, *, max_len: int) -> tuple[bool, list[str]]:
    """Returns (ok, messages)."""
    msgs: list[str] = []
    ok = True

    # UTF-8 readability
    if not is_utf8_readable(p):
        ok = False
        msgs.append("Not UTF-8 decodable")
        return ok, msgs  # Bail early; other checks depend on decoding

    text = p.read_text(encoding="utf-8")

    # Syntax check using compile
    try:
        compile(text, str(p), "exec")
    except SyntaxError as e:
        ok = False
        loc = f"line {e.lineno}"
        msgs.append(f"SyntaxError ({loc}): {e.msg}")

    # Line-length check
    long_hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) > max_len:
            long_hits.append(i)
    if long_hits:
        ok = False
        preview = ", ".join(map(str, long_hits[:6]))
        if len(long_hits) > 6:
            preview += ", …"
        msgs.append(f"Long lines >{max_len}: {len(long_hits)} (e.g., {preview})")

    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("code"), help="root folder to scan")
    ap.add_argument("--max-len", type=int, default=85, help="max allowed line length")
    ap.add_argument("--quiet", action="store_true", help="suppress per-file OK lines")
    ap.add_argument("--dry-run", action="store_true", help="list files and exit")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"Root not found: {args.root}", file=sys.stderr)
        return 2

    files = iter_code_files(args.root)
    if args.dry_run:
        print(f"Dry run: {len(files)} file(s) under {args.root}")
        for p in sorted(files):
            print(p)
        return 0
    total = len(files)
    errors = 0
    long_viol = 0

    for p in sorted(files):
        ok, msgs = check_file(p, max_len=args.max_len)
        if ok:
            if not args.quiet:
                print(f"OK   {p}")
        else:
            errors += 1
            # Count long-line messages for summary
            for m in msgs:
                if m.startswith("Long lines"):
                    # parse count between ':' and '(' -> simplistic
                    try:
                        count_str = m.split(":", 1)[1].split("(")[0].strip()
                        long_viol += int(count_str)
                    except Exception:
                        pass
            print(f"FAIL {p}")
            for m in msgs:
                print(f"  - {m}")

    print(f"Summary: checked {total} files. Failures: {errors}. Long-line violations: {long_viol}.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
