#!/usr/bin/env python3
# Deep Learning with PyTorch
# (c) Dr. Yves J. Hilpisch
# AI-powered by GPT-5.x
"""Run chapter scripts to verify they execute without error.

This discovers `code/chNN/*.py` and runs each using a small wrapper that
enforces a headless Matplotlib backend and intercepts `plt.show()` to
auto-save all open figures to a non-tracked folder, then close them. This
prevents GUI windows from blocking execution.

Environment variables (overridden by CLI flags if provided):
- `DL_TIMEOUT_SEC`  per-script timeout (default: 120)
- `DL_FILTER`       substring filter on filenames (e.g. "ch07")
- `DL_OUTPUT_DIR`   where to save figures (default: `outputs/chapters`)

CLI flags:
- `--timeout`, `--filter`, `--output-dir`, `--dry-run`
"""
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import Iterable, Tuple, List
import time
import ast
import importlib


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "code"


def iter_scripts() -> Iterable[Path]:
    if not SCRIPTS_ROOT.exists():
        return []
    for sub in sorted(SCRIPTS_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("ch"):
            continue
        if len(sub.name) < 3 or not sub.name[2:4].isdigit():
            continue
        yield from sorted(sub.rglob("*.py"))


def run_script(path: Path, timeout: float, outdir: Path) -> tuple[int, str]:
    runner_code = f"""
import os, sys, runpy
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
outdir = r"{str(outdir)}"
os.makedirs(outdir, exist_ok=True)
_counter = {{'i': 0}}
script_path = sys.argv[1]
base = os.path.splitext(os.path.basename(script_path))[0]

def _save_show(*a, **k):
    _counter['i'] += 1
    for num in plt.get_fignums():
        fn = os.path.join(outdir, f"{{base}}_{{_counter['i']:03d}}_{{num}}.png")
        plt.figure(num).savefig(fn, dpi=200)
    plt.close('all')

plt.show = _save_show
sys.argv = [script_path]
runpy.run_path(script_path, run_name='__main__')
"""

    runner_path = ROOT / "outputs" / "_runner.py"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(runner_code, encoding="utf-8")

    cmd = [sys.executable, str(runner_path), str(path)]
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


def fmt_duration(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000.0:
        return f"{ms:.0f}ms"
    return f"{seconds:.2f}s"


def scan_imports(src: str) -> List[str]:
    mods: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for m in mods:
        if m not in seen and m not in ("__future__",):
            seen.add(m)
            uniq.append(m)
    return uniq


def probe_imports(mods: List[str]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    loaded: List[str] = []
    for m in mods:
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception:
            missing.append(m)
    return loaded, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=None, help="per-script timeout in seconds")
    ap.add_argument("--filter", type=str, default=None, help="substring filter on filenames")
    ap.add_argument("--output-dir", type=str, default=None, help="where to save figures")
    ap.add_argument("--dry-run", action="store_true", help="list scripts and exit")
    args = ap.parse_args(argv)

    timeout = args.timeout if args.timeout is not None else float(os.getenv("DL_TIMEOUT_SEC", "120"))
    filt = args.filter if args.filter is not None else os.getenv("DL_FILTER", "")
    outdir = Path(
        args.output_dir
        if args.output_dir is not None
        else os.getenv("DL_OUTPUT_DIR", str(ROOT / "outputs" / "chapters"))
    )

    paths = [p for p in iter_scripts() if (filt in str(p.relative_to(ROOT)))]
    if not paths:
        print(f"No chapter scripts found under {SCRIPTS_ROOT / 'chNN'}")
        return 0
    if args.dry_run:
        print(f"Dry run: {len(paths)} script(s) under {SCRIPTS_ROOT}")
        for p in paths:
            print(p.relative_to(ROOT))
        print(f"Output dir: {outdir}")
        return 0

    print(f"Running {len(paths)} script(s) with timeout={timeout:.0f}s each\n")
    failures: list[tuple[Path, int]] = []

    for i, path in enumerate(paths, 1):
        rel = path.relative_to(ROOT)
        print(f"[{i}/{len(paths)}] Validating {rel}")

        total_t0 = time.perf_counter()

        # Structure: AST parse and basic header check
        src = path.read_text(encoding="utf-8", errors="replace")
        s_t0 = time.perf_counter()
        structure_ok = True
        struct_msg = "OK"
        try:
            ast.parse(src)
            # Optional: check for meta header
            head = "\n".join(src.splitlines()[:6])
            if "Deep Learning with PyTorch" not in head:
                struct_msg = "OK (missing meta header)"
        except SyntaxError as e:
            structure_ok = False
            struct_msg = f"FAIL (syntax error: {e.msg})"
        s_dt = time.perf_counter() - s_t0
        print(f"  • Structure: {struct_msg}   ({fmt_duration(s_dt)})")

        # Imports: scan and probe
        im_scan_t0 = time.perf_counter()
        mods = scan_imports(src)
        im_scan_dt = time.perf_counter() - im_scan_t0
        im_probe_t0 = time.perf_counter()
        loaded, missing = probe_imports(mods)
        im_probe_dt = time.perf_counter() - im_probe_t0
        if missing:
            im_msg = f"FAIL (missing: {', '.join(sorted(set(missing)))})"
        else:
            im_msg = "OK"
        print(
            f"  • Imports: {im_msg}   ({fmt_duration(im_scan_dt)} scan, {fmt_duration(im_probe_dt)} probe)"
        )

        # Execute
        ex_t0 = time.perf_counter()
        if structure_ok:
            code, out = run_script(path, timeout, outdir)
            ex_dt = time.perf_counter() - ex_t0
            ex_msg = "OK" if code == 0 else f"FAIL (code={code})"
        else:
            code, out = (1, "structure failed; execution skipped")
            ex_dt = time.perf_counter() - ex_t0
            ex_msg = "SKIP (structure failed)"
        print(f"  • Execute: {ex_msg}   ({fmt_duration(ex_dt)})")

        total_dt = time.perf_counter() - total_t0
        print(f"  • Total: {fmt_duration(total_dt)}\n")

        if code != 0:
            failures.append((path, code))
            # keep the output concise; show only on failure
            print("----- output (truncated to last 200 lines) -----")
            lines = out.splitlines()[-200:]
            print("\n".join(lines))
            print("----- end output -----\n")

    if failures:
        print("Summary: FAIL")
        for path, code in failures:
            print(f"  - {path.name}: exit {code}")
        return 1
    else:
        print("Summary: OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
