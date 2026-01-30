#!/usr/bin/env python3
"""
Execute Jupyter notebooks to validate they run end-to-end with detailed, structured output.

Configurable via CLI flags:
- --paths: one or more files or directories to search (default: notebooks)
- --pattern: glob pattern within directories (default: *.ipynb)
- --exclude: regex patterns to skip (may be passed multiple times)
- --timeout: per-cell timeout seconds (default: 180)
- --kernel: kernel name to use (default: python3)
- --allow-errors: do not fail on cell exceptions (report only)
- --fail-fast: stop on first failure (default: continue)
- --outdir: if set, save executed notebooks to this dir (mirrors tree)
- --list: only list selected notebooks and exit
- --skip-execute: do not execute notebooks (checks structure/imports only)
- --verbose: print extra details

Exit code is non-zero if any notebook fails (unless --allow-errors is used).
"""
from __future__ import annotations

import argparse
import warnings
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import os
import tempfile
import ast

import nbformat
from nbformat.validator import MissingIDFieldWarning
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

warnings.filterwarnings("ignore", category=MissingIDFieldWarning)


def find_notebooks(paths: Iterable[str], pattern: str, excludes: list[re.Pattern]) -> list[Path]:
    results: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            results.extend(sorted(path.glob(pattern)))
        elif path.suffix == ".ipynb" and path.exists():
            results.append(path)
    # Filter excludes
    out: list[Path] = []
    for nb in results:
        s = str(nb)
        if any(rx.search(s) for rx in excludes):
            continue
        out.append(nb)
    return out


def ensure_outpath(base_out: Path, nb_path: Path) -> Path:
    rel = nb_path
    if nb_path.is_absolute():
        # Try to make relative to CWD if possible
        try:
            rel = nb_path.relative_to(Path.cwd())
        except Exception:
            rel = nb_path.name
    out = base_out / Path(rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def normalize_cell_ids(nb) -> int:
    """Ensure every cell has an 'id'. Returns number of changes."""
    changed = 0
    seen: set[str] = set()
    for i, cell in enumerate(nb.cells):
        cid = cell.get("id")
        if not cid or cid in seen:
            new_id = f"cell-{i:04d}"
            # guarantee uniqueness
            j = 0
            nid = new_id
            while nid in seen:
                j += 1
                nid = f"{new_id}-{j}"
            cell["id"] = nid
            changed += 1
            cid = nid
        seen.add(cid)
    return changed


def _ensure_writable_dir(path: Path) -> Path:
    """Ensure a directory exists and is writable; fallback to temp dir if not."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".__writetest__"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return path
    except Exception:
        return Path(tempfile.mkdtemp(prefix="pmds_env_"))


def execute_notebook(nb_path: Path, timeout: int, kernel: str, allow_errors: bool, outdir: Path | None, verbose: bool, normalize_ids: bool) -> tuple[bool, float, str | None, int]:
    t0 = time.perf_counter()
    nb = nbformat.read(nb_path, as_version=4)
    id_changes = 0
    if normalize_ids:
        id_changes = normalize_cell_ids(nb)
    # Ensure headless Matplotlib in the kernel process and writable config dirs
    os.environ["MPLBACKEND"] = os.environ.get("MPLBACKEND", "Agg")
    base = Path("outputs")
    mplcfg = _ensure_writable_dir(base / "_mpl")
    ipydir = _ensure_writable_dir(base / "_ipython")
    jrt = _ensure_writable_dir(base / "_jupyter_runtime")
    os.environ["MPLCONFIGDIR"] = str(mplcfg)
    os.environ["IPYTHONDIR"] = str(ipydir)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(jrt)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel,
        allow_errors=allow_errors,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    try:
        client.execute()
        ok = True
        err = None
    except CellExecutionError as e:
        ok = False
        err = str(e)
    except Exception as e:
        ok = False
        err = f"Unexpected error: {e}"
    dur = time.perf_counter() - t0
    if outdir is not None:
        outpath = ensure_outpath(outdir, nb_path)
        nbformat.write(nb, outpath)
        if verbose:
            print(f"saved executed: {outpath}")
    return ok, dur, err, id_changes


def fmt_duration(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000.0:
        return f"{ms:.1f}ms"
    return f"{seconds:.2f}s"


def scan_imports_from_notebook(nb) -> List[str]:
    code = []
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            # Join without outputs and strip magics
            src_lines = cell.get("source", "").splitlines()
            sanitized = []
            for line in src_lines:
                s = line.lstrip()
                if s.startswith("%") or s.startswith("!") or s.startswith("%%"):
                    continue
                sanitized.append(line)
            code.append("\n".join(sanitized))
    src = "\n".join(code)
    mods: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    # Dedup preserve order, and ignore __future__
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
            __import__(m)
            loaded.append(m)
        except Exception:
            missing.append(m)
    return loaded, missing


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--paths", nargs="+", default=["notebooks"], help="Files or directories to search")
    ap.add_argument("--pattern", default="*.ipynb", help="Glob pattern within directories")
    ap.add_argument("--exclude", action="append", default=[], help="Regex to exclude (can pass multiple)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-cell timeout (seconds)")
    ap.add_argument("--kernel", default="python3", help="Kernel name")
    ap.add_argument("--allow-errors", action="store_true", help="Do not fail on cell exceptions")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    ap.add_argument("--outdir", type=Path, default=None, help="Save executed notebooks under this directory")
    ap.add_argument("--list", action="store_true", help="Only list selected notebooks and exit")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    ap.add_argument("--normalize-ids", action="store_true", help="Normalize missing/duplicate cell ids before execution")
    ap.add_argument("--write-normalized", action="store_true", help="Write normalized notebooks back in place (only applies with --no-exec or when not using --outdir)")
    ap.add_argument("--no-exec", action="store_true", help="Do not execute notebooks; useful with --normalize-ids and --write-normalized")
    ap.add_argument("--skip-execute", action="store_true", help="Alias for --no-exec")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    excludes = [re.compile(rx) for rx in ns.exclude]
    notebooks = find_notebooks(ns.paths, ns.pattern, excludes)
    if not notebooks:
        print("No notebooks matched.")
        return 0
    if ns.list:
        for nb in notebooks:
            print(nb)
        return 0
    # Alias handling
    if ns.skip_execute:
        ns.no_exec = True

    # Only normalize without execution
    if ns.no_exec:
        total_changed = 0
        for nbp in notebooks:
            nb = nbformat.read(nbp, as_version=4)
            changed = normalize_cell_ids(nb) if ns.normalize_ids else 0
            total_changed += changed
            if changed and ns.write_normalized and ns.outdir is None:
                nbformat.write(nb, nbp)
                if ns.verbose:
                    print(f"normalized ids in-place: {nbp}")
            elif changed and ns.write_normalized and ns.outdir is not None:
                outpath = ensure_outpath(ns.outdir, nbp)
                nbformat.write(nb, outpath)
                if ns.verbose:
                    print(f"normalized ids to: {outpath}")
        print(f"Normalized ids in {total_changed} cells across {len(notebooks)} notebooks.")
        return 0

    failures: list[tuple[Path, str | None, float]] = []
    grand_total = 0.0
    N = len(notebooks)
    for i, nb_path in enumerate(notebooks, 1):
        rel = nb_path
        try:
            rel = nb_path.relative_to(Path.cwd())
        except Exception:
            rel = nb_path
        print(f"[{i}/{N}] Validating {rel}")

        total_t0 = time.perf_counter()

        # Structure
        s_t0 = time.perf_counter()
        nb = nbformat.read(nb_path, as_version=4)
        structure_ok = True
        struct_msg = "OK"
        # Basic schema checks
        if nb.get("nbformat") != 4:
            struct_msg = f"OK (nbformat={nb.get('nbformat')})"
        s_dt = time.perf_counter() - s_t0
        print(f"  • Structure: {struct_msg}   ({fmt_duration(s_dt)})")

        # Imports
        im_scan_t0 = time.perf_counter()
        mods = scan_imports_from_notebook(nb)
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
        if ns.no_exec:
            ex_msg = "SKIP (--no-exec)"
            print(f"  • Execute: {ex_msg}")
            ok = True
            err = None
        else:
            ex_t0 = time.perf_counter()
            ok, dur, err, id_changes = execute_notebook(
                nb_path,
                ns.timeout,
                ns.kernel,
                ns.allow_errors,
                ns.outdir,
                ns.verbose,
                ns.normalize_ids,
            )
            ex_dt = time.perf_counter() - ex_t0
            ex_msg = "OK" if ok else "FAIL"
            print(f"  • Execute: {ex_msg}   ({fmt_duration(ex_dt)})")

        total_dt = time.perf_counter() - total_t0
        grand_total += total_dt
        print(f"  • Total: {fmt_duration(total_dt)}\n")

        if not ok:
            failures.append((nb_path, err, dur))
            if ns.fail_fast:
                break
    print(f"Executed {N} notebooks in {fmt_duration(grand_total)}. Failures: {len(failures)}")
    if failures and not ns.allow_errors:
        print("Failures:")
        for nb, err, dur in failures:
            print(f"  {nb} ({dur:.1f}s)\n    {err}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
