# Validation Tools Guide

# Deep Learning with PyTorch
# (c) Dr. Yves J. Hilpisch
# AI-powered by GPT-5.x

This folder contains validators to help you run and verify all code assets non‑interactively.

## 1) Validate Chapter Scripts

Script: `tools/validate_code.py`

What it does:
- Runs code checks (syntax + line length) across `code/`.
- Executes `code/chNN/*.py` headlessly and captures errors.

Usage examples:

  python tools/validate_code.py

  python tools/validate_code.py --skip-execute

  python tools/validate_code.py --filter ch07

Options:
- `--skip-execute`: skip running scripts (only lint).
- `--max-len`: line-length limit (default: 85).
- `--filter`: substring filter for chapter scripts.
- `--dry-run`: list what would run.

## 2) Validate Figure Scripts

Script: `tools/validate_figures.py`

What it does:
- Runs code checks (syntax + line length) across `code/figures`.
- Executes all figure scripts headlessly and captures errors.

Usage examples:

  python tools/validate_figures.py

  python tools/validate_figures.py --skip-execute

  python tools/validate_figures.py --filter ch10

Notes:
- Figure scripts write outputs into `figures/` (often SVG; some scripts emit PNG/PDF).

## 3) Validate Notebooks

Script: `tools/validate_notebooks.py`

What it does:
- Finds notebooks (default: `notebooks/*.ipynb`).
- Scans structure/imports and optionally executes notebooks.

Usage examples:

  python tools/validate_notebooks.py

  python tools/validate_notebooks.py --skip-execute --normalize-ids --write-normalized

Options:
- `--skip-execute`: skip execution (structure/imports only).
- `--timeout`: per-cell timeout seconds (default: 180).
- `--kernel`: kernel name (default: `python3`).
- `--exclude`: regex to skip certain notebooks (repeatable).
- `--normalize-ids`: normalize missing/duplicate cell IDs.
- `--write-normalized`: write normalized notebooks.

Notes:
- The validators configure `MPLBACKEND=Agg`, `MPLCONFIGDIR`, `IPYTHONDIR`, and
  `JUPYTER_RUNTIME_DIR` to writable locations automatically.
- If you run into missing packages, install them via `pip install -r requirements.txt`.

## Tips

- Prefer running from the repository root for consistent relative paths.
- For headless servers, no GUI is required. Figures are saved automatically.
