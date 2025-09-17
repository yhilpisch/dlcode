<p align="center">
  <img src="https://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="300" />
</p>

# Deep Learning with PyTorch — Code Repository

This repository accompanies the book “Deep Learning Basics with PyTorch”. It contains runnable Python scripts and Jupyter notebooks that mirror the book’s content and examples.

- Book (HTML): https://hilpisch.com/tae/dl.html
- Repository: https://github.com/yhilpisch/dlcode
- Author: Dr. Yves J. Hilpisch — The Python Quants GmbH

## What’s Inside

- `code/` — Stand‑alone Python scripts organized by chapter (e.g., `ch05/`, `ch14/`).
- `notebooks/` — Per‑chapter notebooks designed to be Colab‑friendly.
- `requirements.txt` — Core Python dependencies used across examples.

Notes:
- PyTorch is intentionally not pinned in `requirements.txt` due to platform‑specific builds. See install instructions below.
- Figures and manuscript sources are not included here to keep the repo lean and focused on code.

## Quickstart (Local)

1) Clone and create a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install core dependencies

```bash
pip install -r requirements.txt
```

3) Install PyTorch for your platform

- Visit https://pytorch.org/get-started/locally/ and follow the selector for your OS, Python, CUDA/Metal/CPU.
- Examples (subject to change; prefer the official selector):
  - CPU‑only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - CUDA 12.x: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

4) Run code and notebooks

```bash
# Scripts
python code/ch01/minimal_regression_sklearn.py

# Notebooks (locally)
pip install jupyterlab ipykernel
python -m ipykernel install --user --name dlcode
jupyter lab
```

## Using Google Colab

- Open a notebook from `notebooks/` in Colab (upload or via GitHub once the repo is public).
- Set runtime hardware to GPU if needed: Runtime → Change runtime type → Hardware accelerator → GPU.
- Install dependencies at the top of the notebook cell:

```python
!pip -q install -r https://raw.githubusercontent.com/yhilpisch/dlcode/main/requirements.txt
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or CPU index
```

Tips:
- Colab sessions are ephemeral: save work to Google Drive (e.g., `from google.colab import drive; drive.mount('/content/drive')`).
- Large models/datasets may require additional storage or runtime configuration.

## Repository Structure and Conventions

- Scripts are self‑contained for clarity and instructional value; they avoid hidden side effects.
- Notebooks favor visualization and step‑by‑step exploration; each cell should be runnable top‑to‑bottom.
- Reproducibility: examples set seeds where helpful; behavior may still vary across hardware/backends.
- When adapting to your data, start from the minimal patterns (training loops, dataloaders) and iterate.

## Disclaimer

This repository and its contents are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind—express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, medical advice, or production deployments without rigorous review, testing, and validation.

Some examples may reference third‑party datasets, services, or APIs subject to their own licenses and terms; you are responsible for ensuring compliance.

## Contact

- Email: team@tpq.io
- Linktree: https://linktr.ee/dyjh
