# Deep Learning with PyTorch — Code Repo Makefile

PY ?= python3
VENV ?= .venv

ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif

.PHONY: help venv install dev jupyter kernel clean run-example

help:
	@echo "Targets:"
	@echo "  venv       Create virtual environment ($(VENV))"
	@echo "  install    Upgrade pip and install requirements.txt"
	@echo "  dev        venv + install (convenience)"
	@echo "  jupyter    Install JupyterLab and ipykernel"
	@echo "  kernel     Register ipykernel named 'dlcode'"
	@echo "  run-example Run a small example script"
	@echo "  clean      Remove caches and temporary files"

venv:
	$(PY) -m venv $(VENV)

install: venv
	"$(BIN)/python" -m pip install --upgrade pip
	"$(BIN)/pip" install -r requirements.txt

dev: install jupyter kernel

jupyter: venv
	"$(BIN)/pip" install jupyterlab ipykernel

kernel: venv
	"$(BIN)/python" -m ipykernel install --user --name dlcode --display-name "Python (dlcode)"

run-example:
	"$(BIN)/python" code/ch01/minimal_regression_sklearn.py || true

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .ipynb_checkpoints 2>/dev/null || true
	rm -f .coverage coverage.xml 2>/dev/null || true

