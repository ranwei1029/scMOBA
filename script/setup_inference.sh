#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${SCMOBA_PYTHON:-python3.11}"
venv="${SCMOBA_VENV:-.venv}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12),(3,13)], "Use Python 3.11, 3.12 or 3.13"'
test ! -e "$venv" || { echo "Environment already exists: $venv" >&2; exit 1; }
unset PYTHONPATH
export PYTHONNOUSERSITE=1
"$python_bin" -m venv "$venv"
"$venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
"$venv/bin/python" -m pip install --index-url https://pypi.org/simple -r requirements-inference.txt
"$venv/bin/python" -m pip check
"$venv/bin/python" -c 'import torch, transformers, peft, datasets, anndata; print("CUDA available:", torch.cuda.is_available()); print("" if torch.cuda.is_available() else "Installation checks passed. Run GPU inference on a host with a visible compatible CUDA GPU.")'
