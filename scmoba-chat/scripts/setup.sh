#!/usr/bin/env bash
set -euo pipefail
# Linux/WSL2. Never modify an existing training environment.
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONNOUSERSITE=1
SCMOBA_VENV="${SCMOBA_VENV:-.venv}"
SCMOBA_PYTHON="${SCMOBA_PYTHON:-python3}"
if [ -e "$SCMOBA_VENV" ]; then
  echo "Refusing to overwrite existing environment: $SCMOBA_VENV" >&2
  echo "Choose a new path with SCMOBA_VENV=.venv-new bash scripts/setup.sh" >&2
  exit 1
fi
"$SCMOBA_PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Python 3.11-3.13 required"'
"$SCMOBA_PYTHON" -m venv "$SCMOBA_VENV"
"$SCMOBA_VENV/bin/python" -m pip install '.[test]'
"$SCMOBA_VENV/bin/scmoba-chat" doctor
"$SCMOBA_VENV/bin/python" -m pytest -q tests
