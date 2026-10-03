#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python}" -m poise.eval "$@"
