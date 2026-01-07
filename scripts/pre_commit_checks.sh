#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  uv sync --group dev
fi

make format
make lint
