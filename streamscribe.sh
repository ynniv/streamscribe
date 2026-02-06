#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-setup on first run
if [ ! -d "$DIR/.venv" ]; then
    "$DIR/setup.sh"
    echo
fi

exec "$DIR/.venv/bin/python" -m streamscribe "$@"
