#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-setup on first run
if [ ! -d "$DIR/.venv" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        "$DIR/setup-macos.sh"
    else
        "$DIR/setup-nemo.sh"
    fi
    echo
fi

exec "$DIR/.venv/bin/python" -m streamscribe "$@"
