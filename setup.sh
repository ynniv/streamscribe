#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== transtream setup ==="
echo

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg is required and Homebrew is not available to install it."
        echo "Install ffmpeg manually, then re-run this script."
        exit 1
    fi
else
    echo "ffmpeg: $(command -v ffmpeg)"
fi

# Create virtualenv
if [ ! -d .venv ]; then
    echo
    echo "Creating virtualenv..."
    python3 -m venv .venv
else
    echo "Virtualenv already exists."
fi

echo
echo "Installing transtream + dependencies (this may take a few minutes)..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

# Fix onnx/protobuf version conflict (onnx wheels require protobuf 6.x gencode)
echo "Fixing protobuf compatibility..."
.venv/bin/pip install "protobuf>=6.0" --quiet

echo
echo "=== Setup complete ==="
echo "Run:  ./transtream.sh \"https://www.youtube.com/watch?v=VIDEO_ID\""
