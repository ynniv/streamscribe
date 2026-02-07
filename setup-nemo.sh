#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== streamscribe NeMo setup ==="
echo

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Attempting to install..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg is required and no supported package manager was found."
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

# Detect CUDA
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "CUDA GPU detected — installing with GPU support..."
    VARIANT="gpu"
    EXTRA_INDEX=""
else
    echo "No CUDA GPU detected — installing CPU-only..."
    VARIANT="cpu"
    EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
fi

echo "Installing streamscribe + dependencies (this may take a few minutes)..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -e ".[$VARIANT]" $EXTRA_INDEX

# onnx's PyPI wheel was compiled with protobuf 6.x gencode, but NeMo pins
# protobuf~=5.29.  Force protobuf 6.x — NeMo works fine with it at runtime.
.venv/bin/pip install "protobuf>=6.0" --quiet

# Set default engine
echo "engine=nemo" > streamscribe.conf

echo
echo "=== Setup complete ==="
echo "Default engine: nemo"
echo "Run:  ./streamscribe.sh \"https://www.youtube.com/watch?v=VIDEO_ID\""
