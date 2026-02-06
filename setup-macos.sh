#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== streamscribe macOS setup ==="
echo "Lightweight install using Apple's built-in speech recognition."
echo "No NeMo/PyTorch download required."
echo

# macOS check
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only."
    echo "Use setup.sh for the full NeMo install on other platforms."
    exit 1
fi

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Attempting to install via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg is required. Install Homebrew (https://brew.sh) then run:"
        echo "  brew install ffmpeg"
        exit 1
    fi
else
    echo "ffmpeg: $(command -v ffmpeg)"
fi

# Check for yt-dlp
if ! command -v yt-dlp &>/dev/null; then
    echo "yt-dlp not found. Attempting to install via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install yt-dlp
    else
        echo "Error: yt-dlp is required. Install with:"
        echo "  brew install yt-dlp"
        exit 1
    fi
else
    echo "yt-dlp: $(command -v yt-dlp)"
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
echo "Installing streamscribe + Apple Speech engine..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -e ".[apple]" --quiet

# Set default engine
echo "engine=apple" > streamscribe.conf

echo
echo "=== Setup complete ==="
echo "Default engine: apple (macOS native)"
echo "Run:  ./streamscribe.sh \"https://www.youtube.com/watch?v=VIDEO_ID\""
echo
echo "The first run will prompt for Speech Recognition permission in System Settings."
