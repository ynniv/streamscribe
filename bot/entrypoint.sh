#!/usr/bin/env bash
set -euo pipefail

# --- PulseAudio setup ---
export XDG_RUNTIME_DIR="/tmp/pulse-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
rm -f "$XDG_RUNTIME_DIR/pulse/pid"

pulseaudio --disable-shm=true --exit-idle-time=-1 2>/dev/null &
PA_PID=$!
sleep 1

export PULSE_SERVER="unix:${XDG_RUNTIME_DIR}/pulse/native"
echo "PulseAudio ready"

# --- Virtual display (Chromium needs a display to output audio) ---
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp 2>/dev/null &
XVFB_PID=$!
export DISPLAY=:99
sleep 1

# --- Cleanup ---
cleanup() {
    echo "Shutting down..."
    kill "$PA_PID" 2>/dev/null || true
    kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

# --- Start API server (loads model, auto-joins rooms from env) ---
exec python bot/server.py
