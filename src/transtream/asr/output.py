"""Terminal display formatting for transcription output."""

from __future__ import annotations

import sys


class TranscriptionDisplay:
    """Prints transcription results to stdout with optional timestamps.

    Status messages go to stderr so stdout can be redirected to a file.
    """

    def __init__(self, show_timestamps: bool = True) -> None:
        self._show_timestamps = show_timestamps

    def show_text(self, text: str, timestamp: float) -> None:
        """Print a transcription segment."""
        if self._show_timestamps:
            ts = self._format_timestamp(timestamp)
            print(f"[{ts}] {text}", flush=True)
        else:
            print(text, flush=True)

    def status(self, message: str) -> None:
        """Print a status message to stderr."""
        print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
