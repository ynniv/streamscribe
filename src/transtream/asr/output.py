"""Terminal display formatting for transcription output."""

from __future__ import annotations

import sys


class TranscriptionDisplay:
    """Prints transcription results to stdout with optional timestamps.

    Status messages go to stderr so stdout can be redirected to a file.
    """

    def __init__(self, show_timestamps: bool = True) -> None:
        self._show_timestamps = show_timestamps

    def show_text(
        self, text: str, timestamp: float, speaker: str | None = None
    ) -> None:
        """Print a transcription segment."""
        parts: list[str] = []
        if self._show_timestamps:
            parts.append(f"[{self._format_timestamp(timestamp)}]")
        if speaker:
            parts.append(f"[{speaker}]")
        parts.append(text)
        print(" ".join(parts), flush=True)

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
