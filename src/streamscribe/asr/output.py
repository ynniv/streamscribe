"""Terminal display formatting for transcription output."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


class TranscriptionDisplay:
    """Prints transcription results to stdout with optional timestamps.

    Status messages go to stderr so stdout can be redirected to a file.
    When an output file is specified, transcript lines are written there
    instead of stdout, keeping the terminal free of NeMo noise.
    """

    def __init__(
        self, show_timestamps: bool = True, output_file: str | None = None
    ) -> None:
        self._show_timestamps = show_timestamps
        self._outfile = open(Path(output_file), "w") if output_file else None

    def write_header(
        self,
        title: str,
        url: str,
        stream_type: str,
        duration: float | None,
    ) -> None:
        """Write metadata header to the output file (no-op without one)."""
        if not self._outfile:
            return
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        dur = f"{duration:.0f}s" if duration else "live"
        lines = [
            f"# {title}",
            f"# {url}",
            f"# {stream_type} | {dur} | {date}",
            "",
        ]
        for line in lines:
            print(line, file=self._outfile, flush=True)

    def show_text(
        self, text: str, timestamp: float, speaker: str | None = None
    ) -> None:
        """Print a transcription segment."""
        prefix = ""
        if self._show_timestamps:
            ts = self._format_timestamp(timestamp)
            if speaker:
                prefix = f"[{ts} {speaker}] "
            else:
                prefix = f"[{ts}] "
        elif speaker:
            prefix = f"[{speaker}] "

        line = f"{prefix}{text}"
        print(line, flush=True)
        if self._outfile:
            print(line, file=self._outfile, flush=True)

    def status(self, message: str) -> None:
        """Print a status message to stderr."""
        print(message, file=sys.stderr, flush=True)

    def close(self) -> None:
        """Close the output file if one was opened."""
        if self._outfile:
            self._outfile.close()
            self._outfile = None

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
