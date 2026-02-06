"""Decode audio streams to raw PCM via ffmpeg subprocess."""

from __future__ import annotations

import subprocess
import sys
from typing import Self

import numpy as np

from transtream.exceptions import AudioDecodingError

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


class AudioDecoder:
    """Launches ffmpeg to convert an audio URL to raw 16kHz mono PCM.

    Use as a context manager to ensure the subprocess is cleaned up.
    """

    def __init__(self, audio_url: str, is_live: bool = False) -> None:
        self._audio_url = audio_url
        self._is_live = is_live
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        self._start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _start(self) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
        ]
        if self._is_live:
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1"]
        cmd += [
            "-i", self._audio_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise AudioDecodingError(
                "ffmpeg not found on PATH. Install ffmpeg to use transtream."
            )

    def read_chunk(self, num_samples: int) -> np.ndarray | None:
        """Read num_samples from the ffmpeg pipe.

        Returns a float32 numpy array normalized to [-1, 1], or None at EOF.
        """
        if self._process is None or self._process.stdout is None:
            raise AudioDecodingError("Decoder not started — use as context manager")

        num_bytes = num_samples * SAMPLE_WIDTH
        data = self._process.stdout.read(num_bytes)

        if not data:
            return None

        # Convert raw bytes to float32 in [-1, 1]
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def close(self) -> None:
        """Terminate the ffmpeg subprocess."""
        if self._process is not None:
            self._process.stdout.close()  # type: ignore[union-attr]
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

            # Print any ffmpeg errors to stderr for debugging
            stderr_output = self._process.stderr.read()  # type: ignore[union-attr]
            if stderr_output:
                print(
                    f"ffmpeg: {stderr_output.decode(errors='replace').strip()}",
                    file=sys.stderr,
                )
            self._process.stderr.close()  # type: ignore[union-attr]
            self._process = None
