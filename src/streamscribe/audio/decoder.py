"""Decode audio streams to raw PCM via ffmpeg subprocess."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import Self

import numpy as np

from streamscribe.exceptions import AudioDecodingError

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

DEVICE_PREFIX = "device:"


def _is_device_input(audio_url: str) -> bool:
    """Check whether audio_url refers to a local audio device."""
    return audio_url.startswith(DEVICE_PREFIX)


def _parse_device_id(audio_url: str) -> str:
    """Extract the device identifier from a device: URL."""
    return audio_url[len(DEVICE_PREFIX):]


def _list_devices_darwin() -> str:
    """Parse avfoundation device list and return only audio devices."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise AudioDecodingError(
            "ffmpeg not found on PATH. Install ffmpeg to use streamscribe."
        )
    # Parse audio devices from stderr — lines like:
    #   [AVFoundation indev @ 0x...] [0] MacBook Pro Microphone
    in_audio = False
    devices: list[str] = []
    for line in result.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if in_audio:
            m = re.search(r"\[(\d+)]\s+(.+)$", line)
            if m:
                devices.append(f"  [{m.group(1)}] {m.group(2)}")
            else:
                break  # end of device list
    if not devices:
        return "No audio input devices found."
    return "Audio input devices:\n" + "\n".join(devices)


def _list_devices_linux() -> str:
    """List audio sources via pactl or arecord."""
    for cmd_args in (
        ["pactl", "list", "sources", "short"],
        ["arecord", "-l"],
    ):
        try:
            result = subprocess.run(
                cmd_args, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            continue
    raise AudioDecodingError(
        "Could not list audio devices. Install pulseaudio or alsa-utils."
    )


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
        if _is_device_input(self._audio_url):
            cmd = self._build_device_cmd(_parse_device_id(self._audio_url))
        else:
            cmd = self._build_url_cmd()
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise AudioDecodingError(
                "ffmpeg not found on PATH. Install ffmpeg to use streamscribe."
            )

    def _build_url_cmd(self) -> list[str]:
        """Build ffmpeg command for a URL or file input."""
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
        return cmd

    def _build_device_cmd(self, device_id: str) -> list[str]:
        """Build ffmpeg command to capture from a local audio device."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
        ]
        system = platform.system()
        if system == "Darwin":
            # avfoundation uses ":N" for audio-only (no video)
            cmd += ["-f", "avfoundation", "-i", f":{device_id}"]
        elif system == "Linux":
            # Try PulseAudio device name; user can also pass an ALSA hw: id
            if device_id == "default":
                cmd += ["-f", "pulse", "-i", "default"]
            elif device_id.startswith("hw:"):
                cmd += ["-f", "alsa", "-i", device_id]
            else:
                cmd += ["-f", "pulse", "-i", device_id]
        else:
            raise AudioDecodingError(
                f"Audio device capture is not supported on {system}."
            )
        cmd += [
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]
        return cmd

    @staticmethod
    def list_devices() -> str:
        """List available audio input devices.

        Returns a cleaned-up, human-readable string for display.
        """
        system = platform.system()
        if system == "Darwin":
            return _list_devices_darwin()
        elif system == "Linux":
            return _list_devices_linux()
        else:
            raise AudioDecodingError(
                f"Audio device listing is not supported on {system}."
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
