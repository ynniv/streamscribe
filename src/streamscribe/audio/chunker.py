"""Sliding window audio chunker with left context overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from streamscribe.audio.decoder import SAMPLE_RATE, AudioDecoder


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""

    samples: np.ndarray  # float32, shape (num_samples,)
    timestamp: float  # start time in seconds (of the non-context portion)
    duration: float  # duration of the non-context portion in seconds
    context_samples: int  # number of leading samples that are left context


class AudioChunker:
    """Yields overlapping audio chunks from an AudioDecoder.

    Each chunk contains `context_duration` seconds of left context from the
    previous chunk, followed by `chunk_duration` seconds of new audio.
    """

    def __init__(
        self,
        decoder: AudioDecoder,
        chunk_duration: float = 5.0,
        context_duration: float = 1.0,
    ) -> None:
        self._decoder = decoder
        self._chunk_duration = chunk_duration
        self._context_duration = context_duration
        self._chunk_samples = int(chunk_duration * SAMPLE_RATE)
        self._context_samples = int(context_duration * SAMPLE_RATE)

    def chunks(self) -> Iterator[AudioChunk]:
        """Yield AudioChunk objects from the decoder stream."""
        previous_tail: np.ndarray | None = None
        position = 0.0  # current time position in seconds

        while True:
            new_audio = self._decoder.read_chunk(self._chunk_samples)
            if new_audio is None:
                break

            actual_new_samples = len(new_audio)

            # Build chunk: left context + new audio
            if previous_tail is not None and len(previous_tail) > 0:
                chunk_audio = np.concatenate([previous_tail, new_audio])
                context_count = len(previous_tail)
            else:
                chunk_audio = new_audio
                context_count = 0

            yield AudioChunk(
                samples=chunk_audio,
                timestamp=position,
                duration=actual_new_samples / SAMPLE_RATE,
                context_samples=context_count,
            )

            position += actual_new_samples / SAMPLE_RATE

            # Save tail of new audio as context for next chunk
            if self._context_samples > 0 and len(new_audio) >= self._context_samples:
                previous_tail = new_audio[-self._context_samples :]
            elif self._context_samples > 0:
                previous_tail = new_audio.copy()
            else:
                previous_tail = None
