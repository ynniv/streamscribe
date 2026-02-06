"""Chunked transcription with text deduplication."""

from __future__ import annotations

import numpy as np

from transtream.audio.chunker import AudioChunk
from transtream.exceptions import TranscriptionError

# RMS energy below this threshold is treated as silence
SILENCE_THRESHOLD = 0.005


class ChunkedTranscriber:
    """Transcribes audio chunks and deduplicates overlapping text."""

    def __init__(self, model, verbose: bool = False) -> None:
        self._model = model
        self._verbose = verbose
        self._previous_words: list[str] = []

    def transcribe(self, chunk: AudioChunk) -> str | None:
        """Transcribe an audio chunk and return deduplicated new text.

        Returns None if the chunk is silent or produces no new text.
        """
        # Skip silent chunks
        rms = float(np.sqrt(np.mean(chunk.samples**2)))
        if rms < SILENCE_THRESHOLD:
            return None

        try:
            import torch

            with torch.no_grad():
                results = self._model.transcribe(
                    [chunk.samples], verbose=self._verbose
                )
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        # NeMo returns list of strings or list of Hypothesis objects
        if not results:
            return None

        text = results[0] if isinstance(results[0], str) else results[0].text
        text = text.strip()
        if not text:
            return None

        current_words = text.split()
        new_text = self._deduplicate(current_words)
        self._previous_words = current_words

        return new_text if new_text else None

    def _deduplicate(self, current_words: list[str]) -> str:
        """Remove overlapping text from the beginning of current output.

        Finds the longest suffix of previous_words that matches a prefix of
        current_words, and returns only the non-overlapping portion.
        """
        if not self._previous_words or not current_words:
            return " ".join(current_words)

        prev = self._previous_words
        curr = current_words

        # Find longest suffix of prev that matches a prefix of curr
        best_overlap = 0
        max_check = min(len(prev), len(curr))

        for overlap_len in range(1, max_check + 1):
            suffix = prev[-overlap_len:]
            prefix = curr[:overlap_len]
            if self._words_match(suffix, prefix):
                best_overlap = overlap_len

        if best_overlap > 0:
            new_words = curr[best_overlap:]
        else:
            new_words = curr

        return " ".join(new_words)

    @staticmethod
    def _words_match(a: list[str], b: list[str]) -> bool:
        """Case-insensitive word list comparison."""
        if len(a) != len(b):
            return False
        return all(x.lower() == y.lower() for x, y in zip(a, b))
