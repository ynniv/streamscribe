"""Shared text deduplication for overlapping transcription chunks."""

from __future__ import annotations


class TextDeduplicator:
    """Removes overlapping text between consecutive transcription outputs.

    Finds the longest suffix of the previous transcription that matches
    a prefix of the current transcription, and returns only the new portion.
    """

    def __init__(self) -> None:
        self._previous_words: list[str] = []

    def deduplicate(self, text: str) -> str | None:
        """Remove overlap with previous text and return only new content.

        Returns None if the text is empty or entirely overlapping.
        Updates internal state with the current words for next call.
        """
        text = text.strip()
        if not text:
            return None

        current_words = text.split()
        new_text = self._remove_overlap(current_words)
        self._previous_words = current_words

        return new_text if new_text else None

    def _remove_overlap(self, current_words: list[str]) -> str:
        if not self._previous_words or not current_words:
            return " ".join(current_words)

        prev = self._previous_words
        curr = current_words

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
        if len(a) != len(b):
            return False
        return all(x.lower() == y.lower() for x, y in zip(a, b))
