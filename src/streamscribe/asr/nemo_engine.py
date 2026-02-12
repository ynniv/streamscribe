"""NeMo ASR engine — wraps existing model manager and transcriber."""

from __future__ import annotations

import numpy as np

from streamscribe.asr.dedup import TextDeduplicator
from streamscribe.asr.model import ASRModelManager, DEFAULT_MODEL
from streamscribe.asr.transcriber import SILENCE_THRESHOLD
from streamscribe.audio.chunker import AudioChunk
from streamscribe.exceptions import TranscriptionError


class NeMoEngine:
    """ASR engine using NVIDIA NeMo models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self._model_mgr = ASRModelManager(model_name, device)
        self._verbose = verbose
        self._dedup = TextDeduplicator()
        self._model = None

    @property
    def name(self) -> str:
        return f"NeMo ({self._model_mgr._model_name})"

    @property
    def supports_diarization(self) -> bool:
        return True

    @property
    def device(self):
        """Expose the model device for diarizer compatibility."""
        return self._model.device if self._model else "cpu"

    def load(self) -> None:
        self._model_mgr.load()
        self._model = self._model_mgr.model

    def transcribe_samples(self, samples: "np.ndarray") -> str | None:
        """Raw inference on samples — no silence check or dedup."""
        try:
            import torch

            with torch.no_grad():
                results = self._model.transcribe([samples], verbose=self._verbose)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e
        if not results:
            return None
        return results[0] if isinstance(results[0], str) else results[0].text

    def transcribe(self, chunk: AudioChunk) -> str | None:
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

        if not results:
            return None

        text = results[0] if isinstance(results[0], str) else results[0].text
        return self._dedup.deduplicate(text)
