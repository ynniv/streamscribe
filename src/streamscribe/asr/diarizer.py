"""Simple online speaker diarization using TitaNet embeddings."""

from __future__ import annotations

import sys

import numpy as np

from streamscribe.asr.transcriber import SILENCE_THRESHOLD
from streamscribe.exceptions import ModelLoadError

# Cosine similarity threshold for matching a known speaker
_SIMILARITY_THRESHOLD = 0.5


class SpeakerDiarizer:
    """Identifies speakers by comparing TitaNet embeddings via cosine similarity.

    Maintains a running average centroid per speaker and assigns new labels
    ("Speaker 1", "Speaker 2", ...) when no existing centroid is close enough.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model = None
        # List of (label, centroid_embedding, count) tuples
        self._speakers: list[tuple[str, np.ndarray, int]] = []

    def load(self) -> None:
        """Load the TitaNet speaker embedding model."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ModelLoadError(
                "NeMo ASR toolkit is required for speaker detection."
            )

        print("Loading speaker embedding model...", file=sys.stderr, flush=True)
        try:
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet_small"
            )
            model = model.to(self._device)
            model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load speaker model: {e}") from e

        self._model = model
        print("Speaker model loaded.", file=sys.stderr, flush=True)

    def identify(self, samples: np.ndarray) -> str | None:
        """Extract a speaker embedding and return a speaker label.

        Returns None for silent chunks.
        """
        rms = float(np.sqrt(np.mean(samples**2)))
        if rms < SILENCE_THRESHOLD:
            return None

        embedding = self._extract_embedding(samples)
        if embedding is None:
            return None

        return self._assign_speaker(embedding)

    def _extract_embedding(self, samples: np.ndarray) -> np.ndarray | None:
        """Run TitaNet forward pass to get a speaker embedding vector."""
        import torch

        with torch.no_grad():
            audio_tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
            audio_len = torch.tensor([samples.shape[0]], dtype=torch.long)

            audio_tensor = audio_tensor.to(self._device)
            audio_len = audio_len.to(self._device)

            _, embedding = self._model.forward(
                input_signal=audio_tensor, input_signal_length=audio_len
            )

        emb = embedding.squeeze().cpu().numpy()
        # Normalize
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        return emb / norm

    def _assign_speaker(self, embedding: np.ndarray) -> str:
        """Match embedding to known speakers or create a new one."""
        best_idx = -1
        best_sim = -1.0

        for i, (_, centroid, _) in enumerate(self._speakers):
            sim = float(np.dot(embedding, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= _SIMILARITY_THRESHOLD:
            # Update running average centroid
            label, centroid, count = self._speakers[best_idx]
            new_count = count + 1
            new_centroid = (centroid * count + embedding) / new_count
            # Re-normalize
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            self._speakers[best_idx] = (label, new_centroid, new_count)
            return label

        # New speaker
        label = f"Speaker {len(self._speakers) + 1}"
        self._speakers.append((label, embedding.copy(), 1))
        return label
