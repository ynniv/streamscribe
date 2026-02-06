"""NeMo ASR model loading and device management."""

from __future__ import annotations

import sys

from transtream.exceptions import ModelLoadError

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ASRModelManager:
    """Loads and manages a NeMo ASR model."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str = "auto"
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def _resolve_device(self) -> str:
        """Determine the best available device."""
        if self._device != "auto":
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def load(self) -> None:
        """Load the NeMo ASR model."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ModelLoadError(
                "NeMo ASR toolkit is required. Install with:\n"
                '  pip install -e ".[cpu]"  (for CPU)\n'
                '  pip install -e ".[gpu]"  (for CUDA GPU)'
            )

        device = self._resolve_device()
        print(f"Loading model {self._model_name} on {device}...", file=sys.stderr)

        try:
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._model_name
            )
            model = model.to(device)
            model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

        self._model = model
        print("Model loaded.", file=sys.stderr)

    @property
    def model(self):
        """Return the loaded model, raising if not yet loaded."""
        if self._model is None:
            raise ModelLoadError("Model not loaded — call load() first")
        return self._model
