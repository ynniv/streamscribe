"""NeMo ASR model loading and device management."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys

from streamscribe.exceptions import ModelLoadError


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout, stderr, and all loggers during NeMo's verbose setup."""
    old_out, old_err = sys.stdout, sys.stderr
    old_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    # Also redirect the underlying file descriptors (catches C-level writes)
    out_fd = old_out.fileno()
    err_fd = old_err.fileno()
    out_dup = os.dup(out_fd)
    err_dup = os.dup(err_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, out_fd)
    os.dup2(devnull, err_fd)
    try:
        yield
    finally:
        os.dup2(out_dup, out_fd)
        os.dup2(err_dup, err_fd)
        os.close(out_dup)
        os.close(err_dup)
        os.close(devnull)
        sys.stdout = old_out
        sys.stderr = old_err
        logging.root.setLevel(old_level)

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
            with _suppress_output():
                import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ModelLoadError(
                "NeMo ASR toolkit is required. Install with:\n"
                '  pip install -e ".[cpu]"  (for CPU)\n'
                '  pip install -e ".[gpu]"  (for CUDA GPU)'
            )
        except Exception as e:
            import platform

            hint = ""
            if platform.system() == "Darwin":
                hint = "\nOn macOS, consider using --engine apple instead."
            raise ModelLoadError(
                f"Failed to import NeMo: {e}{hint}"
            ) from e

        device = self._resolve_device()
        print(f"Loading model {self._model_name} on {device}...", file=sys.stderr)

        try:
            with _suppress_output():
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
