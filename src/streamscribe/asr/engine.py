"""ASR engine abstraction layer."""

from __future__ import annotations

import pathlib
import platform
from typing import Protocol

from streamscribe.audio.chunker import AudioChunk
from streamscribe.exceptions import EngineUnavailableError


class ASREngine(Protocol):
    """Protocol that all ASR engines must implement."""

    @property
    def name(self) -> str:
        """Human-readable engine name for display."""
        ...

    @property
    def supports_diarization(self) -> bool:
        """Whether this engine supports speaker diarization."""
        ...

    def load(self) -> None:
        """Load models or initialize the recognizer."""
        ...

    def transcribe(self, chunk: AudioChunk) -> str | None:
        """Transcribe an audio chunk and return deduplicated new text.

        Returns None if the chunk is silent or produces no new text.
        """
        ...


def _read_config_engine() -> str | None:
    """Read the default engine from streamscribe.conf if it exists."""
    # Walk up from this file to find the project root
    conf = pathlib.Path(__file__).resolve().parents[3] / "streamscribe.conf"
    if not conf.is_file():
        return None
    for line in conf.read_text().splitlines():
        line = line.strip()
        if line.startswith("engine="):
            return line.split("=", 1)[1].strip()
    return None


def resolve_engine_name(engine: str) -> str:
    """Resolve 'auto' to a concrete engine name.

    Checks streamscribe.conf first, then falls back to platform detection.
    """
    if engine != "auto":
        return engine
    configured = _read_config_engine()
    if configured:
        return configured
    if platform.system() == "Darwin":
        return "apple"
    return "nemo"


def create_engine(
    engine: str,
    model_name: str = "",
    device: str = "auto",
    verbose: bool = False,
) -> ASREngine:
    """Factory that creates the appropriate ASR engine.

    Returns an initialized (but not yet loaded) ASREngine instance.
    """
    engine = resolve_engine_name(engine)

    if engine == "nemo":
        from streamscribe.asr.nemo_engine import NeMoEngine

        return NeMoEngine(model_name=model_name, device=device, verbose=verbose)

    if engine == "apple":
        if platform.system() != "Darwin":
            raise EngineUnavailableError(
                "The Apple Speech engine is only available on macOS. "
                "Use --engine nemo instead."
            )
        try:
            from streamscribe.asr.apple_engine import AppleSpeechEngine
        except ImportError:
            raise EngineUnavailableError(
                "pyobjc-framework-Speech is required for the Apple engine.\n"
                "Install with: pip install pyobjc-framework-Speech\n"
                "Or use --engine nemo to use the NeMo engine instead."
            )
        return AppleSpeechEngine(verbose=verbose)

    raise EngineUnavailableError(
        f"Unknown engine '{engine}'. Choose from: nemo, apple, auto"
    )
