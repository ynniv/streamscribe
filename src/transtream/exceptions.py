"""Custom exception hierarchy for transtream."""


class TranstreamError(Exception):
    """Base exception for all transtream errors."""


class AudioExtractionError(TranstreamError):
    """Failed to extract audio URL from the given source."""


class AudioDecodingError(TranstreamError):
    """Failed to decode audio via ffmpeg."""


class ModelLoadError(TranstreamError):
    """Failed to load the ASR model."""


class TranscriptionError(TranstreamError):
    """Error during transcription inference."""
