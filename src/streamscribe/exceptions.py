"""Custom exception hierarchy for streamscribe."""


class StreamscribeError(Exception):
    """Base exception for all streamscribe errors."""


class AudioExtractionError(StreamscribeError):
    """Failed to extract audio URL from the given source."""


class AudioDecodingError(StreamscribeError):
    """Failed to decode audio via ffmpeg."""


class ModelLoadError(StreamscribeError):
    """Failed to load the ASR model."""


class TranscriptionError(StreamscribeError):
    """Error during transcription inference."""
