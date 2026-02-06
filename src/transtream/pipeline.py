"""Main transcription pipeline wiring all components together."""

from __future__ import annotations

import sys

from transtream.asr.model import ASRModelManager
from transtream.asr.output import TranscriptionDisplay
from transtream.asr.transcriber import ChunkedTranscriber
from transtream.audio.chunker import AudioChunker
from transtream.audio.decoder import AudioDecoder
from transtream.audio.extractor import AudioExtractor, StreamInfo
from transtream.exceptions import TranstreamError


class TranscriptionPipeline:
    """Orchestrates the full audio extraction → ASR → display pipeline."""

    def __init__(
        self,
        url: str,
        model_name: str,
        device: str = "auto",
        chunk_duration: float = 5.0,
        context_duration: float = 1.0,
        show_timestamps: bool = True,
        verbose: bool = False,
        from_start: bool = False,
        speakers: bool = False,
    ) -> None:
        self._url = url
        self._model_name = model_name
        self._device = device
        self._chunk_duration = chunk_duration
        self._context_duration = context_duration
        self._show_timestamps = show_timestamps
        self._verbose = verbose
        self._from_start = from_start
        self._speakers = speakers
        self._display = TranscriptionDisplay(show_timestamps)

    def run(self) -> None:
        """Run the full pipeline. Blocks until complete or interrupted."""
        # Phase 1: Extract audio URL
        self._display.status("Extracting audio stream URL...")
        extractor = AudioExtractor()
        stream_info = extractor.extract(self._url, from_start=self._from_start)
        self._print_stream_info(stream_info)

        # Adjust chunk size for live streams (but not when replaying from start)
        chunk_duration = self._chunk_duration
        if stream_info.is_live and not self._from_start and chunk_duration > 3.0:
            chunk_duration = 2.0
            self._display.status(
                f"Live stream detected — using {chunk_duration}s chunks for lower latency."
            )
        if self._from_start and stream_info.is_live:
            self._display.status("Replaying from start (faster than realtime).")

        # Phase 2: Load ASR model
        model_mgr = ASRModelManager(self._model_name, self._device)
        model_mgr.load()

        # Phase 2b: Load speaker model if requested
        diarizer = None
        if self._speakers:
            from transtream.asr.diarizer import SpeakerDiarizer

            diarizer = SpeakerDiarizer(device=model_mgr.model.device)
            diarizer.load()

        # Phase 3: Decode + transcribe
        transcriber = ChunkedTranscriber(model_mgr.model, verbose=self._verbose)
        segments_count = 0
        total_duration = 0.0

        self._display.status("Starting transcription...\n")

        try:
            with AudioDecoder(stream_info.audio_url, stream_info.is_live) as decoder:
                chunker = AudioChunker(decoder, chunk_duration, self._context_duration)

                for chunk in chunker.chunks():
                    text = transcriber.transcribe(chunk)
                    if text:
                        speaker = None
                        if diarizer is not None:
                            speaker = diarizer.identify(chunk.samples)
                        self._display.show_text(text, chunk.timestamp, speaker)
                        segments_count += 1
                    total_duration = chunk.timestamp + chunk.duration

        except KeyboardInterrupt:
            pass  # Handled below
        except TranstreamError:
            raise
        except Exception as e:
            if self._verbose:
                raise
            print(f"\nError during transcription: {e}", file=sys.stderr)

        # Summary
        self._display.status(
            f"\nDone — {segments_count} segments, "
            f"{total_duration:.1f}s of audio processed."
        )

    def _print_stream_info(self, info: StreamInfo) -> None:
        """Print stream metadata to stderr."""
        stream_type = "LIVE" if info.is_live else "video"
        duration_str = (
            f"{info.duration:.0f}s" if info.duration else "unknown duration"
        )
        self._display.status(f"Title: {info.title}")
        self._display.status(f"Type: {stream_type}, Duration: {duration_str}")
