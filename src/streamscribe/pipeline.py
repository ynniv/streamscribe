"""Main transcription pipeline wiring all components together."""

from __future__ import annotations

import re
import sys

from streamscribe.asr.engine import create_engine
from streamscribe.asr.output import TranscriptionDisplay
from streamscribe.audio.chunker import AudioChunker
from streamscribe.audio.decoder import AudioDecoder
from streamscribe.audio.extractor import AudioExtractor, StreamInfo
from streamscribe.exceptions import StreamscribeError


def _slugify(title: str) -> str:
    """Turn a stream title into a safe, descriptive filename."""
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return f"{slug[:80]}.txt" if slug else "transcript.txt"


class TranscriptionPipeline:
    """Orchestrates the full audio extraction → ASR → display pipeline."""

    def __init__(
        self,
        url: str,
        engine: str = "auto",
        model_name: str = "",
        device: str = "auto",
        chunk_duration: float = 5.0,
        context_duration: float = 1.0,
        show_timestamps: bool = True,
        verbose: bool = False,
        from_start: bool = False,
        speakers: bool = False,
        output_file: str | None = None,
        auto_output: bool = False,
    ) -> None:
        self._url = url
        self._engine_name = engine
        self._model_name = model_name
        self._device = device
        self._chunk_duration = chunk_duration
        self._context_duration = context_duration
        self._show_timestamps = show_timestamps
        self._verbose = verbose
        self._from_start = from_start
        self._speakers = speakers
        self._output_file = output_file
        self._auto_output = auto_output

    def run(self) -> None:
        """Run the full pipeline. Blocks until complete or interrupted."""
        # Use a temporary display for pre-extraction status messages
        display = TranscriptionDisplay(self._show_timestamps)

        # Phase 1: Extract audio URL
        display.status("Extracting audio stream URL...")
        extractor = AudioExtractor()
        stream_info = extractor.extract(self._url, from_start=self._from_start)

        # Resolve output file: -O derives name from stream title
        output_file = self._output_file
        if self._auto_output:
            output_file = _slugify(stream_info.title)

        # Now create the real display (possibly with an output file)
        display = TranscriptionDisplay(self._show_timestamps, output_file)
        self._print_stream_info(display, stream_info)
        if output_file:
            display.status(f"Writing transcript to {output_file}")
        display.write_header(
            title=stream_info.title,
            url=self._url,
            stream_type="live" if stream_info.is_live else "video",
            duration=stream_info.duration,
        )

        # Adjust chunk size for live streams (but not when replaying from start)
        chunk_duration = self._chunk_duration
        if stream_info.is_live and not self._from_start and chunk_duration > 3.0:
            chunk_duration = 2.0
            display.status(
                f"Live stream detected — using {chunk_duration}s chunks for lower latency."
            )
        if self._from_start and stream_info.is_live:
            display.status("Replaying from start (faster than realtime).")

        # Phase 2: Create and load ASR engine
        engine = create_engine(
            self._engine_name,
            model_name=self._model_name,
            device=self._device,
            verbose=self._verbose,
        )
        display.status(f"Engine: {engine.name}")
        engine.load()

        # Phase 2b: Load speaker model if requested
        diarizer = None
        if self._speakers:
            if not engine.supports_diarization:
                display.status(
                    f"Warning: Speaker diarization not supported with "
                    f"{engine.name}. Skipping."
                )
            else:
                from streamscribe.asr.diarizer import SpeakerDiarizer

                device = getattr(engine, "device", "cpu")
                diarizer = SpeakerDiarizer(device=device)
                diarizer.load()
        segments_count = 0
        total_duration = 0.0

        display.status("Starting transcription...\n")

        try:
            with AudioDecoder(stream_info.audio_url, stream_info.is_live) as decoder:
                chunker = AudioChunker(decoder, chunk_duration, self._context_duration)

                for chunk in chunker.chunks():
                    text = engine.transcribe(chunk)
                    if text:
                        speaker = None
                        if diarizer is not None:
                            speaker = diarizer.identify(chunk.samples)
                        display.show_text(text, chunk.timestamp, speaker)
                        segments_count += 1
                    total_duration = chunk.timestamp + chunk.duration

        except KeyboardInterrupt:
            pass  # Handled below
        except StreamscribeError:
            raise
        except Exception as e:
            if self._verbose:
                raise
            print(f"\nError during transcription: {e}", file=sys.stderr)

        # Summary
        display.close()
        display.status(
            f"\nDone — {segments_count} segments, "
            f"{total_duration:.1f}s of audio processed."
        )

    @staticmethod
    def _print_stream_info(display: TranscriptionDisplay, info: StreamInfo) -> None:
        """Print stream metadata to stderr."""
        stream_type = "LIVE" if info.is_live else "video"
        duration_str = (
            f"{info.duration:.0f}s" if info.duration else "unknown duration"
        )
        display.status(f"Title: {info.title}")
        display.status(f"Type: {stream_type}, Duration: {duration_str}")
