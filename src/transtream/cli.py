"""CLI entry point and argument parsing for transtream."""

from __future__ import annotations

import argparse
import shutil
import sys

from transtream.asr.model import DEFAULT_MODEL
from transtream.exceptions import TranstreamError


def _check_dependencies() -> list[str]:
    """Check that required external tools are available."""
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("yt-dlp") is None:
        missing.append("yt-dlp")
    return missing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transtream",
        description="Transcribe YouTube videos and live streams in real-time.",
    )
    parser.add_argument(
        "url",
        help="YouTube video or live stream URL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"NeMo ASR model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Audio chunk duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--context-duration",
        type=float,
        default=1.0,
        help="Left context overlap in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Start live stream from beginning (faster than realtime)",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="Enable speaker detection (labels output with Speaker 1, Speaker 2, ...)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Omit timestamps from output (useful for file redirection)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full tracebacks on error",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate external dependencies
    missing = _check_dependencies()
    if missing:
        print(
            f"Error: required tools not found on PATH: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Lazy import to keep --help fast
    from transtream.pipeline import TranscriptionPipeline

    pipeline = TranscriptionPipeline(
        url=args.url,
        model_name=args.model,
        device=args.device,
        chunk_duration=args.chunk_duration,
        context_duration=args.context_duration,
        show_timestamps=not args.no_timestamps,
        verbose=args.verbose,
        from_start=args.from_start,
        speakers=args.speakers,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except TranstreamError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
