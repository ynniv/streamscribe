"""Multi-room transcription: one model, many audio streams."""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass

from streamscribe.asr.engine import create_engine, ASREngine
from streamscribe.asr.output import TranscriptionDisplay
from streamscribe.audio.chunker import AudioChunk, AudioChunker
from streamscribe.audio.decoder import AudioDecoder


@dataclass
class Room:
    name: str
    sink_monitor: str
    output_file: str | None


def transcribe_room(
    room: Room,
    engine: ASREngine,
    lock: threading.Lock,
    chunk_duration: float = 2.0,
    context_duration: float = 1.0,
) -> None:
    """Read audio from one room's sink monitor and transcribe it."""
    display = TranscriptionDisplay(show_timestamps=True, output_file=room.output_file)
    display.write_header(
        title=room.name,
        url=room.sink_monitor,
        stream_type="live",
        duration=None,
    )

    try:
        with AudioDecoder(f"device:{room.sink_monitor}", is_live=True) as decoder:
            chunker = AudioChunker(decoder, chunk_duration, context_duration)
            for chunk in chunker.chunks():
                with lock:
                    text = engine.transcribe(chunk)
                if text:
                    display.show_text(text, chunk.timestamp)
    except Exception as e:
        print(f"[{room.name}] Error: {e}", file=sys.stderr)
    finally:
        display.close()
        print(f"[{room.name}] stopped", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-room transcription")
    parser.add_argument(
        "rooms",
        nargs="+",
        help="Room specs as name:sink_monitor[:output_file]",
    )
    parser.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Parse room specs
    rooms = []
    for spec in args.rooms:
        parts = spec.split(":", 2)
        if len(parts) < 2:
            print(f"Invalid room spec '{spec}', expected name:sink_monitor[:output]",
                  file=sys.stderr)
            sys.exit(1)
        rooms.append(Room(
            name=parts[0],
            sink_monitor=parts[1],
            output_file=parts[2] if len(parts) > 2 else None,
        ))

    # Load model once
    print(f"Loading model {args.model}...", file=sys.stderr)
    engine = create_engine("nemo", model_name=args.model, device=args.device)
    engine.load()
    print("Model loaded.", file=sys.stderr)

    # Lock for serialized access to the engine
    lock = threading.Lock()

    # Start a thread per room
    threads = []
    for room in rooms:
        print(f"[{room.name}] listening on {room.sink_monitor}", file=sys.stderr)
        t = threading.Thread(
            target=transcribe_room,
            args=(room, engine, lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for all threads (Ctrl-C will exit)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
