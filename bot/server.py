"""HTTP API for managing transcription streams.

Source types:
    jitsi   — Jitsi Meet rooms (headless browser + PulseAudio)
    url     — YouTube / sites supported by yt-dlp
    direct  — raw audio/stream URLs (HLS, DASH, RTMP, mp3, etc.) → ffmpeg

Endpoints:
    GET    /streams              — list all streams
    POST   /streams              — add a stream
    GET    /streams/<name>       — stream info + transcript
    DELETE /streams/<name>       — stop and remove a stream
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from streamscribe.asr.dedup import TextDeduplicator
from streamscribe.asr.engine import create_engine, ASREngine
from streamscribe.asr.output import TranscriptionDisplay
from streamscribe.asr.transcriber import SILENCE_THRESHOLD
from streamscribe.audio.chunker import AudioChunker
from streamscribe.audio.decoder import AudioDecoder
from streamscribe.audio.extractor import AudioExtractor, StreamInfo, device_stream_info

# Jitsi hostnames we recognize for auto-detection
_JITSI_HOSTS = {"meet.jit.si", "8x8.vc"}

# Extensions that ffmpeg handles directly (no yt-dlp needed)
_DIRECT_EXTENSIONS = {
    ".m3u8", ".m3u", ".mpd",  # HLS, DASH
    ".mp3", ".mp4", ".m4a", ".ogg", ".opus", ".flac", ".wav", ".aac",
    ".mkv", ".webm", ".ts",
}

# URL schemes that go straight to ffmpeg
_DIRECT_SCHEMES = {"rtmp", "rtmps", "rtsp", "rtp", "srt", "udp", "tcp"}


def _detect_source_type(url: str) -> str:
    """Guess whether a URL is jitsi, a direct stream, or a yt-dlp site."""
    parsed = urlparse(url)
    if parsed.hostname in _JITSI_HOSTS:
        return "jitsi"
    if parsed.scheme in _DIRECT_SCHEMES:
        return "direct"
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in _DIRECT_EXTENSIONS):
        return "direct"
    return "url"



def _derive_name(url: str) -> str:
    """Derive a short name from a URL."""
    from urllib.parse import parse_qs

    parsed = urlparse(url)
    # YouTube: use the video ID instead of "watch"
    if parsed.hostname and (
        "youtube.com" in parsed.hostname or "youtu.be" in parsed.hostname
    ):
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        # /live/VIDEO_ID or youtu.be/VIDEO_ID — use last path segment
        last_seg = parsed.path.strip("/").rsplit("/", 1)[-1]
        if last_seg:
            return last_seg
    path = parsed.path.rstrip("/")
    name = Path(path).name
    return name if name else parsed.hostname or "stream"


@dataclass
class StreamState:
    name: str
    url: str
    source_type: str  # "jitsi" or "url"
    thread: threading.Thread
    stop_event: threading.Event
    started_at: float
    transcript: list[dict] = field(default_factory=list)
    output_file: str | None = None
    status: str = "starting"  # starting, active, error, stopped
    error_message: str | None = None
    # Jitsi-only fields
    sink_name: str | None = None
    sink_module_id: int | None = None
    browser_proc: subprocess.Popen | None = None
    # URL-only fields
    stream_info: StreamInfo | None = None
    # Wall-clock epoch corresponding to chunk.timestamp=0 (live streams)
    epoch_start: float | None = None
    # Per-stream deduplicator (engine's built-in one is shared/stateful)
    dedup: TextDeduplicator = field(default_factory=TextDeduplicator)


class _YtdlpPipedDecoder:
    """Pipe yt-dlp output through ffmpeg for live stream decoding.

    More reliable than extracting a URL and handing it to ffmpeg separately,
    because yt-dlp handles auth-token refresh, cookie management, and
    adaptive-bitrate selection internally.
    """

    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2  # 16-bit

    def __init__(self, url: str) -> None:
        self._url = url
        self._ytdlp: subprocess.Popen | None = None
        self._ffmpeg: subprocess.Popen | None = None

    def __enter__(self) -> "_YtdlpPipedDecoder":
        self._start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _start(self) -> None:
        self._ytdlp = subprocess.Popen(
            [
                "yt-dlp", "-f", "bestaudio/best",
                "--js-runtimes", "node",
                "-o", "-", "--quiet", "--no-warnings",
                self._url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.SAMPLE_RATE),
                "-ac", "1", "-f", "s16le", "pipe:1",
            ],
            stdin=self._ytdlp.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Allow yt-dlp to receive SIGPIPE if ffmpeg exits first
        self._ytdlp.stdout.close()

        # Background threads to surface errors immediately
        def _log_stderr(proc: subprocess.Popen, label: str) -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                msg = line.decode(errors="replace").rstrip()
                if msg:
                    print(f"[{label}] {msg}", file=sys.stderr)

        threading.Thread(
            target=_log_stderr, args=(self._ytdlp, "yt-dlp"), daemon=True,
        ).start()
        threading.Thread(
            target=_log_stderr, args=(self._ffmpeg, "ffmpeg-pipe"), daemon=True,
        ).start()

    def read_chunk(self, num_samples: int) -> "np.ndarray | None":
        import numpy as np

        if self._ffmpeg is None or self._ffmpeg.stdout is None:
            return None
        # Check if yt-dlp died before producing any output
        if self._ytdlp is not None and self._ytdlp.poll() is not None:
            rc = self._ytdlp.returncode
            if rc != 0:
                stderr = ""
                if self._ytdlp.stderr:
                    stderr = self._ytdlp.stderr.read().decode(errors="replace").strip()
                msg = f"yt-dlp exited with code {rc}"
                if stderr:
                    msg += f": {stderr}"
                print(f"[yt-dlp pipe] {msg}", file=sys.stderr)
                return None
        data = self._ffmpeg.stdout.read(num_samples * self.SAMPLE_WIDTH)
        if not data:
            return None
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        for proc in (self._ffmpeg, self._ytdlp):
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except OSError:
                pass
        self._ffmpeg = None
        self._ytdlp = None


class StreamManager:
    """Manages the lifecycle of transcription streams."""

    def __init__(
        self,
        engine: ASREngine,
        bot_name: str = "Transcription Bot",
        diarizer: "SpeakerDiarizer | None" = None,
    ) -> None:
        self.engine = engine
        self.diarizer = diarizer
        self.bot_name = bot_name
        self._lock = threading.Lock()  # serializes engine.transcribe()
        self._streams: dict[str, StreamState] = {}
        self._sink_counter = 0

    def list_streams(self) -> list[dict]:
        return [self._summary(s) for s in self._streams.values()]

    def get_stream(self, name: str) -> dict | None:
        s = self._streams.get(name)
        if s is None:
            return None
        info = self._summary(s)
        info["transcript"] = s.transcript
        return info

    def _summary(self, s: StreamState) -> dict:
        d: dict = {
            "name": s.name,
            "url": s.url,
            "type": s.source_type,
            "status": s.status,
            "uptime_s": int(time.time() - s.started_at),
            "segments": len(s.transcript),
        }
        if s.epoch_start is not None:
            d["epoch_start"] = s.epoch_start
        if s.error_message:
            d["error"] = s.error_message
        if s.source_type == "jitsi":
            d["sink"] = s.sink_name
        if s.stream_info:
            if s.stream_info.duration:
                d["duration_s"] = s.stream_info.duration
            if s.stream_info.release_timestamp:
                d["release_timestamp"] = s.stream_info.release_timestamp
        return d

    def add_stream(
        self,
        url: str,
        name: str | None = None,
        source_type: str | None = None,
    ) -> dict:
        if source_type is None:
            source_type = _detect_source_type(url)
        if name is None:
            name = _derive_name(url)
        if name in self._streams:
            raise ValueError(f"Stream '{name}' already exists")

        output_file = f"/output/{name}.txt" if Path("/output").is_dir() else None
        stop_event = threading.Event()

        if source_type == "jitsi":
            state = self._setup_jitsi(url, name, output_file, stop_event)
        elif source_type == "browser":
            state = self._setup_browser(url, name, output_file, stop_event)
        elif source_type == "direct":
            state = self._setup_direct(url, name, output_file, stop_event)
        else:
            state = self._setup_url(url, name, output_file, stop_event)

        self._streams[name] = state
        state.thread.start()

        print(f"[{name}] added ({source_type})", file=sys.stderr)
        return self._summary(state)

    def _setup_jitsi(
        self, url: str, name: str, output_file: str | None, stop_event: threading.Event,
    ) -> StreamState:
        sink_name = f"jitsi_sink_{self._sink_counter}"
        self._sink_counter += 1

        result = subprocess.run(
            ["pactl", "load-module", "module-null-sink",
             f"sink_name={sink_name}",
             f"sink_properties=device.description={name}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sink: {result.stderr}")
        module_id = int(result.stdout.strip())

        env = os.environ.copy()
        env["PULSE_SINK"] = sink_name
        browser_proc = subprocess.Popen(
            [sys.executable, "bot/join.py", url, "--name", self.bot_name],
            env=env,
        )

        return StreamState(
            name=name, url=url, source_type="jitsi",
            sink_name=sink_name, sink_module_id=module_id,
            browser_proc=browser_proc,
            thread=threading.Thread(
                target=self._transcribe_jitsi, args=(name,), daemon=True,
            ),
            stop_event=stop_event, started_at=time.time(),
            output_file=output_file,
        )

    def _setup_browser(
        self, url: str, name: str, output_file: str | None, stop_event: threading.Event,
    ) -> StreamState:
        """Open a URL in headless Chromium and capture its audio output.

        Uses the same PulseAudio sink approach as Jitsi, but with a simpler
        browser script (bot/play.py) that just navigates and plays.
        """
        sink_name = f"browser_sink_{self._sink_counter}"
        self._sink_counter += 1

        result = subprocess.run(
            ["pactl", "load-module", "module-null-sink",
             f"sink_name={sink_name}",
             f"sink_properties=device.description={name}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sink: {result.stderr}")
        module_id = int(result.stdout.strip())

        env = os.environ.copy()
        env["PULSE_SINK"] = sink_name
        browser_proc = subprocess.Popen(
            [sys.executable, "bot/play.py", url],
            env=env,
        )

        return StreamState(
            name=name, url=url, source_type="browser",
            sink_name=sink_name, sink_module_id=module_id,
            browser_proc=browser_proc,
            thread=threading.Thread(
                target=self._transcribe_jitsi, args=(name,), daemon=True,
            ),
            stop_event=stop_event, started_at=time.time(),
            output_file=output_file,
        )

    def _setup_url(
        self, url: str, name: str, output_file: str | None, stop_event: threading.Event,
    ) -> StreamState:
        # Stream info will be extracted in the background thread
        return StreamState(
            name=name, url=url, source_type="url",
            thread=threading.Thread(
                target=self._transcribe_url, args=(name,), daemon=True,
            ),
            stop_event=stop_event, started_at=time.time(),
            output_file=output_file,
        )

    def _setup_direct(
        self, url: str, name: str, output_file: str | None, stop_event: threading.Event,
    ) -> StreamState:
        # Treat streaming protocols as live; files/downloads as non-live
        parsed = urlparse(url)
        is_live = parsed.scheme in _DIRECT_SCHEMES or url.endswith(".m3u8")

        stream_info = StreamInfo(
            audio_url=url, is_live=is_live,
            title=name, duration=None,
        )

        return StreamState(
            name=name, url=url, source_type="direct",
            stream_info=stream_info,
            thread=threading.Thread(
                target=self._transcribe_url, args=(name,), daemon=True,
            ),
            stop_event=stop_event, started_at=time.time(),
            output_file=output_file,
        )

    def remove_stream(self, name: str) -> bool:
        state = self._streams.pop(name, None)
        if state is None:
            return False

        state.stop_event.set()

        if state.browser_proc is not None:
            state.browser_proc.terminate()
            try:
                state.browser_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.browser_proc.kill()

        if state.sink_module_id is not None:
            subprocess.run(
                ["pactl", "unload-module", str(state.sink_module_id)],
                capture_output=True,
            )

        state.thread.join(timeout=5)
        print(f"[{name}] removed", file=sys.stderr)
        return True

    def shutdown(self) -> None:
        for name in list(self._streams):
            self.remove_stream(name)

    def _start_browser_capture(self, state: StreamState) -> None:
        """Set up browser audio capture on an existing stream state."""
        sink_name = f"browser_sink_{self._sink_counter}"
        self._sink_counter += 1

        result = subprocess.run(
            ["pactl", "load-module", "module-null-sink",
             f"sink_name={sink_name}",
             f"sink_properties=device.description={state.name}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sink: {result.stderr}")

        env = os.environ.copy()
        env["PULSE_SINK"] = sink_name
        state.browser_proc = subprocess.Popen(
            [sys.executable, "bot/play.py", state.url],
            env=env,
        )
        state.sink_name = sink_name
        state.sink_module_id = int(result.stdout.strip())

    # --- Transcription loops ---

    def _transcribe_jitsi(self, name: str) -> None:
        """Transcription worker for a browser-based source (Jitsi or browser)."""
        state = self._streams.get(name)
        # YouTube needs longer to load (ads, consent, heavy JS player)
        is_youtube = state and (
            "youtube.com" in state.url or "youtu.be" in state.url
        )
        wait = 40 if is_youtube else 10
        print(f"[{name}] waiting {wait}s for browser audio...", file=sys.stderr)
        time.sleep(wait)
        state = self._streams.get(name)
        if state is None or state.sink_name is None:
            return
        audio_url = f"device:{state.sink_name}.monitor"
        state.status = "active"
        self._run_transcription(name, audio_url, is_live=True)

    def _transcribe_url(self, name: str) -> None:
        """Transcription worker for a URL/file source."""
        state = self._streams.get(name)
        if state is None:
            return

        # If stream_info isn't set yet, extract it (direct streams skip this)
        if state.stream_info is None:
            try:
                extractor = AudioExtractor()
                print(f"[{name}] extracting stream info via yt-dlp...", file=sys.stderr)
                stream_info = extractor.extract(state.url)
                print(
                    f"[{name}] yt-dlp: title={stream_info.title!r} "
                    f"live={stream_info.is_live} duration={stream_info.duration}",
                    file=sys.stderr,
                )
                if not stream_info.is_live:
                    print(f"[{name}] downloading audio...", file=sys.stderr)
                    stream_info = extractor.download(state.url)
                    print(f"[{name}] downloaded: {stream_info.audio_url}", file=sys.stderr)
                state.stream_info = stream_info
            except Exception as e:
                parsed = urlparse(state.url)
                is_youtube = parsed.hostname and (
                    "youtube.com" in parsed.hostname or "youtu.be" in parsed.hostname
                )
                if is_youtube:
                    print(f"[{name}] yt-dlp failed ({e}), falling back to browser", file=sys.stderr)
                    try:
                        self._start_browser_capture(state)
                        state.source_type = "url"
                        state.status = "active"
                        self._transcribe_jitsi(name)
                        return
                    except (FileNotFoundError, RuntimeError) as browser_err:
                        state.status = "error"
                        state.error_message = (
                            f"yt-dlp failed ({e}) and browser fallback unavailable ({browser_err})"
                        )
                        print(f"[{name}] {state.error_message}", file=sys.stderr)
                        return
                else:
                    state.status = "error"
                    state.error_message = str(e)
                    print(f"[{name}] extraction failed: {e}", file=sys.stderr)
                    return

        state.status = "active"
        self._run_transcription(
            name, state.stream_info.audio_url, is_live=state.stream_info.is_live,
        )

    def _process_chunks(
        self,
        name: str,
        state: StreamState,
        display: TranscriptionDisplay,
        decoder_ctx: AudioDecoder,
        is_live: bool,
        ts_offset: float = 0.0,
    ) -> tuple[int, int]:
        """Run the chunk-processing loop on a decoder. Returns (chunk_count, silent_count)."""
        import numpy as np

        chunk_duration = 2.0 if is_live else 5.0
        chunk_count = 0
        silent_count = 0

        with decoder_ctx as decoder:
            chunker = AudioChunker(decoder, chunk_duration, context_duration=1.0)
            for chunk in chunker.chunks():
                if state.stop_event.is_set():
                    break
                chunk_count += 1
                ts = chunk.timestamp + ts_offset
                if chunk_count == 1:
                    if is_live and state.epoch_start is None:
                        state.epoch_start = time.time() - ts
                    print(f"[{name}] first audio chunk received", file=sys.stderr)
                if chunk_count <= 5 or chunk_count % 50 == 0:
                    print(
                        f"[{name}] chunk #{chunk_count}  "
                        f"samples={len(chunk.samples)}  "
                        f"ts={ts:.1f}s",
                        file=sys.stderr,
                    )
                rms = float(np.sqrt(np.mean(chunk.samples**2)))
                if rms < SILENCE_THRESHOLD:
                    silent_count += 1
                    if silent_count <= 3 or silent_count % 50 == 0:
                        print(
                            f"[{name}] silent chunk (rms={rms:.6f}, "
                            f"threshold={SILENCE_THRESHOLD}), "
                            f"silent so far: {silent_count}/{chunk_count}",
                            file=sys.stderr,
                        )
                    continue
                with self._lock:
                    raw = self.engine.transcribe_samples(chunk.samples)
                if not raw:
                    continue
                text = state.dedup.deduplicate(raw)
                if chunk_count <= 20 or chunk_count % 50 == 0:
                    print(f"[{name}] raw={raw!r}  dedup={text!r}", file=sys.stderr)
                if text:
                    speaker = None
                    if self.diarizer is not None:
                        with self._lock:
                            speaker = self.diarizer.identify(chunk.samples)
                    display.show_text(text, ts, speaker)
                    seg: dict = {"time": round(ts, 1), "text": text}
                    if speaker:
                        seg["speaker"] = speaker
                    state.transcript.append(seg)
        return chunk_count, silent_count

    def _run_transcription(self, name: str, audio_url: str, is_live: bool) -> None:
        state = self._streams.get(name)
        if state is None:
            return

        display = TranscriptionDisplay(
            show_timestamps=True, output_file=state.output_file,
        )
        display.write_header(
            title=state.name, url=state.url,
            stream_type="live" if is_live else "video", duration=None,
        )

        try:
            decoder = AudioDecoder(audio_url, is_live=is_live)
            total_chunks, total_silent = self._process_chunks(
                name, state, display, decoder, is_live=is_live,
            )

            if total_chunks == 0:
                print(
                    f"[{name}] WARNING: no audio chunks received — "
                    f"source may have failed silently",
                    file=sys.stderr,
                )
            elif total_silent == total_chunks:
                print(
                    f"[{name}] WARNING: all {total_chunks} chunks were silent",
                    file=sys.stderr,
                )
        except Exception as e:
            if not state.stop_event.is_set():
                print(f"[{name}] Error: {e}", file=sys.stderr)
                state.status = "error"
                state.error_message = str(e)
        finally:
            display.close()
            if not is_live and name in self._streams:
                self._streams[name].stop_event.set()
            if state.status == "active":
                state.status = "stopped"
            print(f"[{name}] transcription stopped", file=sys.stderr)


_STATIC_DIR = Path(__file__).parent / "static"


class APIHandler(BaseHTTPRequestHandler):
    manager: StreamManager  # set on the class before starting the server

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(_STATIC_DIR / "index.html", "text/html")
        elif self.path == "/streams":
            self._json_response(200, self.manager.list_streams())
        elif self.path.startswith("/streams/"):
            name = self.path[len("/streams/"):]
            data = self.manager.get_stream(name)
            if data is None:
                self._json_response(404, {"error": f"Stream '{name}' not found"})
            else:
                self._json_response(200, data)
        else:
            self._json_response(404, {"error": "Not found"})

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/streams":
            body = self._read_body()
            if body is None:
                return
            url = body.get("url")
            if not url:
                self._json_response(400, {"error": "Missing 'url' field"})
                return
            try:
                result = self.manager.add_stream(
                    url,
                    name=body.get("name"),
                    source_type=body.get("type"),
                )
                self._json_response(201, result)
            except (ValueError, RuntimeError) as e:
                self._json_response(409, {"error": str(e)})
            except Exception as e:
                self._json_response(500, {"error": str(e)})
        else:
            self._json_response(404, {"error": "Not found"})

    def do_DELETE(self) -> None:
        if self.path.startswith("/streams/"):
            name = self.path[len("/streams/"):]
            if self.manager.remove_stream(name):
                self._json_response(200, {"removed": name})
            else:
                self._json_response(404, {"error": f"Stream '{name}' not found"})
        else:
            self._json_response(404, {"error": "Not found"})

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json_response(400, {"error": "Empty request body"})
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return None

    def _json_response(self, status: int, data: object) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"API: {fmt % args}", file=sys.stderr)


def main() -> None:
    port = int(os.environ.get("API_PORT", "8080"))
    engine_name = os.environ.get("ENGINE", "auto")
    model = os.environ.get("MODEL", "nvidia/nemotron-speech-streaming-en-0.6b")
    device = os.environ.get("COMPUTE_DEVICE", "auto")
    bot_name = os.environ.get("BOT_NAME", "Transcription Bot")

    print(f"Loading engine={engine_name} model={model} device={device}...", file=sys.stderr)
    engine = create_engine(engine_name, model_name=model, device=device)
    engine.load()
    print(f"Engine ready: {engine.name}", file=sys.stderr)

    # Load speaker diarizer when the engine supports it
    diarizer = None
    if engine.supports_diarization:
        from streamscribe.asr.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device=getattr(engine, "device", "cpu"))
        diarizer.load()

    manager = StreamManager(engine, bot_name=bot_name, diarizer=diarizer)
    APIHandler.manager = manager

    # Auto-join rooms from env vars
    room_urls = os.environ.get("JITSI_ROOM_URLS", os.environ.get("JITSI_ROOM_URL", ""))
    for url in room_urls.split():
        if url:
            try:
                manager.add_stream(url)
            except Exception as e:
                print(f"Failed to auto-add {url}: {e}", file=sys.stderr)

    server = HTTPServer(("0.0.0.0", port), APIHandler)
    print(f"API listening on :{port}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
