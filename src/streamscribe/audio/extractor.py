"""Extract audio stream URLs from YouTube using yt-dlp."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from streamscribe.audio.decoder import DEVICE_PREFIX
from streamscribe.exceptions import AudioExtractionError

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "streamscribe"


@dataclass
class StreamInfo:
    """Information about an extracted audio stream."""

    audio_url: str
    is_live: bool
    title: str
    duration: float | None  # None for live streams
    local_path: Path | None = field(default=None, repr=False)


class AudioExtractor:
    """Uses yt-dlp to extract the best audio stream URL."""

    def extract(
        self,
        url: str,
        from_start: bool = False,
        cookies_from_browser: str | None = None,
        cookies: str | None = None,
    ) -> StreamInfo:
        """Extract audio stream info from a YouTube URL.

        Does not download — only resolves the direct audio URL.
        If from_start is True, requests the live stream from its beginning.
        """
        try:
            import yt_dlp
        except ImportError:
            raise AudioExtractionError(
                "yt-dlp is required. Install with: pip install yt-dlp"
            )

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "ignore_no_formats_error": True,
            "remote_components": {"ejs:github"},
        }
        if from_start:
            ydl_opts["live_from_start"] = True
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        if cookies:
            ydl_opts["cookiefile"] = cookies

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise AudioExtractionError(f"Failed to extract audio: {e}") from e

        if info is None:
            raise AudioExtractionError("yt-dlp returned no info for the URL")

        audio_url = info.get("url")
        if not audio_url:
            # For formats with separate audio, pick the best audio format
            formats = info.get("formats", [])
            audio_formats = [
                f for f in formats if f.get("acodec") != "none" and f.get("url")
            ]
            if not audio_formats:
                raise AudioExtractionError("No audio stream found in the URL")
            # Prefer formats with audio only (no video codec)
            audio_only = [f for f in audio_formats if f.get("vcodec") == "none"]
            best = audio_only[-1] if audio_only else audio_formats[-1]
            audio_url = best["url"]

        return StreamInfo(
            audio_url=audio_url,
            is_live=bool(info.get("is_live")),
            title=info.get("title", "Unknown"),
            duration=info.get("duration"),
        )

    def download(
        self,
        url: str,
        cookies_from_browser: str | None = None,
        cookies: str | None = None,
    ) -> StreamInfo:
        """Download audio for a non-live video, returning a StreamInfo with local_path."""
        try:
            import yt_dlp
        except ImportError:
            raise AudioExtractionError(
                "yt-dlp is required. Install with: pip install yt-dlp"
            )

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Return cached file if it exists
        cached = list(_CACHE_DIR.glob(f"{url_hash}.*"))
        if cached:
            local = cached[0]
            # Quick metadata-only call for title/duration
            info = self._extract_metadata(
                url, cookies_from_browser=cookies_from_browser, cookies=cookies
            )
            return StreamInfo(
                audio_url=str(local),
                is_live=False,
                title=info.get("title", "Unknown"),
                duration=info.get("duration"),
                local_path=local,
            )

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": str(_CACHE_DIR / url_hash) + ".%(ext)s",
            "remote_components": {"ejs:github"},
        }
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        if cookies:
            ydl_opts["cookiefile"] = cookies

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise AudioExtractionError(f"Failed to download audio: {e}") from e

        if info is None:
            raise AudioExtractionError("yt-dlp returned no info for the URL")

        # Find the downloaded file
        local = Path(ydl.prepare_filename(info))
        if not local.exists():
            for candidate in _CACHE_DIR.glob(f"{url_hash}.*"):
                local = candidate
                break

        if not local.exists():
            raise AudioExtractionError(f"Downloaded file not found: {local}")

        return StreamInfo(
            audio_url=str(local),
            is_live=False,
            title=info.get("title", "Unknown"),
            duration=info.get("duration"),
            local_path=local,
        )

    def _extract_metadata(
        self,
        url: str,
        cookies_from_browser: str | None = None,
        cookies: str | None = None,
    ) -> dict:
        """Quick metadata-only extraction (no download)."""
        import yt_dlp

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "ignore_no_formats_error": True,
            "remote_components": {"ejs:github"},
        }
        if cookies_from_browser:
            opts["cookiesfrombrowser"] = (cookies_from_browser,)
        if cookies:
            opts["cookiefile"] = cookies
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception:
            return {}


def device_stream_info(device_id: str) -> StreamInfo:
    """Create a StreamInfo for a local audio device (no yt-dlp needed)."""
    return StreamInfo(
        audio_url=f"{DEVICE_PREFIX}{device_id}",
        is_live=True,
        title=f"Audio Device ({device_id})",
        duration=None,
    )
