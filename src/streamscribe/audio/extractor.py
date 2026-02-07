"""Extract audio stream URLs from YouTube using yt-dlp."""

from __future__ import annotations

from dataclasses import dataclass

from streamscribe.exceptions import AudioExtractionError


@dataclass
class StreamInfo:
    """Information about an extracted audio stream."""

    audio_url: str
    is_live: bool
    title: str
    duration: float | None  # None for live streams


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
            "format": "bestaudio*/best*",
            "quiet": True,
            "no_warnings": True,
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
