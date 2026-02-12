"""Playwright script to open a URL in Firefox and let it play audio.

Uses Firefox instead of Chromium because Playwright's Chromium lacks
proprietary codecs (AAC/H.264) needed by YouTube.  Audio is routed
through PulseAudio (via PULSE_SINK env var) and captured from the
sink's monitor source.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a URL in headless Firefox")
    parser.add_argument("url", help="URL to open and play")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shutdown = threading.Event()

    def handle_signal(signum: int, frame: object) -> None:
        print(f"Received signal {signum}, shutting down...", file=sys.stderr)
        shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=False,
            firefox_user_prefs={
                "media.autoplay.default": 0,  # 0 = allow all
                "media.autoplay.blocking_policy": 0,
                "media.autoplay.allow-extension-background-pages": True,
                "media.autoplay.block-event.enabled": False,
                "privacy.trackingprotection.enabled": False,
                "dom.webdriver.enabled": False,
            },
        )

        context = browser.new_context(ignore_https_errors=True)

        # Hide automation signals that YouTube uses to block bots
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        # Pre-set YouTube consent cookies to skip the consent page
        if "youtube.com" in args.url or "youtu.be" in args.url:
            context.add_cookies([
                {
                    "name": "SOCS",
                    "value": "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjMwODI5LjA3X3AxGgJlbiACGgYIgJnPpwY",
                    "domain": ".youtube.com",
                    "path": "/",
                },
                {
                    "name": "CONSENT",
                    "value": "PENDING+987",
                    "domain": ".youtube.com",
                    "path": "/",
                },
            ])

        page = context.new_page()

        # Capture browser console errors for debugging (skip noisy JS warnings)
        page.on("console", lambda msg: print(
            f"[browser:{msg.type}] {msg.text}", file=sys.stderr,
        ) if msg.type == "error" else None)
        page.on("pageerror", lambda err: print(
            f"[browser:exception] {err}", file=sys.stderr,
        ))

        # Navigate to YouTube homepage first so init scripts and cookies
        # are active before loading the actual video (avoids bot detection
        # on the first page load)
        if "youtube.com" in args.url or "youtu.be" in args.url:
            print("Priming YouTube session...", file=sys.stderr)
            try:
                page.goto("https://www.youtube.com", wait_until="domcontentloaded", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(2000)

        print(f"Opening {args.url}", file=sys.stderr)
        try:
            page.goto(args.url, wait_until="load", timeout=60000)
            print("Page loaded", file=sys.stderr)
        except Exception as e:
            print(f"goto warning (continuing): {e}", file=sys.stderr)

        # Log page state for debugging
        print(f"Page title: {page.title()}", file=sys.stderr)
        print(f"Page URL: {page.url}", file=sys.stderr)

        # Dismiss any consent/cookie banners
        for selector in (
            "button[aria-label='Accept all']",
            "button[aria-label='Reject all']",
            "button.yt-spec-button-shape-next--filled[aria-label]",
            "tp-yt-paper-button#button[aria-label='Agree']",
        ):
            try:
                btn = page.query_selector(selector)
                if btn and btn.is_visible():
                    btn.click()
                    print(f"Clicked consent: {selector}", file=sys.stderr)
                    page.wait_for_timeout(2000)
                    break
            except Exception:
                continue

        # Check for YouTube playback errors (e.g. "Video unavailable")
        def _check_yt_error() -> str | None:
            try:
                return page.evaluate("""() => {
                    const reason = document.querySelector(
                        '#reason, .ytp-error-content-wrap-reason, '
                        + 'yt-playability-error-supported-renderers #reason'
                    );
                    if (reason && reason.textContent.trim()) return reason.textContent.trim();
                    const subreason = document.querySelector(
                        '#subreason, .ytp-error-content-wrap-subreason'
                    );
                    if (subreason && subreason.textContent.trim()) return subreason.textContent.trim();
                    return null;
                }""")
            except Exception:
                return None

        # Wait for video to actually have media loaded (readyState >= 2)
        print("Waiting for video media to load...", file=sys.stderr)
        for attempt in range(60):
            # Early exit if YouTube says the video is unavailable
            yt_err = _check_yt_error()
            if yt_err:
                print(f"YouTube error: {yt_err}", file=sys.stderr)
                shutdown.wait()
                context.close()
                browser.close()
                return

            try:
                state = page.evaluate("""() => {
                    const v = document.querySelector('video');
                    if (!v) return null;
                    return {
                        readyState: v.readyState,
                        networkState: v.networkState,
                        currentTime: v.currentTime,
                        paused: v.paused,
                        error: v.error?.code,
                        src: v.src?.substring(0, 100),
                        srcObject: !!v.srcObject,
                        mediaKeys: !!v.mediaKeys,
                    };
                }""")
            except Exception:
                state = None

            if state and state.get("readyState", 0) >= 2:
                print(f"Video ready (attempt {attempt+1}): {state}", file=sys.stderr)
                break

            # Try to skip ads
            try:
                skip_btn = page.query_selector(
                    "button.ytp-skip-ad-button, "
                    "button.ytp-ad-skip-button, "
                    "button.ytp-ad-skip-button-modern, "
                    ".ytp-skip-ad button"
                )
                if skip_btn and skip_btn.is_visible():
                    skip_btn.click()
                    print("Skipped ad", file=sys.stderr)
                    page.wait_for_timeout(1000)
            except Exception:
                pass

            if attempt % 5 == 0:
                print(f"  waiting... (attempt {attempt+1}, state={state})", file=sys.stderr)
                # Periodically try to click play / dismiss overlays
                try:
                    page.evaluate("""() => {
                        const v = document.querySelector('video');
                        if (v) { v.muted = false; v.volume = 1.0; v.play(); }
                        // Click play button if visible
                        document.querySelector('.ytp-play-button')?.click();
                        // Dismiss "Are you still watching?" overlay
                        document.querySelector('.ytp-unmute-text')?.click();
                        document.querySelector('button.ytp-large-play-button')?.click();
                    }""")
                except Exception:
                    pass

            page.wait_for_timeout(1000)
        else:
            print(f"Video did not reach ready state after 60s: {state}", file=sys.stderr)

        # Final playback attempt
        try:
            result = page.evaluate("""() => {
                const v = document.querySelector('video');
                if (!v) return 'no video';
                v.muted = false;
                v.volume = 1.0;
                v.play();
                return {
                    paused: v.paused,
                    readyState: v.readyState,
                    currentTime: v.currentTime,
                    duration: v.duration,
                    buffered: v.buffered?.length,
                };
            }""")
            print(f"Final video state: {result}", file=sys.stderr)
        except Exception as e:
            print(f"Final play error: {e}", file=sys.stderr)

        print("Playing, capturing audio...", file=sys.stderr)
        shutdown.wait()

        print("Closing browser...", file=sys.stderr)
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
