"""Playwright script to join a Jitsi Meet room with headless Chromium."""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join a Jitsi Meet room")
    parser.add_argument("room_url", help="Full Jitsi Meet room URL")
    parser.add_argument(
        "--name", default="Transcription Bot", help="Display name in the meeting"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Append config hash params to mute audio/video on join
    separator = "&" if "#" in args.room_url else "#"
    room_url = (
        f"{args.room_url}{separator}"
        "config.startWithAudioMuted=true"
        "&config.startWithVideoMuted=true"
        "&config.prejoinConfig.enabled=false"
    )

    shutdown = threading.Event()

    def handle_signal(signum: int, frame: object) -> None:
        print(f"Received signal {signum}, shutting down...", file=sys.stderr)
        shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            permissions=["microphone", "camera"],
            ignore_https_errors=True,
        )
        page = context.new_page()

        print(f"Navigating to {room_url}", file=sys.stderr)
        page.goto(room_url, wait_until="networkidle", timeout=60000)

        # Handle pre-join screen if it appears (prejoinConfig.enabled=false
        # should skip it, but some Jitsi deployments ignore that config)
        try:
            name_input = page.wait_for_selector(
                "input#premeeting-name-input, input[data-testid='prejoin.input']",
                timeout=5000,
            )
            if name_input:
                name_input.fill(args.name)
                # Click the join button
                join_btn = page.query_selector(
                    "button[data-testid='prejoin.joinMeeting'], "
                    "[role='button'][aria-label='Join meeting']"
                )
                if join_btn:
                    join_btn.click()
                    print("Clicked join button on pre-join screen", file=sys.stderr)
        except Exception:
            # No pre-join screen or already joined
            pass

        print("Joined room, waiting for audio...", file=sys.stderr)

        # Block until signalled to shut down
        shutdown.wait()

        print("Closing browser...", file=sys.stderr)
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
