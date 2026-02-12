FROM python:3.12-slim

# System dependencies: ffmpeg, PulseAudio, Xvfb (virtual display for audio),
# C++ compiler (texterrors needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        pulseaudio \
        xvfb \
        g++ \
        nodejs \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JS runtime for full YouTube format extraction
RUN echo '--js-runtimes node' > /etc/yt-dlp.conf

# Prefer IPv4 — Docker containers often lack IPv6 routes, causing yt-dlp to fail
RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf

# Non-root user (PulseAudio requires non-root)
RUN useradd -m -s /bin/bash bot
WORKDIR /home/bot/app

# Copy project files for pip install
COPY pyproject.toml .
COPY src/ ./src/

# Playwright browsers go in a shared path (not user home, which gets a volume mount)
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# Install streamscribe with CPU extras and pin protobuf to avoid conflicts
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[cpu]" \
        "protobuf>=3.20,<6" \
    && pip install --no-cache-dir playwright \
    && playwright install --with-deps chromium \
    && playwright install --with-deps firefox

# Copy bot code
COPY bot/ ./bot/

# Xvfb needs this directory to exist (non-root can't create it at runtime)
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Fix ownership
RUN chown -R bot:bot /home/bot

USER bot

ENTRYPOINT ["bash", "bot/entrypoint.sh"]
