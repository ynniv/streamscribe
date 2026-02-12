"""Apple Speech engine — uses macOS SFSpeechRecognizer via PyObjC."""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np

from streamscribe.asr.dedup import TextDeduplicator
from streamscribe.asr.transcriber import SILENCE_THRESHOLD
from streamscribe.audio.chunker import AudioChunk
from streamscribe.audio.decoder import SAMPLE_RATE
from streamscribe.exceptions import ModelLoadError, TranscriptionError


class AppleSpeechEngine:
    """ASR engine using macOS built-in SFSpeechRecognizer.

    Uses on-device recognition (no network required). Each AudioChunk is
    recognized as a separate request, which naturally respects the ~1 minute
    per-request limitation of SFSpeechRecognizer.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._dedup = TextDeduplicator()
        self._recognizer: Any = None
        self._Speech: Any = None
        self._AVFoundation: Any = None

    @property
    def name(self) -> str:
        return "Apple Speech (on-device)"

    @property
    def supports_diarization(self) -> bool:
        return False

    def load(self) -> None:
        """Initialize SFSpeechRecognizer."""
        try:
            import Speech
            import AVFoundation
        except ImportError:
            raise ModelLoadError(
                "PyObjC frameworks are required for the Apple engine.\n"
                "Install with: pip install pyobjc-framework-Speech pyobjc-framework-AVFoundation\n"
                "Or run: ./setup-macos.sh"
            )

        print("Initializing Apple Speech recognizer...", file=sys.stderr)

        # Check if authorization was explicitly denied
        auth_status = Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status == Speech.SFSpeechRecognizerAuthorizationStatusDenied:
            raise ModelLoadError(
                "Speech recognition permission denied. "
                "Enable it in System Settings > Privacy & Security > Speech Recognition."
            )
        if auth_status == Speech.SFSpeechRecognizerAuthorizationStatusRestricted:
            raise ModelLoadError(
                "Speech recognition is restricted on this device."
            )

        recognizer = Speech.SFSpeechRecognizer.alloc().init()
        if recognizer is None:
            raise ModelLoadError(
                "SFSpeechRecognizer could not be initialized. "
                "Speech recognition may not be available for the current locale."
            )

        if not recognizer.isAvailable():
            raise ModelLoadError(
                "Speech recognition is not currently available on this device."
            )

        if recognizer.supportsOnDeviceRecognition():
            print("On-device recognition available.", file=sys.stderr)
        else:
            print(
                "Warning: On-device recognition not supported for this locale. "
                "Recognition will use Apple servers.",
                file=sys.stderr,
            )

        # Use a background operation queue so recognition callbacks
        # don't require the main thread's RunLoop (needed for server use)
        from Foundation import NSOperationQueue
        bg_queue = NSOperationQueue.alloc().init()
        bg_queue.setName_("SpeechRecognition")
        recognizer.setQueue_(bg_queue)

        self._recognizer = recognizer
        self._Speech = Speech
        self._AVFoundation = AVFoundation
        print("Apple Speech recognizer ready.", file=sys.stderr)

    def transcribe_samples(self, samples: "np.ndarray") -> str | None:
        """Raw inference on samples — no silence check or dedup."""
        try:
            return self._recognize_buffer(samples)
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Apple Speech recognition failed: {e}"
            ) from e

    def transcribe(self, chunk: AudioChunk) -> str | None:
        rms = float(np.sqrt(np.mean(chunk.samples**2)))
        if rms < SILENCE_THRESHOLD:
            return None

        try:
            text = self._recognize_buffer(chunk.samples)
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Apple Speech recognition failed: {e}"
            ) from e

        if not text:
            return None

        return self._dedup.deduplicate(text)

    def _recognize_buffer(self, samples: np.ndarray) -> str | None:
        """Feed PCM samples to SFSpeechAudioBufferRecognitionRequest."""
        Speech = self._Speech
        AVFoundation = self._AVFoundation

        # Create audio format: 16kHz, mono, float32
        audio_format = (
            AVFoundation.AVAudioFormat.alloc()
            .initWithCommonFormat_sampleRate_channels_interleaved_(
                AVFoundation.AVAudioPCMFormatFloat32,
                float(SAMPLE_RATE),
                1,
                False,
            )
        )
        if audio_format is None:
            return None

        # Create PCM buffer and copy samples via numpy view
        frame_count = len(samples)
        pcm_buffer = (
            AVFoundation.AVAudioPCMBuffer.alloc()
            .initWithPCMFormat_frameCapacity_(audio_format, frame_count)
        )
        if pcm_buffer is None:
            return None

        pcm_buffer.setFrameLength_(frame_count)

        channel_data = pcm_buffer.floatChannelData()[0]
        dest = np.frombuffer(
            channel_data.as_buffer(frame_count * 4), dtype=np.float32
        )[:frame_count]
        dest[:] = samples

        # Create recognition request
        request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        request.setShouldReportPartialResults_(False)

        if self._recognizer.supportsOnDeviceRecognition():
            request.setRequiresOnDeviceRecognition_(True)

        request.appendAudioPCMBuffer_(pcm_buffer)
        request.endAudio()

        # Wait for result by spinning the RunLoop
        event = threading.Event()
        result_text: list[str | None] = [None]
        error_msg: list[str | None] = [None]

        def handler(result: Any, error: Any) -> None:
            if error is not None:
                error_msg[0] = str(error)
                event.set()
            elif result is not None and result.isFinal():
                best = result.bestTranscription()
                if best is not None:
                    result_text[0] = best.formattedString()
                event.set()

        self._recognizer.recognitionTaskWithRequest_resultHandler_(
            request, handler
        )

        # Wait for the callback — using Event.wait() instead of NSRunLoop
        # because NSRunLoop doesn't process callbacks in background threads.
        if not event.wait(timeout=30.0):
            if self._verbose:
                print(
                    "Warning: Apple Speech recognition timed out",
                    file=sys.stderr,
                )
            return None

        if error_msg[0] is not None:
            if self._verbose:
                print(
                    f"Warning: Apple Speech error: {error_msg[0]}",
                    file=sys.stderr,
                )
            return None

        return result_text[0]
