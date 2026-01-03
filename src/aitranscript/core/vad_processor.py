"""Voice Activity Detection (VAD) using Silero VAD"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pydub import AudioSegment

from aitranscript.models.config import VADConfig
from aitranscript.models.segment import SpeechSegment
from aitranscript.utils.logger import get_logger

logger = get_logger(__name__)


class VADProcessingError(Exception):
    """Raised when VAD processing fails"""

    pass


# Global model cache (singleton pattern)
_SILERO_VAD_MODEL = None
_SILERO_VAD_UTILS = None


def load_silero_vad() -> tuple:
    """Load Silero VAD model (cached singleton)

    Returns:
        Tuple of (model, utils) from Silero VAD

    Example:
        >>> model, utils = load_silero_vad()
        >>> get_speech_timestamps = utils[0]
    """
    global _SILERO_VAD_MODEL, _SILERO_VAD_UTILS

    if _SILERO_VAD_MODEL is None:
        logger.info("Loading Silero VAD model...")

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        _SILERO_VAD_MODEL = model
        _SILERO_VAD_UTILS = utils

        logger.info("Silero VAD model loaded successfully")

    return _SILERO_VAD_MODEL, _SILERO_VAD_UTILS


class VADProcessor:
    """Process audio files to detect speech segments using Silero VAD

    Uses Silero VAD to identify speech regions in audio and create
    SpeechSegment objects optimized for subtitle generation (1.5-6s).

    Attributes:
        config: VADConfig with detection parameters
        model: Silero VAD model instance

    Example:
        >>> vad = VADProcessor()
        >>> segments = vad.process("audio.wav")
        >>> for seg in segments:
        ...     print(f"{seg.start:.2f}s - {seg.end:.2f}s")
    """

    def __init__(self, config: Optional[VADConfig] = None):
        """Initialize VADProcessor

        Args:
            config: VADConfig instance (default: VADConfig())
        """
        self.config = config or VADConfig()
        self.model, self.utils = load_silero_vad()

        # Unpack utils
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        logger.info(f"VADProcessor initialized with config: {self.config}")

    def process(self, audio_path: Union[str, Path]) -> list[SpeechSegment]:
        """Detect speech segments in audio file

        Args:
            audio_path: Path to audio file (WAV format preferred)

        Returns:
            List of SpeechSegment objects with detected speech

        Raises:
            VADProcessingError: If processing fails

        Example:
            >>> vad = VADProcessor()
            >>> segments = vad.process("audio.wav")
            >>> print(f"Found {len(segments)} speech segments")
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise VADProcessingError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Processing audio with VAD: {audio_path}")

            # Load audio using pydub (avoids torchaudio dependency)
            audio = AudioSegment.from_file(str(audio_path))

            # Convert to 16kHz mono (Silero VAD requirement)
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Convert to numpy array and normalize to [-1, 1]
            samples = np.array(audio.get_array_of_samples())
            samples = samples.astype(np.float32) / 32768.0  # int16 to float32

            # Convert to torch tensor
            wav = torch.from_numpy(samples)

            # Get speech timestamps using Silero VAD
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=self.config.threshold,
                sampling_rate=16000,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                max_speech_duration_s=self.config.max_segment_duration_s,
                min_silence_duration_ms=self.config.max_pause_duration_ms,
                window_size_samples=512,
                speech_pad_ms=30,
            )

            # Convert to SpeechSegment objects
            segments = []
            for ts in speech_timestamps:
                start = ts["start"] / 16000  # Convert samples to seconds
                end = ts["end"] / 16000

                segment = SpeechSegment(start=start, end=end)
                segments.append(segment)

            logger.info(
                f"VAD detected {len(segments)} speech segments "
                f"(total: {sum(s.duration for s in segments):.1f}s)"
            )

            return segments

        except Exception as e:
            raise VADProcessingError(f"VAD processing failed: {e}") from e

    def _split_long_segments(
        self, segments: list[SpeechSegment]
    ) -> list[SpeechSegment]:
        """Split segments that exceed max duration

        Args:
            segments: List of speech segments

        Returns:
            List with long segments split

        Note:
            This is called internally if needed for post-processing
        """
        result = []
        max_duration = self.config.max_segment_duration_s

        for seg in segments:
            if seg.duration <= max_duration:
                result.append(seg)
            else:
                # Split into chunks
                current_start = seg.start
                while current_start < seg.end:
                    chunk_end = min(current_start + max_duration, seg.end)
                    result.append(SpeechSegment(start=current_start, end=chunk_end))
                    current_start = chunk_end

        return result
