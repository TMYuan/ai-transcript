"""Speech transcription using Faster-Whisper with GPU optimization"""

from pathlib import Path
from typing import Optional, Union

import torch
from faster_whisper import WhisperModel

from aitranscript.models.config import ASRConfig
from aitranscript.models.segment import SpeechSegment, TranscriptSegment
from aitranscript.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails"""

    pass


def detect_device(prefer: Optional[str] = None) -> str:
    """Detect CUDA availability and return appropriate device

    Args:
        prefer: Optional device preference ("cpu" or "cuda")

    Returns:
        Device string ("cpu" or "cuda")

    Example:
        >>> device = detect_device()
        >>> print(device)  # "cuda" if available, else "cpu"
        >>> device = detect_device(prefer="cpu")
        >>> print(device)  # Always "cpu"
    """
    if prefer == "cpu":
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    """Singleton pattern for Whisper model caching

    Caches models by (model_size, device, compute_type) to avoid
    reloading on every transcription request.

    Example:
        >>> model1 = ModelManager.get_model("tiny", "cpu", "int8")
        >>> model2 = ModelManager.get_model("tiny", "cpu", "int8")
        >>> assert model1 is model2  # Same instance (cached)
    """

    _models: dict[str, WhisperModel] = {}

    @classmethod
    def get_model(cls, model_size: str, device: str, compute_type: str) -> WhisperModel:
        """Get or create cached Whisper model

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Compute type (int8, float16, float32)

        Returns:
            Cached or newly created WhisperModel instance
        """
        cache_key = f"{model_size}-{device}-{compute_type}"

        if cache_key not in cls._models:
            logger.info(
                f"Loading Whisper model: {model_size} on {device} ({compute_type})"
            )
            cls._models[cache_key] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
            logger.info(f"Whisper model loaded: {cache_key}")

        return cls._models[cache_key]


class Transcriber:
    """Transcribe speech segments using Faster-Whisper

    Provides GPU-optimized transcription with automatic CPU fallback,
    model caching, and batch processing for memory efficiency.

    Attributes:
        config: ASRConfig with model and processing parameters
        model: Cached WhisperModel instance
        BATCH_SIZE: Default batch size for memory management

    Example:
        >>> transcriber = Transcriber()
        >>> segment = SpeechSegment(start=0.0, end=2.5)
        >>> result = transcriber.transcribe_segment("audio.wav", segment)
        >>> print(result.text)
        "Hello world"
    """

    BATCH_SIZE = 50  # Process in batches for memory management

    def __init__(self, config: Optional[ASRConfig] = None):
        """Initialize Transcriber

        Args:
            config: ASRConfig instance (default: ASRConfig())
        """
        self.config = config or ASRConfig()

        # Validate CUDA availability and fallback if needed
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.config.device = "cpu"
            self.config.compute_type = "int8"

        # Try to load model, fallback to CPU on error
        try:
            self.model = ModelManager.get_model(
                self.config.model_size, self.config.device, self.config.compute_type
            )
        except ValueError as e:
            # Fallback if compute type not supported (e.g., float16 on old GPUs)
            if "float16" in str(e) and self.config.device == "cuda":
                logger.warning(f"float16 not supported, falling back to int8: {e}")
                self.config.compute_type = "int8"
                self.model = ModelManager.get_model(
                    self.config.model_size, self.config.device, self.config.compute_type
                )
            else:
                raise

        logger.info(f"Transcriber initialized with config: {self.config}")

    def transcribe_segment(
        self, audio_path: Union[str, Path], segment: SpeechSegment
    ) -> TranscriptSegment:
        """Transcribe a single speech segment

        Args:
            audio_path: Path to audio file
            segment: SpeechSegment with start/end timing

        Returns:
            TranscriptSegment with transcribed text and original timing

        Raises:
            TranscriptionError: If transcription fails

        Example:
            >>> transcriber = Transcriber()
            >>> segment = SpeechSegment(start=1.0, end=3.5)
            >>> result = transcriber.transcribe_segment("audio.wav", segment)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        try:
            logger.debug(
                f"Transcribing segment {segment.start:.2f}s - {segment.end:.2f}s"
            )

            # Transcribe with Faster-Whisper using clip_timestamps
            # This only processes the specified time range, avoiding redundant work
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                beam_size=self.config.beam_size,
                language=self.config.language,
                vad_filter=False,  # We already did VAD
                clip_timestamps=[segment.start, segment.end],  # Only this range
            )

            # Collect all text from the clipped segment
            text_parts = []
            for whisper_seg in segments_iter:
                text_parts.append(whisper_seg.text.strip())

            # Combine all text parts
            text = " ".join(text_parts).strip()

            # Handle empty transcription (silence or non-speech audio)
            # TranscriptSegment requires non-empty, non-whitespace text
            if not text:
                text = "[no speech]"

            # Create TranscriptSegment with original timing
            result = TranscriptSegment(start=segment.start, end=segment.end, text=text)

            logger.debug(f"Transcribed: '{text[:50]}...'")

            return result

        except RuntimeError as e:
            # Catch CUDA/cuDNN errors specifically
            error_msg = str(e).lower()
            if "cuda" in error_msg or "cudnn" in error_msg or "gpu" in error_msg:
                raise TranscriptionError(
                    f"GPU/CUDA error during transcription. Your GPU may not be "
                    f"supported or drivers may be incompatible. "
                    f"Try using --device cpu. Error: {e}"
                ) from e
            raise TranscriptionError(
                f"Transcription failed for segment "
                f"{segment.start:.2f}-{segment.end:.2f}: {e}"
            ) from e
        except Exception as e:
            raise TranscriptionError(
                f"Transcription failed for segment "
                f"{segment.start:.2f}-{segment.end:.2f}: {e}"
            ) from e

    def transcribe_segments(
        self, audio_path: Union[str, Path], segments: list[SpeechSegment]
    ) -> list[TranscriptSegment]:
        """Transcribe multiple speech segments with batch processing

        Args:
            audio_path: Path to audio file
            segments: List of SpeechSegment objects

        Returns:
            List of TranscriptSegment objects (same order as input)

        Raises:
            TranscriptionError: If transcription fails

        Example:
            >>> transcriber = Transcriber()
            >>> segments = [SpeechSegment(0, 2), SpeechSegment(3, 5)]
            >>> results = transcriber.transcribe_segments("audio.wav", segments)
        """
        if not segments:
            return []

        logger.info(f"Transcribing {len(segments)} segments")

        results = []

        # Process in batches for memory management
        for i in range(0, len(segments), self.BATCH_SIZE):
            batch = segments[i : i + self.BATCH_SIZE]

            logger.debug(
                f"Processing batch {i // self.BATCH_SIZE + 1}/"
                f"{(len(segments) + self.BATCH_SIZE - 1) // self.BATCH_SIZE}"
            )

            # Transcribe each segment in batch
            for segment in batch:
                result = self.transcribe_segment(audio_path, segment)
                results.append(result)

            # Clean up GPU memory after each batch
            if self.config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"Transcription complete: {len(results)} segments "
            f"({sum(len(r.text) for r in results)} chars)"
        )

        return results
