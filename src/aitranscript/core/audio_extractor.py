"""Audio extraction from video files using pydub"""

from pathlib import Path
from typing import Union

from pydub import AudioSegment

from aitranscript.utils.logger import get_logger

logger = get_logger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails"""

    pass


class AudioExtractor:
    """Extract and convert audio from video files to ASR-optimized format

    Converts video/audio files to 16kHz mono WAV format suitable for
    Whisper ASR processing.

    Attributes:
        sample_rate: Target sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        output_format: Output audio format (default: "wav")

    Example:
        >>> extractor = AudioExtractor()
        >>> output = extractor.extract("video.mp4", "audio.wav")
        >>> print(output)  # Path to extracted audio
    """

    SUPPORTED_FORMATS = {
        "mp4",
        "mkv",
        "avi",
        "mov",
        "flv",
        "wmv",  # Video
        "mp3",
        "wav",
        "flac",
        "aac",
        "m4a",
        "ogg",  # Audio
    }

    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, output_format: str = "wav"
    ):
        """Initialize AudioExtractor

        Args:
            sample_rate: Target sample rate (default: 16000 Hz)
            channels: Number of channels (default: 1 for mono)
            output_format: Output format (default: "wav")
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_format = output_format

    def extract(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Path:
        """Extract audio from video/audio file

        Args:
            input_path: Path to input video/audio file
            output_path: Path for output WAV file

        Returns:
            Path to extracted audio file

        Raises:
            AudioExtractionError: If extraction fails

        Example:
            >>> extractor = AudioExtractor()
            >>> output = extractor.extract("video.mp4", "audio.wav")
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Validation
        if not input_path.exists():
            raise AudioExtractionError(f"Input file not found: {input_path}")

        if input_path.suffix.lower().strip(".") not in self.SUPPORTED_FORMATS:
            raise AudioExtractionError(f"Unsupported format: {input_path.suffix}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Extracting audio from {input_path}")

            # Load audio (pydub auto-detects format)
            audio = AudioSegment.from_file(str(input_path))

            # Convert to target format
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(self.channels)

            # Export with ffmpeg
            audio.export(
                str(output_path),
                format=self.output_format,
                parameters=["-ac", "1"],  # Force mono
            )

            logger.info(
                f"Audio extracted: {output_path} "
                f"({self.sample_rate}Hz, {self.channels}ch)"
            )

            return output_path

        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio: {e}") from e


def validate_audio_properties(
    audio_path: Union[str, Path], expected_rate: int = 16000, expected_channels: int = 1
) -> bool:
    """Validate audio file properties

    Args:
        audio_path: Path to audio file
        expected_rate: Expected sample rate
        expected_channels: Expected number of channels

    Returns:
        True if validation passes

    Raises:
        AudioExtractionError: If properties don't match
    """
    audio = AudioSegment.from_wav(audio_path)

    if audio.frame_rate != expected_rate:
        raise AudioExtractionError(
            f"Invalid sample rate: {audio.frame_rate} (expected {expected_rate})"
        )

    if audio.channels != expected_channels:
        raise AudioExtractionError(
            f"Invalid channels: {audio.channels} (expected {expected_channels})"
        )

    return True


def extract_audio(input_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """Convenience function to extract audio with default settings

    Args:
        input_path: Path to input video/audio file
        output_path: Path for output WAV file

    Returns:
        Path to extracted audio file

    Example:
        >>> output = extract_audio("video.mp4", "audio.wav")
    """
    extractor = AudioExtractor()
    return extractor.extract(input_path, output_path)
