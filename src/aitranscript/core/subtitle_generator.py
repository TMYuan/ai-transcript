"""Generate subtitle files (SRT/VTT) from transcript segments"""

from pathlib import Path
from typing import Union

from aitranscript.models.segment import TranscriptSegment
from aitranscript.utils.logger import get_logger

logger = get_logger(__name__)


class SubtitleGeneratorError(Exception):
    """Raised when subtitle generation fails"""

    pass


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string in VTT format

    Example:
        >>> _format_timestamp_vtt(65.5)
        '00:01:05.500'
        >>> _format_timestamp_vtt(3661.0)
        '01:01:01.000'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class SubtitleGenerator:
    """Generate subtitle files from transcript segments

    Supports SRT (SubRip) and VTT (WebVTT) subtitle formats.

    Example:
        >>> segments = [
        ...     TranscriptSegment(0.0, 2.5, "Hello world"),
        ...     TranscriptSegment(3.0, 5.5, "How are you?")
        ... ]
        >>> generator = SubtitleGenerator()
        >>> generator.generate(segments, "output.srt", format="srt")
    """

    def generate(
        self,
        segments: list[TranscriptSegment],
        output_path: Union[str, Path],
        format: str = "srt",
    ) -> Path:
        """Generate subtitle file from transcript segments

        Args:
            segments: List of TranscriptSegment objects with transcribed text
            output_path: Path to output subtitle file
            format: Subtitle format - 'srt' or 'vtt' (default: 'srt')

        Returns:
            Path to generated subtitle file

        Raises:
            SubtitleGeneratorError: If generation fails
            ValueError: If format is invalid

        Example:
            >>> generator = SubtitleGenerator()
            >>> segments = [TranscriptSegment(0.0, 2.5, "Hello")]
            >>> generator.generate(segments, "output.srt")
            PosixPath('output.srt')
        """
        output_path = Path(output_path)

        if format not in ["srt", "vtt"]:
            raise ValueError("format must be 'srt' or 'vtt'")

        if not segments:
            logger.warning("No segments provided, creating empty subtitle file")

        try:
            logger.info(
                f"Generating {format.upper()} subtitle file: {output_path} "
                f"({len(segments)} segments)"
            )

            if format == "srt":
                content = self._generate_srt(segments)
            else:  # vtt
                content = self._generate_vtt(segments)

            # Write to file
            output_path.write_text(content, encoding="utf-8")

            logger.info(f"Subtitle file generated successfully: {output_path}")
            return output_path

        except Exception as e:
            raise SubtitleGeneratorError(f"Failed to generate subtitles: {e}") from e

    def _generate_srt(self, segments: list[TranscriptSegment]) -> str:
        """Generate SRT format content

        Args:
            segments: List of transcript segments

        Returns:
            Complete SRT file content as string

        Example:
            >>> generator = SubtitleGenerator()
            >>> segments = [TranscriptSegment(0.0, 2.5, "Hello")]
            >>> print(generator._generate_srt(segments))
            1
            00:00:00,000 --> 00:00:02,500
            Hello
            <BLANKLINE>
        """
        parts = []
        for index, segment in enumerate(segments, start=1):
            parts.append(segment.to_srt_format(index))

        return "\n".join(parts)

    def _generate_vtt(self, segments: list[TranscriptSegment]) -> str:
        """Generate VTT (WebVTT) format content

        Args:
            segments: List of transcript segments

        Returns:
            Complete VTT file content as string

        Example:
            >>> generator = SubtitleGenerator()
            >>> segments = [TranscriptSegment(0.0, 2.5, "Hello")]
            >>> print(generator._generate_vtt(segments))
            WEBVTT
            <BLANKLINE>
            1
            00:00:00.000 --> 00:00:02.500
            Hello
            <BLANKLINE>
        """
        parts = ["WEBVTT\n"]

        for index, segment in enumerate(segments, start=1):
            start_time = _format_timestamp_vtt(segment.start)
            end_time = _format_timestamp_vtt(segment.end)
            parts.append(f"{index}\n{start_time} --> {end_time}\n{segment.text}\n")

        return "\n".join(parts)

    def generate_srt(
        self, segments: list[TranscriptSegment], output_path: Union[str, Path]
    ) -> Path:
        """Generate SRT subtitle file (convenience method)

        Args:
            segments: List of transcript segments
            output_path: Path to output .srt file

        Returns:
            Path to generated SRT file

        Example:
            >>> generator = SubtitleGenerator()
            >>> segments = [TranscriptSegment(0.0, 2.5, "Hello")]
            >>> generator.generate_srt(segments, "output.srt")
            PosixPath('output.srt')
        """
        return self.generate(segments, output_path, format="srt")

    def generate_vtt(
        self, segments: list[TranscriptSegment], output_path: Union[str, Path]
    ) -> Path:
        """Generate VTT subtitle file (convenience method)

        Args:
            segments: List of transcript segments
            output_path: Path to output .vtt file

        Returns:
            Path to generated VTT file

        Example:
            >>> generator = SubtitleGenerator()
            >>> segments = [TranscriptSegment(0.0, 2.5, "Hello")]
            >>> generator.generate_vtt(segments, "output.vtt")
            PosixPath('output.vtt')
        """
        return self.generate(segments, output_path, format="vtt")
