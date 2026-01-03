"""Data models for speech and transcript segments"""

from dataclasses import dataclass


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string in SRT format

    Example:
        >>> _format_timestamp(65.5)
        '00:01:05,500'
        >>> _format_timestamp(3661.0)
        '01:01:01,000'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


@dataclass
class SpeechSegment:
    """Represents a detected speech segment from VAD

    Attributes:
        start: Start time in seconds
        end: End time in seconds

    Example:
        >>> segment = SpeechSegment(start=0.0, end=2.5)
        >>> segment.duration
        2.5
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds

        Returns:
            Duration in seconds (end - start)
        """
        return self.end - self.start

    def __post_init__(self):
        """Validate segment times

        Raises:
            ValueError: If start is negative or start >= end
        """
        if self.start < 0:
            raise ValueError("start time must be non-negative")
        if self.start >= self.end:
            raise ValueError("start time must be before end time")


@dataclass
class TranscriptSegment:
    """Represents transcribed text with timing information

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        text: Transcribed text content

    Example:
        >>> segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        >>> segment.duration
        2.5
        >>> srt = segment.to_srt_format(index=1)
    """

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds

        Returns:
            Duration in seconds (end - start)
        """
        return self.end - self.start

    def to_srt_format(self, index: int) -> str:
        """Convert to SRT subtitle format

        Args:
            index: Subtitle index number (1-based)

        Returns:
            Formatted SRT subtitle block

        Example:
            >>> segment = TranscriptSegment(0.0, 2.5, "Hello world")
            >>> print(segment.to_srt_format(1))
            1
            00:00:00,000 --> 00:00:02,500
            Hello world
        """
        start_time = _format_timestamp(self.start)
        end_time = _format_timestamp(self.end)
        return f"{index}\n{start_time} --> {end_time}\n{self.text}\n"

    def to_dict(self) -> dict:
        """Serialize to dictionary

        Returns:
            Dictionary with segment data including computed duration

        Example:
            >>> segment = TranscriptSegment(0.0, 2.5, "Hello")
            >>> segment.to_dict()
            {'start': 0.0, 'end': 2.5, 'text': 'Hello', 'duration': 2.5}
        """
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "duration": self.duration,
        }

    def __post_init__(self):
        """Validate segment data

        Raises:
            ValueError: If start is negative, start >= end, or text is empty
        """
        if self.start < 0:
            raise ValueError("start time must be non-negative")
        if self.start >= self.end:
            raise ValueError("start time must be before end time")
        if not self.text or not self.text.strip():
            raise ValueError("text cannot be empty")
