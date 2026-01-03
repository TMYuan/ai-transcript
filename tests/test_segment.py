"""Tests for segment data models"""

import pytest

from aitranscript.models.segment import SpeechSegment, TranscriptSegment


class TestSpeechSegment:
    """Test suite for SpeechSegment dataclass"""

    def test_speech_segment_creation(self):
        """Verify SpeechSegment initialization with valid data"""
        segment = SpeechSegment(start=0.0, end=2.5)
        assert segment.start == 0.0
        assert segment.end == 2.5

    def test_speech_segment_duration(self):
        """Test duration calculation (end - start)"""
        segment = SpeechSegment(start=1.0, end=4.5)
        assert segment.duration == 3.5

    def test_speech_segment_validation_negative_start(self):
        """Test that negative start time raises ValueError"""
        with pytest.raises(ValueError, match="start time must be non-negative"):
            SpeechSegment(start=-1.0, end=2.0)

    def test_speech_segment_validation_start_after_end(self):
        """Test that start > end raises ValueError"""
        with pytest.raises(ValueError, match="start time must be before end time"):
            SpeechSegment(start=5.0, end=2.0)

    def test_speech_segment_validation_equal_times(self):
        """Test that start == end raises ValueError"""
        with pytest.raises(ValueError, match="start time must be before end time"):
            SpeechSegment(start=2.0, end=2.0)

    def test_speech_segment_zero_start(self):
        """Test that zero start time is valid"""
        segment = SpeechSegment(start=0.0, end=1.0)
        assert segment.start == 0.0
        assert segment.duration == 1.0


class TestTranscriptSegment:
    """Test suite for TranscriptSegment dataclass"""

    def test_transcript_segment_creation(self):
        """Verify TranscriptSegment initialization with valid data"""
        segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.text == "Hello world"

    def test_transcript_segment_duration(self):
        """Test duration calculation"""
        segment = TranscriptSegment(start=1.0, end=4.5, text="Test")
        assert segment.duration == 3.5

    def test_transcript_segment_to_srt_format(self):
        """Verify SRT timestamp format (HH:MM:SS,mmm --> HH:MM:SS,mmm)"""
        segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        srt_text = segment.to_srt_format(index=1)

        # Verify structure
        assert "1\n" in srt_text  # Index
        assert "00:00:00,000 --> 00:00:02,500" in srt_text  # Timestamp
        assert "Hello world" in srt_text  # Text

    def test_transcript_segment_to_srt_format_with_minutes(self):
        """Test SRT format with minutes and hours"""
        segment = TranscriptSegment(
            start=65.5, end=125.750, text="One minute mark"
        )  # 1:05.5 to 2:05.75
        srt_text = segment.to_srt_format(index=2)

        assert "2\n" in srt_text
        assert "00:01:05,500 --> 00:02:05,750" in srt_text
        assert "One minute mark" in srt_text

    def test_transcript_segment_to_srt_format_with_hours(self):
        """Test SRT format with hours"""
        segment = TranscriptSegment(
            start=3661.0, end=3665.5, text="Over an hour"
        )  # 1:01:01 to 1:01:05.5
        srt_text = segment.to_srt_format(index=3)

        assert "3\n" in srt_text
        assert "01:01:01,000 --> 01:01:05,500" in srt_text
        assert "Over an hour" in srt_text

    def test_transcript_segment_to_dict(self):
        """Test serialization to dictionary"""
        segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        data = segment.to_dict()

        assert data["start"] == 0.0
        assert data["end"] == 2.5
        assert data["text"] == "Hello world"
        assert data["duration"] == 2.5

    def test_transcript_segment_validation_empty_text(self):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError, match="text cannot be empty"):
            TranscriptSegment(start=0.0, end=2.0, text="")

    def test_transcript_segment_validation_whitespace_only(self):
        """Test that whitespace-only text raises ValueError"""
        with pytest.raises(ValueError, match="text cannot be empty"):
            TranscriptSegment(start=0.0, end=2.0, text="   ")

    def test_transcript_segment_validation_negative_start(self):
        """Test that negative start time raises ValueError"""
        with pytest.raises(ValueError, match="start time must be non-negative"):
            TranscriptSegment(start=-1.0, end=2.0, text="Test")

    def test_transcript_segment_validation_start_after_end(self):
        """Test that start > end raises ValueError"""
        with pytest.raises(ValueError, match="start time must be before end time"):
            TranscriptSegment(start=5.0, end=2.0, text="Test")

    def test_transcript_segment_preserves_whitespace_in_text(self):
        """Test that leading/trailing whitespace in meaningful text is preserved"""
        segment = TranscriptSegment(start=0.0, end=2.0, text=" Hello ")
        assert segment.text == " Hello "
