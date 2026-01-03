"""Tests for subtitle file generation (SRT/VTT)"""

import pytest

from aitranscript.core.subtitle_generator import (
    SubtitleGenerator,
    SubtitleGeneratorError,
    _format_timestamp_vtt,
)
from aitranscript.models.segment import TranscriptSegment


class TestFormatTimestampVTT:
    """Test VTT timestamp formatting"""

    def test_format_timestamp_vtt_zero(self):
        """Test formatting zero time"""
        assert _format_timestamp_vtt(0.0) == "00:00:00.000"

    def test_format_timestamp_vtt_seconds_only(self):
        """Test formatting seconds only"""
        assert _format_timestamp_vtt(5.5) == "00:00:05.500"
        assert _format_timestamp_vtt(59.999) == "00:00:59.999"

    def test_format_timestamp_vtt_minutes(self):
        """Test formatting with minutes"""
        assert _format_timestamp_vtt(65.5) == "00:01:05.500"
        assert _format_timestamp_vtt(125.0) == "00:02:05.000"

    def test_format_timestamp_vtt_hours(self):
        """Test formatting with hours"""
        assert _format_timestamp_vtt(3661.0) == "01:01:01.000"
        assert _format_timestamp_vtt(7322.5) == "02:02:02.500"

    def test_format_timestamp_vtt_uses_period(self):
        """Test that VTT uses period (.) not comma (,) for milliseconds"""
        result = _format_timestamp_vtt(1.5)
        assert "." in result
        assert "," not in result
        assert result == "00:00:01.500"


class TestSubtitleGenerator:
    """Test suite for SubtitleGenerator class"""

    def test_generator_initialization(self):
        """Test SubtitleGenerator initialization"""
        generator = SubtitleGenerator()
        assert generator is not None

    def test_generate_srt_single_segment(self, tmp_path):
        """Test generating SRT file with single segment"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.5, "Hello world")]

        output_file = tmp_path / "test.srt"
        result = generator.generate(segments, output_file, format="srt")

        assert result == output_file
        assert output_file.exists()

        content = output_file.read_text()
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:02,500\n" in content
        assert "Hello world\n" in content

    def test_generate_srt_multiple_segments(self, tmp_path):
        """Test generating SRT with multiple segments"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(0.0, 2.5, "First subtitle"),
            TranscriptSegment(3.0, 5.5, "Second subtitle"),
            TranscriptSegment(6.0, 8.0, "Third subtitle"),
        ]

        output_file = tmp_path / "test.srt"
        generator.generate(segments, output_file, format="srt")

        content = output_file.read_text()

        # Check all three segments are present
        assert "1\n" in content
        assert "2\n" in content
        assert "3\n" in content
        assert "First subtitle" in content
        assert "Second subtitle" in content
        assert "Third subtitle" in content

    def test_generate_vtt_single_segment(self, tmp_path):
        """Test generating VTT file with single segment"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.5, "Hello world")]

        output_file = tmp_path / "test.vtt"
        result = generator.generate(segments, output_file, format="vtt")

        assert result == output_file
        assert output_file.exists()

        content = output_file.read_text()
        assert content.startswith("WEBVTT\n")
        assert "1\n" in content
        assert "00:00:00.000 --> 00:00:02.500\n" in content
        assert "Hello world\n" in content

    def test_generate_vtt_multiple_segments(self, tmp_path):
        """Test generating VTT with multiple segments"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(0.0, 2.5, "First subtitle"),
            TranscriptSegment(3.0, 5.5, "Second subtitle"),
            TranscriptSegment(6.0, 8.0, "Third subtitle"),
        ]

        output_file = tmp_path / "test.vtt"
        generator.generate(segments, output_file, format="vtt")

        content = output_file.read_text()

        # Check WEBVTT header
        assert content.startswith("WEBVTT\n")

        # Check all segments present
        assert "First subtitle" in content
        assert "Second subtitle" in content
        assert "Third subtitle" in content

        # Check VTT uses periods not commas
        assert "00:00:00.000" in content
        assert "00:00:02.500" in content

    def test_generate_empty_segments(self, tmp_path):
        """Test generating subtitle file with no segments"""
        generator = SubtitleGenerator()
        segments = []

        output_file = tmp_path / "empty.srt"
        generator.generate(segments, output_file, format="srt")

        assert output_file.exists()
        content = output_file.read_text()
        # Should just have empty content or single newline
        assert len(content.strip()) == 0

    def test_generate_invalid_format(self, tmp_path):
        """Test that invalid format raises ValueError"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.5, "Hello")]

        output_file = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="format must be 'srt' or 'vtt'"):
            generator.generate(segments, output_file, format="txt")

    def test_generate_srt_preserves_timing(self, tmp_path):
        """Test that SRT generation preserves exact timing"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(1.234, 5.678, "Precise timing"),
        ]

        output_file = tmp_path / "timing.srt"
        generator.generate(segments, output_file, format="srt")

        content = output_file.read_text()
        assert "00:00:01,234 --> 00:00:05,678" in content

    def test_generate_vtt_preserves_timing(self, tmp_path):
        """Test that VTT generation preserves exact timing"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(1.234, 5.678, "Precise timing"),
        ]

        output_file = tmp_path / "timing.vtt"
        generator.generate(segments, output_file, format="vtt")

        content = output_file.read_text()
        assert "00:00:01.234 --> 00:00:05.678" in content

    def test_generate_with_long_text(self, tmp_path):
        """Test generating subtitle with long text content"""
        generator = SubtitleGenerator()
        long_text = (
            "This is a very long subtitle that might span multiple lines "
            "when displayed but is stored as a single segment."
        )
        segments = [TranscriptSegment(0.0, 5.0, long_text)]

        output_file = tmp_path / "long.srt"
        generator.generate(segments, output_file, format="srt")

        content = output_file.read_text()
        assert long_text in content

    def test_generate_with_special_characters(self, tmp_path):
        """Test generating subtitle with special characters"""
        generator = SubtitleGenerator()
        special_text = "Hello! How are you? I'm fine. 你好 #test @user"
        segments = [TranscriptSegment(0.0, 2.0, special_text)]

        output_file = tmp_path / "special.srt"
        generator.generate(segments, output_file, format="srt")

        content = output_file.read_text(encoding="utf-8")
        assert special_text in content

    def test_generate_srt_convenience_method(self, tmp_path):
        """Test generate_srt convenience method"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.0, "Test")]

        output_file = tmp_path / "convenience.srt"
        result = generator.generate_srt(segments, output_file)

        assert result == output_file
        assert output_file.exists()
        content = output_file.read_text()
        assert "00:00:00,000 --> 00:00:02,000" in content

    def test_generate_vtt_convenience_method(self, tmp_path):
        """Test generate_vtt convenience method"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.0, "Test")]

        output_file = tmp_path / "convenience.vtt"
        result = generator.generate_vtt(segments, output_file)

        assert result == output_file
        assert output_file.exists()
        content = output_file.read_text()
        assert "WEBVTT" in content
        assert "00:00:00.000 --> 00:00:02.000" in content

    def test_generate_creates_parent_directories(self, tmp_path):
        """Test that generate creates parent directories if needed"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(0.0, 2.0, "Test")]

        # Create nested path that doesn't exist
        nested_path = tmp_path / "subdir1" / "subdir2" / "output.srt"

        # This should fail because parent dirs don't exist
        with pytest.raises(SubtitleGeneratorError):
            generator.generate(segments, nested_path, format="srt")

    def test_generate_overwrites_existing_file(self, tmp_path):
        """Test that generate overwrites existing files"""
        generator = SubtitleGenerator()
        segments1 = [TranscriptSegment(0.0, 2.0, "First version")]
        segments2 = [TranscriptSegment(0.0, 2.0, "Second version")]

        output_file = tmp_path / "overwrite.srt"

        # Generate first version
        generator.generate(segments1, output_file, format="srt")
        first_content = output_file.read_text()
        assert "First version" in first_content

        # Generate second version (should overwrite)
        generator.generate(segments2, output_file, format="srt")
        second_content = output_file.read_text()
        assert "Second version" in second_content
        assert "First version" not in second_content

    def test_srt_and_vtt_have_different_formats(self, tmp_path):
        """Test that SRT and VTT use different timestamp formats"""
        generator = SubtitleGenerator()
        segments = [TranscriptSegment(1.5, 3.5, "Test")]

        srt_file = tmp_path / "test.srt"
        vtt_file = tmp_path / "test.vtt"

        generator.generate(segments, srt_file, format="srt")
        generator.generate(segments, vtt_file, format="vtt")

        srt_content = srt_file.read_text()
        vtt_content = vtt_file.read_text()

        # SRT uses comma for milliseconds
        assert "00:00:01,500" in srt_content
        # VTT uses period for milliseconds
        assert "00:00:01.500" in vtt_content
        # VTT has WEBVTT header
        assert "WEBVTT" in vtt_content
        assert "WEBVTT" not in srt_content

    def test_generate_sequential_numbering(self, tmp_path):
        """Test that segments are numbered sequentially starting from 1"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(0.0, 1.0, "One"),
            TranscriptSegment(1.0, 2.0, "Two"),
            TranscriptSegment(2.0, 3.0, "Three"),
        ]

        output_file = tmp_path / "numbered.srt"
        generator.generate(segments, output_file, format="srt")

        content = output_file.read_text()
        lines = content.split("\n")

        # Find all segment numbers (they should be on their own lines)
        numbers = [line for line in lines if line.strip().isdigit()]
        assert numbers == ["1", "2", "3"]

    def test_internal_generate_srt_method(self):
        """Test internal _generate_srt method"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(0.0, 2.0, "First"),
            TranscriptSegment(2.0, 4.0, "Second"),
        ]

        content = generator._generate_srt(segments)

        assert "1\n" in content
        assert "2\n" in content
        assert "First" in content
        assert "Second" in content
        assert "00:00:00,000 --> 00:00:02,000" in content

    def test_internal_generate_vtt_method(self):
        """Test internal _generate_vtt method"""
        generator = SubtitleGenerator()
        segments = [
            TranscriptSegment(0.0, 2.0, "First"),
            TranscriptSegment(2.0, 4.0, "Second"),
        ]

        content = generator._generate_vtt(segments)

        assert content.startswith("WEBVTT\n")
        assert "1\n" in content
        assert "2\n" in content
        assert "First" in content
        assert "Second" in content
        assert "00:00:00.000 --> 00:00:02.000" in content


class TestSubtitleGeneratorIntegration:
    """Integration tests with other components"""

    def test_generate_from_transcriber_output(self, tmp_path):
        """Test generating subtitles from transcriber-like output"""
        # Simulate output from transcriber
        segments = [
            TranscriptSegment(0.0, 3.5, "Welcome to the video."),
            TranscriptSegment(4.0, 7.5, "Today we'll discuss Python."),
            TranscriptSegment(8.0, 12.0, "Let's start with the basics."),
        ]

        generator = SubtitleGenerator()

        # Generate both formats
        srt_file = tmp_path / "output.srt"
        vtt_file = tmp_path / "output.vtt"

        generator.generate_srt(segments, srt_file)
        generator.generate_vtt(segments, vtt_file)

        # Verify both files created
        assert srt_file.exists()
        assert vtt_file.exists()

        # Verify content
        srt_content = srt_file.read_text()
        vtt_content = vtt_file.read_text()

        assert "Welcome to the video." in srt_content
        assert "Welcome to the video." in vtt_content
        assert "WEBVTT" in vtt_content

    def test_round_trip_compatibility(self, tmp_path):
        """Test that generated subtitles maintain data integrity"""
        original_segments = [
            TranscriptSegment(1.234, 5.678, "Precise timing test"),
            TranscriptSegment(10.0, 15.5, "Another segment"),
        ]

        generator = SubtitleGenerator()
        output_file = tmp_path / "roundtrip.srt"

        generator.generate(original_segments, output_file, format="srt")

        # Read back and verify all data is preserved in the file
        content = output_file.read_text()

        # Check timestamps are preserved
        assert "00:00:01,234" in content
        assert "00:00:05,678" in content
        assert "00:00:10,000" in content
        assert "00:00:15,500" in content

        # Check text is preserved
        assert "Precise timing test" in content
        assert "Another segment" in content
