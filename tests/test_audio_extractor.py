"""Tests for audio extraction from video files"""

from pathlib import Path

import pytest
from pydub import AudioSegment

from aitranscript.core.audio_extractor import (
    AudioExtractionError,
    AudioExtractor,
    extract_audio,
    validate_audio_properties,
)


class TestAudioExtractor:
    """Test suite for AudioExtractor class"""

    def test_extractor_initialization_default(self):
        """Verify AudioExtractor initializes with default settings"""
        extractor = AudioExtractor()

        assert extractor.sample_rate == 16000
        assert extractor.channels == 1
        assert extractor.output_format == "wav"

    def test_extractor_initialization_custom(self):
        """Test custom initialization parameters"""
        extractor = AudioExtractor(sample_rate=22050, channels=2)

        assert extractor.sample_rate == 22050
        assert extractor.channels == 2

    def test_extract_audio_from_wav(self, sample_audio_wav, tmp_path):
        """Test extracting audio from WAV file (passthrough with conversion)"""
        output_path = tmp_path / "output.wav"
        extractor = AudioExtractor()

        result = extractor.extract(sample_audio_wav, output_path)

        assert result.exists()
        assert result.suffix == ".wav"
        assert result == output_path

    def test_extract_audio_properties_16khz_mono(self, sample_audio_wav, tmp_path):
        """Verify extracted audio has correct properties: 16kHz mono"""
        output_path = tmp_path / "output.wav"
        extractor = AudioExtractor()

        result = extractor.extract(sample_audio_wav, output_path)

        # Validate audio properties
        audio = AudioSegment.from_wav(result)
        assert audio.frame_rate == 16000
        assert audio.channels == 1

    def test_extract_audio_from_different_rate(self, tmp_path):
        """Test extracting and converting audio with different sample rate"""
        # Create audio with 44100 Hz
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=1000)
        tone = tone.set_frame_rate(44100).set_channels(2)  # Stereo, 44.1kHz

        input_path = tmp_path / "input_44100.wav"
        tone.export(str(input_path), format="wav")

        # Extract and convert
        output_path = tmp_path / "output.wav"
        extractor = AudioExtractor()
        result = extractor.extract(input_path, output_path)

        # Should be converted to 16kHz mono
        audio = AudioSegment.from_wav(result)
        assert audio.frame_rate == 16000
        assert audio.channels == 1

    def test_extract_audio_invalid_input_file(self, tmp_path):
        """Test that nonexistent input file raises AudioExtractionError"""
        extractor = AudioExtractor()

        with pytest.raises(AudioExtractionError, match="Input file not found"):
            extractor.extract("nonexistent.mp4", tmp_path / "output.wav")

    def test_extract_audio_invalid_format(self, tmp_path):
        """Test that unsupported file format raises error"""
        extractor = AudioExtractor()

        # Create dummy file with unsupported extension
        invalid_file = tmp_path / "test.xyz"
        invalid_file.write_text("dummy content")

        with pytest.raises(AudioExtractionError, match="Unsupported format"):
            extractor.extract(invalid_file, tmp_path / "output.wav")

    def test_extract_audio_output_dir_created(self, sample_audio_wav, tmp_path):
        """Test that output directory is created if it doesn't exist"""
        output_path = tmp_path / "subdir" / "nested" / "output.wav"
        extractor = AudioExtractor()

        result = extractor.extract(sample_audio_wav, output_path)

        assert output_path.parent.exists()
        assert result.exists()

    def test_extract_audio_overwrite_existing(self, sample_audio_wav, tmp_path):
        """Test that existing output file is overwritten"""
        output_path = tmp_path / "output.wav"
        output_path.write_text("old content")

        extractor = AudioExtractor()
        result = extractor.extract(sample_audio_wav, output_path)

        assert result.exists()
        # File should be valid audio, not "old content"
        audio = AudioSegment.from_wav(result)
        assert audio.frame_rate == 16000

    def test_extract_audio_returns_path(self, sample_audio_wav, tmp_path):
        """Test that extract() returns Path object"""
        output_path = tmp_path / "output.wav"
        extractor = AudioExtractor()

        result = extractor.extract(sample_audio_wav, output_path)

        assert isinstance(result, Path)


class TestValidateAudioProperties:
    """Test audio property validation helper"""

    def test_validate_audio_properties_valid(self, sample_audio_wav):
        """Test validation passes for correct audio properties"""
        result = validate_audio_properties(
            sample_audio_wav, expected_rate=16000, expected_channels=1
        )

        assert result is True

    def test_validate_audio_properties_wrong_rate(self, tmp_path):
        """Test validation fails for wrong sample rate"""
        # Create audio with wrong rate
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=1000)
        tone = tone.set_frame_rate(22050).set_channels(1)

        wrong_rate_file = tmp_path / "wrong_rate.wav"
        tone.export(str(wrong_rate_file), format="wav")

        with pytest.raises(AudioExtractionError, match="Invalid sample rate"):
            validate_audio_properties(
                wrong_rate_file, expected_rate=16000, expected_channels=1
            )

    def test_validate_audio_properties_wrong_channels(self, tmp_path):
        """Test validation fails for wrong channel count"""
        # Create stereo audio
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=1000)
        tone = tone.set_frame_rate(16000).set_channels(2)  # Stereo

        stereo_file = tmp_path / "stereo.wav"
        tone.export(str(stereo_file), format="wav")

        with pytest.raises(AudioExtractionError, match="Invalid channels"):
            validate_audio_properties(
                stereo_file, expected_rate=16000, expected_channels=1
            )


class TestExtractAudioFunction:
    """Test standalone extract_audio() convenience function"""

    def test_extract_audio_function(self, sample_audio_wav, tmp_path):
        """Test convenience function with default parameters"""
        output_path = tmp_path / "output.wav"

        result = extract_audio(sample_audio_wav, output_path)

        assert result.exists()
        assert result.suffix == ".wav"

    def test_extract_audio_function_returns_path(self, sample_audio_wav, tmp_path):
        """Test that function returns Path object"""
        output_path = tmp_path / "output.wav"

        result = extract_audio(sample_audio_wav, output_path)

        assert isinstance(result, Path)

    def test_extract_audio_function_converts_properties(
        self, sample_audio_wav, tmp_path
    ):
        """Test that convenience function converts to 16kHz mono"""
        output_path = tmp_path / "output.wav"

        result = extract_audio(sample_audio_wav, output_path)

        audio = AudioSegment.from_wav(result)
        assert audio.frame_rate == 16000
        assert audio.channels == 1
