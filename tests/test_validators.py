"""Tests for input/output validation utilities

Tests follow TDD RED phase - these tests will fail until validators.py is implemented.
"""

from pathlib import Path

import pytest

from aitranscript.utils.validators import (
    ValidationError,
    ensure_directory,
    is_audio_file,
    is_video_file,
    validate_file_format,
    validate_input_file,
    validate_output_path,
)


class TestValidateInputFile:
    """Test input file validation"""

    def test_validate_input_file_exists(self, sample_audio_wav):
        """Should accept existing audio file"""
        result = validate_input_file(sample_audio_wav)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_file()

    def test_validate_input_file_nonexistent(self):
        """Should raise ValidationError for nonexistent file"""
        with pytest.raises(ValidationError, match="Input file not found"):
            validate_input_file("/nonexistent/file.mp4")

    def test_validate_input_file_accepts_string(self, sample_audio_wav):
        """Should accept string path and convert to Path"""
        result = validate_input_file(str(sample_audio_wav))
        assert isinstance(result, Path)
        assert result == sample_audio_wav

    def test_validate_input_file_directory_fails(self, tmp_path):
        """Should raise ValidationError when path is a directory"""
        with pytest.raises(ValidationError, match="Path is not a file"):
            validate_input_file(tmp_path)


class TestValidateOutputPath:
    """Test output path validation"""

    def test_validate_output_path_new_file(self, tmp_output_dir):
        """Should accept new file path"""
        output_file = tmp_output_dir / "output.srt"
        result = validate_output_path(output_file)
        assert isinstance(result, Path)
        assert result == output_file.resolve()

    def test_validate_output_path_creates_parent_dirs(self, tmp_path):
        """Should create parent directories when create_dirs=True"""
        output_file = tmp_path / "nested" / "dirs" / "output.srt"
        result = validate_output_path(output_file, create_dirs=True)
        assert result.parent.exists()
        assert result.parent.is_dir()

    def test_validate_output_path_overwrite_false(self, tmp_output_dir):
        """Should raise ValidationError when file exists and overwrite=False"""
        existing_file = tmp_output_dir / "existing.srt"
        existing_file.write_text("existing content")

        with pytest.raises(ValidationError, match="Output file already exists"):
            validate_output_path(existing_file, overwrite=False)

    def test_validate_output_path_overwrite_true(self, tmp_output_dir):
        """Should accept existing file when overwrite=True"""
        existing_file = tmp_output_dir / "existing.srt"
        existing_file.write_text("existing content")

        result = validate_output_path(existing_file, overwrite=True)
        assert isinstance(result, Path)
        assert result == existing_file.resolve()


class TestValidateFileFormat:
    """Test file format validation"""

    def test_validate_format_video_extensions(self):
        """Should accept common video formats"""
        video_files = [
            "video.mp4",
            "movie.mkv",
            "clip.avi",
            "recording.mov",
            "stream.flv",
            "film.wmv",
        ]
        for video_file in video_files:
            # Should not raise
            validate_file_format(video_file, ["video"])

    def test_validate_format_audio_extensions(self):
        """Should accept common audio formats"""
        audio_files = [
            "audio.mp3",
            "sound.wav",
            "music.flac",
            "voice.aac",
            "podcast.m4a",
            "song.ogg",
        ]
        for audio_file in audio_files:
            # Should not raise
            validate_file_format(audio_file, ["audio"])

    def test_validate_format_invalid_extension(self):
        """Should raise ValidationError for unsupported format"""
        with pytest.raises(ValidationError, match="Unsupported format: .txt"):
            validate_file_format("document.txt", ["video", "audio"])

    def test_validate_format_case_insensitive(self):
        """Should accept uppercase extensions"""
        # Should not raise
        validate_file_format("VIDEO.MP4", ["video"])
        validate_file_format("AUDIO.WAV", ["audio"])


class TestFileTypeCheckers:
    """Test is_video_file and is_audio_file helpers"""

    def test_is_video_file_true(self):
        """Should return True for video files"""
        assert is_video_file("video.mp4") is True
        assert is_video_file("movie.mkv") is True
        assert is_video_file(Path("clip.avi")) is True

    def test_is_video_file_false(self):
        """Should return False for non-video files"""
        assert is_video_file("audio.mp3") is False
        assert is_video_file("document.txt") is False

    def test_is_audio_file_true(self):
        """Should return True for audio files"""
        assert is_audio_file("audio.mp3") is True
        assert is_audio_file("sound.wav") is True
        assert is_audio_file(Path("music.flac")) is True

    def test_is_audio_file_false(self):
        """Should return False for non-audio files"""
        assert is_audio_file("video.mp4") is False
        assert is_audio_file("document.txt") is False


class TestEnsureDirectory:
    """Test directory creation utility"""

    def test_ensure_directory_creates_new(self, tmp_path):
        """Should create new directory"""
        new_dir = tmp_path / "new_directory"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()
        assert result == new_dir

    def test_ensure_directory_idempotent(self, tmp_path):
        """Should not fail if directory already exists"""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_directory(existing_dir)
        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_nested(self, tmp_path):
        """Should create nested directories"""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = ensure_directory(nested_dir)
        assert result.exists()
        assert result.is_dir()
        assert (tmp_path / "level1").exists()
        assert (tmp_path / "level1" / "level2").exists()
