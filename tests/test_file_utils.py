"""Tests for file utility functions

Tests follow TDD RED phase - these tests will fail until file_utils.py is implemented.
"""

from pathlib import Path

import pytest

from aitranscript.utils.file_utils import (
    TempFileManager,
    create_temp_audio_path,
    safe_cleanup,
)


class TestTempFileManager:
    """Test TempFileManager context manager for automatic cleanup"""

    def test_temp_file_manager_creates_context(self, tmp_path):
        """Should create context and return path"""
        temp_file = tmp_path / "temp.wav"
        with TempFileManager(temp_file) as path:
            assert isinstance(path, Path)
            assert path == temp_file

    def test_temp_file_manager_cleanup_on_success(self, tmp_path):
        """Should delete file after successful context exit"""
        temp_file = tmp_path / "temp.wav"
        temp_file.write_text("test content")

        with TempFileManager(temp_file):
            assert temp_file.exists()

        # File should be deleted after context
        assert not temp_file.exists()

    def test_temp_file_manager_cleanup_on_exception(self, tmp_path):
        """Should delete file even when exception occurs"""
        temp_file = tmp_path / "temp.wav"
        temp_file.write_text("test content")

        with pytest.raises(ValueError):
            with TempFileManager(temp_file):
                assert temp_file.exists()
                raise ValueError("Test error")

        # File should still be deleted
        assert not temp_file.exists()

    def test_temp_file_manager_no_cleanup_if_keep(self, tmp_path):
        """Should not delete file when keep=True"""
        temp_file = tmp_path / "temp.wav"
        temp_file.write_text("test content")

        with TempFileManager(temp_file, keep=True):
            assert temp_file.exists()

        # File should still exist
        assert temp_file.exists()

    def test_temp_file_manager_no_error_if_file_not_exists(self, tmp_path):
        """Should not raise error if file doesn't exist during cleanup"""
        temp_file = tmp_path / "nonexistent.wav"

        # Should not raise even though file doesn't exist
        with TempFileManager(temp_file):
            pass

    def test_temp_file_manager_nested_contexts(self, tmp_path):
        """Should handle nested context managers correctly"""
        temp_file1 = tmp_path / "temp1.wav"
        temp_file2 = tmp_path / "temp2.wav"
        temp_file1.write_text("content1")
        temp_file2.write_text("content2")

        with TempFileManager(temp_file1):
            assert temp_file1.exists()
            with TempFileManager(temp_file2):
                assert temp_file2.exists()
            assert not temp_file2.exists()
        assert not temp_file1.exists()


class TestCreateTempAudioPath:
    """Test temporary audio path generation"""

    def test_create_temp_audio_path_unique(self):
        """Should generate unique paths on each call"""
        path1 = create_temp_audio_path()
        path2 = create_temp_audio_path()
        assert path1 != path2

    def test_create_temp_audio_path_uses_prefix(self):
        """Should use custom prefix in filename"""
        path = create_temp_audio_path(prefix="custom_audio")
        assert path.name.startswith("custom_audio_")

    def test_create_temp_audio_path_wav_extension(self):
        """Should always use .wav extension"""
        path = create_temp_audio_path()
        assert path.suffix == ".wav"

    def test_create_temp_audio_path_custom_dir(self, tmp_path):
        """Should create path in custom directory"""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        path = create_temp_audio_path(dir=custom_dir)
        assert path.parent == custom_dir

    def test_create_temp_audio_path_default_cwd(self):
        """Should use current working directory by default"""
        path = create_temp_audio_path()
        assert path.parent == Path.cwd()


class TestSafeCleanup:
    """Test safe file cleanup utility"""

    def test_safe_cleanup_removes_file(self, tmp_path):
        """Should remove existing file"""
        test_file = tmp_path / "test.wav"
        test_file.write_text("test content")

        safe_cleanup(test_file)
        assert not test_file.exists()

    def test_safe_cleanup_nonexistent_file(self, tmp_path):
        """Should not raise error for nonexistent file"""
        nonexistent = tmp_path / "nonexistent.wav"
        # Should not raise
        safe_cleanup(nonexistent)

    def test_safe_cleanup_multiple_files(self, tmp_path):
        """Should handle multiple files in one call"""
        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"
        file3 = tmp_path / "file3.wav"

        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        safe_cleanup(file1, file2, file3)

        assert not file1.exists()
        assert not file2.exists()
        assert not file3.exists()

    def test_safe_cleanup_partial_failure_continues(self, tmp_path):
        """Should continue cleanup even if one file fails"""
        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "nonexistent.wav"  # Doesn't exist
        file3 = tmp_path / "file3.wav"

        file1.write_text("content1")
        file3.write_text("content3")

        # Should not raise, should cleanup file1 and file3
        safe_cleanup(file1, file2, file3)

        assert not file1.exists()
        assert not file3.exists()
