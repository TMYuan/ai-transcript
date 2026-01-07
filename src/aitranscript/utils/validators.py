"""Input/output validation utilities for pipeline processing

Provides validation functions for file paths, formats, and directory management.
"""

from pathlib import Path
from typing import Union

# Supported file formats
VIDEO_FORMATS = ["mp4", "mkv", "avi", "mov", "flv", "wmv"]
AUDIO_FORMATS = ["mp3", "wav", "flac", "aac", "m4a", "ogg"]


class ValidationError(Exception):
    """Raised when validation fails"""


def validate_input_file(path: Union[str, Path]) -> Path:
    """Validate input file exists and is readable

    Args:
        path: Path to input file (string or Path object)

    Returns:
        Path: Resolved absolute path to the file

    Raises:
        ValidationError: If file doesn't exist or is not a file

    Example:
        >>> validate_input_file("video.mp4")
        PosixPath('/absolute/path/to/video.mp4')
    """
    path = Path(path).resolve()

    if not path.exists():
        raise ValidationError(f"Input file not found: {path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")

    return path


def validate_output_path(
    path: Union[str, Path], overwrite: bool = True, create_dirs: bool = True
) -> Path:
    """Validate output path is writable

    Args:
        path: Path to output file (string or Path object)
        overwrite: If True, allow overwriting existing files (default: True)
        create_dirs: If True, create parent directories if needed (default: True)

    Returns:
        Path: Resolved absolute path to the output file

    Raises:
        ValidationError: If file exists and overwrite=False

    Example:
        >>> validate_output_path("output/subtitles.srt")
        PosixPath('/absolute/path/to/output/subtitles.srt')
    """
    path = Path(path).resolve()

    if path.exists() and not overwrite:
        raise ValidationError(f"Output file already exists: {path}")

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def validate_file_format(path: str, allowed: list[str]) -> bool:
    """Check if file extension is in allowed list

    Args:
        path: File path to check
        allowed: List of allowed format types ("video", "audio")

    Returns:
        bool: True if format is valid

    Raises:
        ValidationError: If file extension is not supported

    Example:
        >>> validate_file_format("video.mp4", ["video"])
        True
        >>> validate_file_format("doc.txt", ["video", "audio"])
        ValidationError: Unsupported format: .txt
    """
    ext = Path(path).suffix.lower().lstrip(".")

    if "video" in allowed and ext in VIDEO_FORMATS:
        return True

    if "audio" in allowed and ext in AUDIO_FORMATS:
        return True

    raise ValidationError(f"Unsupported format: .{ext}")


def is_video_file(path: Union[str, Path]) -> bool:
    """Check if file is a video

    Args:
        path: File path to check

    Returns:
        bool: True if file has a video extension, False otherwise

    Example:
        >>> is_video_file("movie.mp4")
        True
        >>> is_video_file("audio.mp3")
        False
    """
    try:
        validate_file_format(str(path), ["video"])
        return True
    except ValidationError:
        return False


def is_audio_file(path: Union[str, Path]) -> bool:
    """Check if file is audio

    Args:
        path: File path to check

    Returns:
        bool: True if file has an audio extension, False otherwise

    Example:
        >>> is_audio_file("music.mp3")
        True
        >>> is_audio_file("video.mp4")
        False
    """
    try:
        validate_file_format(str(path), ["audio"])
        return True
    except ValidationError:
        return False


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating if needed

    Args:
        path: Directory path to create

    Returns:
        Path: Path to the created/existing directory

    Example:
        >>> ensure_directory("output/subtitles")
        PosixPath('/absolute/path/to/output/subtitles')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
