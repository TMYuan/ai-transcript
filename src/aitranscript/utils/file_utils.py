"""File management utilities for pipeline processing

Provides utilities for temporary file management, path generation, and cleanup.
"""

import uuid
from pathlib import Path
from typing import Optional

from aitranscript.utils.logger import get_logger

logger = get_logger(__name__)


class TempFileManager:
    """Context manager for automatic temporary file cleanup

    Ensures temporary files are deleted after use, even if exceptions occur.

    Example:
        >>> temp_path = Path("temp_audio.wav")
        >>> with TempFileManager(temp_path) as temp:
        ...     temp.write_text("audio data")
        ...     # File is automatically deleted on exit
    """

    def __init__(self, path: Path, keep: bool = False):
        """Initialize temp file manager

        Args:
            path: Path to temporary file
            keep: If True, do not delete file on exit (default: False)
        """
        self.path = Path(path)
        self.keep = keep

    def __enter__(self) -> Path:
        """Enter context, return path"""
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, cleanup file if not keeping"""
        if not self.keep and self.path.exists():
            try:
                self.path.unlink()
                logger.debug(f"Cleaned up temporary file: {self.path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {self.path}: {e}")
        return False  # Don't suppress exceptions


def create_temp_audio_path(
    prefix: str = "temp_audio", dir: Optional[Path] = None
) -> Path:
    """Create unique temporary audio file path

    Args:
        prefix: Filename prefix (default: "temp_audio")
        dir: Directory for temp file (default: current working directory)

    Returns:
        Path: Unique path for temporary audio file with .wav extension

    Example:
        >>> create_temp_audio_path()
        PosixPath('/current/dir/temp_audio_a1b2c3d4.wav')
        >>> create_temp_audio_path(prefix="custom", dir=Path("/tmp"))
        PosixPath('/tmp/custom_e5f6g7h8.wav')
    """
    dir = dir or Path.cwd()
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    return dir / filename


def safe_cleanup(*paths: Path) -> None:
    """Safely remove files, ignoring errors

    Attempts to delete all provided files. Continues even if some deletions fail.
    Logs debug messages for failures but does not raise exceptions.

    Args:
        *paths: Variable number of file paths to delete

    Example:
        >>> safe_cleanup(Path("temp1.wav"), Path("temp2.wav"))
        # Both files deleted, no errors even if files don't exist
    """
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            logger.debug(f"Cleaned up file: {path}")
        except Exception as e:
            logger.debug(f"Cleanup failed for {path}: {e}")
