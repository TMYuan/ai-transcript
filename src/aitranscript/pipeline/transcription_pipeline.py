"""End-to-end transcription pipeline orchestration

Provides seamless workflow from video/audio input to subtitle output.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from aitranscript.core.audio_extractor import AudioExtractor
from aitranscript.core.subtitle_generator import SubtitleGenerator
from aitranscript.core.transcriber import Transcriber
from aitranscript.core.vad_processor import VADProcessor
from aitranscript.models.config import PipelineConfig
from aitranscript.utils.file_utils import create_temp_audio_path
from aitranscript.utils.logger import get_logger
from aitranscript.utils.validators import (
    ensure_directory,
    is_video_file,
    validate_input_file,
    validate_output_path,
)

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline processing

    Attributes:
        success: Whether processing completed successfully
        output_path: Path to generated subtitle file
        segment_count: Number of transcript segments generated
        duration_seconds: Total duration of speech detected
        processing_time_seconds: Time taken to process
        error: Error message if processing failed (None if successful)

    Example:
        >>> result = PipelineResult(
        ...     success=True,
        ...     output_path=Path("output.srt"),
        ...     segment_count=10,
        ...     duration_seconds=30.5,
        ...     processing_time_seconds=45.2
        ... )
        >>> result.success
        True
    """

    success: bool
    output_path: Path
    segment_count: int
    duration_seconds: float
    processing_time_seconds: float
    error: Optional[str] = None


class PipelineError(Exception):
    """Raised when pipeline processing fails"""


class TranscriptionPipeline:
    """End-to-end transcription pipeline

    Orchestrates the complete workflow:
    1. Validate inputs
    2. Extract audio from video (if needed)
    3. Detect speech segments with VAD
    4. Transcribe segments with Whisper
    5. Generate subtitle file (SRT or VTT)

    Example:
        >>> pipeline = TranscriptionPipeline()
        >>> result = pipeline.process("video.mp4", "output.srt")
        >>> print(f"Processed {result.segment_count} segments")

        >>> # Custom configuration
        >>> from aitranscript.models.config import PipelineConfig, ASRConfig
        >>> config = PipelineConfig(
        ...     asr=ASRConfig(model_size="tiny", device="cpu"),
        ...     output_format="vtt"
        ... )
        >>> pipeline = TranscriptionPipeline(config)
        >>> result = pipeline.process("audio.wav", "output.vtt")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize transcription pipeline

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        logger.info(
            f"Pipeline initialized with {self.config.asr.model_size} "
            f"model, {self.config.output_format.upper()} output"
        )

    def process(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> PipelineResult:
        """Process single file: video/audio → subtitles

        Args:
            input_path: Path to video or audio file
            output_path: Path to output subtitle file (.srt or .vtt)

        Returns:
            PipelineResult with metrics and status

        Example:
            >>> pipeline = TranscriptionPipeline()
            >>> result = pipeline.process("video.mp4", "output.srt")
            >>> if result.success:
            ...     print(f"Success! {result.segment_count} segments")
            ... else:
            ...     print(f"Failed: {result.error}")
        """
        start_time = time.time()
        temp_audio = None

        try:
            # 1. Validate inputs
            input_path = validate_input_file(input_path)
            output_path = validate_output_path(output_path)

            # 2. Extract audio if video
            if self.config.extract_audio and is_video_file(input_path):
                temp_audio = create_temp_audio_path()
                logger.info(f"Extracting audio from video: {input_path}")
                extractor = AudioExtractor()
                audio_path = extractor.extract(str(input_path), str(temp_audio))
            else:
                audio_path = str(input_path)

            # 3. VAD - Detect speech segments
            logger.info("Detecting speech segments...")
            vad = VADProcessor(self.config.vad)
            speech_segments = vad.process(audio_path)
            logger.info(f"Found {len(speech_segments)} speech segments")

            # 4. Transcription
            logger.info(f"Transcribing with {self.config.asr.model_size} model...")
            transcriber = Transcriber(self.config.asr)
            transcripts = transcriber.transcribe_segments(audio_path, speech_segments)

            # 5. Generate subtitles
            logger.info(f"Generating {self.config.output_format.upper()} subtitles...")
            generator = SubtitleGenerator()
            generator.generate(
                transcripts, output_path, format=self.config.output_format
            )

            # 6. Calculate metrics
            duration = sum(s.duration for s in speech_segments)
            processing_time = time.time() - start_time

            logger.info(
                f"✅ Success: {len(transcripts)} segments in {processing_time:.1f}s"
            )

            return PipelineResult(
                success=True,
                output_path=output_path,
                segment_count=len(transcripts),
                duration_seconds=duration,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            processing_time = time.time() - start_time
            return PipelineResult(
                success=False,
                output_path=Path(output_path) if output_path else Path(),
                segment_count=0,
                duration_seconds=0.0,
                processing_time_seconds=processing_time,
                error=str(e),
            )

        finally:
            # Always cleanup temp files
            if temp_audio and temp_audio.exists():
                try:
                    temp_audio.unlink()
                    logger.debug(f"Cleaned up temporary audio: {temp_audio}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup {temp_audio}: {cleanup_error}")

    def process_batch(
        self,
        input_paths: list[Union[str, Path]],
        output_dir: Union[str, Path],
    ) -> list[PipelineResult]:
        """Process multiple files with model reuse

        Args:
            input_paths: List of video/audio files
            output_dir: Directory for output subtitles

        Returns:
            List of PipelineResult (one per input)

        Example:
            >>> pipeline = TranscriptionPipeline()
            >>> files = ["video1.mp4", "video2.mp4", "video3.mp4"]
            >>> results = pipeline.process_batch(files, "output_dir")
            >>> successful = [r for r in results if r.success]
            >>> print(f"{len(successful)}/{len(results)} succeeded")
        """
        output_dir = ensure_directory(output_dir)
        results = []

        logger.info(f"Batch processing {len(input_paths)} files...")

        for i, input_path in enumerate(input_paths, 1):
            logger.info(f"[{i}/{len(input_paths)}] Processing {input_path}")

            # Determine output filename
            input_name = Path(input_path).stem
            output_name = f"{input_name}.{self.config.output_format}"
            output_path = output_dir / output_name

            # Process single file
            result = self.process(input_path, output_path)
            results.append(result)

            # Clear GPU cache between files if using CUDA
            if self.config.asr.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        successful = sum(r.success for r in results)
        logger.info(f"Batch complete: {successful}/{len(results)} succeeded")

        return results
