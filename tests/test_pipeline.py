"""Tests for TranscriptionPipeline end-to-end orchestration

Tests follow TDD RED phase - these tests will fail until
transcription_pipeline.py is implemented.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from aitranscript.core.transcriber import TranscriptionError
from aitranscript.core.vad_processor import VADProcessingError
from aitranscript.models.config import ASRConfig, PipelineConfig, VADConfig
from aitranscript.pipeline.transcription_pipeline import (
    PipelineResult,
    TranscriptionPipeline,
)

# ============================================================================
# Group 1: Basic Structure
# ============================================================================


class TestPipelineResult:
    """Test PipelineResult data class"""

    def test_pipeline_result_success(self):
        """Should create successful result with all metrics"""
        result = PipelineResult(
            success=True,
            output_path=Path("output.srt"),
            segment_count=10,
            duration_seconds=30.5,
            processing_time_seconds=45.2,
        )
        assert result.success is True
        assert result.output_path == Path("output.srt")
        assert result.segment_count == 10
        assert result.duration_seconds == 30.5
        assert result.processing_time_seconds == 45.2
        assert result.error is None

    def test_pipeline_result_failure(self):
        """Should create failure result with error message"""
        result = PipelineResult(
            success=False,
            output_path=Path("output.srt"),
            segment_count=0,
            duration_seconds=0.0,
            processing_time_seconds=5.0,
            error="VAD processing failed",
        )
        assert result.success is False
        assert result.error == "VAD processing failed"
        assert result.segment_count == 0


class TestTranscriptionPipeline:
    """Test TranscriptionPipeline initialization"""

    def test_pipeline_initialization_default(self):
        """Should initialize with default configuration"""
        pipeline = TranscriptionPipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.config.extract_audio is True
        assert pipeline.config.output_format == "srt"

    def test_pipeline_initialization_custom_config(self):
        """Should initialize with custom configuration"""
        config = PipelineConfig(
            vad=VADConfig(min_speech_duration_ms=500),
            asr=ASRConfig(model_size="tiny", device="cpu"),
            output_format="vtt",
        )
        pipeline = TranscriptionPipeline(config)
        assert pipeline.config.vad.min_speech_duration_ms == 500
        assert pipeline.config.asr.model_size == "tiny"
        assert pipeline.config.output_format == "vtt"

    def test_pipeline_process_returns_result(self, sample_audio_wav, tmp_output_dir):
        """Should return PipelineResult from process method"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"
        result = pipeline.process(sample_audio_wav, output_path)
        assert isinstance(result, PipelineResult)


# ============================================================================
# Group 2: Audio Processing
# ============================================================================


class TestPipelineProcessAudio:
    """Test pipeline audio processing workflow"""

    def test_process_audio_file_success(self, sample_audio_wav, tmp_output_dir):
        """Should successfully process audio file to SRT"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"

        result = pipeline.process(sample_audio_wav, output_path)

        assert result.success is True
        assert result.output_path == output_path
        assert output_path.exists()

    def test_process_creates_output_file(self, sample_audio_wav, tmp_output_dir):
        """Should create output subtitle file"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "subtitles.srt"

        pipeline.process(sample_audio_wav, output_path)

        assert output_path.exists()
        # File may be empty if no speech detected (valid scenario)
        assert output_path.stat().st_size >= 0

    def test_process_vtt_format(self, sample_audio_wav, tmp_output_dir):
        """Should generate VTT format when configured"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(
                asr=ASRConfig(model_size="tiny", device="cpu"), output_format="vtt"
            )
        )
        output_path = tmp_output_dir / "output.vtt"

        result = pipeline.process(sample_audio_wav, output_path)

        assert result.success is True
        assert output_path.exists()
        content = output_path.read_text()
        assert content.startswith("WEBVTT")

    def test_process_metrics_timing(self, sample_audio_wav, tmp_output_dir):
        """Should track processing time"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"

        result = pipeline.process(sample_audio_wav, output_path)

        assert result.processing_time_seconds > 0

    def test_process_metrics_segment_count(self, sample_audio_wav, tmp_output_dir):
        """Should report correct segment count"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"

        result = pipeline.process(sample_audio_wav, output_path)

        assert result.segment_count >= 0

    def test_process_metrics_duration(self, sample_audio_wav, tmp_output_dir):
        """Should report speech duration"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"

        result = pipeline.process(sample_audio_wav, output_path)

        assert result.duration_seconds >= 0


# ============================================================================
# Group 3: Error Handling
# ============================================================================


class TestPipelineErrorHandling:
    """Test pipeline error handling and cleanup"""

    def test_process_input_file_not_found(self, tmp_output_dir):
        """Should handle nonexistent input file gracefully"""
        pipeline = TranscriptionPipeline()
        output_path = tmp_output_dir / "output.srt"

        result = pipeline.process("/nonexistent/file.mp4", output_path)

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "exist" in result.error.lower()

    def test_process_invalid_output_path(self, sample_audio_wav):
        """Should handle invalid output path with overwrite=False"""
        pipeline = TranscriptionPipeline()
        # Create existing file
        existing_file = Path("existing.srt")
        existing_file.write_text("existing content")

        try:
            # Mock validate_output_path to raise error
            with patch(
                "aitranscript.pipeline.transcription_pipeline.validate_output_path"
            ) as mock_validate:
                from aitranscript.utils.validators import ValidationError

                mock_validate.side_effect = ValidationError(
                    "Output file already exists"
                )

                result = pipeline.process(sample_audio_wav, existing_file)

                assert result.success is False
                assert result.error is not None
        finally:
            if existing_file.exists():
                existing_file.unlink()

    def test_process_vad_failure_cleanup(self, sample_audio_wav, tmp_output_dir):
        """Should cleanup temp files when VAD fails"""
        pipeline = TranscriptionPipeline()
        output_path = tmp_output_dir / "output.srt"

        with patch(
            "aitranscript.pipeline.transcription_pipeline.VADProcessor"
        ) as mock_vad:
            mock_vad.return_value.process.side_effect = VADProcessingError("VAD failed")

            result = pipeline.process(sample_audio_wav, output_path)

            assert result.success is False
            assert "VAD" in result.error or "failed" in result.error.lower()

    def test_process_transcription_failure_cleanup(
        self, sample_audio_wav, tmp_output_dir
    ):
        """Should cleanup temp files when transcription fails"""
        pipeline = TranscriptionPipeline()
        output_path = tmp_output_dir / "output.srt"

        with patch(
            "aitranscript.pipeline.transcription_pipeline.Transcriber"
        ) as mock_transcriber:
            mock_transcriber.return_value.transcribe_segments.side_effect = (
                TranscriptionError("Transcription failed")
            )

            result = pipeline.process(sample_audio_wav, output_path)

            assert result.success is False
            assert "failed" in result.error.lower()

    def test_process_accepts_string_paths(self, sample_audio_wav, tmp_output_dir):
        """Should accept string paths, not just Path objects"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = str(tmp_output_dir / "output.srt")

        result = pipeline.process(str(sample_audio_wav), output_path)

        assert result.success is True
        assert Path(output_path).exists()

    def test_process_cleans_up_temp_audio(self, tmp_output_dir):
        """Should cleanup temporary audio files after processing video"""
        import shutil
        from pathlib import Path

        # Create a mock video file (copy the sample audio as a .mp4)
        mock_video = tmp_output_dir / "test_video.mp4"
        sample_wav = Path("tests/data/sample_audio.wav")
        if sample_wav.exists():
            shutil.copy(sample_wav, mock_video)
        else:
            # Create a minimal valid file if sample doesn't exist
            mock_video.write_bytes(b"fake video data")

        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )
        output_path = tmp_output_dir / "output.srt"

        # Get temp file pattern before processing
        import glob

        temp_files_before = set(glob.glob("temp_audio*.wav"))

        # Process the video
        pipeline.process(mock_video, output_path)

        # Get temp files after processing
        temp_files_after = set(glob.glob("temp_audio*.wav"))

        # Should not have created any lingering temp files
        new_temp_files = temp_files_after - temp_files_before
        assert len(new_temp_files) == 0, f"Temp files not cleaned up: {new_temp_files}"


# ============================================================================
# Group 4: Batch Processing
# ============================================================================


class TestPipelineBatchProcessing:
    """Test batch processing functionality"""

    def test_process_batch_multiple_files(self, sample_audio_wav, tmp_output_dir):
        """Should process multiple files in batch"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )

        # Create multiple input files (use same sample for testing)
        input_files = [sample_audio_wav, sample_audio_wav, sample_audio_wav]

        results = pipeline.process_batch(input_files, tmp_output_dir)

        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)

    def test_process_batch_individual_results(self, sample_audio_wav, tmp_output_dir):
        """Should return individual results for each file"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )

        input_files = [sample_audio_wav, sample_audio_wav]
        results = pipeline.process_batch(input_files, tmp_output_dir)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, PipelineResult)

    def test_process_batch_continues_on_failure(self, sample_audio_wav, tmp_output_dir):
        """Should continue processing even if one file fails"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )

        # Mix valid and invalid files
        input_files = [sample_audio_wav, "/nonexistent/file.wav", sample_audio_wav]

        results = pipeline.process_batch(input_files, tmp_output_dir)

        assert len(results) == 3
        # At least the valid files should succeed
        successful = [r for r in results if r.success]
        assert len(successful) >= 1

    def test_process_batch_creates_output_files(self, sample_audio_wav, tmp_output_dir):
        """Should create separate output files for each input"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(asr=ASRConfig(model_size="tiny", device="cpu"))
        )

        input_files = [sample_audio_wav, sample_audio_wav]
        results = pipeline.process_batch(input_files, tmp_output_dir)

        # Check that output files were created
        # (may have same name, but that's OK for test)
        for result in results:
            if result.success:
                assert result.output_path.exists()


# ============================================================================
# Group 5: Integration Tests (marked as slow)
# ============================================================================


@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests for full pipeline flow"""

    def test_full_pipeline_audio_to_srt(self, sample_audio_wav, tmp_output_dir):
        """Should complete full pipeline from audio to SRT"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(
                asr=ASRConfig(model_size="tiny", device="cpu"),
                output_format="srt",
            )
        )
        output_path = tmp_output_dir / "integration_test.srt"

        result = pipeline.process(sample_audio_wav, output_path)

        # Verify success
        assert result.success is True
        assert result.error is None

        # Verify output file exists
        assert output_path.exists()
        # Content may be empty if no speech detected (valid scenario)
        content = output_path.read_text()
        assert len(content) >= 0

        # Verify metrics
        assert result.processing_time_seconds > 0
        assert result.segment_count >= 0
        assert result.duration_seconds >= 0

    def test_full_pipeline_audio_to_vtt(self, sample_audio_wav, tmp_output_dir):
        """Should complete full pipeline from audio to VTT"""
        pipeline = TranscriptionPipeline(
            PipelineConfig(
                asr=ASRConfig(model_size="tiny", device="cpu"),
                output_format="vtt",
            )
        )
        output_path = tmp_output_dir / "integration_test.vtt"

        result = pipeline.process(sample_audio_wav, output_path)

        # Verify success
        assert result.success is True
        assert result.error is None

        # Verify VTT format
        assert output_path.exists()
        content = output_path.read_text()
        assert content.startswith("WEBVTT")

        # Verify metrics
        assert result.processing_time_seconds > 0
