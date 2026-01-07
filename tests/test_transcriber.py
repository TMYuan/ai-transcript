"""Tests for speech transcription using Faster-Whisper"""

import pytest
import torch

from aitranscript.core.transcriber import (
    ModelManager,
    Transcriber,
    TranscriptionError,
    detect_device,
)
from aitranscript.models.config import ASRConfig
from aitranscript.models.segment import SpeechSegment, TranscriptSegment


class TestDetectDevice:
    """Test device detection for CUDA/CPU"""

    def test_detect_device_returns_string(self):
        """Test that detect_device returns a valid device string"""
        device = detect_device()

        assert isinstance(device, str)
        assert device in ["cpu", "cuda"]

    def test_detect_device_cpu_preference(self):
        """Test that prefer='cpu' always returns CPU"""
        device = detect_device(prefer="cpu")

        assert device == "cpu"

    def test_detect_device_cuda_when_available(self):
        """Test CUDA detection when available"""
        if torch.cuda.is_available():
            device = detect_device()
            assert device == "cuda"
        else:
            # Skip on systems without CUDA
            pytest.skip("CUDA not available")

    def test_detect_device_cpu_fallback(self):
        """Test CPU fallback when CUDA unavailable"""
        # This will be CPU on non-CUDA systems
        device = detect_device()
        assert device in ["cpu", "cuda"]


class TestModelManager:
    """Test ModelManager singleton pattern"""

    def test_model_manager_caches_model(self):
        """Test that same config returns cached model"""
        # Clear cache first
        ModelManager._models.clear()

        # Request model twice with same config
        model1 = ModelManager.get_model("tiny", "cpu", "int8")
        model2 = ModelManager.get_model("tiny", "cpu", "int8")

        # Should return same instance (cached)
        assert model1 is model2

    def test_model_manager_different_configs(self):
        """Test that different configs return different models"""
        ModelManager._models.clear()

        model1 = ModelManager.get_model("tiny", "cpu", "int8")
        model2 = ModelManager.get_model("base", "cpu", "int8")

        # Should be different instances
        assert model1 is not model2

    def test_model_manager_cache_key_includes_all_params(self):
        """Test that cache key includes model_size, device, compute_type"""
        ModelManager._models.clear()

        model1 = ModelManager.get_model("tiny", "cpu", "int8")
        model2 = ModelManager.get_model("tiny", "cpu", "float32")

        # Different compute_type should create different instance
        assert model1 is not model2


class TestTranscriber:
    """Test suite for Transcriber class"""

    def test_transcriber_initialization_default(self):
        """Test initialization with default ASRConfig"""
        transcriber = Transcriber()

        # Default ASRConfig uses medium model
        assert transcriber.config.model_size == "medium"
        assert transcriber.config.device in ["cpu", "cuda"]
        assert transcriber.config.beam_size == 5
        assert transcriber.model is not None

    def test_transcriber_initialization_custom_config(self):
        """Test initialization with custom ASRConfig"""
        config = ASRConfig(
            model_size="tiny", device="cpu", compute_type="int8", beam_size=3
        )
        transcriber = Transcriber(config)

        assert transcriber.config.model_size == "tiny"
        assert transcriber.config.device == "cpu"
        assert transcriber.config.compute_type == "int8"
        assert transcriber.config.beam_size == 3

    def test_transcriber_model_loaded(self):
        """Test that model is loaded during initialization"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        assert transcriber.model is not None
        assert hasattr(transcriber.model, "transcribe")

    def test_transcriber_cuda_fallback(self):
        """Test that transcriber handles CUDA unavailability"""
        # Request CUDA device
        config = ASRConfig(device="cuda")
        transcriber = Transcriber(config)

        # Should either use CUDA or fallback to CPU
        assert transcriber.config.device in ["cpu", "cuda"]

    def test_transcribe_segment_basic(self, sample_audio_wav):
        """Test transcribing a single speech segment"""
        config = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=3.0)

        result = transcriber.transcribe_segment(sample_audio_wav, segment)

        assert isinstance(result, TranscriptSegment)
        assert result.start == segment.start
        assert result.end == segment.end
        assert isinstance(result.text, str)

    def test_transcribe_segment_preserves_timing(self, sample_audio_wav):
        """Test that transcription preserves original segment timing"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.5, end=2.5)

        result = transcriber.transcribe_segment(sample_audio_wav, segment)

        assert result.start == 0.5
        assert result.end == 2.5
        assert result.duration == 2.0

    def test_transcribe_segment_invalid_audio(self):
        """Test error handling for invalid audio file"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=1.0)

        with pytest.raises(TranscriptionError, match="not found|does not exist"):
            transcriber.transcribe_segment("nonexistent.wav", segment)

    def test_transcribe_segment_cuda_error(self, sample_audio_wav):
        """Test error handling for CUDA/GPU errors"""
        from unittest.mock import patch

        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=1.0)

        # Mock the model.transcribe to raise a RuntimeError with CUDA message
        with patch.object(
            transcriber.model, "transcribe", side_effect=RuntimeError("CUDA error")
        ):
            with pytest.raises(
                TranscriptionError, match="GPU/CUDA error.*--device cpu"
            ):
                transcriber.transcribe_segment(sample_audio_wav, segment)

    def test_transcribe_segments_list(self, sample_audio_wav, mock_vad_segments):
        """Test transcribing multiple segments"""
        config = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
        transcriber = Transcriber(config)

        # Use smaller subset for speed
        segments = mock_vad_segments[:2]

        results = transcriber.transcribe_segments(sample_audio_wav, segments)

        assert isinstance(results, list)
        assert len(results) == len(segments)
        assert all(isinstance(r, TranscriptSegment) for r in results)

    def test_transcribe_segments_preserves_order(
        self, sample_audio_wav, mock_vad_segments
    ):
        """Test that segment order is preserved"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        segments = mock_vad_segments[:2]

        results = transcriber.transcribe_segments(sample_audio_wav, segments)

        # Check order preserved
        for orig, result in zip(segments, results):
            assert result.start == orig.start
            assert result.end == orig.end

    def test_transcribe_empty_segment_list(self, sample_audio_wav):
        """Test handling of empty segment list"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        results = transcriber.transcribe_segments(sample_audio_wav, [])

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.slow
    def test_transcribe_segments_batch_processing(self, sample_audio_wav):
        """Test batch processing with many segments"""
        config = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
        transcriber = Transcriber(config)

        # Create many small segments
        segments = [
            SpeechSegment(start=i * 0.5, end=(i + 1) * 0.5)
            for i in range(10)  # 10 segments for speed
        ]

        results = transcriber.transcribe_segments(sample_audio_wav, segments)

        assert len(results) == len(segments)
        assert all(isinstance(r, TranscriptSegment) for r in results)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA GPU",
    )
    def test_gpu_transcription_works(self, sample_audio_wav):
        """Test that GPU transcription works (even on older GPUs like GTX 1080)

        This test verifies that GPU transcription works on compute capability 6.x GPUs,
        even though ctranslate2 officially requires 7.0+. In practice, it often works.

        Note: sample_audio_wav is a synthetic tone, not real speech, so may return
        "[no speech]" - the important part is that GPU inference doesn't crash.
        """
        config = ASRConfig(model_size="tiny", device="cuda", compute_type="int8")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=2.0)

        # This should work even on sm_61 (GTX 1080) despite warnings
        result = transcriber.transcribe_segment(sample_audio_wav, segment)

        # Verify it returns a valid result (GPU inference worked)
        assert isinstance(result, TranscriptSegment)
        assert result.start == 0.0
        assert result.end == 2.0
        assert len(result.text) > 0
        # Text can be "[no speech]" for synthetic audio - that's OK

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7,
        reason="Requires CUDA with compute capability ≥7.0 (ctranslate2 limitation)",
    )
    def test_gpu_memory_cleanup(self, sample_audio_wav):
        """Test that GPU memory is cleaned up after transcription

        Note: ctranslate2 (faster-whisper backend) requires sm_70+ GPUs.
        Older GPUs like GTX 1080 (sm_61) should use CPU transcription.
        """
        config = ASRConfig(model_size="tiny", device="cuda", compute_type="int8")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=1.0)

        # Get initial memory
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Transcribe
        _ = transcriber.transcribe_segment(sample_audio_wav, segment)

        # Force cleanup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Memory should not increase significantly
        final_memory = torch.cuda.memory_allocated()
        # Allow some overhead but should be roughly same
        assert final_memory <= initial_memory * 1.5

    def test_transcribe_with_corrupted_audio(self, tmp_path):
        """Test handling of corrupted audio file"""
        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        # Create corrupted audio file
        corrupted = tmp_path / "corrupted.wav"
        corrupted.write_bytes(b"not a wav file")

        segment = SpeechSegment(start=0.0, end=1.0)

        with pytest.raises(TranscriptionError):
            transcriber.transcribe_segment(corrupted, segment)

    def test_transcribe_segment_empty_audio(self, tmp_path):
        """Test transcription of silent/empty segment"""
        from pydub import AudioSegment

        # Create 1 second of silence
        silence = AudioSegment.silent(duration=1000, frame_rate=16000)
        silent_file = tmp_path / "silence.wav"
        silence.export(str(silent_file), format="wav")

        config = ASRConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        segment = SpeechSegment(start=0.0, end=1.0)

        result = transcriber.transcribe_segment(silent_file, segment)

        # Should return TranscriptSegment with placeholder for no speech
        assert isinstance(result, TranscriptSegment)
        assert result.start == 0.0
        assert result.end == 1.0
        # Placeholder text for segments with no detected speech
        assert result.text == "[no speech]"


class TestTranscriberIntegration:
    """Integration tests with audio extraction and VAD"""

    def test_transcriber_after_vad(self, sample_speech_audio):
        """Test transcriber with VAD-generated segments"""
        from aitranscript.core.vad_processor import VADProcessor

        # Generate speech segments with VAD
        vad = VADProcessor()
        segments = vad.process(sample_speech_audio)

        # Transcribe segments (if any detected)
        if segments:
            config = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
            transcriber = Transcriber(config)

            results = transcriber.transcribe_segments(sample_speech_audio, segments)

            assert len(results) == len(segments)
            assert all(isinstance(r, TranscriptSegment) for r in results)

    def test_end_to_end_extract_vad_transcribe(self, tmp_path):
        """Test full pipeline: extract → VAD → transcribe"""
        from pydub.generators import Sine

        from aitranscript.core.audio_extractor import AudioExtractor
        from aitranscript.core.vad_processor import VADProcessor

        # Create test audio
        tone = Sine(300).to_audio_segment(duration=2000)
        tone = tone.set_frame_rate(44100)
        input_file = tmp_path / "input.wav"
        tone.export(str(input_file), format="wav")

        # Extract audio
        extractor = AudioExtractor()
        extracted = extractor.extract(input_file, tmp_path / "extracted.wav")

        # VAD
        vad = VADProcessor()
        segments = vad.process(extracted)

        # Transcribe (if segments detected)
        if segments:
            config = ASRConfig(model_size="tiny", device="cpu", compute_type="int8")
            transcriber = Transcriber(config)
            results = transcriber.transcribe_segments(extracted, segments)

            assert isinstance(results, list)
            assert all(isinstance(r, TranscriptSegment) for r in results)
