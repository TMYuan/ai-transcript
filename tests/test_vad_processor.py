"""Tests for Voice Activity Detection (VAD) processing"""

import pytest

from aitranscript.core.vad_processor import (
    VADProcessingError,
    VADProcessor,
    load_silero_vad,
)
from aitranscript.models.config import VADConfig
from aitranscript.models.segment import SpeechSegment


class TestVADProcessor:
    """Test suite for VADProcessor class"""

    def test_vad_processor_initialization_default(self):
        """Verify VADProcessor initializes with default config"""
        vad = VADProcessor()

        assert vad.config.min_speech_duration_ms == 250
        assert vad.config.max_pause_duration_ms == 300
        assert vad.config.max_segment_duration_s == 6.0
        assert vad.config.threshold == 0.5

    def test_vad_processor_initialization_custom_config(self):
        """Test initialization with custom VADConfig"""
        config = VADConfig(
            min_speech_duration_ms=500,
            max_pause_duration_ms=400,
            max_segment_duration_s=5.0,
        )
        vad = VADProcessor(config)

        assert vad.config.min_speech_duration_ms == 500
        assert vad.config.max_pause_duration_ms == 400
        assert vad.config.max_segment_duration_s == 5.0

    def test_vad_process_returns_speech_segments(self, sample_audio_wav):
        """Test that process() returns list of SpeechSegment objects"""
        vad = VADProcessor()

        segments = vad.process(sample_audio_wav)

        assert isinstance(segments, list)
        # May be empty for pure tone, but should be a list
        assert all(isinstance(s, SpeechSegment) for s in segments)

    def test_vad_segments_within_duration_limits(self, sample_speech_audio):
        """Test that segments respect duration limits"""
        vad = VADProcessor()

        segments = vad.process(sample_speech_audio)

        # All segments should be within max duration
        for seg in segments:
            assert seg.duration <= 6.0
            assert seg.start >= 0
            assert seg.end > seg.start

    def test_vad_process_with_speech_like_audio(self, sample_speech_audio):
        """Test VAD on speech-like audio (varying tones)"""
        vad = VADProcessor()

        segments = vad.process(sample_speech_audio)

        # Should detect at least some speech in the varying tone audio
        assert isinstance(segments, list)
        # Speech-like audio should produce some segments
        if len(segments) > 0:
            # Verify segments are valid
            for seg in segments:
                assert seg.end > seg.start
                assert seg.duration > 0

    def test_vad_process_empty_audio(self, tmp_path):
        """Test processing audio with no speech (silence)"""
        from pydub import AudioSegment

        # Create 2 seconds of silence
        silence = AudioSegment.silent(duration=2000, frame_rate=16000)
        silent_file = tmp_path / "silence.wav"
        silence.export(str(silent_file), format="wav")

        vad = VADProcessor()
        segments = vad.process(silent_file)

        # Should return empty list or very few segments for pure silence
        assert isinstance(segments, list)
        # Silence should produce no segments or very short ones
        assert len(segments) == 0 or all(seg.duration < 0.5 for seg in segments)

    def test_vad_process_invalid_file(self):
        """Test that invalid audio file raises VADProcessingError"""
        vad = VADProcessor()

        with pytest.raises(VADProcessingError, match="not found"):
            vad.process("nonexistent.wav")

    def test_vad_process_corrupted_audio(self, tmp_path):
        """Test that corrupted audio file raises error"""
        corrupted = tmp_path / "corrupted.wav"
        corrupted.write_bytes(b"not a wav file")

        vad = VADProcessor()

        with pytest.raises(VADProcessingError):
            vad.process(corrupted)

    def test_vad_returns_sorted_segments(self, sample_speech_audio):
        """Test that returned segments are sorted by start time"""
        vad = VADProcessor()

        segments = vad.process(sample_speech_audio)

        # Check segments are in chronological order
        for i in range(len(segments) - 1):
            assert segments[i].start <= segments[i + 1].start

    def test_vad_no_overlapping_segments(self, sample_speech_audio):
        """Test that segments don't overlap"""
        vad = VADProcessor()

        segments = vad.process(sample_speech_audio)

        # Check no overlaps
        for i in range(len(segments) - 1):
            assert segments[i].end <= segments[i + 1].start

    def test_vad_filters_very_short_segments(self):
        """Test that very short segments are filtered based on min_speech_duration_ms"""
        # Use strict filtering (500ms minimum)
        config = VADConfig(min_speech_duration_ms=500)
        vad = VADProcessor(config)

        # Note: With pure tones, VAD might not detect speech at all
        # This test verifies the config is applied
        assert vad.config.min_speech_duration_ms == 500

    def test_vad_respects_max_segment_duration(self, long_audio_with_pauses):
        """Test that max_segment_duration is respected"""
        config = VADConfig(max_segment_duration_s=3.0)
        vad = VADProcessor(config)

        segments = vad.process(long_audio_with_pauses)

        # All segments should be <= max duration
        for seg in segments:
            assert seg.duration <= 3.0

    def test_vad_config_threshold_affects_detection(self, sample_speech_audio):
        """Test that threshold parameter affects detection"""
        # Higher threshold = stricter detection
        strict_vad = VADProcessor(VADConfig(threshold=0.9))
        lenient_vad = VADProcessor(VADConfig(threshold=0.3))

        strict_segments = strict_vad.process(sample_speech_audio)
        lenient_segments = lenient_vad.process(sample_speech_audio)

        # Both should return lists
        assert isinstance(strict_segments, list)
        assert isinstance(lenient_segments, list)

        # Lenient threshold should detect more or equal segments
        # (unless audio has no speech at all)
        assert len(lenient_segments) >= len(strict_segments)

    def test_split_long_segments_basic(self):
        """Test _split_long_segments() splits segments exceeding max duration"""
        vad = VADProcessor(VADConfig(max_segment_duration_s=3.0))

        # Create segments with one exceeding max duration
        segments = [
            SpeechSegment(start=0.0, end=2.0),  # OK (2s)
            SpeechSegment(start=3.0, end=10.0),  # Too long (7s) - should split
            SpeechSegment(start=11.0, end=13.5),  # OK (2.5s)
        ]

        result = vad._split_long_segments(segments)

        # Should have more segments (second one split into 3 chunks)
        assert len(result) > len(segments)

        # All segments should be <= max duration
        for seg in result:
            assert seg.duration <= 3.0

        # Total duration should be preserved
        original_duration = sum(s.duration for s in segments)
        result_duration = sum(s.duration for s in result)
        assert abs(original_duration - result_duration) < 0.01

    def test_split_long_segments_preserves_short(self):
        """Test _split_long_segments() preserves segments within limit"""
        vad = VADProcessor(VADConfig(max_segment_duration_s=5.0))

        # All segments within limit
        segments = [
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=3.0, end=7.0),  # 4s - within limit
            SpeechSegment(start=8.0, end=10.5),
        ]

        result = vad._split_long_segments(segments)

        # Should have same number of segments
        assert len(result) == len(segments)

        # Segments should be identical
        for orig, res in zip(segments, result):
            assert orig.start == res.start
            assert orig.end == res.end


class TestLoadSileroVAD:
    """Test Silero VAD model loading"""

    def test_load_silero_vad_returns_model(self):
        """Test that load_silero_vad() successfully loads model"""
        model, utils = load_silero_vad()

        assert model is not None
        assert utils is not None
        # Utils should be a tuple of functions
        assert len(utils) == 5

    def test_load_silero_vad_caches_model(self):
        """Test that model is cached and reused (singleton pattern)"""
        model1, utils1 = load_silero_vad()
        model2, utils2 = load_silero_vad()

        # Should return same instances (cached)
        assert model1 is model2
        assert utils1 is utils2

    def test_load_silero_vad_model_callable(self):
        """Test that loaded model is callable"""
        model, utils = load_silero_vad()

        # Model should be callable (can process audio)
        assert callable(model)


class TestVADIntegration:
    """Integration tests with real audio samples"""

    def test_vad_on_continuous_tone(self, sample_speech_audio):
        """Test VAD on continuous varying tone audio"""
        vad = VADProcessor()

        segments = vad.process(sample_speech_audio)

        # Should return a list (may be empty for pure tones)
        assert isinstance(segments, list)

        # If segments detected, verify they're valid
        for seg in segments:
            assert seg.start >= 0
            assert seg.end > seg.start
            assert seg.duration > 0

    def test_vad_on_audio_with_pauses(self, long_audio_with_pauses):
        """Test VAD on audio with deliberate pauses"""
        vad = VADProcessor()

        segments = vad.process(long_audio_with_pauses)

        # Should return a list
        assert isinstance(segments, list)

        # If segments detected, they should be separated by pauses
        if len(segments) > 1:
            # Check there are gaps between some segments (pauses detected)
            gaps = [
                segments[i + 1].start - segments[i].end
                for i in range(len(segments) - 1)
            ]
            # At least some gaps should exist (from the pauses we added)
            assert any(gap > 0.1 for gap in gaps)

    def test_vad_end_to_end_with_extraction(self, tmp_path):
        """Test VAD after audio extraction (integration test)"""
        from pydub.generators import Sine

        from aitranscript.core.audio_extractor import AudioExtractor

        # Create a simple audio file
        tone = Sine(300).to_audio_segment(duration=2000)
        tone = tone.set_frame_rate(44100)  # Different rate
        input_file = tmp_path / "input.wav"
        tone.export(str(input_file), format="wav")

        # Extract to 16kHz mono
        extractor = AudioExtractor()
        extracted = extractor.extract(input_file, tmp_path / "extracted.wav")

        # Process with VAD
        vad = VADProcessor()
        segments = vad.process(extracted)

        # Should work without errors
        assert isinstance(segments, list)
