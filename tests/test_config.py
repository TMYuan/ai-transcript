"""Tests for configuration data models"""

import pytest

from aitranscript.models.config import ASRConfig, PipelineConfig, VADConfig


class TestVADConfig:
    """Test suite for VADConfig dataclass"""

    def test_vad_config_defaults(self):
        """Verify default VAD parameters"""
        config = VADConfig()
        assert config.min_speech_duration_ms == 250
        assert config.max_pause_duration_ms == 300
        assert config.max_segment_duration_s == 6.0
        assert config.threshold == 0.5

    def test_vad_config_custom_values(self):
        """Test custom configuration values"""
        config = VADConfig(
            min_speech_duration_ms=500,
            max_pause_duration_ms=400,
            max_segment_duration_s=8.0,
            threshold=0.6,
        )
        assert config.min_speech_duration_ms == 500
        assert config.max_pause_duration_ms == 400
        assert config.max_segment_duration_s == 8.0
        assert config.threshold == 0.6

    def test_vad_config_validation_negative_min_duration(self):
        """Test that negative min_speech_duration_ms raises ValueError"""
        with pytest.raises(ValueError, match="min_speech_duration_ms must be positive"):
            VADConfig(min_speech_duration_ms=-100)

    def test_vad_config_validation_negative_max_pause(self):
        """Test that negative max_pause_duration_ms raises ValueError"""
        with pytest.raises(ValueError, match="max_pause_duration_ms must be positive"):
            VADConfig(max_pause_duration_ms=-100)

    def test_vad_config_validation_negative_max_segment(self):
        """Test that negative max_segment_duration_s raises ValueError"""
        with pytest.raises(ValueError, match="max_segment_duration_s must be positive"):
            VADConfig(max_segment_duration_s=-1.0)

    def test_vad_config_validation_threshold_too_low(self):
        """Test that threshold < 0 raises ValueError"""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            VADConfig(threshold=-0.1)

    def test_vad_config_validation_threshold_too_high(self):
        """Test that threshold > 1 raises ValueError"""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            VADConfig(threshold=1.5)

    def test_vad_config_threshold_boundary_values(self):
        """Test that threshold boundary values (0 and 1) are valid"""
        config_zero = VADConfig(threshold=0.0)
        assert config_zero.threshold == 0.0

        config_one = VADConfig(threshold=1.0)
        assert config_one.threshold == 1.0


class TestASRConfig:
    """Test suite for ASRConfig dataclass"""

    def test_asr_config_defaults(self):
        """Verify default ASR parameters"""
        config = ASRConfig()
        assert config.model_size == "medium"
        assert config.device == "cuda"
        assert config.compute_type == "int8"  # Default to int8 for GPU compatibility
        assert config.language == "en"
        assert config.beam_size == 5

    def test_asr_config_cuda(self):
        """Test CUDA device with int8 compute type (compatible with all GPUs)"""
        config = ASRConfig(device="cuda")
        assert config.device == "cuda"
        assert config.compute_type == "int8"  # Default int8 works on sm_61+ GPUs

    def test_asr_config_cpu(self):
        """Test CPU device auto-configures to int8"""
        config = ASRConfig(device="cpu")
        assert config.device == "cpu"
        assert config.compute_type == "int8"

    def test_asr_config_cpu_overrides_float16(self):
        """Test that CPU device forces int8 even if float16 is specified"""
        config = ASRConfig(device="cpu", compute_type="float16")
        assert config.device == "cpu"
        assert config.compute_type == "int8"  # Auto-corrected

    def test_asr_config_custom_values(self):
        """Test custom configuration values"""
        config = ASRConfig(model_size="base", device="cpu", language="zh", beam_size=3)
        assert config.model_size == "base"
        assert config.device == "cpu"
        assert config.language == "zh"
        assert config.beam_size == 3

    def test_asr_config_validation_invalid_device(self):
        """Test that invalid device raises ValueError"""
        with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
            ASRConfig(device="gpu")

    def test_asr_config_validation_invalid_model_size(self):
        """Test that invalid model_size raises ValueError"""
        with pytest.raises(
            ValueError,
            match="model_size must be one of: tiny, base, small, medium, large",
        ):
            ASRConfig(model_size="huge")

    def test_asr_config_all_valid_model_sizes(self):
        """Test that all valid model sizes are accepted"""
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        for size in valid_sizes:
            config = ASRConfig(model_size=size)
            assert config.model_size == size

    def test_asr_config_validation_invalid_beam_size(self):
        """Test that beam_size < 1 raises ValueError"""
        with pytest.raises(ValueError, match="beam_size must be at least 1"):
            ASRConfig(beam_size=0)


class TestPipelineConfig:
    """Test suite for PipelineConfig dataclass"""

    def test_pipeline_config_defaults(self):
        """Verify default pipeline configuration"""
        config = PipelineConfig()
        assert config.vad is not None
        assert config.asr is not None
        assert config.extract_audio is True
        assert config.output_format == "srt"

    def test_pipeline_config_composition(self):
        """Test that VADConfig and ASRConfig are properly composed"""
        config = PipelineConfig()

        # Check VAD defaults
        assert isinstance(config.vad, VADConfig)
        assert config.vad.min_speech_duration_ms == 250

        # Check ASR defaults
        assert isinstance(config.asr, ASRConfig)
        assert config.asr.model_size == "medium"

    def test_pipeline_config_custom_vad(self):
        """Test custom VAD configuration"""
        custom_vad = VADConfig(min_speech_duration_ms=500)
        config = PipelineConfig(vad=custom_vad)

        assert config.vad.min_speech_duration_ms == 500

    def test_pipeline_config_custom_asr(self):
        """Test custom ASR configuration"""
        custom_asr = ASRConfig(model_size="base", device="cpu")
        config = PipelineConfig(asr=custom_asr)

        assert config.asr.model_size == "base"
        assert config.asr.device == "cpu"

    def test_pipeline_config_custom_values(self):
        """Test custom pipeline configuration values"""
        config = PipelineConfig(extract_audio=False, output_format="vtt")

        assert config.extract_audio is False
        assert config.output_format == "vtt"

    def test_pipeline_config_validation_invalid_output_format(self):
        """Test that invalid output_format raises ValueError"""
        with pytest.raises(ValueError, match="output_format must be 'srt' or 'vtt'"):
            PipelineConfig(output_format="txt")

    def test_pipeline_config_all_valid_output_formats(self):
        """Test that all valid output formats are accepted"""
        for fmt in ["srt", "vtt"]:
            config = PipelineConfig(output_format=fmt)
            assert config.output_format == fmt

    def test_pipeline_config_full_custom(self):
        """Test fully customized pipeline configuration"""
        custom_vad = VADConfig(min_speech_duration_ms=300, max_segment_duration_s=8.0)
        custom_asr = ASRConfig(model_size="small", device="cpu", language="zh")

        config = PipelineConfig(
            vad=custom_vad, asr=custom_asr, extract_audio=False, output_format="vtt"
        )

        assert config.vad.min_speech_duration_ms == 300
        assert config.vad.max_segment_duration_s == 8.0
        assert config.asr.model_size == "small"
        assert config.asr.device == "cpu"
        assert config.asr.language == "zh"
        assert config.extract_audio is False
        assert config.output_format == "vtt"
