"""Configuration data models for transcription pipeline"""

from dataclasses import dataclass, field


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection

    Attributes:
        min_speech_duration_ms: Minimum speech duration to keep (filters short sounds)
        max_pause_duration_ms: Maximum pause duration within same segment
        max_segment_duration_s: Maximum segment duration to prevent long subtitles
        threshold: VAD confidence threshold (0.0 to 1.0)

    Example:
        >>> config = VADConfig()
        >>> config.min_speech_duration_ms
        250
        >>> custom_config = VADConfig(min_speech_duration_ms=500)
    """

    min_speech_duration_ms: int = 250
    max_pause_duration_ms: int = 300
    max_segment_duration_s: float = 6.0
    threshold: float = 0.5

    def __post_init__(self):
        """Validate configuration parameters

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.min_speech_duration_ms <= 0:
            raise ValueError("min_speech_duration_ms must be positive")
        if self.max_pause_duration_ms <= 0:
            raise ValueError("max_pause_duration_ms must be positive")
        if self.max_segment_duration_s <= 0:
            raise ValueError("max_segment_duration_s must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")


@dataclass
class ASRConfig:
    """Configuration for Automatic Speech Recognition (Whisper)

    Attributes:
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Compute device ('cuda' or 'cpu')
        compute_type: Precision type ('float16' for GPU, 'int8' for CPU)
        language: Target language code (e.g., 'en', 'zh')
        beam_size: Beam search width for decoding

    Example:
        >>> config = ASRConfig()
        >>> config.model_size
        'medium'
        >>> cpu_config = ASRConfig(device='cpu')
        >>> cpu_config.compute_type  # Auto-configured to int8
        'int8'
    """

    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "int8"  # Default to int8 (works on all GPUs, including sm_61)
    language: str = "en"
    beam_size: int = 5

    def __post_init__(self):
        """Validate and auto-configure parameters

        Automatically sets compute_type to 'int8' for CPU or when not specified.
        Note: int8 works on all GPUs including older ones (sm_61+).
        float16 requires compute capability >= 7.0 (sm_70+).

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate device
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")

        # Validate model_size
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of: {', '.join(valid_sizes)}")

        # Validate beam_size
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")

        # Auto-configure compute_type for CPU
        if self.device == "cpu" and self.compute_type != "int8":
            self.compute_type = "int8"


@dataclass
class PipelineConfig:
    """Main pipeline configuration

    Attributes:
        vad: Voice Activity Detection configuration
        asr: Automatic Speech Recognition configuration
        extract_audio: Whether to extract audio from video files
        output_format: Subtitle output format ('srt' or 'vtt')

    Example:
        >>> config = PipelineConfig()
        >>> config.vad.min_speech_duration_ms
        250
        >>> config.asr.model_size
        'medium'
        >>> custom_config = PipelineConfig(
        ...     vad=VADConfig(min_speech_duration_ms=500),
        ...     asr=ASRConfig(device='cpu'),
        ...     output_format='vtt'
        ... )
    """

    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    extract_audio: bool = True
    output_format: str = "srt"

    def __post_init__(self):
        """Validate configuration

        Raises:
            ValueError: If output_format is invalid
        """
        if self.output_format not in ["srt", "vtt"]:
            raise ValueError("output_format must be 'srt' or 'vtt'")
