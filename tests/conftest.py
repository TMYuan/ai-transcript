"""Pytest fixtures for ai-transcript tests"""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide temporary directory for test outputs with auto-cleanup

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path object for temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_audio_path():
    """Return path to test audio fixture

    Returns:
        Path object for sample audio file (placeholder for now)

    Note:
        This is a placeholder. Actual audio file will be added in Phase 2
    """
    return Path("tests/fixtures/sample_audio.wav")


@pytest.fixture
def sample_audio_wav(tmp_path):
    """Generate 3-second test WAV file (16kHz mono)

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path object for generated test audio file

    Note:
        Generates a 3-second 300Hz tone (speech-like frequency)
        at 16kHz sample rate, mono channel
    """
    from pydub.generators import Sine

    # Generate 3 seconds of 300Hz tone (speech-like frequency)
    tone = Sine(300).to_audio_segment(duration=3000)
    tone = tone.set_frame_rate(16000).set_channels(1)

    output = tmp_path / "fixtures" / "sample_audio.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    tone.export(str(output), format="wav")

    return output


@pytest.fixture
def sample_speech_audio(tmp_path):
    """Generate audio with speech-like characteristics (varying tones)

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path object for generated speech-like audio file

    Note:
        Creates 3 seconds of varying tones to simulate speech pattern.
        Uses frequencies 200-350 Hz which are in speech range.
    """
    from pydub.generators import Sine

    # Create varying tones to simulate speech pattern (750ms each)
    tones = [Sine(freq).to_audio_segment(duration=750) for freq in [200, 300, 250, 350]]
    combined = sum(tones)
    combined = combined.set_frame_rate(16000).set_channels(1)

    output = tmp_path / "fixtures" / "speech_like.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output), format="wav")

    return output


@pytest.fixture
def long_audio_with_pauses(tmp_path):
    """Generate audio with speech segments and pauses

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path object for generated audio file with pauses

    Note:
        Creates pattern: speech (1s) - pause (500ms) - speech (1s) -
        pause (500ms) - speech (1s). Total duration: ~3.5 seconds
    """
    from pydub import AudioSegment
    from pydub.generators import Sine

    # Create speech-like tone (1 second at 300 Hz)
    speech = Sine(300).to_audio_segment(duration=1000)
    # Create silence (500ms pause)
    silence = AudioSegment.silent(duration=500, frame_rate=16000)

    # Pattern: speech - pause - speech - pause - speech
    audio = speech + silence + speech + silence + speech
    audio = audio.set_frame_rate(16000).set_channels(1)

    output = tmp_path / "fixtures" / "speech_with_pauses.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(output), format="wav")

    return output


@pytest.fixture
def mock_vad_segments():
    """Pre-defined SpeechSegment objects for testing

    Returns:
        List of SpeechSegment objects with realistic timings (1.5-6s)

    Note:
        Requires SpeechSegment to be importable. Will be used after
        models/segment.py is implemented
    """
    from aitranscript.models.segment import SpeechSegment

    return [
        SpeechSegment(start=0.0, end=2.5),  # 2.5s segment
        SpeechSegment(start=3.0, end=5.5),  # 2.5s segment
        SpeechSegment(start=6.0, end=9.5),  # 3.5s segment
        SpeechSegment(start=10.0, end=15.0),  # 5.0s segment (near max)
    ]


@pytest.fixture
def test_vad_config():
    """Default VADConfig for tests

    Returns:
        VADConfig with default settings

    Note:
        Requires VADConfig to be importable. Will be used after
        models/config.py is implemented
    """
    from aitranscript.models.config import VADConfig

    return VADConfig()


@pytest.fixture
def test_asr_config():
    """Default ASRConfig for tests (CPU, base model for speed)

    Returns:
        ASRConfig configured for fast testing on CPU

    Note:
        Requires ASRConfig to be importable. Will be used after
        models/config.py is implemented
    """
    from aitranscript.models.config import ASRConfig

    return ASRConfig(model_size="base", device="cpu", compute_type="int8")


@pytest.fixture
def test_asr_config_fast():
    """Fast ASRConfig for testing (tiny model, CPU, optimized)

    Returns:
        ASRConfig configured for very fast testing with tiny model

    Note:
        Use this fixture for tests that need quick transcription.
        The tiny model is much faster but less accurate.
    """
    from aitranscript.models.config import ASRConfig

    return ASRConfig(model_size="tiny", device="cpu", compute_type="int8", beam_size=5)
