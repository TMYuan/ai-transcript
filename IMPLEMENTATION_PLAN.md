# AI Transcript Implementation Plan

## Overview
Build a production-ready video transcription tool that generates English subtitles using VAD + Faster-Whisper pipeline. Architected for future Chinese translation capability.

**Tech Stack**: faster-whisper, Silero VAD, pydub, python-srt
**Target**: GPU (CUDA) with float16 optimization
**Interface**: Python library + CLI wrapper (maximum flexibility)

---

## Implementation Status

**Phase 1: Project Setup & Core Data Models** ✅ **COMPLETE** (2026-01-01)
- ✅ Project initialized with uv (70 packages installed)
- ✅ Directory structure created (src layout)
- ✅ Data models: `SpeechSegment`, `TranscriptSegment` with SRT formatting
- ✅ Config models: `VADConfig`, `ASRConfig`, `PipelineConfig` with validation
- ✅ Test infrastructure: 42 tests passing, 87% coverage
- ✅ Logger utility with Rich formatting
- ✅ All code quality checks passing (ruff)

**Phase 2: Core Processing Components** ⏳ **NEXT**
- Pending: audio_extractor.py, vad_processor.py, transcriber.py, subtitle_generator.py

**Phase 3: Pipeline Orchestration** ⏳ **PENDING**

**Phase 4: CLI Interface** ⏳ **PENDING**

**Phase 5: Integration Testing & Documentation** ⏳ **PENDING**

---

## Project Structure

```
ai-transcript/
├── src/aitranscript/
│   ├── core/
│   │   ├── audio_extractor.py      # Video → wav 16kHz mono
│   │   ├── vad_processor.py        # Silero VAD segmentation
│   │   ├── transcriber.py          # Faster-Whisper ASR
│   │   └── subtitle_generator.py   # SRT/VTT generation
│   ├── models/
│   │   ├── segment.py              # SpeechSegment, TranscriptSegment
│   │   └── config.py               # VADConfig, ASRConfig, PipelineConfig
│   ├── pipeline/
│   │   └── transcription_pipeline.py  # Main orchestration
│   ├── utils/
│   │   ├── logger.py               # Rich logging
│   │   ├── validators.py           # Input validation
│   │   └── file_utils.py           # File operations
│   └── cli/
│       └── main.py                 # Click CLI interface
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_vad_processor.py
│   ├── test_transcriber.py
│   ├── test_pipeline.py
│   └── fixtures/
└── pyproject.toml
```

---

## Implementation Phases (TDD Approach)

> **TDD Workflow**: For each component, follow Red-Green-Refactor cycle:
> 1. 🔴 **RED**: Write failing test first
> 2. 🟢 **GREEN**: Write minimal code to pass test
> 3. 🔵 **REFACTOR**: Improve code quality while keeping tests green

### Phase 1: Project Setup & Core Data Models
**Goal**: Establish project foundation with TDD infrastructure

**1. Initialize project with uv**
```bash
uv init
uv add faster-whisper torch torchaudio pydub numpy silero-vad srt click rich pydantic
uv add --dev ruff pytest pytest-cov pytest-mock
```

**2. Create directory structure** (src layout)
```bash
mkdir -p src/aitranscript/{core,models,pipeline,utils,cli}
mkdir -p tests/fixtures
touch src/aitranscript/__init__.py
```

**3. Setup test infrastructure**
- Create `tests/conftest.py` with pytest fixtures:
  - `sample_audio_path`: Path to test audio file
  - `mock_vad_segments`: Pre-defined speech segments
  - `test_config`: Test configuration objects
  - `tmp_output_dir`: Temporary directory for test outputs

**4. TDD: Implement models/segment.py** ⭐
- 🔴 Write `tests/test_segment.py` FIRST:
  ```python
  def test_speech_segment_creation():
      segment = SpeechSegment(start=0.0, end=2.5)
      assert segment.start == 0.0
      assert segment.end == 2.5
      assert segment.duration == 2.5

  def test_transcript_segment_to_srt_format():
      segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
      srt_text = segment.to_srt_format()
      assert "00:00:00,000 --> 00:00:02,500" in srt_text
  ```
- 🟢 Implement `SpeechSegment` and `TranscriptSegment` dataclasses
- 🔵 Add validation and helper methods

**5. TDD: Implement models/config.py**
- 🔴 Write `tests/test_config.py` FIRST:
  ```python
  def test_vad_config_defaults():
      config = VADConfig()
      assert config.min_speech_duration_ms == 250
      assert config.max_segment_duration_s == 6.0

  def test_asr_config_cuda():
      config = ASRConfig(device="cuda")
      assert config.compute_type == "float16"
  ```
- 🟢 Implement config dataclasses with defaults
- 🔵 Add validation (e.g., device must be "cuda" or "cpu")

**6. Setup utils/logger.py** (no tests needed, utility only)

### Phase 2: Core Processing Components (TDD)
**Goal**: Test-drive each core component

**7. TDD: Implement core/audio_extractor.py**
- 🔴 Write `tests/test_audio_extractor.py` FIRST:
  ```python
  def test_extract_audio_from_video(tmp_path):
      # Test with mock video file
      output = extract_audio("tests/fixtures/sample.mp4", tmp_path / "output.wav")
      assert output.exists()
      assert output.suffix == ".wav"
      # Verify audio properties: 16kHz, mono

  def test_extract_audio_invalid_file():
      with pytest.raises(AudioExtractionError):
          extract_audio("nonexistent.mp4")
  ```
- 🟢 Implement `extract_audio()` function
- 🔵 Add error handling, cleanup, format validation

**8. TDD: Implement core/vad_processor.py** ⭐ CRITICAL
- 🔴 Write `tests/test_vad_processor.py` FIRST:
  ```python
  def test_vad_processor_initialization():
      vad = VADProcessor()
      assert vad.min_speech_duration_ms == 250

  def test_vad_process_audio(sample_audio_path):
      vad = VADProcessor()
      segments = vad.process(sample_audio_path)
      assert len(segments) > 0
      assert all(isinstance(s, SpeechSegment) for s in segments)
      assert all(1.5 <= s.duration <= 6.0 for s in segments)  # Timing targets

  def test_vad_filters_short_segments():
      vad = VADProcessor(min_speech_duration_ms=500)
      segments = vad.process("tests/fixtures/short_clips.wav")
      assert all(s.duration >= 0.5 for s in segments)
  ```
- 🟢 Implement `VADProcessor` class with Silero VAD
- 🔵 Optimize parameters, add chunking for long audio

**9. TDD: Implement core/transcriber.py** ⭐ CRITICAL
- 🔴 Write `tests/test_transcriber.py` FIRST:
  ```python
  def test_transcriber_initialization_cuda():
      transcriber = Transcriber(device="cuda")
      # Should fallback to CPU if CUDA unavailable in test

  def test_transcriber_device_fallback(mocker):
      mocker.patch('torch.cuda.is_available', return_value=False)
      transcriber = Transcriber(device="cuda")
      assert transcriber.device == "cpu"

  def test_transcribe_segments(sample_audio_path, mock_vad_segments):
      transcriber = Transcriber(model_size="base", device="cpu")  # Use base for speed
      results = transcriber.transcribe_segments(sample_audio_path, mock_vad_segments)
      assert len(results) == len(mock_vad_segments)
      assert all(r.text for r in results)  # All have text
      assert all(r.start >= 0 for r in results)  # Valid timestamps

  def test_model_manager_singleton():
      model1 = ModelManager.get_model("base", "cpu", "int8")
      model2 = ModelManager.get_model("base", "cpu", "int8")
      assert model1 is model2  # Same instance
  ```
- 🟢 Implement `Transcriber` class and `ModelManager`
- 🔵 Add GPU optimization, memory management, error handling

**10. TDD: Implement core/subtitle_generator.py**
- 🔴 Write `tests/test_subtitle_generator.py` FIRST:
  ```python
  def test_generate_srt(tmp_path):
      segments = [
          TranscriptSegment(0.0, 2.5, "Hello world"),
          TranscriptSegment(3.0, 5.5, "This is a test"),
      ]
      output = tmp_path / "output.srt"
      result = generate_srt(segments, output)

      assert output.exists()
      content = output.read_text()
      assert "1\n" in content  # Subtitle number
      assert "00:00:00,000 --> 00:00:02,500" in content
      assert "Hello world" in content

  def test_merge_short_segments():
      segments = [
          TranscriptSegment(0.0, 0.8, "Hi"),  # Too short
          TranscriptSegment(1.0, 1.6, "there"),  # Too short
          TranscriptSegment(2.0, 5.0, "How are you?"),  # Good
      ]
      merged = merge_short_segments(segments, min_duration=1.5)
      assert len(merged) < len(segments)
      assert merged[0].text == "Hi there"  # Merged

  def test_split_long_segments():
      segments = [TranscriptSegment(0.0, 10.0, "A very long sentence...")]
      split = split_long_segments(segments, max_duration=6.0)
      assert len(split) > 1
  ```
- 🟢 Implement SRT/VTT generation functions
- 🔵 Add segment merging/splitting, word wrapping

### Phase 3: Pipeline Orchestration (TDD)
**Goal**: Test-drive end-to-end workflow

**11. TDD: Implement utils/validators.py and utils/file_utils.py**
- 🔴 Write tests for validation and file utilities FIRST
- 🟢 Implement validators and file utilities
- 🔵 Add edge case handling

**12. TDD: Implement pipeline/transcription_pipeline.py** ⭐ CRITICAL
- 🔴 Write `tests/test_pipeline.py` FIRST:
  ```python
  def test_pipeline_initialization():
      pipeline = TranscriptionPipeline()
      assert pipeline.config is not None

  def test_pipeline_process_audio(sample_audio_path, tmp_path):
      config = PipelineConfig(
          asr=ASRConfig(model_size="base", device="cpu"),  # Fast for testing
          extract_audio=False  # Already have audio
      )
      pipeline = TranscriptionPipeline(config)

      output = tmp_path / "output.srt"
      result = pipeline.process(sample_audio_path, output)

      assert output.exists()
      assert result.segment_count > 0
      assert result.duration_seconds > 0
      assert result.processing_time > 0

  def test_pipeline_process_video(tmp_path):
      # Test with video input (requires audio extraction)
      config = PipelineConfig(extract_audio=True)
      pipeline = TranscriptionPipeline(config)
      # ... test video processing

  def test_pipeline_batch_processing(tmp_path):
      pipeline = TranscriptionPipeline()
      input_files = ["audio1.wav", "audio2.wav"]
      results = pipeline.process_batch(input_files, tmp_path)
      assert len(results) == 2

  def test_pipeline_error_cleanup(mocker):
      # Test that temp files are cleaned up on error
      mocker.patch('aitranscript.core.vad_processor.VADProcessor.process',
                   side_effect=VADProcessingError("Test error"))
      pipeline = TranscriptionPipeline()
      with pytest.raises(VADProcessingError):
          pipeline.process("audio.wav")
      # Verify no temp files left behind
  ```
- 🟢 Implement `TranscriptionPipeline` class
- 🔵 Add error handling, progress callbacks, resource cleanup

### Phase 4: CLI Interface (TDD)
**Goal**: Test-drive command-line interface

**13. TDD: Implement cli/main.py**
- 🔴 Write `tests/test_cli.py` FIRST:
  ```python
  from click.testing import CliRunner
  from aitranscript.cli.main import cli

  def test_cli_transcribe_command():
      runner = CliRunner()
      result = runner.invoke(cli, ['transcribe', 'tests/fixtures/sample.wav'])
      assert result.exit_code == 0

  def test_cli_batch_command():
      runner = CliRunner()
      result = runner.invoke(cli, ['batch', 'tests/fixtures/', '-o', 'output/'])
      assert result.exit_code == 0

  def test_cli_invalid_file():
      runner = CliRunner()
      result = runner.invoke(cli, ['transcribe', 'nonexistent.mp4'])
      assert result.exit_code != 0
      assert "Error" in result.output
  ```
- 🟢 Implement Click CLI with `transcribe` and `batch` commands
- 🔵 Add Rich progress bars, better error messages

**14. Update pyproject.toml** with entry point and run `uv sync`

### Phase 5: Integration Testing & Documentation
**Goal**: Ensure end-to-end functionality and documentation

**15. Integration tests**
- 🔴 Write comprehensive integration tests:
  ```python
  def test_full_workflow_video_to_srt():
      # End-to-end: video → audio → VAD → ASR → SRT
      # Use real (small) video file from fixtures

  def test_batch_processing_performance():
      # Verify model reuse speeds up batch processing

  def test_gpu_vs_cpu_compatibility():
      # Same results on CPU and GPU (within tolerance)
  ```
- 🟢 Ensure all integration tests pass
- Run full test suite: `uv run pytest -v --cov`

**16. Documentation & final polish**
- Create comprehensive README.md
- Add .gitignore
- Code quality: `uv run ruff format . && uv run ruff check --fix`
- Final test run with coverage report

**17. Create sample test fixtures**
- Add `tests/fixtures/sample_audio.wav` (5-10s audio clip)
- Add `tests/fixtures/sample_video.mp4` (short video for testing)
- Document where to get test files (or generate synthetic ones)

---

## Critical Implementation Details

### GPU Setup & CUDA Optimization
```python
# In transcriber.py initialization
if device == "cuda":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
        compute_type = "int8"
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

### VAD Configuration for Subtitle Quality
```python
# Recommended defaults in VADConfig
min_speech_duration_ms = 250    # Filter very short sounds
max_pause_duration_ms = 300     # Allow brief pauses in same subtitle
max_segment_duration_s = 6.0    # Prevent overly long subtitles
```

### Model Management Strategy
```python
# Singleton pattern for model reuse in batch processing
class ModelManager:
    _model = None

    @classmethod
    def get_model(cls, model_size, device, compute_type):
        if cls._model is None:
            cls._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root="./models"  # Cache locally
            )
        return cls._model
```

### Memory Management for Long Videos
```python
# Process in batches for videos ≥60 minutes
BATCH_SIZE = 50  # Process 50 segments at a time
for i in range(0, len(segments), BATCH_SIZE):
    batch = segments[i:i + BATCH_SIZE]
    results.extend(self._transcribe_batch(batch))
    if device == "cuda":
        torch.cuda.empty_cache()  # Clear cache periodically
```

---

## Future Extension: Translation Module

**Design for extensibility** - Translation can be added without refactoring:

```python
# Future structure
src/aitranscript/translation/
    ├── translator.py           # Base interface
    ├── openai_translator.py    # GPT API implementation
    └── nllb_translator.py      # Offline NLLB-200

# Integration point in PipelineConfig
@dataclass
class PipelineConfig:
    # ... existing fields ...
    translation: Optional[TranslationConfig] = None  # None = no translation
```

**Key principles**:
- Interface-based design (all translators implement common interface)
- Timestamp preservation (translation doesn't modify timing)
- Plugin architecture (easy to add providers)
- Optional dual-language subtitle output

---

## Critical Files to Implement (Priority Order)

1. **src/aitranscript/models/segment.py** - Foundation data structures
2. **src/aitranscript/models/config.py** - Configuration architecture
3. **src/aitranscript/core/vad_processor.py** - Most impactful on subtitle quality
4. **src/aitranscript/core/transcriber.py** - Core ASR engine with GPU optimization
5. **src/aitranscript/pipeline/transcription_pipeline.py** - Main orchestration

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "aitranscript"
version = "0.1.0"
description = "AI-powered video transcription with Whisper and VAD"
requires-python = ">=3.9"
dependencies = [
    "faster-whisper>=1.0.0",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "pydub>=0.25.1",
    "numpy>=1.24.0",
    "silero-vad>=5.0.0",
    "srt>=3.5.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
]

[project.scripts]
aitranscript = "aitranscript.cli.main:cli"
```

---

## Development Workflow

```bash
# Code quality
uv run ruff format . && uv run ruff check --fix

# Testing
uv run pytest                    # All tests
uv run pytest -v --cov          # With coverage

# Run CLI
uv run aitranscript transcribe video.mp4
uv run aitranscript batch ./videos/ -o ./output/
```

---

## Success Criteria

- ✅ Process 10-20 min video in 3-8 minutes (GPU medium model)
- ✅ Generate subtitles with 1.5-6s segments
- ✅ Natural sentence breaks (no mid-phrase cuts)
- ✅ Support batch processing with model reuse
- ✅ Graceful CPU fallback when CUDA unavailable
- ✅ 70%+ test coverage for core modules
- ✅ Clear error messages and progress indication
- ✅ Ready for future translation module integration

---

## TDD Benefits for This Project

**Why TDD for AI Transcript?**
1. ✅ **Complex Components**: VAD and ASR have many edge cases - tests catch them early
2. ✅ **GPU/CPU Compatibility**: Tests ensure graceful fallback and consistent behavior
3. ✅ **Segment Timing Critical**: Tests verify 1.5-6s subtitle timing constraints
4. ✅ **File I/O Heavy**: Tests catch path issues, cleanup failures, format errors
5. ✅ **Refactoring Safety**: Can optimize (e.g., memory management) with confidence
6. ✅ **Documentation**: Tests serve as executable examples of how components work
7. ✅ **Future Translation**: Strong test suite makes adding translation safer

**TDD Daily Workflow:**
```bash
# 1. Write failing test
uv run pytest tests/test_vad_processor.py -v  # 🔴 RED

# 2. Implement minimal code
# ... edit vad_processor.py ...

# 3. Make test pass
uv run pytest tests/test_vad_processor.py -v  # 🟢 GREEN

# 4. Refactor while keeping tests green
uv run ruff format . && uv run ruff check --fix
uv run pytest tests/test_vad_processor.py -v  # 🔵 Still GREEN

# 5. Run full suite periodically
uv run pytest -v --cov
```

---

## Estimated Implementation Time: 9-10 days

**Phase 1**: 1.5 days (setup + models + test infrastructure)
**Phase 2**: 3.5 days (TDD core components with tests)
**Phase 3**: 2 days (TDD pipeline)
**Phase 4**: 1 day (TDD CLI)
**Phase 5**: 1 day (integration tests + docs)

*Note: TDD adds ~10-15% upfront time but reduces debugging time and ensures production quality*
