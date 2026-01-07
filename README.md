# AI Transcript

> AI-powered video transcription with Whisper and VAD for accurate subtitle generation

Generate high-quality subtitles from video or audio files using OpenAI's Whisper model with intelligent voice activity detection (VAD) for optimal accuracy.

## Features

- **🎯 High Accuracy**: Combines Silero VAD + Faster-Whisper for precise speech detection and transcription
- **⚡ GPU Accelerated**: CUDA support with 2.8x faster processing (GTX 1080 tested)
- **📝 Multiple Formats**: Export to SRT or VTT subtitle formats
- **🎬 Video & Audio**: Supports MP4, AVI, MKV, MOV, WAV, MP3, FLAC, and more
- **🛠️ Flexible Models**: Choose from tiny to large Whisper models based on accuracy/speed needs
- **🌍 Multi-language**: Supports 99+ languages including English, Chinese, Spanish, French, etc.
- **💻 Clean CLI**: Professional command-line interface with progress indicators
- **🧪 Well Tested**: 206 tests with comprehensive coverage

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- FFmpeg (for video processing)
- CUDA-capable GPU (optional, for acceleration)

### Install with uv (Recommended)

```bash
# Clone repository
git clone https://github.com/TMYuan/ai-transcript.git
cd ai-transcript

# Install dependencies
uv sync

# Install FFmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```

### Install with pip

```bash
git clone https://github.com/TMYuan/ai-transcript.git
cd ai-transcript
pip install -e .
```

## Quick Start

```bash
# Basic usage (auto-detects GPU)
uv run aitranscript transcribe video.mp4

# Specify model size and device
uv run aitranscript transcribe video.mp4 --model medium --device cuda

# Generate VTT format with custom output
uv run aitranscript transcribe audio.wav --format vtt -o subtitles.vtt

# Transcribe in Chinese
uv run aitranscript transcribe video.mp4 --language zh
```

## CLI Usage

### Transcribe Command

```bash
uv run aitranscript transcribe [OPTIONS] INPUT_FILE
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output PATH` | Output subtitle file path | Auto-generated |
| `--model MODEL` | Whisper model size: tiny, base, small, medium, large | medium |
| `--device DEVICE` | Compute device: cuda, cpu | cuda |
| `--format FORMAT` | Subtitle format: srt, vtt | srt |
| `--language CODE` | Language code (en, zh, es, fr, etc.) | en |
| `-q, --quiet` | Minimal output (no progress bars) | - |
| `-v, --verbose` | Detailed logging for debugging | - |

**Examples:**

```bash
# Fast transcription with tiny model on CPU
uv run aitranscript transcribe video.mp4 --model tiny --device cpu

# High accuracy with large model on GPU
uv run aitranscript transcribe video.mp4 --model large --device cuda

# Quiet mode for scripting
uv run aitranscript transcribe video.mp4 --quiet -o output.srt

# Verbose output for debugging
uv run aitranscript transcribe video.mp4 --verbose
```

### Help

```bash
# Show help
uv run aitranscript --help

# Show transcribe command help
uv run aitranscript transcribe --help

# Show version
uv run aitranscript --version
```

## Supported Formats

**Video:** MP4, AVI, MKV, MOV, FLV, WMV, WebM
**Audio:** WAV, MP3, FLAC, AAC, OGG, M4A

## Project Structure

```
ai-transcript/
├── src/aitranscript/
│   ├── cli/
│   │   └── main.py              # Click-based CLI interface
│   ├── core/
│   │   ├── audio_extractor.py   # Video → Audio extraction
│   │   ├── transcriber.py       # Whisper transcription
│   │   ├── vad_processor.py     # Speech detection with Silero VAD
│   │   └── subtitle_generator.py # SRT/VTT generation
│   ├── models/
│   │   ├── config.py            # Configuration dataclasses
│   │   └── segment.py           # Speech/Transcript segments
│   ├── pipeline/
│   │   └── transcription_pipeline.py # End-to-end orchestration
│   └── utils/
│       ├── file_utils.py        # File operations
│       ├── validators.py        # Input/output validation
│       └── logger.py            # Centralized logging
├── tests/                       # Comprehensive test suite (206 tests)
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

## Support

- **Issues**: [GitHub Issues](https://github.com/TMYuan/ai-transcript/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TMYuan/ai-transcript/discussions)
