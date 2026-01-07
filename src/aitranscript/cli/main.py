"""CLI interface for aitranscript

Provides commands for video/audio transcription with VAD + Whisper.

Usage:
    aitranscript transcribe video.mp4
    aitranscript transcribe audio.wav -o output.srt --model tiny
    aitranscript transcribe video.mp4 --format vtt --device cpu
"""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from aitranscript.models.config import ASRConfig, PipelineConfig
from aitranscript.pipeline.transcription_pipeline import TranscriptionPipeline

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="aitranscript")
@click.pass_context
def cli(ctx):
    """AI-powered video transcription with Whisper and VAD

    Generate accurate subtitles from video or audio files.

    Examples:
        aitranscript transcribe video.mp4
        aitranscript transcribe audio.wav -o output.srt
        aitranscript transcribe video.mp4 --model tiny --device cpu
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output subtitle file path (auto-generated if not specified)",
)
@click.option(
    "--model",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default="medium",
    help="Whisper model size (default: medium)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Compute device (default: cuda)",
)
@click.option(
    "--format",
    type=click.Choice(["srt", "vtt"]),
    default="srt",
    help="Subtitle format (default: srt)",
)
@click.option("--language", default="en", help="Language code (default: en)")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output (no progress bars)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output (detailed logging)")
def transcribe(input_file, output, model, device, format, language, quiet, verbose):
    """Transcribe a video or audio file to subtitles

    Examples:
        aitranscript transcribe video.mp4
        aitranscript transcribe audio.wav -o output.srt --model tiny
        aitranscript transcribe video.mp4 --format vtt --device cpu
    """

    # Configure logging level
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.WARNING
    else:
        # Normal mode: suppress INFO logs to avoid cluttering spinner
        log_level = logging.WARNING

    # Configure all aitranscript loggers AFTER import
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("aitranscript"):
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(log_level)
            # Also configure all handlers
            for handler in logger_obj.handlers:
                handler.setLevel(log_level)

    try:
        # Determine output path
        if output is None:
            input_path = Path(input_file)
            output = input_path.stem + f".{format}"

        # Create configuration
        config = PipelineConfig(
            asr=ASRConfig(model_size=model, device=device, language=language),
            output_format=format,
        )

        # Initialize pipeline
        pipeline = TranscriptionPipeline(config)

        # Process with progress
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(
                    f"Transcribing {Path(input_file).name}...", total=None
                )
                result = pipeline.process(input_file, output)
        else:
            result = pipeline.process(input_file, output)

        # Report results
        if result.success:
            console.print("[green]✅ Success![/green]")
            console.print(f"Output: {result.output_path}")
            console.print(f"Segments: {result.segment_count}")
            console.print(f"Duration: {result.duration_seconds:.1f}s")
            console.print(f"Processing time: {result.processing_time_seconds:.1f}s")
        else:
            console.print(f"[red]❌ Error: {result.error}[/red]")
            raise click.Abort()

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise click.Abort() from e
