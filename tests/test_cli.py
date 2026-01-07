"""Tests for CLI interface

Tests follow TDD RED phase - these tests will fail until cli/main.py is implemented.
"""

from click.testing import CliRunner

from aitranscript.cli.main import cli

# ============================================================================
# Phase 4.1: CLI Core Tests
# ============================================================================


class TestCLICore:
    """Test basic CLI functionality"""

    def test_cli_help(self):
        """Should display help message"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert (
            "aitranscript" in result.output.lower()
            or "transcribe" in result.output.lower()
        )

    def test_cli_version(self):
        """Should display version"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output or "version" in result.output.lower()

    def test_cli_no_args(self):
        """Should error when no command provided"""
        runner = CliRunner()
        result = runner.invoke(cli)
        # Click exits with code 2 when missing required command
        assert result.exit_code == 2

    def test_cli_transcribe_command_exists(self):
        """Should have transcribe command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower()

    def test_cli_invalid_command(self):
        """Should error on invalid command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0

    def test_cli_help_shows_usage(self):
        """Should show usage instructions in help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "commands" in result.output.lower()

    def test_cli_transcribe_help(self):
        """Should show transcribe command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower()

    def test_cli_creates_context(self):
        """Should create Click context object"""
        runner = CliRunner()
        # This tests that the CLI group properly initializes
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0


# ============================================================================
# Phase 4.2: Transcribe Command Tests - Basic
# ============================================================================


class TestTranscribeCommandBasic:
    """Test basic transcribe command functionality"""

    def test_transcribe_audio_file(self, sample_audio_wav, tmp_output_dir):
        """Should transcribe audio file successfully"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli, ["transcribe", str(sample_audio_wav), "-o", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Success" in result.output or "✅" in result.output

    def test_transcribe_default_output_name(self, sample_audio_wav):
        """Should auto-generate output filename"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Copy sample file to isolated filesystem
            import shutil
            from pathlib import Path

            test_audio = Path("test_audio.wav")
            shutil.copy(sample_audio_wav, test_audio)

            result = runner.invoke(cli, ["transcribe", str(test_audio)])

            assert result.exit_code == 0
            # Should create test_audio.srt in current directory
            assert Path("test_audio.srt").exists()

    def test_transcribe_with_model_size(self, sample_audio_wav, tmp_output_dir):
        """Should accept model size parameter"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--model",
                "tiny",
            ],
        )

        assert result.exit_code == 0

    def test_transcribe_vtt_format(self, sample_audio_wav, tmp_output_dir):
        """Should support VTT format"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.vtt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--format",
                "vtt",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert content.startswith("WEBVTT")

    def test_transcribe_nonexistent_file(self, tmp_output_dir):
        """Should handle missing input file gracefully"""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "transcribe",
                "/nonexistent/file.mp4",
                "-o",
                str(tmp_output_dir / "output.srt"),
            ],
        )

        assert result.exit_code != 0

    def test_transcribe_cpu_device(self, sample_audio_wav, tmp_output_dir):
        """Should accept device parameter"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0


# ============================================================================
# Phase 4.2: Transcribe Command Tests - Advanced
# ============================================================================


class TestTranscribeCommandAdvanced:
    """Test advanced transcribe command features"""

    def test_transcribe_language(self, sample_audio_wav, tmp_output_dir):
        """Should accept language parameter"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--language",
                "en",
            ],
        )

        assert result.exit_code == 0

    def test_transcribe_quiet_mode(self, sample_audio_wav, tmp_output_dir):
        """Should support quiet mode"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--quiet",
            ],
        )

        assert result.exit_code == 0
        # Quiet mode should have minimal output

    def test_transcribe_verbose_mode(self, sample_audio_wav, tmp_output_dir):
        """Should support verbose mode"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        # Verbose mode should have detailed output

    def test_transcribe_combined_options(self, sample_audio_wav, tmp_output_dir):
        """Should handle multiple options together"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.vtt"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_audio_wav),
                "-o",
                str(output_file),
                "--model",
                "tiny",
                "--device",
                "cpu",
                "--format",
                "vtt",
                "--language",
                "en",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_transcribe_short_option_o(self, sample_audio_wav, tmp_output_dir):
        """Should accept short -o option for output"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli, ["transcribe", str(sample_audio_wav), "-o", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_transcribe_short_option_q(self, sample_audio_wav, tmp_output_dir):
        """Should accept short -q option for quiet mode"""
        runner = CliRunner()
        output_file = tmp_output_dir / "output.srt"

        result = runner.invoke(
            cli, ["transcribe", str(sample_audio_wav), "-o", str(output_file), "-q"]
        )

        assert result.exit_code == 0
