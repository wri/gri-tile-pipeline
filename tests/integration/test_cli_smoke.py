"""CLI smoke tests: verify all subcommands respond to --help without import errors."""

import subprocess
import sys

import pytest


CLI_COMMANDS = [
    ["gri-ttc", "--help"],
    ["gri-ttc", "ingest", "--help"],
    ["gri-ttc", "check", "--help"],
    ["gri-ttc", "download", "--help"],
    ["gri-ttc", "download-s1", "--help"],
    ["gri-ttc", "predict", "--help"],
    ["gri-ttc", "stats", "--help"],
    ["gri-ttc", "cost", "--help"],
    ["gri-ttc", "run", "--help"],
]


@pytest.mark.parametrize("cmd", CLI_COMMANDS, ids=[" ".join(c) for c in CLI_COMMANDS])
def test_cli_help(cmd):
    """Each CLI subcommand should respond to --help without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "gri_tile_pipeline.cli"] + cmd[1:]
        if cmd[0] == "gri-ttc"
        else cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Fall back to invoking via click directly if the entry point isn't installed
    if result.returncode != 0:
        from click.testing import CliRunner

        from gri_tile_pipeline.cli import gri_ttc

        runner = CliRunner()
        cli_result = runner.invoke(gri_ttc, cmd[1:])
        assert cli_result.exit_code == 0, (
            f"Command {' '.join(cmd)} failed:\n{cli_result.output}"
        )


def test_cli_version():
    """gri-ttc --version should print version info."""
    from click.testing import CliRunner

    from gri_tile_pipeline.cli import gri_ttc

    runner = CliRunner()
    result = runner.invoke(gri_ttc, ["--version"])
    assert result.exit_code == 0
    assert "gri-ttc" in result.output or "version" in result.output.lower()
