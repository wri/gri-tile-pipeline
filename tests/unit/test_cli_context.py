"""Unit tests for gri_tile_pipeline.cli_context."""

import json
from io import StringIO

import click
import pytest
from click.testing import CliRunner

from gri_tile_pipeline.cli_context import (
    CliContext,
    attach,
    confirm_or_abort,
    emit_json,
    get,
)
from gri_tile_pipeline.config import PipelineConfig


def _make_ctx(**overrides):
    base = dict(
        cfg=PipelineConfig(),
        run_id="testrun1",
    )
    base.update(overrides)
    return CliContext(**base)


def test_attach_and_get_roundtrip():
    @click.command()
    @click.pass_context
    def cmd(ctx):
        gri = _make_ctx(verbose=2)
        attach(ctx, gri)
        assert get(ctx) is gri
        assert ctx.obj["cfg"] is gri.cfg
        assert ctx.obj["run_id"] == "testrun1"

    CliRunner().invoke(cmd, [], catch_exceptions=False)


def test_get_walks_parent_chain():
    @click.group()
    @click.pass_context
    def grp(ctx):
        attach(ctx, _make_ctx(run_id="parent1"))

    @grp.command()
    @click.pass_context
    def child(ctx):
        assert get(ctx).run_id == "parent1"

    result = CliRunner().invoke(grp, ["child"], catch_exceptions=False)
    assert result.exit_code == 0


def test_get_raises_without_attach():
    @click.command()
    @click.pass_context
    def cmd(ctx):
        get(ctx)

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def _capture_stdout(fn):
    """Run *fn* with sys.stdout redirected; return what it wrote."""
    import sys as _sys
    buf = StringIO()
    old, _sys.stdout = _sys.stdout, buf
    try:
        fn()
    finally:
        _sys.stdout = old
    return buf.getvalue()


def test_emit_json_writes_payload_in_json_mode():
    ctx = click.Context(click.Command("c"))
    attach(ctx, _make_ctx(json_mode=True))
    out = _capture_stdout(lambda: emit_json(ctx, {"status": "ok", "count": 3}))
    payload = json.loads(out)
    assert payload == {"count": 3, "status": "ok"}


def test_emit_json_noop_when_disabled():
    ctx = click.Context(click.Command("c"))
    attach(ctx, _make_ctx(json_mode=False))
    out = _capture_stdout(lambda: emit_json(ctx, {"x": 1}))
    assert out == ""


def test_confirm_or_abort_honors_yes():
    ctx = click.Context(click.Command("c"))
    attach(ctx, _make_ctx(yes=True))
    confirm_or_abort(ctx, "proceed?")  # must not raise


def test_confirm_or_abort_honors_dry_run():
    ctx = click.Context(click.Command("c"))
    attach(ctx, _make_ctx(dry_run=True))
    confirm_or_abort(ctx, "proceed?")  # must not raise


def test_confirm_or_abort_honors_json_mode():
    ctx = click.Context(click.Command("c"))
    attach(ctx, _make_ctx(json_mode=True))
    confirm_or_abort(ctx, "proceed?")  # must not raise


def test_root_group_global_flags_populate_ctx():
    """-v, --json, --workers, --aws-profile get stashed on CliContext."""
    from gri_tile_pipeline.cli import gri_ttc

    # Invoke a dummy subcommand that inspects CliContext.
    @gri_ttc.command("_test_ctx")
    @click.pass_context
    def _test_ctx(ctx):
        gri = get(ctx)
        click.echo(f"v={gri.verbose} q={gri.quiet} json={gri.json_mode} "
                   f"workers={gri.workers} dry={gri.dry_run} yes={gri.yes} "
                   f"profile={gri.aws_profile} histdir={gri.run_history_dir}")

    runner = CliRunner()
    result = runner.invoke(
        gri_ttc,
        ["-vv", "--json", "--workers", "8", "--dry-run", "--yes",
         "--aws-profile", "land", "--run-history-dir", "/tmp/runs", "_test_ctx"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "v=2" in result.output
    assert "json=True" in result.output
    assert "workers=8" in result.output
    assert "dry=True" in result.output
    assert "yes=True" in result.output
    assert "profile=land" in result.output
    assert "histdir=/tmp/runs" in result.output


def test_root_group_rejects_verbose_and_quiet_together():
    from gri_tile_pipeline.cli import gri_ttc

    result = CliRunner().invoke(gri_ttc, ["-v", "-q"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
