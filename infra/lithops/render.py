"""Render Lithops config templates from Terraform outputs.

Usage:
    python infra/lithops/render.py --env land-research

Reads:
    infra/lithops/*.yaml.tmpl           (templates with ${VAR} placeholders)
    terraform -chdir=infra/terraform/envs/<env> output -json   (values)

Writes:
    .lithops/<env>/*.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from string import Template


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "infra" / "lithops"


def terraform_outputs(env: str) -> dict[str, str]:
    env_dir = REPO_ROOT / "infra" / "terraform" / "envs" / env
    if not env_dir.exists():
        sys.exit(f"No Terraform env at {env_dir}")
    result = subprocess.run(
        ["terraform", f"-chdir={env_dir}", "output", "-json"],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = json.loads(result.stdout)
    return {k.upper(): v["value"] for k, v in raw.items()}


def render(env: str) -> None:
    values = terraform_outputs(env)
    out_dir = REPO_ROOT / ".lithops" / env
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = sorted(TEMPLATES_DIR.glob("*.yaml.tmpl"))
    if not templates:
        sys.exit(f"No templates found in {TEMPLATES_DIR}")

    for tmpl in templates:
        body = Template(tmpl.read_text())
        try:
            rendered = body.substitute(values)
        except KeyError as e:
            sys.exit(
                f"{tmpl.name}: template references ${{{e.args[0]}}} but Terraform "
                f"did not output that value. Available: {sorted(values)}"
            )
        dest = out_dir / tmpl.name.removesuffix(".tmpl")
        dest.write_text(rendered)
        print(f"  {tmpl.name} -> {dest.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Terraform env name (e.g. land-research).")
    args = parser.parse_args()
    print(f"Rendering Lithops configs for env={args.env}")
    render(args.env)


if __name__ == "__main__":
    main()
