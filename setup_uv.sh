#!/bin/bash

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.https://${KENNC_REPO_PAT}@github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="https://github.com/"
uv lock --upgrade # -U
uv sync
source .venv/bin/activate
