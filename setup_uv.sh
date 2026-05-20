#!/bin/bash

uv lock --upgrade # -U
uv sync
source .venv/bin/activate
