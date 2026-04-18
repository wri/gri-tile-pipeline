"""Ensure the repo root is on sys.path for Lithops module resolution.

Lithops uses ``imp.find_module()`` to locate ``include_modules`` entries.
When running via the installed CLI (``gri-ttc``), only ``src/`` is on
sys.path — the repo root (where ``loaders/`` and ``lithops_workers.py``
live) is not.  Importing this module adds it.
"""
from __future__ import annotations

import os
import sys

# Repo root = two levels up from this file (steps/ -> gri_tile_pipeline/ -> src/ -> repo)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
