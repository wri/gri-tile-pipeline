from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

REFERENCE_TIF = REPO_ROOT / "example" / "1000X871Y_FINAL.tif"
MODEL_DIR = REPO_ROOT / "models"

ARD_DIR = REPO_ROOT / "example" / "sample_ard"
ARD_RAW = ARD_DIR/ "raw_v2"

GOLDEN_DIR = REPO_ROOT / "example" / "golden"
GOLDEN_RAW = GOLDEN_DIR / "raw"
GOLDEN_TILES = ["1000X798Y", "1000X799Y", "1000X800Y"]