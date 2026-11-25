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
# Cases of golden-sets/tree-cover/v1 in the model registry (see scripts/download_golden.py).
# 1000X79{8,9}Y/800Y: 2023, Brazil. 236{2,3}X110{3,4,6}Y: 2025, closed-canopy Ethiopia
# tiles where the pre-#21 S1 edge-padding produced visible block artifacts.
GOLDEN_TILES = [
    "1000X798Y", "1000X799Y", "1000X800Y",
    "2362X1103Y", "2362X1104Y", "2363X1106Y",
]