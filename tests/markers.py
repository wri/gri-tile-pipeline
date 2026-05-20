import pytest
import boto3

from tests.constants import ARD_DIR, REFERENCE_TIF, MODEL_DIR, GOLDEN_DIR

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

has_ard = pytest.mark.skipif(
    not ARD_DIR.is_dir(),
    reason=f"ARD directory not found: {ARD_DIR}",
)
has_reference = pytest.mark.skipif(
    not REFERENCE_TIF.is_file(),
    reason=f"Reference TIF not found: {REFERENCE_TIF}",
)
has_model = pytest.mark.skipif(
    not (MODEL_DIR / "predict_graph-172.pb").is_file(),
    reason=f"Model not found: {MODEL_DIR / 'predict_graph-172.pb'}",
)

_tf_available = False
try:
    import tensorflow  # noqa: F401
    _tf_available = True
except ImportError:
    pass

has_tf = pytest.mark.skipif(not _tf_available, reason="TensorFlow not installed")

has_golden = pytest.mark.skipif(
    not GOLDEN_DIR.is_dir(),
    reason=f"Golden test data not found: {GOLDEN_DIR}",
)

def _aws_available() -> bool:
    return boto3.Session().get_credentials() is not None

has_aws = pytest.mark.skipif(not _aws_available(), reason="No AWS credentials")

