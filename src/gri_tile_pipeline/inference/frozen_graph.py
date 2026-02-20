"""TensorFlow frozen-graph loading and session management.

Ported from reference ``download_and_predict_job.py`` lines 1905-1947.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class PredictSession:
    """Holds a TF v1 session plus the tensors needed for inference."""

    sess: "tf.compat.v1.Session"
    logits: "tf.Tensor"
    inp: "tf.Tensor"
    length: "tf.Tensor"


@dataclass
class SuperResolveSession:
    """Holds a TF v1 session for DSen2 super-resolution."""

    sess: "tf.compat.v1.Session"
    logits: "tf.Tensor"
    inp: "tf.Tensor"
    inp_bilinear: "tf.Tensor"


def load_predict_graph(
    model_dir: str,
    size: int = 172,
    length: int = 4,
) -> PredictSession:
    """Load the tree-cover prediction frozen graph.

    Args:
        model_dir: Directory containing ``predict_graph-{size}.pb``.
        size: Spatial size encoded in the graph filename.
        length: Temporal sequence length (default 4 = quarterly).

    Returns:
        :class:`PredictSession` with session and tensor handles.
    """
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    pb_path = str(Path(model_dir) / f"predict_graph-{size}.pb")
    logger.info(f"Loading prediction graph from {pb_path}")

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    # TF 2.x requires explicit Graph context for import_graph_def
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="predict")
    sess = tf.compat.v1.Session(graph=graph)

    if length == 12:
        logits_name = "predict/conv2d_13/Sigmoid:0"
    else:
        logits_name = "predict/conv2d/Sigmoid:0"

    logits = sess.graph.get_tensor_by_name(logits_name)
    inp = sess.graph.get_tensor_by_name("predict/Placeholder:0")
    length_tensor = sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")

    return PredictSession(sess=sess, logits=logits, inp=inp, length=length_tensor)


def load_superresolve_graph(model_dir: str) -> SuperResolveSession:
    """Load the DSen2 super-resolution frozen graph.

    Args:
        model_dir: Directory containing ``superresolve_graph.pb``.

    Returns:
        :class:`SuperResolveSession` with session and tensor handles.
    """
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    pb_path = Path(model_dir) / "superresolve_graph.pb"
    if not pb_path.exists():
        raise FileNotFoundError(pb_path)
    pb_path = str(pb_path)
    logger.info(f"Loading super-resolution graph from {pb_path}")

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="superresolve")
    sess = tf.compat.v1.Session(graph=graph)

    logits = sess.graph.get_tensor_by_name("superresolve/Add_2:0")
    inp = sess.graph.get_tensor_by_name("superresolve/Placeholder:0")
    inp_bilinear = sess.graph.get_tensor_by_name("superresolve/Placeholder_1:0")

    return SuperResolveSession(
        sess=sess, logits=logits, inp=inp, inp_bilinear=inp_bilinear,
    )
