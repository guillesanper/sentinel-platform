"""
Two-stage cascades of the flow-based track.

:class:`~ml.flow.pipelines.cascade_xgb.CascadeXGBPipeline` is the production
path: Isolation Forest for "is this anomalous", XGBoost for "which attack".

``CascadeMLPPipeline`` is **not** re-exported. It shares Stage 1 with the
XGBoost cascade but scores F1-macro 0.7508 against 0.9516, and it exists as a
reference implementation rather than a deployable model. Importing
``ml.flow.pipelines`` should not pull in a NumPy MLP nobody serves:

    from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline
"""

from ml.flow.pipelines.base import (
    FEATURE_NAMES_FILE,
    FEATURE_NAMES_FORMAT_VERSION,
    BaseCascadePipeline,
)
from ml.flow.pipelines.cascade_xgb import CascadeXGBPipeline

__all__ = [
    "FEATURE_NAMES_FILE",
    "FEATURE_NAMES_FORMAT_VERSION",
    "BaseCascadePipeline",
    "CascadeXGBPipeline",
]
