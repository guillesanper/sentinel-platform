"""Evaluation utilities shared by both Sentinel ML tracks (events and flows).

Only :mod:`ml.evaluation.metrics` is re-exported here, and that is deliberate:
``ml.evaluation.plots`` depends on matplotlib/seaborn, which live in
``requirements-dev.txt``. ``api/Dockerfile`` copies ``ml/`` wholesale while
installing only ``requirements.txt``, so any import chain reaching ``plots``
from package import time would break the production image (or bloat it by
~120 MB of plotting stack and CVE surface). Import ``plots`` explicitly, inside
the branch that actually needs a figure.
"""

from ml.evaluation.metrics import best_fbeta_threshold, compare_models, compute_metrics

__all__ = [
    "best_fbeta_threshold",
    "compare_models",
    "compute_metrics",
]
