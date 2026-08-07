"""
Flow-based intrusion detection track.

Models network flows (78 CICFlowMeter features, taxonomy
``BENIGN|Botnet|BruteForce|DDoS|DoS|Scan|WebAttack``), in parallel to the
event-based track in ``ml/models/``. The two ontologies are deliberately kept
separate: they share neither feature space nor label space.

Nothing in this package may import matplotlib at module level — ``ml/`` is
copied wholesale into the API image, which installs only ``requirements.txt``.
"""

from ml.flow.config import (
    FlowConfig,
    IsolationForestConfig,
    NeuralNetworkConfig,
    XGBoostConfig,
)
from ml.flow.labels import (
    ATTACK_GROUPS,
    BENIGN_LABEL,
    CAPTURE_DAYS,
    CLASS_ORDER,
    UNKNOWN_CAPTURE_DAY,
    map_capture_day,
    map_label,
)

__all__ = [
    "ATTACK_GROUPS",
    "BENIGN_LABEL",
    "CAPTURE_DAYS",
    "CLASS_ORDER",
    "UNKNOWN_CAPTURE_DAY",
    "FlowConfig",
    "IsolationForestConfig",
    "NeuralNetworkConfig",
    "XGBoostConfig",
    "map_capture_day",
    "map_label",
]
