"""
Models of the flow-based cascade.

Stage 1 is :class:`~ml.flow.models.isolation_forest.FlowAnomalyDetector`, an
Isolation Forest trained on BENIGN traffic only. Stage 2 is either
:class:`~ml.flow.models.xgboost_classifier.FlowAttackClassifier` (the serving
path) or the from-scratch NumPy MLP in ``neural_network``.

``NeuralNetwork`` is deliberately **not** re-exported here. It scores F1-macro
0.7508 against 0.9516 for XGBoost, so it never enters the serving path, and
importing ``ml.flow.models`` should not pull in an experimental branch nobody
serves. Import it explicitly:

    from ml.flow.models.neural_network import NeuralNetwork
"""

from ml.flow.models.isolation_forest import FlowAnomalyDetector
from ml.flow.models.xgboost_classifier import FlowAttackClassifier

__all__ = [
    "FlowAnomalyDetector",
    "FlowAttackClassifier",
]
