"""
Training script for Sentinel ML models.

Trains the anomaly detector (Isolation Forest) and attack classifier (XGBoost)
using synthetic or real security event data. Integrates with MLflow for
experiment tracking and model versioning.

Usage:
    python -m ml.training.train --n-samples 10000 --output-dir models/
    python -m ml.training.train --data-path data/events.json --output-dir models/
"""

import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import mlflow
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.features.pipeline import CombinedFeatureExtractor, FeatureExtractor  # noqa: E402
from ml.models.anomaly_detector import NORMAL_LABEL, AnomalyDetector  # noqa: E402
from ml.models.classifier import ATTACK_TYPES, LABEL_TO_IDX, AttackClassifier  # noqa: E402

logger = structlog.get_logger()

# Threshold-tuning strategies exposed by --threshold-strategy.
THRESHOLD_STRATEGIES = ["recall", "fbeta", "none"]

# Binary classification labels (normal vs attack)
BINARY_ATTACK_TYPES = ["normal", "brute_force"]


def _infer_attack_type(row: pd.Series) -> str:
    """
    Infer attack type from row features using heuristics.

    Args:
        row: A pandas Series containing the row data.

    Returns:
        The inferred attack type string.
    """
    if row["attack_detected"] == 0:
        return "normal"

    # Determine attack type based on feature patterns
    if row["failed_logins"] >= 3 and row["login_attempts"] >= 5:
        attack_type = "brute_force"
    elif row["failed_logins"] >= 2 and row["ip_reputation_score"] < 0.3:
        attack_type = "credential_stuffing"
    elif row["unusual_time_access"] == 1 and row["session_duration"] > 600:
        attack_type = "lateral_movement"
    elif row["network_packet_size"] > 700 and row["session_duration"] > 500:
        attack_type = "data_exfiltration"
    elif row["protocol_type"] == "ICMP" or row["session_duration"] < 60:
        attack_type = "reconnaissance"
    else:
        attack_type = "privilege_escalation"

    return attack_type


# ============================================================================
# Data Loading
# ============================================================================


def load_or_generate_data(
    data_path: str | Path | None = None,
    n_samples: int = 10000,
    attack_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load data from file or generate synthetic data.

    Args:
        data_path: Path to JSON file with events. If None, generates synthetic data.
        n_samples: Number of samples to generate (if data_path is None).
        attack_ratio: Ratio of attack events in synthetic data.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y, labels) where:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Encoded labels of shape (n_samples,)
        - labels: String labels list

    Raises:
        FileNotFoundError: If data_path is specified but file doesn't exist.
        ValueError: If data file format is invalid.
    """
    extractor = FeatureExtractor()

    if data_path is not None:
        # Load from file
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("loading_data_from_file", path=str(data_path))

        with data_path.open() as f:
            data = json.load(f)

        # Expected format: {"events": [...], "labels": [...]}
        # or just a list of {"event": {...}, "label": "..."}
        if isinstance(data, dict):
            events = data.get("events", [])
            labels = data.get("labels", [])
        elif isinstance(data, list):
            events = [item.get("event", item) for item in data]
            labels = [item.get("label", "normal") for item in data]
        else:
            raise ValueError("Invalid data format. Expected dict or list.")

        if len(events) != len(labels):
            raise ValueError(f"Mismatch: {len(events)} events but {len(labels)} labels")

        logger.info("data_loaded", n_events=len(events))

    else:
        # Generate synthetic data
        logger.info(
            "generating_synthetic_data",
            n_samples=n_samples,
            attack_ratio=attack_ratio,
            seed=seed,
        )

        # Import here to avoid circular imports
        from data.generators.synthetic import generate_dataset

        events, labels = generate_dataset(
            n_samples=n_samples,
            attack_ratio=attack_ratio,
            seed=seed,
        )

        logger.info("synthetic_data_generated", n_events=len(events))

    # Extract features
    logger.info("extracting_features", n_events=len(events))
    X = extractor.transform_batch(events)

    # Encode labels
    y = np.array([LABEL_TO_IDX.get(label, 0) for label in labels], dtype=np.int32)

    logger.info(
        "data_prepared",
        X_shape=X.shape,
        y_shape=y.shape,
        n_classes=len(set(labels)),
    )

    return X, y, labels


def load_kaggle_csv(
    csv_path: str | Path,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load data directly from Kaggle CSV with pre-computed features.

    The CSV is expected to have columns: session_id, network_packet_size,
    protocol_type, login_attempts, session_duration, encryption_used,
    ip_reputation_score, failed_logins, browser_type, unusual_time_access,
    attack_detected.

    Args:
        csv_path: Path to the Kaggle CSV file.
        binary: If True, use binary classification (normal vs attack).
                If False, infer attack types using heuristics.

    Returns:
        Tuple of (X, y, labels) where:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Encoded labels of shape (n_samples,)
        - labels: String labels list

    Raises:
        FileNotFoundError: If csv_path doesn't exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("loading_kaggle_csv", path=str(csv_path))

    df = pd.read_csv(csv_path)
    logger.info("csv_loaded", n_rows=len(df), columns=list(df.columns))

    # Encode categorical columns
    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
    encryption_map = {"None": 0, "DES": 1, "AES": 2}
    browser_map = {"Unknown": 0, "Chrome": 1, "Firefox": 2, "Edge": 3, "Safari": 4}

    df["protocol_encoded"] = df["protocol_type"].map(protocol_map).fillna(0).astype(int)
    df["encryption_encoded"] = df["encryption_used"].map(encryption_map).fillna(0).astype(int)
    df["browser_encoded"] = df["browser_type"].map(browser_map).fillna(0).astype(int)

    # Select feature columns (numeric + encoded categoricals)
    feature_cols = [
        "network_packet_size",
        "login_attempts",
        "session_duration",
        "ip_reputation_score",
        "failed_logins",
        "unusual_time_access",
        "protocol_encoded",
        "encryption_encoded",
        "browser_encoded",
    ]

    X = df[feature_cols].values.astype(np.float32)

    # Assign labels based on attack_detected column
    if binary:
        # Binary classification: normal (0) vs brute_force (1) as generic "attack"
        labels = [
            "normal" if row["attack_detected"] == 0 else "brute_force" for _, row in df.iterrows()
        ]
    else:
        # Multi-class: infer attack types from features (heuristics)
        labels = [_infer_attack_type(row) for _, row in df.iterrows()]

    # Encode labels to integers
    y = np.array([LABEL_TO_IDX.get(label, 0) for label in labels], dtype=np.int32)

    logger.info(
        "kaggle_data_prepared",
        X_shape=X.shape,
        y_shape=y.shape,
        label_distribution={k: labels.count(k) for k in set(labels)},
    )

    return X, y, labels


def load_or_generate_data_with_campaigns(
    n_samples: int = 10000,
    attack_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic data with campaigns and extract combined features.

    Uses CombinedFeatureExtractor (17 event + 8 window = 25 features) to
    properly capture attack patterns that differ by user cardinality:
    - brute_force: same user, same IP
    - credential_stuffing: multiple users, same IP

    Args:
        n_samples: Approximate number of events to generate.
        attack_ratio: Ratio of attack campaigns.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y, labels) where:
        - X: Feature matrix of shape (n_campaigns, 25)
        - y: Encoded labels of shape (n_campaigns,)
        - labels: String labels list (one per campaign)
    """
    from data.generators.synthetic import SyntheticEventGenerator

    logger.info(
        "generating_campaign_data",
        n_samples=n_samples,
        attack_ratio=attack_ratio,
        seed=seed,
    )

    generator = SyntheticEventGenerator(seed=seed)
    event_groups, labels, campaign_ids = generator.generate_dataset_with_campaigns(
        n_samples=n_samples,
        attack_ratio=attack_ratio,
    )

    logger.info(
        "campaigns_generated",
        n_campaigns=len(event_groups),
        total_events=sum(len(g) for g in event_groups),
    )

    # Extract combined features (17 event + 8 window = 25 total)
    extractor = CombinedFeatureExtractor()
    X = extractor.transform_batch_groups(event_groups)

    # Encode labels
    y = np.array([LABEL_TO_IDX.get(label, 0) for label in labels], dtype=np.int32)

    logger.info(
        "campaign_features_extracted",
        X_shape=X.shape,
        y_shape=y.shape,
        n_features=X.shape[1],
    )

    return X, y, labels


# ============================================================================
# Splitting
# ============================================================================


def to_anomaly_labels(labels: Sequence[str]) -> np.ndarray:
    """
    Turn string class labels into the binary indicator the detector expects.

    Args:
        labels: Canonical class names, e.g. ``["normal", "brute_force", ...]``.

    Returns:
        Integer array of shape ``(len(labels),)``: 0 = normal, 1 = attack.
    """
    return np.array([0 if label == NORMAL_LABEL else 1 for label in labels], dtype=int)


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
]:
    """
    Split into train/val/test carrying the string labels through **both** cuts.

    ``labels`` travelling alongside ``X`` is the whole point of this function.
    Both cuts shuffle (sklearn's default), so a label array that is not passed
    into ``train_test_split`` cannot be re-aligned afterwards by slicing — the
    permutation is not recoverable. Threshold optimisation reads its ground
    truth from ``labels_val``, so a misalignment there tunes the operating point
    against noise and reports a recall that means nothing.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Encoded labels of shape ``(n_samples,)``, used for stratification.
        labels: String class names aligned with ``X``.
        test_size: Fraction of the whole set held out for test.
        val_size: Fraction of the *remaining* training set held out for
            validation.
        seed: Random seed for both cuts.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test,
        labels_train, labels_val, labels_test)``.
    """
    labels_array = np.asarray(labels, dtype=object)

    X_train_full, X_test, y_train_full, y_test, labels_train_full, labels_test = train_test_split(
        X,
        y,
        labels_array,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X_train_full,
        y_train_full,
        labels_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        list(labels_train),
        list(labels_val),
        list(labels_test),
    )


# ============================================================================
# Model Training
# ============================================================================


def train_anomaly_detector(
    X_train: np.ndarray,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
) -> AnomalyDetector:
    """
    Train the anomaly detection model.

    Args:
        X_train: Training features of shape (n_samples, n_features).
        contamination: Expected proportion of outliers.
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.

    Returns:
        Trained AnomalyDetector instance.
    """
    logger.info(
        "training_anomaly_detector",
        n_samples=X_train.shape[0],
        contamination=contamination,
        n_estimators=n_estimators,
    )

    detector = AnomalyDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    detector.fit(X_train)

    logger.info("anomaly_detector_trained")

    return detector


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    class_weight: str | None = "balanced",
) -> AttackClassifier:
    """
    Train the attack classification model.

    Args:
        X_train: Training features of shape (n_samples, n_features).
        y_train: Training labels of shape (n_samples,).
        X_val: Optional validation features.
        y_val: Optional validation labels.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Random seed for reproducibility.
        class_weight: Class weighting strategy ('balanced' or None).

    Returns:
        Trained AttackClassifier instance.
    """
    logger.info(
        "training_attack_classifier",
        n_samples=X_train.shape[0],
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
    )

    classifier = AttackClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        class_weight=class_weight,
    )

    # Set up validation set if provided
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    classifier.fit(X_train, y_train, eval_set=eval_set)

    logger.info("attack_classifier_trained")

    return classifier


# ============================================================================
# Model Evaluation
# ============================================================================


def evaluate_models(
    detector: AnomalyDetector,
    classifier: AttackClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: list[str],
    binary: bool = False,
) -> dict[str, Any]:
    """
    Evaluate trained models on test data.

    Args:
        detector: Trained anomaly detector.
        classifier: Trained attack classifier.
        X_test: Test features.
        y_test: Test labels (encoded).
        labels_test: Test labels (strings).
        binary: If True, use binary classification labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info("evaluating_models", n_test_samples=X_test.shape[0])

    metrics: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Anomaly Detector Evaluation
    # -------------------------------------------------------------------------

    # For anomaly detection, we consider non-normal as anomaly
    y_anomaly_true = np.array([1 if label != "normal" else 0 for label in labels_test])

    anomaly_results = []
    for i in range(len(X_test)):
        result = detector.predict(X_test[i])
        anomaly_results.append(1 if result["is_anomaly"] else 0)

    y_anomaly_pred = np.array(anomaly_results)

    # Calculate anomaly detection metrics
    anomaly_accuracy = accuracy_score(y_anomaly_true, y_anomaly_pred)
    anomaly_precision = precision_score(y_anomaly_true, y_anomaly_pred, zero_division=0)
    anomaly_recall = recall_score(y_anomaly_true, y_anomaly_pred, zero_division=0)
    anomaly_f1 = f1_score(y_anomaly_true, y_anomaly_pred, zero_division=0)

    metrics["anomaly"] = {
        "accuracy": float(anomaly_accuracy),
        "precision": float(anomaly_precision),
        "recall": float(anomaly_recall),
        "f1": float(anomaly_f1),
    }

    logger.info(
        "anomaly_detector_evaluation",
        accuracy=anomaly_accuracy,
        precision=anomaly_precision,
        recall=anomaly_recall,
        f1=anomaly_f1,
    )

    # -------------------------------------------------------------------------
    # Classifier Evaluation
    # -------------------------------------------------------------------------

    # Get predictions for all test samples
    y_pred_list: list[int] = []
    for i in range(len(X_test)):
        result = classifier.predict(X_test[i])
        y_pred_list.append(LABEL_TO_IDX.get(result["attack_type"], 0))

    y_pred = np.array(y_pred_list)

    # Calculate classification metrics
    class_accuracy = accuracy_score(y_test, y_pred)
    class_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    class_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    class_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    target_names = BINARY_ATTACK_TYPES if binary else ATTACK_TYPES
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics["classifier"] = {
        "accuracy": float(class_accuracy),
        "precision": float(class_precision),
        "recall": float(class_recall),
        "f1": float(class_f1),
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
    }

    logger.info(
        "attack_classifier_evaluation",
        accuracy=class_accuracy,
        precision=class_precision,
        recall=class_recall,
        f1=class_f1,
    )

    # -------------------------------------------------------------------------
    # Per-Attack-Type Metrics
    # -------------------------------------------------------------------------

    per_attack_metrics = {}
    for attack_type in target_names:
        if attack_type in class_report:
            per_attack_metrics[attack_type] = {
                "precision": class_report[attack_type]["precision"],
                "recall": class_report[attack_type]["recall"],
                "f1": class_report[attack_type]["f1-score"],
                "support": class_report[attack_type]["support"],
            }

    metrics["per_attack_type"] = per_attack_metrics

    return metrics


# ============================================================================
# MLflow Integration
# ============================================================================


def log_to_mlflow(
    metrics: dict[str, Any],
    detector: AnomalyDetector,
    classifier: AttackClassifier,
    params: dict[str, Any],
    output_dir: Path,
    binary: bool = False,
) -> str:
    """
    Log training results to MLflow.

    Args:
        metrics: Evaluation metrics dictionary.
        detector: Trained anomaly detector.
        classifier: Trained attack classifier.
        params: Training parameters.
        output_dir: Directory where models are saved.
        binary: If True, use binary classification labels.

    Returns:
        MLflow run ID.
    """
    # Set up MLflow
    mlflow.set_experiment("sentinel-intrusion-detection")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("mlflow_run_started", run_id=run_id)

        # Log parameters
        mlflow.log_params(params)

        # Log anomaly detector metrics
        mlflow.log_metric("anomaly_accuracy", metrics["anomaly"]["accuracy"])
        mlflow.log_metric("anomaly_precision", metrics["anomaly"]["precision"])
        mlflow.log_metric("anomaly_recall", metrics["anomaly"]["recall"])
        mlflow.log_metric("anomaly_f1", metrics["anomaly"]["f1"])

        # Log classifier metrics
        mlflow.log_metric("classifier_accuracy", metrics["classifier"]["accuracy"])
        mlflow.log_metric("classifier_precision", metrics["classifier"]["precision"])
        mlflow.log_metric("classifier_recall", metrics["classifier"]["recall"])
        mlflow.log_metric("classifier_f1", metrics["classifier"]["f1"])

        # Log per-attack-type metrics
        for attack_type, attack_metrics in metrics["per_attack_type"].items():
            mlflow.log_metric(f"{attack_type}_precision", attack_metrics["precision"])
            mlflow.log_metric(f"{attack_type}_recall", attack_metrics["recall"])
            mlflow.log_metric(f"{attack_type}_f1", attack_metrics["f1"])

        # Log model artifacts
        anomaly_model_path = output_dir / "anomaly_detector.joblib"
        classifier_model_path = output_dir / "attack_classifier.joblib"

        if anomaly_model_path.exists():
            mlflow.log_artifact(str(anomaly_model_path), "models")

        if classifier_model_path.exists():
            mlflow.log_artifact(str(classifier_model_path), "models")

        # Log metrics as JSON artifact
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            # Convert numpy types for JSON serialization
            json.dump(metrics, f, indent=2, default=str)
        mlflow.log_artifact(str(metrics_path))

        # Log confusion matrix as artifact
        cm_path = output_dir / "confusion_matrix.json"
        cm_labels = BINARY_ATTACK_TYPES if binary else ATTACK_TYPES
        with cm_path.open("w") as f:
            json.dump(
                {
                    "matrix": metrics["classifier"]["confusion_matrix"],
                    "labels": cm_labels,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(cm_path))

        logger.info("mlflow_artifacts_logged", run_id=run_id)

    return str(run_id)


# ============================================================================
# Main Training Pipeline
# ============================================================================


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=False),
    default=None,
    help="Path to JSON file with training data. If not provided, generates synthetic data.",
)
@click.option(
    "--csv-path",
    type=click.Path(exists=False),
    default=None,
    help="Path to Kaggle CSV file with pre-computed features (bypasses FeatureExtractor).",
)
@click.option(
    "--binary",
    is_flag=True,
    default=False,
    help="Use binary classification (normal vs attack) instead of multi-class. Only with --csv-path.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="models",
    help="Directory to save trained models.",
)
@click.option(
    "--n-samples",
    type=int,
    default=10000,
    help="Number of synthetic samples to generate (if no data-path).",
)
@click.option(
    "--attack-ratio",
    type=float,
    default=0.3,
    help="Ratio of attack events in synthetic data.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing.",
)
@click.option(
    "--val-size",
    type=float,
    default=0.1,
    help="Proportion of training data to use for validation.",
)
@click.option(
    "--contamination",
    type=str,
    default="auto",
    help="Proportion of outliers. Use 'auto' to calculate from attack-ratio.",
)
@click.option(
    "--n-estimators",
    type=int,
    default=100,
    help="Number of trees/boosting rounds for models.",
)
@click.option(
    "--max-depth",
    type=int,
    default=6,
    help="Maximum tree depth for classifier.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.1,
    help="Learning rate for XGBoost classifier.",
)
@click.option(
    "--class-weight",
    type=click.Choice(["balanced", "none"]),
    default="balanced",
    help="Class weighting strategy: 'balanced' (recommended) or 'none'.",
)
@click.option(
    "--optimize-threshold/--no-optimize-threshold",
    default=True,
    help="Tune the anomaly threshold on the validation set. "
    "--no-optimize-threshold forces --threshold-strategy none.",
)
@click.option(
    "--threshold-strategy",
    type=click.Choice(THRESHOLD_STRATEGIES),
    default="recall",
    help="How to tune the anomaly threshold: 'recall' pins recall to "
    "--target-recall, 'fbeta' maximizes F-beta for --beta, 'none' keeps the "
    "contamination percentile from fit().",
)
@click.option(
    "--target-recall",
    type=float,
    default=0.8,
    help="Target recall for --threshold-strategy recall (0.8 = detect 80%% of attacks).",
)
@click.option(
    "--beta",
    type=float,
    default=2.0,
    help="Recall weight for --threshold-strategy fbeta. >1 favours recall, <1 favours precision.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--mlflow/--no-mlflow",
    default=True,
    help="Enable MLflow tracking.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "--use-window-features/--no-window-features",
    default=True,
    help="Use window features (25 total) for better brute_force vs credential_stuffing distinction.",
)
def main(  # noqa: PLR0912, PLR0915
    data_path: str | None,
    csv_path: str | None,
    binary: bool,
    output_dir: str,
    n_samples: int,
    attack_ratio: float,
    test_size: float,
    val_size: float,
    contamination: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    class_weight: str,
    optimize_threshold: bool,
    threshold_strategy: str,
    target_recall: float,
    beta: float,
    seed: int,
    mlflow: bool,
    verbose: bool,
    use_window_features: bool,
) -> None:
    """
    Train Sentinel ML models for intrusion detection.

    This script trains both the anomaly detector (Isolation Forest) and
    attack classifier (XGBoost) models, evaluates their performance,
    and optionally logs results to MLflow.

    Examples:

        # Train with synthetic data
        python -m ml.training.train --n-samples 10000 --output-dir models/

        # Train with real data (JSON)
        python -m ml.training.train --data-path data/events.json

        # Train with Kaggle CSV (pre-computed features)
        python -m ml.training.train --csv-path data/samples/cybersecurity_intrusion_data.csv

        # Train with custom parameters
        python -m ml.training.train --n-estimators 200 --max-depth 8

        # Tune the anomaly threshold for F-beta instead of a fixed recall target
        python -m ml.training.train --threshold-strategy fbeta --beta 2.0
    """
    # Setup logging
    log_level_name = "DEBUG" if verbose else "INFO"
    log_level_int: int = getattr(logging, log_level_name, logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
    )

    logger.info(
        "training_started",
        data_path=data_path,
        csv_path=csv_path,
        output_dir=output_dir,
        n_samples=n_samples,
        attack_ratio=attack_ratio,
        use_window_features=use_window_features,
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load/Generate Data
    # -------------------------------------------------------------------------

    if csv_path is not None:
        # Load directly from Kaggle CSV (bypasses FeatureExtractor)
        X, y, labels = load_kaggle_csv(csv_path, binary=binary)
    elif use_window_features and data_path is None:
        # Generate synthetic data with campaigns for window features (25 features)
        print("\n" + "=" * 60)
        print("Using Window Features (25 total: 17 event + 8 window)")
        print("=" * 60)
        X, y, labels = load_or_generate_data_with_campaigns(
            n_samples=n_samples,
            attack_ratio=attack_ratio,
            seed=seed,
        )
    else:
        # Load from JSON or generate synthetic data (17 features)
        X, y, labels = load_or_generate_data(
            data_path=data_path,
            n_samples=n_samples,
            attack_ratio=attack_ratio,
            seed=seed,
        )

    # Split data. The string labels are carried through *both* cuts: the
    # threshold optimisation below reads its ground truth from labels_val, and
    # both cuts shuffle, so a label array left behind cannot be realigned.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        labels_train,
        labels_val,
        labels_test,
    ) = split_train_val_test(
        X,
        y,
        labels,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    logger.info(
        "data_split",
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
    )

    # Print class distribution
    train_dist = Counter(labels_train)
    logger.info("class_distribution", distribution=dict(train_dist))

    # Calculate actual contamination (proportion of attacks in data)
    if contamination == "auto":
        actual_contamination = attack_ratio
        logger.info(
            "auto_contamination_calculated",
            attack_ratio=attack_ratio,
            actual_contamination=actual_contamination,
        )
    else:
        actual_contamination = float(contamination)

    # -------------------------------------------------------------------------
    # Train Models
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Anomaly Detector (Isolation Forest)")
    print("=" * 60)

    detector = train_anomaly_detector(
        X_train=X_train,
        contamination=actual_contamination,
        n_estimators=n_estimators,
        random_state=seed,
    )

    # Tune the decision threshold. --no-optimize-threshold still wins over
    # --threshold-strategy, so the older invocation keeps its meaning.
    strategy = threshold_strategy if optimize_threshold else "none"

    # Binary ground truth for the validation rows, taken from the labels the
    # split carried along with X_val — never re-derived by slicing.
    y_val_anomaly = to_anomaly_labels(labels_val)
    operating_point: dict[str, Any] | None = None

    if strategy == "recall":
        print(f"Optimizing threshold for {target_recall:.0%} recall...")
        detector.optimize_threshold_for_recall(
            X_val=X_val,
            y_val=y_val_anomaly,
            target_recall=target_recall,
        )
        print(f"Threshold optimized: {detector._threshold:.4f}")
    elif strategy == "fbeta":
        print(f"Optimizing threshold for F-{beta:g}...")
        detector.optimize_threshold_fbeta(
            X_val=X_val,
            y_val=y_val_anomaly,
            beta=beta,
        )
        operating_point = detector.get_operating_point_info()
        print(
            f"Threshold optimized: {detector._threshold:.4f} "
            f"(precision {operating_point['precision']:.4f}, "
            f"recall {operating_point['recall']:.4f})"
        )
    else:
        print("Threshold optimization disabled; keeping the contamination percentile.")

    # Save anomaly detector
    detector_path = output_path / "anomaly_detector.joblib"
    detector.save(detector_path)
    print(f"Anomaly detector saved to: {detector_path}")

    print("\n" + "=" * 60)
    print("Training Attack Classifier (XGBoost)")
    print("=" * 60)

    classifier = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        class_weight=class_weight if class_weight != "none" else None,
    )

    # Save classifier
    classifier_path = output_path / "attack_classifier.joblib"
    classifier.save(classifier_path)
    print(f"Attack classifier saved to: {classifier_path}")

    # -------------------------------------------------------------------------
    # Evaluate Models
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Evaluating Models")
    print("=" * 60)

    metrics = evaluate_models(
        detector=detector,
        classifier=classifier,
        X_test=X_test,
        y_test=y_test,
        labels_test=labels_test,
        binary=binary,
    )

    # Print summary
    print("\n--- Anomaly Detection Metrics ---")
    print(f"  Accuracy:  {metrics['anomaly']['accuracy']:.4f}")
    print(f"  Precision: {metrics['anomaly']['precision']:.4f}")
    print(f"  Recall:    {metrics['anomaly']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['anomaly']['f1']:.4f}")

    print("\n--- Attack Classification Metrics ---")
    print(f"  Accuracy:  {metrics['classifier']['accuracy']:.4f}")
    print(f"  Precision: {metrics['classifier']['precision']:.4f}")
    print(f"  Recall:    {metrics['classifier']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['classifier']['f1']:.4f}")

    print("\n--- Per-Attack-Type F1 Scores ---")
    for attack_type, attack_metrics in metrics["per_attack_type"].items():
        support = attack_metrics.get("support", 0)
        f1 = attack_metrics.get("f1", 0)
        print(f"  {attack_type:25s}: {f1:.4f} (n={support})")

    # -------------------------------------------------------------------------
    # MLflow Logging
    # -------------------------------------------------------------------------

    # Determine data source string
    if csv_path:
        data_source = f"csv:{csv_path}"
    elif data_path:
        data_source = f"json:{data_path}"
    else:
        data_source = "synthetic"

    if mlflow:
        print("\n" + "=" * 60)
        print("Logging to MLflow")
        print("=" * 60)

        params = {
            "n_samples": n_samples,
            "attack_ratio": attack_ratio,
            "test_size": test_size,
            "val_size": val_size,
            "contamination": actual_contamination,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "seed": seed,
            "data_source": data_source,
            "threshold_strategy": strategy,
        }

        run_id = log_to_mlflow(
            metrics=metrics,
            detector=detector,
            classifier=classifier,
            params=params,
            output_dir=output_path,
            binary=binary,
        )

        print(f"MLflow run ID: {run_id}")
        print("View at: mlflow ui")

    # -------------------------------------------------------------------------
    # Save Metrics Summary
    # -------------------------------------------------------------------------

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "source": data_source,
            "n_samples": len(X),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        },
        "parameters": {
            "contamination": actual_contamination,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "threshold": {
            "strategy": strategy,
            "value": float(detector._threshold),
            "target_recall": target_recall if strategy == "recall" else None,
            "beta": beta if strategy == "fbeta" else None,
            # Only the F-beta search measures an operating point. The other two
            # strategies never computed one, and inventing it here would be a
            # fabricated number in the training record.
            "operating_point": operating_point,
        },
        "metrics": metrics,
        "model_paths": {
            "anomaly_detector": str(detector_path),
            "attack_classifier": str(classifier_path),
        },
    }

    summary_path = output_path / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTraining summary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
