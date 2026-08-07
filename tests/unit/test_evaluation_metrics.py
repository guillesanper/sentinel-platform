"""
Unit tests for ml.evaluation.metrics.

Shared by both tracks, so the tests use plain string labels and a stub model
rather than anything track-specific.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation import compare_models, compute_metrics
from ml.evaluation.metrics import SCALAR_METRICS

# ============================================================================
# Fixtures
# ============================================================================


class StubModel:
    """Minimal `predict(X)`-compatible stand-in for a trained model."""

    def __init__(self, predictions: np.ndarray):
        self._predictions = np.asarray(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X carries the row indices into the canned prediction array, which is
        # what lets the same stub answer correctly for cross-validation folds.
        return self._predictions[X[:, 0].astype(int)]


@pytest.fixture
def three_class_labels() -> list[str]:
    return ["BENIGN", "DDoS", "DoS"]


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array(["BENIGN"] * 30 + ["DDoS"] * 30 + ["DoS"] * 30)


@pytest.fixture
def index_matrix(y_true: np.ndarray) -> np.ndarray:
    """Feature matrix whose first column is the row index (see StubModel)."""
    return np.arange(len(y_true), dtype=float).reshape(-1, 1)


# ============================================================================
# compute_metrics
# ============================================================================


class TestComputeMetrics:
    """Scalar and per-class metric computation."""

    def test_perfect_predictions(self, y_true):
        metrics = compute_metrics(y_true, y_true)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)

    def test_returns_all_expected_keys(self, y_true):
        metrics = compute_metrics(y_true, y_true)
        assert set(metrics) == {*SCALAR_METRICS, "f1_per_class"}

    def test_per_class_f1_covers_every_label(self, y_true, three_class_labels):
        metrics = compute_metrics(y_true, y_true, labels=three_class_labels)
        assert set(metrics["f1_per_class"]) == set(three_class_labels)

    def test_labels_default_to_the_union_of_true_and_pred(self):
        y_t = np.array(["BENIGN", "BENIGN"])
        y_p = np.array(["BENIGN", "DoS"])
        metrics = compute_metrics(y_t, y_p)
        assert set(metrics["f1_per_class"]) == {"BENIGN", "DoS"}

    def test_explicit_labels_restrict_the_report(self, y_true):
        metrics = compute_metrics(y_true, y_true, labels=["BENIGN"])
        assert set(metrics["f1_per_class"]) == {"BENIGN"}

    def test_wrong_predictions_score_zero(self):
        y_t = np.array(["BENIGN"] * 10)
        y_p = np.array(["DoS"] * 10)
        metrics = compute_metrics(y_t, y_p)
        assert metrics["accuracy"] == pytest.approx(0.0)
        assert metrics["f1_macro"] == pytest.approx(0.0)

    def test_absent_class_does_not_raise(self, y_true, three_class_labels):
        """zero_division=0 must absorb classes with no support."""
        y_p = np.array(["BENIGN"] * len(y_true))
        metrics = compute_metrics(y_true, y_p, labels=three_class_labels)
        assert metrics["f1_per_class"]["DDoS"] == pytest.approx(0.0)

    def test_metric_values_are_plain_floats(self, y_true):
        metrics = compute_metrics(y_true, y_true)
        for key in SCALAR_METRICS:
            assert isinstance(metrics[key], float)
            assert not isinstance(metrics[key], np.floating)

    def test_no_auc_without_scores(self, y_true):
        assert "auc_roc_macro" not in compute_metrics(y_true, y_true)


class TestComputeMetricsAucRoc:
    """One-vs-Rest AUC-ROC."""

    def test_auc_keys_present_when_scores_given(self, y_true, three_class_labels):
        y_score = _one_hot_scores(y_true, three_class_labels)
        metrics = compute_metrics(y_true, y_true, y_score=y_score)
        assert "auc_roc_macro" in metrics
        assert set(metrics["auc_roc_per_class"]) == set(three_class_labels)

    def test_perfect_scores_give_auc_one(self, y_true, three_class_labels):
        y_score = _one_hot_scores(y_true, three_class_labels)
        metrics = compute_metrics(y_true, y_true, y_score=y_score)
        assert metrics["auc_roc_macro"] == pytest.approx(1.0)

    def test_binary_problem_is_expanded_to_two_columns(self):
        y_t = np.array(["BENIGN"] * 10 + ["DoS"] * 10)
        y_score = np.column_stack(
            [np.r_[np.ones(10), np.zeros(10)], np.r_[np.zeros(10), np.ones(10)]]
        )
        metrics = compute_metrics(y_t, y_t, y_score=y_score)
        assert metrics["auc_roc_macro"] == pytest.approx(1.0)

    def test_single_class_yields_nan_instead_of_raising(self):
        """roc_auc_score raises on a single-class target; we degrade to NaN."""
        y_t = np.array(["BENIGN"] * 10)
        y_score = np.ones((10, 1))
        metrics = compute_metrics(y_t, y_t, y_score=y_score)
        assert np.isnan(metrics["auc_roc_macro"])

    def test_fewer_score_columns_than_classes_is_tolerated(self, y_true, three_class_labels):
        y_score = _one_hot_scores(y_true, three_class_labels)[:, :2]
        metrics = compute_metrics(y_true, y_true, y_score=y_score)
        assert len(metrics["auc_roc_per_class"]) == 3


# ============================================================================
# compare_models
# ============================================================================


class TestCompareModels:
    """Head-to-head comparison of two trained models."""

    def test_returns_the_documented_keys(self, y_true, index_matrix):
        result = compare_models(StubModel(y_true), StubModel(y_true), index_matrix, y_true)
        assert set(result) == {"model_a", "model_b", "comparison"}

    def test_better_model_b_wins(self, y_true, index_matrix, three_class_labels):
        degraded = y_true.copy()
        degraded[:20] = "DoS"
        result = compare_models(
            StubModel(degraded),
            StubModel(y_true),
            index_matrix,
            y_true,
            labels=three_class_labels,
        )
        assert result["comparison"]["accuracy"]["winner"] == "model_b"
        assert result["comparison"]["accuracy"]["delta"] > 0

    def test_better_model_a_wins(self, y_true, index_matrix, three_class_labels):
        degraded = y_true.copy()
        degraded[:20] = "DoS"
        result = compare_models(
            StubModel(y_true),
            StubModel(degraded),
            index_matrix,
            y_true,
            labels=three_class_labels,
        )
        assert result["comparison"]["accuracy"]["winner"] == "model_a"
        assert result["comparison"]["accuracy"]["delta"] < 0

    def test_identical_models_tie(self, y_true, index_matrix):
        result = compare_models(StubModel(y_true), StubModel(y_true), index_matrix, y_true)
        for metric in SCALAR_METRICS:
            assert result["comparison"][metric]["winner"] == "tie"
            assert result["comparison"][metric]["delta"] == pytest.approx(0.0)

    def test_comparison_uses_model_a_b_naming(self, y_true, index_matrix):
        """compare_pipelines/pipeline_a/pipeline_b were renamed when ported."""
        result = compare_models(StubModel(y_true), StubModel(y_true), index_matrix, y_true)
        assert "pipeline_a" not in result
        assert "pipeline_b" not in result
        row = result["comparison"]["accuracy"]
        assert set(row) == {"model_a", "model_b", "delta", "winner"}

    def test_old_compare_pipelines_name_is_gone(self):
        import ml.evaluation.metrics as metrics_module

        assert not hasattr(metrics_module, "compare_pipelines")

    def test_per_class_comparison_covers_the_union_of_classes(
        self, y_true, index_matrix, three_class_labels
    ):
        result = compare_models(
            StubModel(y_true),
            StubModel(y_true),
            index_matrix,
            y_true,
            labels=three_class_labels,
        )
        assert set(result["comparison"]["f1_per_class"]) == set(three_class_labels)

    def test_scores_feed_auc_into_each_model_report(self, y_true, index_matrix, three_class_labels):
        y_score = _one_hot_scores(y_true, three_class_labels)
        result = compare_models(
            StubModel(y_true),
            StubModel(y_true),
            index_matrix,
            y_true,
            labels=three_class_labels,
            y_score_a=y_score,
            y_score_b=y_score,
        )
        assert result["model_a"]["auc_roc_macro"] == pytest.approx(1.0)
        assert result["model_b"]["auc_roc_macro"] == pytest.approx(1.0)

    def test_missing_scalar_metric_is_skipped(self):
        from ml.evaluation.metrics import _build_comparison

        comparison = _build_comparison({"accuracy": 0.9}, {"f1_macro": 0.8})
        assert set(comparison) == {"f1_per_class"}


class TestCompareModelsCrossValidation:
    """StratifiedKFold variance estimation over the test set."""

    def test_cv_disabled_by_default(self, y_true, index_matrix):
        result = compare_models(StubModel(y_true), StubModel(y_true), index_matrix, y_true)
        assert "cv_model_a" not in result

    @pytest.mark.parametrize("n_cv_splits", [0, 1])
    def test_cv_skipped_for_zero_or_one_split(self, y_true, index_matrix, n_cv_splits):
        result = compare_models(
            StubModel(y_true),
            StubModel(y_true),
            index_matrix,
            y_true,
            n_cv_splits=n_cv_splits,
        )
        assert "cv_model_a" not in result

    def test_cv_reports_mean_and_std_per_metric(self, y_true, index_matrix):
        result = compare_models(
            StubModel(y_true),
            StubModel(y_true),
            index_matrix,
            y_true,
            n_cv_splits=3,
        )
        for key in ("cv_model_a", "cv_model_b"):
            assert set(result[key]) == set(SCALAR_METRICS)
            for stats in result[key].values():
                assert set(stats) == {"mean", "std"}

    def test_cv_of_a_perfect_model_has_zero_variance(self, y_true, index_matrix):
        result = compare_models(
            StubModel(y_true),
            StubModel(y_true),
            index_matrix,
            y_true,
            n_cv_splits=3,
        )
        assert result["cv_model_a"]["accuracy"]["mean"] == pytest.approx(1.0)
        assert result["cv_model_a"]["accuracy"]["std"] == pytest.approx(0.0)


# ============================================================================
# Dependency guard
# ============================================================================


def test_importing_evaluation_does_not_pull_matplotlib():
    """
    api/Dockerfile copies ml/ wholesale but installs only requirements.txt.
    If ml.evaluation ever imports plots at module level, the production image
    breaks (or grows ~120 MB of plotting stack).

    Runs in a subprocess so the result does not depend on what the rest of the
    test session happened to import first.
    """
    probe = (
        "import ml.evaluation, sys; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib'; "
        "assert 'seaborn' not in sys.modules, 'seaborn'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_evaluation_package_does_not_export_plots():
    import ml.evaluation

    assert not hasattr(ml.evaluation, "plots")


# ============================================================================
# Helpers
# ============================================================================


def _one_hot_scores(y_true: np.ndarray, labels: list[str]) -> np.ndarray:
    """Build a perfect-confidence probability matrix aligned with `labels`."""
    index = {label: i for i, label in enumerate(sorted(labels))}
    scores = np.zeros((len(y_true), len(labels)))
    for row, label in enumerate(y_true):
        scores[row, index[label]] = 1.0
    return scores
