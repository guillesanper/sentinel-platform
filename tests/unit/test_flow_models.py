"""
Unit tests for the Stage 1 and Stage 2 models of the flow-based track.

Two properties here are worth more than the rest put together, because both fail
without raising anything:

* **F-beta actually steers the trade-off.** If ``optimize_threshold`` silently
  ignored ``beta``, every run would still produce a perfectly plausible
  threshold — just not the one the operator asked for. The test pins the
  monotonicity that gives ``beta`` its meaning.
* **A save/load roundtrip preserves predictions.** A cascade that reloads into a
  slightly different state predicts confidently and wrongly, forever.
"""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from ml.flow.config import IsolationForestConfig, NeuralNetworkConfig, XGBoostConfig
from ml.flow.cv import TemporalIndex
from ml.flow.labels import BENIGN_LABEL
from ml.flow.models import FlowAnomalyDetector, FlowAttackClassifier
from ml.flow.models import xgboost_classifier as xgb_module
from ml.flow.models.isolation_forest import (
    MODEL_FILE,
    THRESHOLD_FILE,
    THRESHOLD_INFO_FILE,
)
from ml.flow.models.neural_network import NeuralNetwork

# ============================================================================
# Synthetic flows
# ============================================================================

_ATTACK_CLASSES: tuple[str, ...] = ("DoS", "Scan", "BruteForce")

# Class separation used by the two data regimes below. "Overlapping" is what
# makes the F-beta test non-vacuous: with well-separated classes every beta
# finds the same perfect threshold and the property holds trivially.
_OVERLAPPING = 1.4
_SEPARABLE = 9.0


def _synthetic_flows(
    *, n_benign: int, per_attack: int, separation: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a labelled feature matrix shaped like the flow track's input.

    Benign rows sit at the origin; each attack class is offset a little further
    out, so ``separation`` controls how much of the benign mass an anomaly
    detector has to cut through.
    """
    rng = np.random.default_rng(seed)

    blocks = [rng.normal(0.0, 1.0, size=(n_benign, 4))]
    labels = [np.array([BENIGN_LABEL] * n_benign, dtype=object)]

    for i, name in enumerate(_ATTACK_CLASSES):
        centre = separation * (1.0 + 0.4 * i)
        blocks.append(rng.normal(centre, 1.0, size=(per_attack, 4)))
        labels.append(np.array([name] * per_attack, dtype=object))

    X = np.vstack(blocks).astype(np.float32)
    y = np.concatenate(labels)

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _train_val(
    separation: float, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A train and a validation set drawn from the same distribution."""
    X_train, y_train = _synthetic_flows(
        n_benign=600, per_attack=100, separation=separation, seed=seed
    )
    X_val, y_val = _synthetic_flows(
        n_benign=300, per_attack=50, separation=separation, seed=seed + 1
    )
    return X_train, y_train, X_val, y_val


@pytest.fixture
def overlapping_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attacks that genuinely sit inside the benign distribution's tail."""
    return _train_val(_OVERLAPPING)


@pytest.fixture
def separable_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attacks far enough out that Stage 1 routes them reliably."""
    return _train_val(_SEPARABLE)


def _fitted_detector(
    X_train: np.ndarray, y_train: np.ndarray, *, n_estimators: int = 60
) -> FlowAnomalyDetector:
    """A detector fitted on the benign rows of ``X_train``, threshold untouched."""
    config = IsolationForestConfig(n_estimators=n_estimators, n_jobs=1)
    return FlowAnomalyDetector(config, random_seed=42).fit(X_train, y_train)


# ============================================================================
# FlowAnomalyDetector — fitting
# ============================================================================


class TestFlowAnomalyDetectorFit:
    """Stage 1 training and its preconditions."""

    def test_fit_returns_self_and_marks_fitted(self, separable_data):
        X_train, y_train, _, _ = separable_data
        detector = FlowAnomalyDetector(IsolationForestConfig(n_estimators=30, n_jobs=1))

        assert detector.is_fitted is False
        assert detector.fit(X_train, y_train) is detector
        assert detector.is_fitted is True

    def test_fit_uses_the_sklearn_default_threshold(self, separable_data):
        X_train, y_train, _, _ = separable_data
        assert _fitted_detector(X_train, y_train).threshold == 0.0

    def test_fit_without_benign_samples_raises(self):
        X, y = _synthetic_flows(n_benign=0, per_attack=50, separation=_SEPARABLE, seed=3)

        with pytest.raises(ValueError, match="No BENIGN samples"):
            FlowAnomalyDetector(IsolationForestConfig(n_estimators=10, n_jobs=1)).fit(X, y)

    def test_predict_binary_before_fit_raises(self, separable_data):
        X_train, _, _, _ = separable_data

        with pytest.raises(RuntimeError, match="must be fit"):
            FlowAnomalyDetector(IsolationForestConfig()).predict_binary(X_train)

    def test_optimize_threshold_before_fit_raises(self, separable_data):
        _, _, X_val, y_val = separable_data

        with pytest.raises(RuntimeError, match="must be fit"):
            FlowAnomalyDetector(IsolationForestConfig()).optimize_threshold(X_val, y_val)


# ============================================================================
# FlowAnomalyDetector — the F-beta trade-off
# ============================================================================


class TestFbetaThreshold:
    """The property that gives ``beta`` its meaning."""

    def test_recall_favouring_beta_never_loses_recall(self, overlapping_data):
        """
        beta=2.0 weighs recall four times as heavily as beta=0.5 does, so the
        threshold it selects can never be the stingier of the two. This is the
        fundamental guarantee of the F-beta family, and the only thing standing
        between an operator asking for high recall and silently not getting it.
        """
        X_train, y_train, X_val, y_val = overlapping_data

        recall_hungry = _fitted_detector(X_train, y_train)
        recall_hungry.optimize_threshold(X_val, y_val, beta=2.0)

        precision_hungry = _fitted_detector(X_train, y_train)
        precision_hungry.optimize_threshold(X_val, y_val, beta=0.5)

        op_recall = recall_hungry.get_operating_point_info()
        op_precision = precision_hungry.get_operating_point_info()

        assert op_recall["recall"] >= op_precision["recall"]
        # Without this the assertion above could pass on a degenerate dataset
        # where both betas reach recall 1.0 and nothing was actually compared.
        assert op_precision["recall"] < 1.0

    def test_precision_favouring_beta_never_loses_precision(self, overlapping_data):
        """The dual: beta<1 must not end up less precise than beta>1."""
        X_train, y_train, X_val, y_val = overlapping_data

        recall_hungry = _fitted_detector(X_train, y_train)
        recall_hungry.optimize_threshold(X_val, y_val, beta=2.0)

        precision_hungry = _fitted_detector(X_train, y_train)
        precision_hungry.optimize_threshold(X_val, y_val, beta=0.5)

        assert (
            precision_hungry.get_operating_point_info()["precision"]
            >= recall_hungry.get_operating_point_info()["precision"]
        )

    def test_higher_beta_flags_at_least_as_many_flows(self, overlapping_data):
        """A higher threshold flags more; the operating point should show it."""
        X_train, y_train, X_val, y_val = overlapping_data

        recall_hungry = _fitted_detector(X_train, y_train)
        recall_hungry.optimize_threshold(X_val, y_val, beta=2.0)

        precision_hungry = _fitted_detector(X_train, y_train)
        precision_hungry.optimize_threshold(X_val, y_val, beta=0.5)

        n_flagged_hi = int(recall_hungry.predict_binary(X_val).sum())
        n_flagged_lo = int(precision_hungry.predict_binary(X_val).sum())
        assert n_flagged_hi >= n_flagged_lo

    def test_optimize_threshold_moves_off_the_default(self, overlapping_data):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)

        threshold = detector.optimize_threshold(X_val, y_val, beta=2.0)

        assert threshold == detector.threshold
        assert threshold != 0.0

    def test_n_candidates_is_honoured(self, overlapping_data):
        """A coarser scan must still land on a valid threshold in range."""
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)

        threshold = detector.optimize_threshold(X_val, y_val, beta=1.0, n_candidates=5)

        assert isinstance(threshold, float)
        assert detector.get_operating_point_info()["threshold"] == threshold


# ============================================================================
# FlowAnomalyDetector — operating point
# ============================================================================


class TestOperatingPointInfo:
    """What the detector reports about the threshold it is using."""

    def test_before_optimizing_raises(self, separable_data):
        """
        Fitting alone is not enough: the default threshold has never been
        measured against anything, so there is no operating point to report and
        returning zeros would be a lie.
        """
        X_train, y_train, _, _ = separable_data
        detector = _fitted_detector(X_train, y_train)

        with pytest.raises(RuntimeError, match="optimize_threshold"):
            detector.get_operating_point_info()

    def test_reports_the_full_confusion_matrix(self, overlapping_data):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        info = detector.get_operating_point_info()

        assert set(info) >= {
            "threshold",
            "beta",
            "f2_score",
            "precision",
            "recall",
            "fpr",
            "fnr",
            "tp",
            "fp",
            "tn",
            "fn",
        }
        assert info["tp"] + info["fp"] + info["tn"] + info["fn"] == len(y_val)
        assert info["beta"] == 2.0

    def test_returned_dict_is_a_copy(self, overlapping_data):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=1.0)

        detector.get_operating_point_info()["recall"] = -1.0

        assert detector.get_operating_point_info()["recall"] != -1.0

    def test_beta_key_is_named_after_beta(self, overlapping_data):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=0.5)

        assert "f0.5_score" in detector.get_operating_point_info()


class TestMeasureOperatingPoint:
    """Measuring the frozen threshold on a set it was not chosen on."""

    def test_measuring_on_the_selection_set_reproduces_the_stored_point(self, overlapping_data):
        """
        The two paths compute the same quantity, so on the same data they must
        agree exactly — including the F-beta, which optimize_threshold takes
        from the PR curve and this one recomputes from the confusion matrix.
        """
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        assert detector.measure_operating_point(X_val, y_val) == (
            detector.get_operating_point_info()
        )

    def test_measures_against_the_set_it_is_given(self, overlapping_data):
        """The confusion matrix must count the rows handed in, not the ones the
        threshold was selected on — that conflation is the bug this exists for."""
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        half = len(y_val) // 2
        info = detector.measure_operating_point(X_val[:half], y_val[:half])

        assert info["tp"] + info["fp"] + info["tn"] + info["fn"] == half

    def test_keeps_the_threshold_frozen(self, overlapping_data):
        """Measuring is not selecting: nothing about the detector may move."""
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)
        before = detector.threshold

        detector.measure_operating_point(X_train, y_train)

        assert detector.threshold == before
        assert detector.get_operating_point_info()["threshold"] == before

    def test_defaults_to_the_beta_the_threshold_was_selected_with(self, overlapping_data):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        info = detector.measure_operating_point(X_train, y_train)

        assert info["beta"] == 2.0
        assert "f2_score" in info

    def test_requires_a_fitted_model(self, separable_data):
        _, _, X_val, y_val = separable_data

        with pytest.raises(RuntimeError, match="fit or loaded"):
            FlowAnomalyDetector(IsolationForestConfig()).measure_operating_point(X_val, y_val)


# ============================================================================
# FlowAnomalyDetector — persistence
# ============================================================================


class TestFlowAnomalyDetectorPersistence:
    """Roundtripping Stage 1 through disk."""

    def test_roundtrip_preserves_predictions(self, overlapping_data, tmp_path: Path):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)
        expected = detector.predict_binary(X_val)

        detector.save(tmp_path / "if")
        restored = FlowAnomalyDetector(IsolationForestConfig(n_jobs=1)).load(tmp_path / "if")

        np.testing.assert_array_equal(restored.predict_binary(X_val), expected)
        assert restored.threshold == detector.threshold

    def test_roundtrip_preserves_the_operating_point(self, overlapping_data, tmp_path: Path):
        """
        Serving has to explain its decisions without a validation set at hand,
        so the operating point travels with the model.
        """
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        detector.save(tmp_path / "if")
        restored = FlowAnomalyDetector(IsolationForestConfig(n_jobs=1)).load(tmp_path / "if")

        assert restored.get_operating_point_info() == detector.get_operating_point_info()

    def test_save_writes_the_expected_files(self, overlapping_data, tmp_path: Path):
        X_train, y_train, X_val, y_val = overlapping_data
        detector = _fitted_detector(X_train, y_train)
        detector.optimize_threshold(X_val, y_val, beta=2.0)

        detector.save(tmp_path / "if")

        assert (tmp_path / "if" / MODEL_FILE).exists()
        assert (tmp_path / "if" / THRESHOLD_FILE).exists()
        info = json.loads((tmp_path / "if" / THRESHOLD_INFO_FILE).read_text(encoding="utf-8"))
        assert info["beta"] == 2.0

    def test_save_without_optimizing_omits_the_operating_point(
        self, separable_data, tmp_path: Path
    ):
        """A detector that was never optimised has no operating point to claim."""
        X_train, y_train, _, _ = separable_data
        _fitted_detector(X_train, y_train).save(tmp_path / "if")

        assert not (tmp_path / "if" / THRESHOLD_INFO_FILE).exists()

        restored = FlowAnomalyDetector(IsolationForestConfig(n_jobs=1)).load(tmp_path / "if")
        assert restored.is_fitted is True
        with pytest.raises(RuntimeError):
            restored.get_operating_point_info()

    def test_save_creates_missing_directories(self, separable_data, tmp_path: Path):
        X_train, y_train, _, _ = separable_data
        target = tmp_path / "deeply" / "nested" / "if"

        _fitted_detector(X_train, y_train).save(target)

        assert (target / MODEL_FILE).exists()


# ============================================================================
# FlowAttackClassifier
# ============================================================================


def _small_xgb_config() -> XGBoostConfig:
    """A booster small enough for unit tests to stay honest about runtime."""
    return XGBoostConfig(n_estimators=20, max_depth=3, early_stopping_rounds=5, n_jobs=1)


def _attacks_only(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Stage 2 input: everything Stage 1 would have routed as anomalous."""
    mask = y != BENIGN_LABEL
    return X[mask], y[mask]


@pytest.fixture
def fitted_classifier(separable_data) -> FlowAttackClassifier:
    """A Stage 2 classifier trained on the attack rows alone."""
    X_train, y_train, X_val, y_val = separable_data
    X_a, y_a = _attacks_only(X_train, y_train)
    X_v, y_v = _attacks_only(X_val, y_val)
    return FlowAttackClassifier(_small_xgb_config(), random_seed=42).fit(X_a, y_a, X_v, y_v)


class TestFlowAttackClassifier:
    """Stage 2 fitting, prediction and introspection."""

    def test_fit_returns_self_and_learns_the_classes(self, fitted_classifier):
        assert fitted_classifier.is_fitted is True
        assert sorted(fitted_classifier.classes_) == sorted(_ATTACK_CLASSES)

    def test_predict_returns_canonical_string_labels(self, fitted_classifier, separable_data):
        _, _, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_val, y_val)

        preds = fitted_classifier.predict(X_a)

        assert preds.shape == y_a.shape
        assert set(preds) <= set(_ATTACK_CLASSES)

    def test_predict_is_accurate_on_separable_classes(self, fitted_classifier, separable_data):
        """A sanity floor: well-separated classes must be learnable."""
        _, _, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_val, y_val)

        accuracy = float((fitted_classifier.predict(X_a) == y_a).mean())

        assert accuracy > 0.85

    def test_predict_proba_rows_sum_to_one(self, fitted_classifier, separable_data):
        _, _, X_val, y_val = separable_data
        X_a, _ = _attacks_only(X_val, y_val)

        probs = fitted_classifier.predict_proba(X_a)

        assert probs.shape == (len(X_a), len(_ATTACK_CLASSES))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_before_fit_raises(self, separable_data):
        X_train, _, _, _ = separable_data

        with pytest.raises(RuntimeError, match="must be fit"):
            FlowAttackClassifier(_small_xgb_config()).predict(X_train)

    def test_classes_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = FlowAttackClassifier(_small_xgb_config()).classes_

    def test_validation_classes_unseen_in_training_are_dropped(self, separable_data):
        """
        A class that never reached Stage 2 during training cannot be encoded.
        Keeping such rows in the eval set would abort the fit outright, so they
        are excluded — loudly, but without failing the run.
        """
        X_train, y_train, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_train, y_train)
        keep = y_a != "BruteForce"

        X_v, y_v = _attacks_only(X_val, y_val)
        classifier = FlowAttackClassifier(_small_xgb_config(), random_seed=42)
        classifier.fit(X_a[keep], y_a[keep], X_v, y_v)

        assert "BruteForce" not in set(classifier.classes_)
        assert set(classifier.predict(X_v)) <= {"DoS", "Scan"}

    def test_feature_importance_uses_the_given_names(self, fitted_classifier):
        names = ["duration", "fwd_packets", "bytes_per_s", "flags"]

        importance = fitted_classifier.get_feature_importance(names)

        assert set(importance) == set(names)
        assert list(importance.values()) == sorted(importance.values(), reverse=True)

    def test_feature_importance_falls_back_to_positional_names(self, fitted_classifier):
        assert set(fitted_classifier.get_feature_importance()) == {"f0", "f1", "f2", "f3"}

    def test_feature_importance_rejects_a_mismatched_name_list(self, fitted_classifier):
        """
        A short or long list would relabel every importance without a word.
        """
        with pytest.raises(ValueError, match="feature_names has 2 entries"):
            fitted_classifier.get_feature_importance(["duration", "flags"])

    def test_roundtrip_preserves_predictions(
        self, fitted_classifier, separable_data, tmp_path: Path
    ):
        _, _, X_val, y_val = separable_data
        X_a, _ = _attacks_only(X_val, y_val)
        expected = fitted_classifier.predict(X_a)

        fitted_classifier.save(tmp_path / "xgb")
        restored = FlowAttackClassifier(_small_xgb_config()).load(tmp_path / "xgb")

        np.testing.assert_array_equal(restored.predict(X_a), expected)
        np.testing.assert_array_equal(restored.classes_, fitted_classifier.classes_)

    def test_load_restores_the_class_count(self, fitted_classifier, tmp_path: Path):
        fitted_classifier.save(tmp_path / "xgb")

        restored = FlowAttackClassifier(_small_xgb_config()).load(tmp_path / "xgb")

        assert restored.is_fitted is True
        assert len(restored.classes_) == len(_ATTACK_CLASSES)

    @pytest.mark.slow
    def test_tune_hyperparameters_finds_params_from_the_search_space(self, separable_data):
        """
        Slow by nature: a randomised search fits ``n_iter * cv_folds`` boosters.
        Trimmed to the smallest shape that still exercises the real code path.
        """
        X_train, y_train, _, _ = separable_data
        X_a, y_a = _attacks_only(X_train, y_train)

        config = XGBoostConfig(
            n_estimators=20,
            max_depth=3,
            n_iter_search=2,
            cv_folds=2,
            n_jobs=1,
            search_space={"max_depth": [2, 3], "learning_rate": [0.1, 0.3]},
        )
        classifier = FlowAttackClassifier(config, random_seed=42)

        best = classifier.tune_hyperparameters(X_a, y_a)

        assert set(best) <= {"max_depth", "learning_rate"}
        assert best["max_depth"] in (2, 3)
        # Tuned values must survive into the final fit.
        assert classifier._merged_params()["max_depth"] == best["max_depth"]


# ============================================================================
# Stage 2 tuning — the cross-validation folds
# ============================================================================


class _SearchSpy:
    """
    Stands in for ``RandomizedSearchCV`` and records what it was handed.

    Every test in :class:`TestTuneHyperparametersFolds` is about *which folds*
    and *which scorer* the search receives, not about what the search finds, so
    fitting ``n_iter * cv_folds`` real boosters would buy nothing but minutes.
    """

    instances: ClassVar[list["_SearchSpy"]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fitted_with: dict | None = None
        self.best_params_ = {"max_depth": 3}
        self.best_score_ = 0.5
        _SearchSpy.instances.append(self)

    def fit(self, X, y, sample_weight=None):
        self.fitted_with = {"X": X, "y": y, "sample_weight": sample_weight}
        return self

    @classmethod
    def last(cls) -> "_SearchSpy":
        return cls.instances[-1]


class _FixedPredictions:
    """An estimator-shaped object that returns predictions decided in advance."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


class _LogSpy:
    """
    Records structlog events emitted by the classifier.

    Same reason as ``test_flow_loader._LogSpy``: ``api.middleware.logging``
    configures structlog with ``cache_logger_on_first_use=True``, so whether an
    event reaches ``caplog`` depends on test order. Swapping the module logger
    does not.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def record(event: str, **kwargs) -> None:
            self.events.append((level, event, kwargs))

        return record

    def names(self, level: str | None = None) -> list[str]:
        return [name for lvl, name, _ in self.events if level is None or lvl == level]


def _tuning_config(cv_folds: int = 3) -> XGBoostConfig:
    """The smallest config that still walks the whole tuning path."""
    return XGBoostConfig(
        n_estimators=10,
        max_depth=3,
        n_iter_search=2,
        cv_folds=cv_folds,
        n_jobs=1,
        search_space={"max_depth": [2, 3]},
    )


def _blocked_anomalies(
    classes: tuple[str, ...] = ("DoS", "Scan", "BruteForce"),
    per_class: int = 12,
) -> tuple[np.ndarray, np.ndarray, TemporalIndex]:
    """
    Routed anomalies shaped like Stage 2's real input: one block per class,
    chronological inside each block, exactly as CICIDS2017's days are.
    """
    rng = np.random.default_rng(11)
    n = len(classes) * per_class
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.repeat(np.array(classes, dtype=object), per_class)
    index = TemporalIndex(
        order=np.arange(n, dtype=np.int64),
        block=np.repeat(np.arange(len(classes)), per_class).astype(np.int32),
    )
    return X, y, index


class TestTuneHyperparametersFolds:
    """
    What the Stage 2 search cuts its folds on.

    None of this changes a published metric — every figure this track reports
    comes from a test set the search never sees. What it changes is whether the
    search is answering the same question the report answers.
    """

    @pytest.fixture(autouse=True)
    def _spy_on_the_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _SearchSpy.instances = []
        monkeypatch.setattr(xgb_module, "RandomizedSearchCV", _SearchSpy)

    @pytest.fixture
    def log_spy(self, monkeypatch: pytest.MonkeyPatch) -> _LogSpy:
        spy = _LogSpy()
        monkeypatch.setattr(xgb_module, "logger", spy)
        return spy

    def test_a_temporal_index_turns_the_folds_chronological(self):
        X, y, index = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y, train_index=index)

        cv = _SearchSpy.last().kwargs["cv"]
        assert isinstance(cv, list)
        assert len(cv) == 3
        for train, val in cv:
            # Positions into X, and every validation row later than every
            # training row of its own block.
            for code in np.unique(index.block):
                in_block = index.block == code
                train_orders = index.order[train][in_block[train]]
                val_orders = index.order[val][in_block[val]]
                assert train_orders.max() < val_orders.min()

    def test_without_an_index_the_folds_stay_the_shuffled_ones(self):
        """
        The pre-existing behaviour, kept as the fallback. It is announced by the
        cascade, which is the only place that knows whether the run claimed a
        temporal protocol at all.
        """
        X, y, _ = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y)

        cv = _SearchSpy.last().kwargs["cv"]
        assert isinstance(cv, StratifiedKFold)
        assert cv.shuffle is True

    def test_an_index_that_does_not_describe_the_matrix_raises(self):
        """
        Not a data condition — a programming one. An index narrowed by the wrong
        mask produces folds that look entirely reasonable and hyperparameters
        chosen on nonsense, so it must not be allowed to degrade quietly.
        """
        X, y, index = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        with pytest.raises(ValueError, match="describes 36 rows but X_train has 12"):
            classifier.tune_hyperparameters(X[:12], y[:12], train_index=index)

    def test_blocks_too_small_to_cut_fall_back_to_shuffled_folds_with_a_warning(
        self, log_spy: _LogSpy
    ):
        """
        Three rows cannot be cut into four chunks, so every validation chunk
        comes back empty and there is nothing temporal left to do.
        """
        X, y, _ = _blocked_anomalies(classes=("DoS", "Scan", "BruteForce"), per_class=1)
        index = TemporalIndex(order=np.arange(3, dtype=np.int64), block=np.zeros(3, dtype=np.int32))
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y, train_index=index)

        assert isinstance(_SearchSpy.last().kwargs["cv"], StratifiedKFold)
        assert "xgb_tuning_temporal_folds_degenerate" in log_spy.names("warning")

    def test_a_single_routed_class_falls_back_instead_of_training_on_one_label(
        self, log_spy: _LogSpy
    ):
        """
        Stage 1 can route one family and nothing else. A fold whose training half
        holds a single class is not a model, and it is also what sends
        ``_objective_params`` down its ``binary:logistic`` branch with one label.
        """
        X, y, index = _blocked_anomalies(classes=("DoS",), per_class=16)
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y, train_index=index)

        assert isinstance(_SearchSpy.last().kwargs["cv"], StratifiedKFold)
        assert "xgb_tuning_temporal_folds_degenerate" in log_spy.names("warning")

    def test_the_fold_shapes_are_logged_for_the_run_record(self, log_spy: _LogSpy):
        """
        The tuning's equivalent of ``split_sublabel_distribution``: without the
        sizes in the log, a fold set that quietly collapsed reads exactly like a
        healthy one.
        """
        X, y, index = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y, train_index=index)

        recorded = [kw for lvl, name, kw in log_spy.events if name == "xgb_tuning_temporal_folds"]
        assert len(recorded) == 1
        sizes = recorded[0]["train_sizes"]
        # Expanding window: each fold trains on strictly more rows than the last.
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
        assert recorded[0]["val_sizes"] and all(n > 0 for n in recorded[0]["val_sizes"])

    def test_the_scorer_averages_over_every_class_not_the_ones_a_fold_holds(self):
        """
        The same defect as the headline ``f1_macro`` computed without ``labels=``,
        one level down. With shuffled stratified folds a missing class was
        impossible; with chronological folds it is routine, and then the class
        set being averaged over varies per fold *and per candidate*, so the
        search's scores stop being comparable to each other.
        """
        X, y, index = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)
        classifier.tune_hyperparameters(X, y, train_index=index)

        scorer = _SearchSpy.last().kwargs["scoring"]
        # A validation chunk holding two of the three classes, predicted
        # perfectly: 1.0 averaged over what is present, 2/3 over all three.
        y_true = np.array([0, 0, 1, 1])
        score = scorer(_FixedPredictions(y_true.copy()), np.zeros((4, 4), np.float32), y_true)

        assert score == pytest.approx(2 / 3)

    def test_the_sample_weights_are_computed_once_over_the_whole_training_set(self):
        """
        Looks like a bug and is not: sklearn slices ``sample_weight`` per fold,
        and the weight is a property of the class, not of the fold. Recomputing
        it per fold would have each fold optimise a slightly different objective.
        """
        X, y, index = _blocked_anomalies()
        classifier = FlowAttackClassifier(_tuning_config(), random_seed=42)

        classifier.tune_hyperparameters(X, y, train_index=index)

        fitted = _SearchSpy.last().fitted_with
        assert fitted is not None
        assert fitted["sample_weight"].shape == (len(y),)


# ============================================================================
# NeuralNetwork — smoke only
# ============================================================================


@pytest.mark.slow
class TestNeuralNetworkSmoke:
    """
    The MLP is excluded from coverage and never served, so these tests exist to
    stop 700 lines of hand-derived backpropagation from rotting unnoticed. They
    assert that it runs, learns something and survives a roundtrip — not that
    its gradients are correct.
    """

    @staticmethod
    def _tiny_config() -> NeuralNetworkConfig:
        return NeuralNetworkConfig(
            hidden_layers=[16, 8, 4],
            dropout_rates=[0.1, 0.1, 0.0],
            epochs=3,
            batch_size=32,
            lr_warmup_epochs=1,
        )

    def test_fit_runs_and_records_history(self, separable_data):
        X_train, y_train, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_train, y_train)
        X_v, y_v = _attacks_only(X_val, y_val)

        network = NeuralNetwork(
            self._tiny_config(),
            n_features=X_a.shape[1],
            n_classes=len(_ATTACK_CLASSES),
            random_seed=42,
        )
        network.fit(X_a, y_a, X_v, y_v)

        assert len(network.history["train_loss"]) == 3
        assert len(network.history["val_acc"]) == 3
        assert all(np.isfinite(network.history["train_loss"]))

    def test_predict_returns_known_labels(self, separable_data):
        X_train, y_train, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_train, y_train)
        X_v, y_v = _attacks_only(X_val, y_val)

        network = NeuralNetwork(
            self._tiny_config(),
            n_features=X_a.shape[1],
            n_classes=len(_ATTACK_CLASSES),
            random_seed=42,
        )
        network.fit(X_a, y_a, X_v, y_v)

        preds = network.predict(X_v)
        probs = network.predict_proba(X_v)

        assert preds.shape == y_v.shape
        assert set(preds) <= set(network.classes_)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_before_fit_raises(self, separable_data):
        _, _, X_val, _ = separable_data
        network = NeuralNetwork(self._tiny_config(), n_features=4, n_classes=3)

        with pytest.raises(RuntimeError, match="must be fit"):
            network.predict(X_val)

    def test_roundtrip_preserves_predictions(self, separable_data, tmp_path: Path):
        X_train, y_train, X_val, y_val = separable_data
        X_a, y_a = _attacks_only(X_train, y_train)
        X_v, y_v = _attacks_only(X_val, y_val)

        network = NeuralNetwork(
            self._tiny_config(),
            n_features=X_a.shape[1],
            n_classes=len(_ATTACK_CLASSES),
            random_seed=42,
        )
        network.fit(X_a, y_a, X_v, y_v)
        expected = network.predict(X_v)

        network.save(tmp_path / "nn")
        restored = NeuralNetwork(
            self._tiny_config(),
            n_features=X_a.shape[1],
            n_classes=len(_ATTACK_CLASSES),
            random_seed=42,
        ).load(tmp_path / "nn")

        np.testing.assert_array_equal(restored.predict(X_v), expected)
