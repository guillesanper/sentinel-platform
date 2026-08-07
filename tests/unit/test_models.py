"""
Unit tests for ML models.

Tests the AnomalyDetector and AttackClassifier classes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.models.anomaly_detector import ARTIFACT_VERSION, AnomalyDetector
from ml.models.classifier import ATTACK_TYPES, AttackClassifier
from ml.models.fusion import FusionLayer

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature data for testing."""
    np.random.seed(42)
    # 100 samples, 17 features (matching FeatureExtractor output)
    return np.random.rand(100, 17).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels for testing."""
    np.random.seed(42)
    # 7 classes (attack types)
    return np.random.randint(0, 7, size=100)


@pytest.fixture
def trained_detector(sample_features: np.ndarray) -> AnomalyDetector:
    """Create a trained anomaly detector."""
    detector = AnomalyDetector(
        contamination=0.1,
        n_estimators=10,  # Small for fast tests
        random_state=42,
    )
    detector.fit(sample_features)
    return detector


@pytest.fixture
def trained_classifier(sample_features: np.ndarray, sample_labels: np.ndarray) -> AttackClassifier:
    """Create a trained attack classifier."""
    classifier = AttackClassifier(
        n_estimators=10,  # Small for fast tests
        max_depth=3,
        random_state=42,
    )
    classifier.fit(sample_features, sample_labels)
    return classifier


# ============================================================================
# AnomalyDetector Tests
# ============================================================================


class TestAnomalyDetectorInit:
    """Tests for AnomalyDetector initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        detector = AnomalyDetector()
        assert detector.contamination == 0.1
        assert detector.n_estimators == 100
        assert detector.random_state == 42
        assert not detector.is_loaded

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        detector = AnomalyDetector(
            contamination=0.2,
            n_estimators=50,
            random_state=123,
        )
        assert detector.contamination == 0.2
        assert detector.n_estimators == 50
        assert detector.random_state == 123

    def test_is_loaded_before_fit(self) -> None:
        """Test is_loaded returns False before training."""
        detector = AnomalyDetector()
        assert not detector.is_loaded


class TestAnomalyDetectorFit:
    """Tests for AnomalyDetector training."""

    def test_fit_returns_self(self, sample_features: np.ndarray) -> None:
        """Test that fit returns self for chaining."""
        detector = AnomalyDetector(n_estimators=10)
        result = detector.fit(sample_features)
        assert result is detector

    def test_fit_sets_is_loaded(self, sample_features: np.ndarray) -> None:
        """Test that fit sets is_loaded to True."""
        detector = AnomalyDetector(n_estimators=10)
        detector.fit(sample_features)
        assert detector.is_loaded

    def test_fit_calculates_threshold(self, sample_features: np.ndarray) -> None:
        """Test that fit calculates a threshold."""
        detector = AnomalyDetector(n_estimators=10, contamination=0.1)
        detector.fit(sample_features)
        assert detector._threshold != 0.0


class TestAnomalyDetectorPredict:
    """Tests for AnomalyDetector prediction."""

    def test_predict_without_fit_raises(self) -> None:
        """Test that predict raises error if model not trained."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not trained or loaded"):
            detector.predict(np.random.rand(17))

    def test_predict_single_sample(self, trained_detector: AnomalyDetector) -> None:
        """Test prediction on a single sample."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_detector.predict(sample)

        assert "score" in result
        assert "is_anomaly" in result
        assert "raw_score" in result
        assert "confidence" in result

    def test_predict_score_range(self, trained_detector: AnomalyDetector) -> None:
        """Test that prediction score is in [0, 1]."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_detector.predict(sample)

        assert 0 <= result["score"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_predict_returns_bool_is_anomaly(self, trained_detector: AnomalyDetector) -> None:
        """Test that is_anomaly is a boolean."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_detector.predict(sample)

        assert isinstance(result["is_anomaly"], bool)

    def test_predict_batch(
        self, trained_detector: AnomalyDetector, sample_features: np.ndarray
    ) -> None:
        """Test prediction on batch of samples."""
        result = trained_detector.predict(sample_features[:10])

        assert "scores" in result
        assert "is_anomaly" in result
        assert len(result["scores"]) == 10
        assert len(result["is_anomaly"]) == 10


class TestAnomalyDetectorSaveLoad:
    """Tests for AnomalyDetector save/load functionality."""

    def test_save_without_fit_raises(self) -> None:
        """Test that save raises error if model not trained."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="No trained model"):
            detector.save()

    def test_save_load_roundtrip(self, trained_detector: AnomalyDetector) -> None:
        """Test save and load preserves model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "detector.joblib"

            # Save
            trained_detector.save(model_path)
            assert model_path.exists()

            # Load into new detector
            new_detector = AnomalyDetector()
            new_detector.load(model_path)

            assert new_detector.is_loaded
            assert new_detector._threshold == trained_detector._threshold

    def test_load_nonexistent_raises(self) -> None:
        """Test that load raises error for nonexistent file."""
        detector = AnomalyDetector()
        with pytest.raises(FileNotFoundError):
            detector.load("/nonexistent/path.joblib")

    def test_predictions_match_after_load(self, trained_detector: AnomalyDetector) -> None:
        """Test that loaded model gives same predictions."""
        sample = np.random.rand(17).astype(np.float32)
        original_result = trained_detector.predict(sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "detector.joblib"
            trained_detector.save(model_path)

            loaded_detector = AnomalyDetector()
            loaded_detector.load(model_path)
            loaded_result = loaded_detector.predict(sample)

            assert original_result["score"] == loaded_result["score"]
            assert original_result["is_anomaly"] == loaded_result["is_anomaly"]


class TestAnomalyDetectorGetParams:
    """Tests for AnomalyDetector.get_params."""

    def test_get_params_returns_dict(self) -> None:
        """Test that get_params returns a dictionary."""
        detector = AnomalyDetector()
        params = detector.get_params()

        assert isinstance(params, dict)
        assert "contamination" in params
        assert "n_estimators" in params
        assert "is_fitted" in params


# ============================================================================
# AnomalyDetector — F-beta threshold (E11)
# ============================================================================

# Class separation used by `overlapping_events`. Deliberately small: with
# well-separated classes every beta finds the same near-perfect threshold and
# the F-beta property below holds trivially, proving nothing.
_ANOMALY_SEPARATION = 0.5

_N_VAL_NORMAL = 300
_N_VAL_ATTACK = 100


@pytest.fixture
def overlapping_events() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A benign-only training matrix plus a validation set whose classes overlap.

    Returns ``(X_train, X_val, y_val_binary, y_val_labels)`` — the same
    validation rows expressed both ways, so the two accepted label shapes can be
    compared directly.
    """
    rng = np.random.default_rng(7)

    X_train = rng.normal(0.0, 1.0, size=(400, 17)).astype(np.float32)
    X_val = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(_N_VAL_NORMAL, 17)),
            rng.normal(_ANOMALY_SEPARATION, 1.0, size=(_N_VAL_ATTACK, 17)),
        ]
    ).astype(np.float32)

    y_binary = np.concatenate(
        [np.zeros(_N_VAL_NORMAL, dtype=int), np.ones(_N_VAL_ATTACK, dtype=int)]
    )
    y_labels = np.array(
        ["normal"] * _N_VAL_NORMAL + ["brute_force"] * _N_VAL_ATTACK,
        dtype=object,
    )
    return X_train, X_val, y_binary, y_labels


def _fitted(X_train: np.ndarray) -> AnomalyDetector:
    """A detector trained on `X_train` with a fast, deterministic forest."""
    return AnomalyDetector(contamination=0.1, n_estimators=50, random_state=42).fit(X_train)


class TestOptimizeThresholdFbeta:
    """The property that gives ``beta`` its meaning."""

    def test_accepts_binary_int_labels(self, overlapping_events) -> None:
        """The event track's native shape: 0 = normal, 1 = attack."""
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)

        threshold = detector.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        assert isinstance(threshold, float)
        assert detector._threshold == threshold
        assert detector.get_operating_point_info()["threshold"] == threshold

    def test_accepts_string_labels(self, overlapping_events) -> None:
        """Canonical class names: anything but "normal" is the positive class."""
        X_train, X_val, _, y_labels = overlapping_events
        detector = _fitted(X_train)

        detector.optimize_threshold_fbeta(X_val, y_labels, beta=2.0)

        info = detector.get_operating_point_info()
        assert info["tp"] + info["fn"] == _N_VAL_ATTACK
        assert info["tn"] + info["fp"] == _N_VAL_NORMAL

    def test_both_label_shapes_agree(self, overlapping_events) -> None:
        """
        The two shapes describe the same validation set, so they must select the
        same threshold. If the dtype sniffing ever mistook one for the other the
        run would still produce a threshold — just one fitted against labels that
        are all 1, which is exactly the kind of silent wrongness worth pinning.
        """
        X_train, X_val, y_binary, y_labels = overlapping_events

        from_ints = _fitted(X_train)
        from_ints.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        from_strings = _fitted(X_train)
        from_strings.optimize_threshold_fbeta(X_val, y_labels, beta=2.0)

        assert from_strings.get_operating_point_info() == from_ints.get_operating_point_info()

    def test_recall_favouring_beta_never_loses_recall(self, overlapping_events) -> None:
        """
        beta=2.0 weighs recall four times as heavily as beta=0.5 does, so the
        threshold it selects can never be the stingier of the two. This is the
        fundamental guarantee of the F-beta family, and the only thing standing
        between an operator asking for high recall and silently not getting it.
        """
        X_train, X_val, y_binary, _ = overlapping_events

        recall_hungry = _fitted(X_train)
        recall_hungry.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        precision_hungry = _fitted(X_train)
        precision_hungry.optimize_threshold_fbeta(X_val, y_binary, beta=0.5)

        op_recall = recall_hungry.get_operating_point_info()
        op_precision = precision_hungry.get_operating_point_info()

        assert op_recall["recall"] >= op_precision["recall"]
        # Without this the assertion above could pass on a degenerate dataset
        # where both betas reach recall 1.0 and nothing was actually compared.
        assert op_precision["recall"] < 1.0

    def test_precision_favouring_beta_never_loses_precision(self, overlapping_events) -> None:
        """The dual: beta<1 must not end up less precise than beta>1."""
        X_train, X_val, y_binary, _ = overlapping_events

        recall_hungry = _fitted(X_train)
        recall_hungry.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        precision_hungry = _fitted(X_train)
        precision_hungry.optimize_threshold_fbeta(X_val, y_binary, beta=0.5)

        assert (
            precision_hungry.get_operating_point_info()["precision"]
            >= recall_hungry.get_operating_point_info()["precision"]
        )

    def test_without_fit_raises(self, overlapping_events) -> None:
        _, X_val, y_binary, _ = overlapping_events
        with pytest.raises(RuntimeError, match="not trained"):
            AnomalyDetector().optimize_threshold_fbeta(X_val, y_binary)

    def test_n_candidates_is_honoured(self, overlapping_events) -> None:
        """A coarser scan must still land on a valid threshold in range."""
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)

        threshold = detector.optimize_threshold_fbeta(X_val, y_binary, beta=1.0, n_candidates=5)

        assert isinstance(threshold, float)
        assert detector.get_operating_point_info()["threshold"] == threshold


class TestAnomalyDetectorOperatingPoint:
    """What the detector reports about the threshold it is using."""

    def test_before_optimizing_raises(self, overlapping_events) -> None:
        """
        Fitting alone is not enough: the percentile threshold `fit` computes has
        never been measured against any label, so there is no operating point to
        report and returning zeros would be a lie.
        """
        X_train, _, _, _ = overlapping_events

        with pytest.raises(RuntimeError, match="optimize_threshold_fbeta"):
            _fitted(X_train).get_operating_point_info()

    def test_reports_the_full_confusion_matrix(self, overlapping_events) -> None:
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)
        detector.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

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
        assert info["tp"] + info["fp"] + info["tn"] + info["fn"] == len(y_binary)
        assert info["beta"] == 2.0

    def test_beta_key_is_named_after_beta(self, overlapping_events) -> None:
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)
        detector.optimize_threshold_fbeta(X_val, y_binary, beta=0.5)

        assert "f0.5_score" in detector.get_operating_point_info()

    def test_returned_dict_is_a_copy(self, overlapping_events) -> None:
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)
        detector.optimize_threshold_fbeta(X_val, y_binary, beta=1.0)

        detector.get_operating_point_info()["recall"] = -1.0

        assert detector.get_operating_point_info()["recall"] != -1.0

    def test_roundtrip_preserves_the_operating_point(
        self, overlapping_events, tmp_path: Path
    ) -> None:
        """Serving must be able to explain its decisions without a val set."""
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)
        detector.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        model_path = tmp_path / "detector.joblib"
        detector.save(model_path)

        restored = AnomalyDetector().load(model_path)

        assert restored.get_operating_point_info() == detector.get_operating_point_info()


class TestOptimizeThresholdCoexistence:
    """`optimize_threshold_for_recall` keeps working exactly as it did."""

    def test_recall_strategy_still_meets_its_target(self, overlapping_events) -> None:
        """
        The threshold picked from the PR curve must translate into roughly the
        requested recall when `predict` actually applies it.

        "Roughly" and not "exactly": the curve's rule is ``raw_score <=
        threshold`` while `predict` uses a strict ``<``, so the sample sitting
        exactly on the boundary changes sides. That is a pre-existing rounding
        seam, not the label misalignment E12 fixes — under that bug the number
        below lands near the base rate instead of near the target.
        """
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)

        detector.optimize_threshold_for_recall(X_val, y_binary, target_recall=0.8)

        flagged = np.array(
            [1 if detector.predict(row)["is_anomaly"] else 0 for row in X_val],
        )
        attacks = y_binary == 1
        assert flagged[attacks].mean() >= 0.78

    def test_recall_strategy_leaves_no_operating_point(self, overlapping_events) -> None:
        """
        It never measured one. Reporting the F-beta operating point of a
        threshold picked by a different rule would be a fabricated number.
        """
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)

        detector.optimize_threshold_for_recall(X_val, y_binary, target_recall=0.8)

        with pytest.raises(RuntimeError):
            detector.get_operating_point_info()

    def test_recall_strategy_without_fit_raises(self, overlapping_events) -> None:
        _, X_val, y_binary, _ = overlapping_events
        with pytest.raises(RuntimeError, match="not trained"):
            AnomalyDetector().optimize_threshold_for_recall(X_val, y_binary)

    def test_unreachable_target_falls_back_to_max_recall(self, overlapping_events) -> None:
        """
        A target above 1.0 cannot be met by any threshold. The detector must
        still return a usable one — the most permissive it found — rather than
        crash a training run at the last step.
        """
        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)

        threshold = detector.optimize_threshold_for_recall(X_val, y_binary, target_recall=1.5)

        assert isinstance(threshold, float)
        assert detector._threshold == threshold

    def test_the_two_strategies_pick_different_thresholds(self, overlapping_events) -> None:
        """Otherwise `--threshold-strategy` would be a no-op knob."""
        X_train, X_val, y_binary, _ = overlapping_events

        by_recall = _fitted(X_train)
        by_recall.optimize_threshold_for_recall(X_val, y_binary, target_recall=0.8)

        by_fbeta = _fitted(X_train)
        by_fbeta.optimize_threshold_fbeta(X_val, y_binary, beta=0.5)

        assert by_recall._threshold != by_fbeta._threshold


class TestAnomalyDetectorArtifactCompatibility:
    """`load()` and the hyperparameters the artifact carries."""

    def test_get_params_after_load_reflects_the_saved_model(
        self, overlapping_events, tmp_path: Path
    ) -> None:
        """
        The bug this pins: `load()` used to restore only the forest and the
        threshold, so `get_params()` reported whatever the *constructor* was
        given — the defaults, for a model trained with something else entirely.
        Every consumer of that dict was being told a number that had nothing to
        do with the model in memory.
        """
        X_train, _, _, _ = overlapping_events
        trained = AnomalyDetector(contamination=0.25, n_estimators=37, random_state=42)
        trained.fit(X_train)

        model_path = tmp_path / "detector.joblib"
        trained.save(model_path)

        restored = AnomalyDetector(contamination=0.1, n_estimators=100).load(model_path)

        params = restored.get_params()
        assert params["contamination"] == 0.25
        assert params["n_estimators"] == 37

    def test_load_accepts_a_v1_0_0_artifact(self, overlapping_events, tmp_path: Path) -> None:
        """
        1.0.0 files are already on disk in deployed clusters and have no
        `threshold_info`. They must keep loading and predicting.
        """
        import joblib

        X_train, _, _, _ = overlapping_events
        trained = _fitted(X_train)

        legacy_path = tmp_path / "legacy.joblib"
        joblib.dump(
            {
                "model": trained._model,
                "threshold": trained._threshold,
                "contamination": trained.contamination,
                "n_estimators": trained.n_estimators,
                "version": "1.0.0",
            },
            legacy_path,
        )

        restored = AnomalyDetector().load(legacy_path)

        assert restored.is_loaded
        assert restored._threshold == trained._threshold
        assert restored.predict(X_train[0])["score"] == trained.predict(X_train[0])["score"]
        with pytest.raises(RuntimeError):
            restored.get_operating_point_info()

    def test_save_writes_the_current_artifact_version(
        self, overlapping_events, tmp_path: Path
    ) -> None:
        import joblib

        X_train, X_val, y_binary, _ = overlapping_events
        detector = _fitted(X_train)
        detector.optimize_threshold_fbeta(X_val, y_binary, beta=2.0)

        model_path = tmp_path / "detector.joblib"
        detector.save(model_path)

        payload = joblib.load(model_path)
        assert payload["version"] == ARTIFACT_VERSION
        assert payload["threshold_info"]["beta"] == 2.0


# ============================================================================
# AttackClassifier Tests
# ============================================================================


class TestAttackClassifierInit:
    """Tests for AttackClassifier initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        classifier = AttackClassifier()
        assert classifier.n_estimators == 100
        assert classifier.max_depth == 6
        assert classifier.learning_rate == 0.1
        assert not classifier.is_loaded

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        classifier = AttackClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
        )
        assert classifier.n_estimators == 50
        assert classifier.max_depth == 4
        assert classifier.learning_rate == 0.05

    def test_classes_property(self) -> None:
        """Test classes property returns attack types."""
        classifier = AttackClassifier()
        classes = classifier.classes
        assert classes == ATTACK_TYPES


class TestAttackClassifierFit:
    """Tests for AttackClassifier training."""

    def test_fit_returns_self(self, sample_features: np.ndarray, sample_labels: np.ndarray) -> None:
        """Test that fit returns self for chaining."""
        classifier = AttackClassifier(n_estimators=10)
        result = classifier.fit(sample_features, sample_labels)
        assert result is classifier

    def test_fit_sets_is_loaded(
        self, sample_features: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test that fit sets is_loaded to True."""
        classifier = AttackClassifier(n_estimators=10)
        classifier.fit(sample_features, sample_labels)
        assert classifier.is_loaded

    def test_fit_with_string_labels(self, sample_features: np.ndarray) -> None:
        """Test fit with string labels."""
        labels = [ATTACK_TYPES[i % len(ATTACK_TYPES)] for i in range(len(sample_features))]
        classifier = AttackClassifier(n_estimators=10)
        classifier.fit(sample_features, labels)
        assert classifier.is_loaded

    def test_fit_with_eval_set(
        self, sample_features: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test fit with validation set."""
        train_size = 80
        X_train = sample_features[:train_size]
        y_train = sample_labels[:train_size]
        X_val = sample_features[train_size:]
        y_val = sample_labels[train_size:]

        classifier = AttackClassifier(n_estimators=10)
        classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        assert classifier.is_loaded


class TestAttackClassifierPredict:
    """Tests for AttackClassifier prediction."""

    def test_predict_without_fit_raises(self) -> None:
        """Test that predict raises error if model not trained."""
        classifier = AttackClassifier()
        with pytest.raises(RuntimeError, match="not trained or loaded"):
            classifier.predict(np.random.rand(17))

    def test_predict_single_sample(self, trained_classifier: AttackClassifier) -> None:
        """Test prediction on a single sample."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_classifier.predict(sample)

        assert "attack_type" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_attack_type_valid(self, trained_classifier: AttackClassifier) -> None:
        """Test that predicted attack type is valid."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_classifier.predict(sample)

        assert result["attack_type"] in ATTACK_TYPES

    def test_predict_confidence_range(self, trained_classifier: AttackClassifier) -> None:
        """Test that confidence is in [0, 1]."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_classifier.predict(sample)

        assert 0 <= result["confidence"] <= 1

    def test_predict_probabilities_sum_to_one(self, trained_classifier: AttackClassifier) -> None:
        """Test that class probabilities sum to 1."""
        sample = np.random.rand(17).astype(np.float32)
        result = trained_classifier.predict(sample)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_predict_batch(
        self, trained_classifier: AttackClassifier, sample_features: np.ndarray
    ) -> None:
        """Test prediction on batch of samples."""
        result = trained_classifier.predict(sample_features[:10])

        assert "results" in result
        assert len(result["results"]) == 10

    def test_predict_proba(
        self, trained_classifier: AttackClassifier, sample_features: np.ndarray
    ) -> None:
        """Test predict_proba method."""
        proba = trained_classifier.predict_proba(sample_features[:10])

        assert proba.shape == (10, len(ATTACK_TYPES))
        # Each row should sum to 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))


class TestAttackClassifierSaveLoad:
    """Tests for AttackClassifier save/load functionality."""

    def test_save_without_fit_raises(self) -> None:
        """Test that save raises error if model not trained."""
        classifier = AttackClassifier()
        with pytest.raises(RuntimeError, match="No trained model"):
            classifier.save()

    def test_save_load_roundtrip(self, trained_classifier: AttackClassifier) -> None:
        """Test save and load preserves model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "classifier.joblib"

            # Save
            trained_classifier.save(model_path)
            assert model_path.exists()

            # Load into new classifier
            new_classifier = AttackClassifier()
            new_classifier.load(model_path)

            assert new_classifier.is_loaded
            assert new_classifier.classes == trained_classifier.classes

    def test_predictions_match_after_load(self, trained_classifier: AttackClassifier) -> None:
        """Test that loaded model gives same predictions."""
        sample = np.random.rand(17).astype(np.float32)
        original_result = trained_classifier.predict(sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "classifier.joblib"
            trained_classifier.save(model_path)

            loaded_classifier = AttackClassifier()
            loaded_classifier.load(model_path)
            loaded_result = loaded_classifier.predict(sample)

            assert original_result["attack_type"] == loaded_result["attack_type"]
            assert original_result["confidence"] == loaded_result["confidence"]


class TestAttackClassifierFeatureImportance:
    """Tests for feature importance methods."""

    def test_get_feature_importance(self, trained_classifier: AttackClassifier) -> None:
        """Test get_feature_importance returns dict."""
        importance = trained_classifier.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 17  # Number of features

    def test_get_feature_importance_before_fit(self) -> None:
        """Test get_feature_importance returns empty dict before fit."""
        classifier = AttackClassifier()
        importance = classifier.get_feature_importance()

        assert importance == {}


# ============================================================================
# FusionLayer Tests
# ============================================================================


class TestFusionLayerInit:
    """Tests for FusionLayer initialization."""

    def test_init_default_weights(self) -> None:
        """Test initialization with default weights."""
        fusion = FusionLayer()
        assert fusion.anomaly_weight == 0.3
        assert fusion.classification_weight == 0.5
        assert fusion.rules_weight == 0.2

    def test_init_custom_weights(self) -> None:
        """Test initialization with custom weights."""
        fusion = FusionLayer(
            anomaly_weight=0.4,
            classification_weight=0.4,
            rules_weight=0.2,
        )
        assert fusion.anomaly_weight == 0.4
        assert fusion.classification_weight == 0.4
        assert fusion.rules_weight == 0.2

    def test_init_normalizes_weights(self) -> None:
        """Test that weights are normalized if they don't sum to 1."""
        fusion = FusionLayer(
            anomaly_weight=0.6,
            classification_weight=1.0,
            rules_weight=0.4,
        )
        # Total is 2.0, should be normalized
        total = fusion.anomaly_weight + fusion.classification_weight + fusion.rules_weight
        assert abs(total - 1.0) < 0.01


class TestFusionLayerCombine:
    """Tests for FusionLayer.combine method."""

    def test_combine_returns_expected_keys(self) -> None:
        """Test that combine returns all expected keys."""
        fusion = FusionLayer()

        anomaly_result = {"score": 0.7, "is_anomaly": True, "confidence": 0.8}
        classification_result = {
            "attack_type": "brute_force",
            "confidence": 0.9,
            "probabilities": {"normal": 0.1, "brute_force": 0.9},
        }
        event = {
            "timestamp": "2024-01-15T14:00:00Z",
            "action": {"result": "failure"},
            "actor": {"source_ip": "203.0.113.45"},
            "source": {"type": "auth"},
        }

        result = fusion.combine(anomaly_result, classification_result, event)

        expected_keys = [
            "threat_score",
            "severity",
            "attack_type",
            "confidence",
            "is_anomaly",
            "explanation",
            "recommended_actions",
        ]
        for key in expected_keys:
            assert key in result

    def test_combine_threat_score_range(self) -> None:
        """Test that threat score is in [0, 1]."""
        fusion = FusionLayer()

        # Test with various inputs
        test_cases = [
            ({"score": 0.0, "is_anomaly": False, "confidence": 0.5}, "normal"),
            ({"score": 1.0, "is_anomaly": True, "confidence": 1.0}, "brute_force"),
            ({"score": 0.5, "is_anomaly": True, "confidence": 0.7}, "privilege_escalation"),
        ]

        for anomaly_result, attack_type in test_cases:
            classification_result = {
                "attack_type": attack_type,
                "confidence": 0.8,
                "probabilities": {"normal": 0.2, attack_type: 0.8},
            }
            event = {"action": {"result": "success"}, "actor": {}, "source": {}}

            result = fusion.combine(anomaly_result, classification_result, event)

            assert 0 <= result["threat_score"] <= 1

    def test_combine_severity_levels(self) -> None:
        """Test severity levels based on threat score."""
        fusion = FusionLayer()

        # High threat score should give HIGH or CRITICAL severity
        anomaly_result = {"score": 0.95, "is_anomaly": True, "confidence": 0.9}
        classification_result = {
            "attack_type": "data_exfiltration",
            "confidence": 0.95,
            "probabilities": {"normal": 0.05, "data_exfiltration": 0.95},
        }
        event = {
            "timestamp": "2024-01-15T03:00:00Z",  # Night time
            "action": {"result": "success", "type": "data_transfer"},
            "actor": {"source_ip": "203.0.113.45"},
            "source": {"type": "network"},
        }

        result = fusion.combine(anomaly_result, classification_result, event)

        assert result["severity"] in ["HIGH", "CRITICAL"]

    def test_combine_normal_event_low_severity(self) -> None:
        """Test that normal events get low severity."""
        fusion = FusionLayer()

        anomaly_result = {"score": 0.1, "is_anomaly": False, "confidence": 0.9}
        classification_result = {
            "attack_type": "normal",
            "confidence": 0.95,
            "probabilities": {"normal": 0.95, "brute_force": 0.05},
        }
        event = {
            "timestamp": "2024-01-15T14:00:00Z",
            "action": {"result": "success"},
            "actor": {"source_ip": "10.0.0.1"},
            "source": {"type": "auth"},
        }

        result = fusion.combine(anomaly_result, classification_result, event)

        assert result["severity"] in ["LOW", "MEDIUM"]

    def test_combine_includes_recommendations(self) -> None:
        """Test that recommendations are included."""
        fusion = FusionLayer()

        anomaly_result = {"score": 0.8, "is_anomaly": True, "confidence": 0.8}
        classification_result = {
            "attack_type": "brute_force",
            "confidence": 0.9,
            "probabilities": {"normal": 0.1, "brute_force": 0.9},
        }
        event = {"action": {"result": "failure"}, "actor": {}, "source": {}}

        result = fusion.combine(anomaly_result, classification_result, event)

        assert len(result["recommended_actions"]) > 0

    def test_combine_includes_explanations(self) -> None:
        """Test that explanations are included."""
        fusion = FusionLayer()

        anomaly_result = {"score": 0.8, "is_anomaly": True, "confidence": 0.8}
        classification_result = {
            "attack_type": "brute_force",
            "confidence": 0.9,
            "probabilities": {"normal": 0.1, "brute_force": 0.9},
        }
        event = {
            "action": {"result": "failure"},
            "actor": {"user": "admin", "source_ip": "1.2.3.4"},
            "source": {},
        }

        result = fusion.combine(anomaly_result, classification_result, event)

        assert len(result["explanation"]) > 0


class TestFusionLayerGetParams:
    """Tests for FusionLayer.get_params."""

    def test_get_params_returns_weights(self) -> None:
        """Test that get_params returns weight configuration."""
        fusion = FusionLayer()
        params = fusion.get_params()

        assert "anomaly_weight" in params
        assert "classification_weight" in params
        assert "rules_weight" in params


# ============================================================================
# Integration Tests
# ============================================================================


class TestModelIntegration:
    """Integration tests combining multiple models."""

    def test_full_pipeline(
        self,
        trained_detector: AnomalyDetector,
        trained_classifier: AttackClassifier,
    ) -> None:
        """Test full prediction pipeline."""
        fusion = FusionLayer()

        # Generate a sample
        sample = np.random.rand(17).astype(np.float32)

        # Get predictions from both models
        anomaly_result = trained_detector.predict(sample)
        classification_result = trained_classifier.predict(sample)

        # Create a mock event
        event = {
            "timestamp": "2024-01-15T14:00:00Z",
            "action": {"result": "failure"},
            "actor": {"user": "test", "source_ip": "10.0.0.1"},
            "source": {"type": "auth"},
        }

        # Combine results
        final_result = fusion.combine(anomaly_result, classification_result, event)

        # Verify final result has all required fields
        assert "threat_score" in final_result
        assert "severity" in final_result
        assert "attack_type" in final_result
        assert "confidence" in final_result
        assert "explanation" in final_result
        assert "recommended_actions" in final_result
