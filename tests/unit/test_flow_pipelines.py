"""
Unit tests for the two-stage cascades of the flow-based track.

The centrepiece is the feature-name contract. A cascade trained on columns in
one order and served data in another does not crash, does not warn, and does not
score badly on its own test set — it simply produces confident nonsense in
production. So ``save()`` refuses to write an artifact without the column order,
``load()`` refuses to read one that lacks it, and the roundtrip is asserted to
preserve the list exactly, order included.

Everything runs on the ``cicids_mini_csv`` fixture, i.e. through the real
loading, cleaning and temporal-partitioning path, so the cascade is exercised
against frames shaped the way training will actually see them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.flow.config import FlowConfig, IsolationForestConfig, XGBoostConfig
from ml.flow.labels import BENIGN_LABEL, CLASS_ORDER
from ml.flow.loader import (
    LABEL_COL,
    clean_dataset,
    get_feature_columns,
    load_cicids2017,
    split_dataset_temporal,
)
from ml.flow.pipelines import FEATURE_NAMES_FILE, BaseCascadePipeline, CascadeXGBPipeline
from ml.flow.pipelines.base import FEATURE_NAMES_FORMAT_VERSION

# ============================================================================
# Fixtures
# ============================================================================


def _small_config() -> FlowConfig:
    """A track config trimmed to unit-test runtimes, semantics unchanged."""
    return FlowConfig(
        isolation_forest=IsolationForestConfig(n_estimators=40, n_jobs=1),
        xgboost=XGBoostConfig(n_estimators=25, max_depth=3, early_stopping_rounds=5, n_jobs=1),
    )


class _Split:
    """The six arrays a cascade needs, plus the column order that produced them."""

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        self.feature_names = get_feature_columns(train)
        self.X_train, self.y_train = self._arrays(train)
        self.X_val, self.y_val = self._arrays(val)
        self.X_test, self.y_test = self._arrays(test)

    def _arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return df[self.feature_names].to_numpy(), df[LABEL_COL].to_numpy()


@pytest.fixture
def flow_split(cicids_mini_csv: Path) -> _Split:
    """The mini dataset, loaded, cleaned and partitioned temporally."""
    config = _small_config()
    df = clean_dataset(load_cicids2017(cicids_mini_csv))
    train, val, test = split_dataset_temporal(df, config)
    return _Split(train, val, test)


@pytest.fixture
def trained_cascade(flow_split: _Split) -> CascadeXGBPipeline:
    """A cascade trained end to end, feature names recorded."""
    pipeline = CascadeXGBPipeline(_small_config())
    pipeline.train(
        flow_split.X_train,
        flow_split.y_train,
        flow_split.X_val,
        flow_split.y_val,
        beta=2.0,
        skip_tuning=True,
        feature_names=flow_split.feature_names,
    )
    return pipeline


# ============================================================================
# Training
# ============================================================================


class TestCascadeXGBTraining:
    """End-to-end training over the real loading path."""

    def test_train_returns_self(self, flow_split: _Split):
        pipeline = CascadeXGBPipeline(_small_config())

        result = pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
            feature_names=flow_split.feature_names,
        )

        assert result is pipeline

    def test_name_is_stable(self):
        assert CascadeXGBPipeline(_small_config()).name == "cascade_xgb"

    def test_is_a_base_cascade_pipeline(self, trained_cascade: CascadeXGBPipeline):
        assert isinstance(trained_cascade, BaseCascadePipeline)

    def test_both_stages_end_up_fitted(self, trained_cascade: CascadeXGBPipeline):
        assert trained_cascade.if_model.is_fitted is True
        assert trained_cascade.stage2_classifier.is_fitted is True

    def test_stage1_threshold_is_optimised_not_default(self, trained_cascade: CascadeXGBPipeline):
        operating_point = trained_cascade.get_if_operating_point()

        assert operating_point["beta"] == 2.0
        assert trained_cascade.if_model.threshold != 0.0

    def test_feature_names_from_train_override_the_constructor(self, flow_split: _Split):
        pipeline = CascadeXGBPipeline(_small_config(), feature_names=["stale", "names"])

        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
            feature_names=flow_split.feature_names,
        )

        assert pipeline.feature_names == flow_split.feature_names

    def test_feature_names_of_the_wrong_length_fail_fast(self, flow_split: _Split):
        """
        A name list that cannot describe the matrix is caught before training,
        not after an hour of fitting — and never silently.
        """
        pipeline = CascadeXGBPipeline(_small_config())

        with pytest.raises(ValueError, match="entries but"):
            pipeline.train(
                flow_split.X_train,
                flow_split.y_train,
                flow_split.X_val,
                flow_split.y_val,
                skip_tuning=True,
                feature_names=["only", "two"],
            )

    def test_training_without_feature_names_warns_but_proceeds(self, flow_split: _Split):
        """Exploration stays possible; it is ``save()`` that draws the line."""
        pipeline = CascadeXGBPipeline(_small_config())

        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
        )

        assert pipeline.feature_names is None
        assert pipeline.stage2_classifier.is_fitted is True

    def test_no_routed_validation_anomalies_falls_back_to_a_pseudo_eval_set(
        self, flow_split: _Split
    ):
        """
        XGBoost's early stopping needs a non-empty ``eval_set``. When Stage 1
        flags training anomalies but no validation ones — a real possibility on
        a small or skewed validation split — training must degrade to a slice
        of the training anomalies rather than crash inside the booster.
        """
        real_detector = CascadeXGBPipeline(_small_config()).if_model

        class _TrainOnlyDetector:
            """Flags anomalies in train, nothing in val, by matching row counts."""

            def __init__(self, inner, n_val: int) -> None:
                self._inner = inner
                self._n_val = n_val

            def fit(self, X, y):
                self._inner.fit(X, y)
                return self

            def optimize_threshold(self, X, y, beta=1.0):
                return self._inner.optimize_threshold(X, y, beta=beta)

            def predict_binary(self, X):
                if X.shape[0] == self._n_val:
                    return np.zeros(X.shape[0], dtype=int)
                return self._inner.predict_binary(X)

            def get_operating_point_info(self):
                return self._inner.get_operating_point_info()

        pipeline = CascadeXGBPipeline(_small_config())
        pipeline._if_model = _TrainOnlyDetector(real_detector, len(flow_split.y_val))

        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
            feature_names=flow_split.feature_names,
        )

        assert pipeline.stage2_classifier.is_fitted is True

    def test_no_routed_anomalies_raises(self, flow_split: _Split):
        """
        A Stage 1 that flags nothing leaves Stage 2 with an empty training set.
        Better a loud error than a classifier fitted on zero rows.
        """
        pipeline = CascadeXGBPipeline(_small_config())
        pipeline.if_model.fit(flow_split.X_train, flow_split.y_train)
        # Push the threshold below every possible score so nothing is flagged.
        pipeline.if_model._threshold = -1e9

        class _FrozenDetector:
            """Stage 1 stand-in that never flags anything."""

            def fit(self, X, y):
                return self

            def optimize_threshold(self, X, y, beta=1.0):
                return 0.0

            def predict_binary(self, X):
                return np.zeros(X.shape[0], dtype=int)

        pipeline._if_model = _FrozenDetector()

        with pytest.raises(RuntimeError, match="routed 0 anomalous samples"):
            pipeline.train(
                flow_split.X_train,
                flow_split.y_train,
                flow_split.X_val,
                flow_split.y_val,
                skip_tuning=True,
                feature_names=flow_split.feature_names,
            )


# ============================================================================
# Prediction and evaluation
# ============================================================================


class TestCascadeXGBPredict:
    """Cascade inference over the two stages."""

    def test_predict_returns_one_canonical_label_per_row(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        preds = trained_cascade.predict(flow_split.X_test)

        assert preds.shape == flow_split.y_test.shape
        assert set(preds) <= set(CLASS_ORDER)

    def test_flows_stage1_calls_benign_never_reach_stage2(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        """The defining property of the cascade, asserted directly."""
        preds = trained_cascade.predict(flow_split.X_test)
        anomaly_mask = trained_cascade.if_model.predict_binary(flow_split.X_test).astype(bool)

        assert set(preds[~anomaly_mask]) <= {BENIGN_LABEL}

    def test_predict_on_an_empty_matrix_returns_an_empty_result(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        empty = flow_split.X_test[:0]

        assert trained_cascade.predict(empty).shape == (0,)

    def test_evaluate_reports_the_mandatory_metrics(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        metrics = trained_cascade.evaluate(flow_split.X_test, flow_split.y_test)

        assert {"accuracy", "f1_macro", "f1_weighted"} <= set(metrics)
        assert all(0.0 <= metrics[k] <= 1.0 for k in ("accuracy", "f1_macro", "f1_weighted"))

    def test_evaluate_includes_the_stage1_operating_point(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        metrics = trained_cascade.evaluate(flow_split.X_test, flow_split.y_test)

        assert {"if_precision", "if_recall", "if_fpr", "if_fnr"} <= set(metrics)

    def test_evaluate_omits_the_operating_point_when_none_was_measured(self, flow_split: _Split):
        """A cascade restored from disk without diagnostics still evaluates."""
        pipeline = CascadeXGBPipeline(_small_config())
        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
            feature_names=flow_split.feature_names,
        )
        pipeline.if_model._threshold_info = {}

        metrics = pipeline.evaluate(flow_split.X_test, flow_split.y_test)

        assert "if_precision" not in metrics
        assert "accuracy" in metrics

    def test_separates_benign_from_attack_traffic(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        """
        A floor, not a benchmark, and deliberately binary.

        The mini fixture draws every attack class from the *same* distribution
        — only the benign blocks differ — so telling DDoS from PortScan in it is
        impossible in principle and end-to-end accuracy is not a meaningful
        number. What the fixture does support is the BENIGN/attack decision,
        which is Stage 1's whole job. A cascade that cannot clear this bar is
        broken, not underfitted.
        """
        predicted_attack = trained_cascade.predict(flow_split.X_test) != BENIGN_LABEL
        actual_attack = flow_split.y_test != BENIGN_LABEL

        assert float((predicted_attack == actual_attack).mean()) > 0.9

    def test_feature_importance_is_keyed_by_column_name(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        importance = trained_cascade.get_feature_importance()

        assert set(importance) == set(flow_split.feature_names)


# ============================================================================
# The feature-name contract
# ============================================================================


class TestFeatureNameContract:
    """
    The single most expensive failure this design can have.

    A column-order mismatch between training and inference raises nothing, logs
    nothing and degrades no visible metric. Every assertion below exists to make
    that failure impossible to reach.
    """

    def test_save_then_load_preserves_the_list_exactly(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split, tmp_path: Path
    ):
        trained_cascade.save(tmp_path / "cascade")

        restored = CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

        # Equality, not set equality: order is the whole point.
        assert restored.feature_names == flow_split.feature_names

    def test_saved_order_survives_a_permuted_looking_list(self, flow_split: _Split, tmp_path: Path):
        """
        Reversing the names must come back reversed. A ``set`` comparison would
        pass on either, which is exactly the bug class this guards.
        """
        reversed_names = list(reversed(flow_split.feature_names))
        pipeline = CascadeXGBPipeline(_small_config())
        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
            feature_names=reversed_names,
        )
        pipeline.save(tmp_path / "cascade")

        restored = CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

        assert restored.feature_names == reversed_names
        assert restored.feature_names != flow_split.feature_names

    def test_save_without_feature_names_refuses(self, flow_split: _Split, tmp_path: Path):
        """
        Refusing to write is deliberate. An artifact without its column order
        cannot be validated at inference, so it must never reach disk.
        """
        pipeline = CascadeXGBPipeline(_small_config())
        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            skip_tuning=True,
        )

        with pytest.raises(RuntimeError, match="cannot be saved without feature names"):
            pipeline.save(tmp_path / "cascade")

    def test_saved_contract_carries_its_metadata(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split, tmp_path: Path
    ):
        trained_cascade.save(tmp_path / "cascade")

        payload = json.loads(
            (tmp_path / "cascade" / FEATURE_NAMES_FILE).read_text(encoding="utf-8")
        )

        assert payload["feature_names"] == flow_split.feature_names
        assert payload["n_features"] == len(flow_split.feature_names)
        assert payload["pipeline"] == "cascade_xgb"
        assert payload["format_version"] == FEATURE_NAMES_FORMAT_VERSION

    def test_load_without_the_contract_file_refuses(
        self, trained_cascade: CascadeXGBPipeline, tmp_path: Path
    ):
        trained_cascade.save(tmp_path / "cascade")
        (tmp_path / "cascade" / FEATURE_NAMES_FILE).unlink()

        with pytest.raises(FileNotFoundError, match=FEATURE_NAMES_FILE):
            CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

    def test_load_rejects_an_inconsistent_contract(
        self, trained_cascade: CascadeXGBPipeline, tmp_path: Path
    ):
        """A declared count that disagrees with the list means a corrupt write."""
        trained_cascade.save(tmp_path / "cascade")
        path = tmp_path / "cascade" / FEATURE_NAMES_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["n_features"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="declares n_features=99"):
            CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

    def test_load_rejects_a_malformed_contract(
        self, trained_cascade: CascadeXGBPipeline, tmp_path: Path
    ):
        trained_cascade.save(tmp_path / "cascade")
        path = tmp_path / "cascade" / FEATURE_NAMES_FILE
        path.write_text(json.dumps({"feature_names": "Flow Duration"}), encoding="utf-8")

        with pytest.raises(ValueError, match="list of feature names"):
            CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

    def test_feature_names_property_hands_out_a_copy(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split
    ):
        names = trained_cascade.feature_names
        assert names is not None
        names.append("injected")

        assert trained_cascade.feature_names == flow_split.feature_names


# ============================================================================
# Persistence
# ============================================================================


class TestCascadeXGBPersistence:
    """Roundtripping the whole cascade through disk."""

    def test_roundtrip_preserves_predictions(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split, tmp_path: Path
    ):
        expected = trained_cascade.predict(flow_split.X_test)

        trained_cascade.save(tmp_path / "cascade")
        restored = CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

        np.testing.assert_array_equal(restored.predict(flow_split.X_test), expected)

    def test_roundtrip_preserves_the_evaluation(
        self, trained_cascade: CascadeXGBPipeline, flow_split: _Split, tmp_path: Path
    ):
        expected = trained_cascade.evaluate(flow_split.X_test, flow_split.y_test)

        trained_cascade.save(tmp_path / "cascade")
        restored = CascadeXGBPipeline(_small_config()).load(tmp_path / "cascade")

        assert restored.evaluate(flow_split.X_test, flow_split.y_test) == expected

    def test_save_writes_both_stage_directories(
        self, trained_cascade: CascadeXGBPipeline, tmp_path: Path
    ):
        trained_cascade.save(tmp_path / "cascade")

        assert (tmp_path / "cascade" / "isolation_forest").is_dir()
        assert (tmp_path / "cascade" / "xgboost").is_dir()
        assert (tmp_path / "cascade" / FEATURE_NAMES_FILE).is_file()

    def test_save_creates_missing_directories(
        self, trained_cascade: CascadeXGBPipeline, tmp_path: Path
    ):
        target = tmp_path / "deeply" / "nested" / "cascade"

        trained_cascade.save(target)

        assert (target / FEATURE_NAMES_FILE).is_file()

    def test_load_returns_self(self, trained_cascade: CascadeXGBPipeline, tmp_path: Path):
        trained_cascade.save(tmp_path / "cascade")
        pipeline = CascadeXGBPipeline(_small_config())

        assert pipeline.load(tmp_path / "cascade") is pipeline

    def test_accepts_a_string_path(self, trained_cascade: CascadeXGBPipeline, tmp_path: Path):
        trained_cascade.save(str(tmp_path / "cascade"))

        restored = CascadeXGBPipeline(_small_config()).load(str(tmp_path / "cascade"))

        assert restored.feature_names == trained_cascade.feature_names


# ============================================================================
# Import hygiene
# ============================================================================


def test_pipelines_package_does_not_import_matplotlib():
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale and installs only
    ``requirements.txt``, where matplotlib is deliberately absent. A
    module-level plotting import anywhere under ``ml.flow`` would either break
    the image or add ~120 MB and its CVE surface to a security product.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.flow.pipelines, sys; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_pipelines_package_does_not_re_export_the_experimental_cascade():
    """
    ``CascadeMLPPipeline`` is a reference implementation with F1-macro 0.7508.
    Importing the package must not drag it in, so that nothing depends on it by
    accident and it stays out of the serving path.
    """
    import ml.flow.pipelines as pipelines_package

    assert not hasattr(pipelines_package, "CascadeMLPPipeline")
    assert "cascade_mlp" not in sys.modules.get("ml.flow.pipelines").__dict__


# ============================================================================
# CascadeMLPPipeline — smoke only
# ============================================================================


@pytest.mark.slow
class TestCascadeMLPSmoke:
    """
    The MLP cascade is excluded from coverage and never served. These tests
    exist so that it fails visibly if the shared Stage 1 or the pipeline
    contract moves under it.
    """

    @staticmethod
    def _config() -> FlowConfig:
        config = _small_config()
        config.neural_network.hidden_layers = [16, 8, 4]
        config.neural_network.dropout_rates = [0.1, 0.1, 0.0]
        config.neural_network.epochs = 3
        config.neural_network.batch_size = 32
        config.neural_network.lr_warmup_epochs = 1
        return config

    def test_train_predict_evaluate_and_roundtrip(self, flow_split: _Split, tmp_path: Path):
        from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline

        pipeline = CascadeMLPPipeline(self._config())
        pipeline.train(
            flow_split.X_train,
            flow_split.y_train,
            flow_split.X_val,
            flow_split.y_val,
            beta=2.0,
            skip_tuning=True,
            feature_names=flow_split.feature_names,
        )

        assert pipeline.name == "cascade_mlp"
        assert set(pipeline.predict(flow_split.X_test)) <= set(CLASS_ORDER)
        assert {"accuracy", "f1_macro"} <= set(
            pipeline.evaluate(flow_split.X_test, flow_split.y_test)
        )
        assert len(pipeline.get_training_history()["train_loss"]) == 3

        expected = pipeline.predict(flow_split.X_test)
        pipeline.save(tmp_path / "mlp")
        restored = CascadeMLPPipeline(self._config()).load(tmp_path / "mlp")

        assert restored.feature_names == flow_split.feature_names
        np.testing.assert_array_equal(restored.predict(flow_split.X_test), expected)

    def test_save_before_training_refuses(self, tmp_path: Path):
        from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline

        with pytest.raises(RuntimeError, match="must be trained before save"):
            CascadeMLPPipeline(self._config(), feature_names=["a"]).save(tmp_path / "mlp")

    def test_predict_before_training_refuses(self, flow_split: _Split):
        from ml.flow.pipelines.cascade_mlp import CascadeMLPPipeline

        with pytest.raises(RuntimeError, match="before predict"):
            CascadeMLPPipeline(self._config()).predict(flow_split.X_test)
