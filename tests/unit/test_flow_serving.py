"""
Unit tests for the flow inference service.

The service exists to enforce one invariant: the column order a cascade was
trained on must equal ``FLOW_FEATURE_NAMES``. A mismatch produces confident
predictions computed from features fed into the wrong columns — no exception, no
warning, no degraded offline metric. So the loudest test in this file is the one
that asserts a mismatched artifact **refuses to load and refuses to predict**.

Everything runs against a cascade trained on a synthetic 78-column dataset built
from the real contract. The ``cicids_mini_csv`` fixture carries three columns,
which is right for the loader but cannot exercise the contract; and the 844 MB
of real captures are off limits to unit tests.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ml.flow.config import FlowConfig, IsolationForestConfig, XGBoostConfig
from ml.flow.features import FLOW_FEATURE_NAMES
from ml.flow.labels import BENIGN_LABEL
from ml.flow.pipelines import FEATURE_NAMES_FILE
from ml.flow.pipelines.cascade_xgb import CascadeXGBPipeline
from ml.flow.serving import CascadeInferenceService, FeatureContractError

# ============================================================================
# A cascade over the real 78-feature contract
# ============================================================================

_N_BENIGN = 400
_N_ATTACK = 120
_ATTACKS = ("DDoS", "Scan")


def _small_config(output_path: Path | None = None) -> FlowConfig:
    """A track config trimmed to unit-test runtimes, semantics unchanged."""
    config = FlowConfig(
        isolation_forest=IsolationForestConfig(n_estimators=40, n_jobs=1),
        xgboost=XGBoostConfig(n_estimators=25, max_depth=3, early_stopping_rounds=5, n_jobs=1),
    )
    if output_path is not None:
        config.output_path = Path(output_path)
    return config


def _synthetic_split(seed: int = 7) -> dict:
    """
    Train/val/test arrays over the genuine 78 columns.

    Benign flows cluster around one centre and attacks around another, well
    separated, so a 40-tree forest and a 25-round booster settle in seconds
    while still learning something. The point is the contract, not the accuracy.
    """
    rng = np.random.default_rng(seed)
    n_features = len(FLOW_FEATURE_NAMES)

    def block(n: int, centre: float, label: str) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(centre, 1.0, size=(n, n_features)).astype(np.float32)
        return X, np.array([label] * n, dtype=object)

    parts = [block(_N_BENIGN, 0.0, BENIGN_LABEL)]
    parts += [block(_N_ATTACK, 8.0 + 4.0 * i, attack) for i, attack in enumerate(_ATTACKS)]

    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])

    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    n_train = int(0.6 * len(y))
    n_val = int(0.2 * len(y))
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :],
        "y_test": y[n_train + n_val :],
        "feature_names": list(FLOW_FEATURE_NAMES),
    }


@pytest.fixture(scope="module")
def synthetic_split() -> dict:
    return _synthetic_split()


@pytest.fixture(scope="module")
def saved_cascade(synthetic_split: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A trained cascade on disk, saved with the real feature contract."""
    directory = tmp_path_factory.mktemp("flow-models") / "cascade_xgb"

    pipeline = CascadeXGBPipeline(_small_config())
    pipeline.train(
        synthetic_split["X_train"],
        synthetic_split["y_train"],
        synthetic_split["X_val"],
        synthetic_split["y_val"],
        beta=2.0,
        skip_tuning=True,
        feature_names=synthetic_split["feature_names"],
    )
    pipeline.save(directory)
    return directory


@pytest.fixture
def service(saved_cascade: Path) -> CascadeInferenceService:
    return CascadeInferenceService().load(saved_cascade)


def _flow_from_vector(vector: np.ndarray) -> dict:
    """Turn a feature row back into a wire-shaped flow payload."""
    return {
        "features": {name: float(vector[i]) for i, name in enumerate(FLOW_FEATURE_NAMES)},
        "src_ip": "10.0.0.5",
        "dst_ip": "10.0.1.20",
        "dst_port": 443,
        "protocol": "TCP",
    }


# ============================================================================
# The feature contract — the reason this service exists
# ============================================================================


class TestFeatureContractEnforcement:
    """
    A cascade whose columns mean something else must never answer a request.
    Failing loudly is the only detectable outcome: the alternative is confident
    nonsense that looks exactly like a correct prediction.
    """

    def test_a_mismatched_contract_refuses_to_load(
        self, saved_cascade: Path, tmp_path: Path, synthetic_split: dict
    ):
        broken = tmp_path / "broken"
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        pipeline._feature_names = [f"totally_different_{i}" for i in range(78)]
        pipeline.save(broken)

        with pytest.raises(FeatureContractError, match="different feature space"):
            CascadeInferenceService().load(broken)

    def test_a_service_that_refused_to_load_does_not_predict(
        self, tmp_path: Path, saved_cascade: Path
    ):
        """
        The half-open state is the dangerous one. After a rejected load the
        service must be as unusable as a fresh one, not partially wired up.
        """
        broken = tmp_path / "broken"
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        pipeline._feature_names = [f"col_{i}" for i in range(78)]
        pipeline.save(broken)

        service = CascadeInferenceService()
        with pytest.raises(FeatureContractError):
            service.load(broken)

        assert service.is_loaded is False
        with pytest.raises(RuntimeError, match="No flow cascade loaded"):
            service.predict({"features": {"Flow Duration": 1.0}})

    def test_the_same_names_in_a_different_order_are_still_a_mismatch(
        self, saved_cascade: Path, tmp_path: Path
    ):
        """
        The subtlest and most dangerous case: every name present, none missing,
        only the order changed. A ``set`` comparison would wave it through, and
        every served prediction afterwards would be wrong.
        """
        reordered = tmp_path / "reordered"
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        names = list(FLOW_FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        pipeline._feature_names = names
        pipeline.save(reordered)

        with pytest.raises(FeatureContractError, match="different order"):
            CascadeInferenceService().load(reordered)

    def test_a_shorter_contract_is_reported_by_length(self, saved_cascade: Path, tmp_path: Path):
        truncated = tmp_path / "truncated"
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        pipeline._feature_names = list(FLOW_FEATURE_NAMES[:40])
        pipeline.save(truncated)

        with pytest.raises(FeatureContractError, match="expected 78 features, artifact has 40"):
            CascadeInferenceService().load(truncated)

    def test_a_renamed_feature_is_reported_by_name(self, saved_cascade: Path, tmp_path: Path):
        renamed = tmp_path / "renamed"
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        names = list(FLOW_FEATURE_NAMES)
        names[14] = "Flow Bytes Per Second"
        pipeline._feature_names = names
        pipeline.save(renamed)

        with pytest.raises(FeatureContractError, match="Flow Bytes/s"):
            CascadeInferenceService().load(renamed)

    def test_a_matching_contract_loads(self, saved_cascade: Path):
        service = CascadeInferenceService().load(saved_cascade)

        assert service.is_loaded is True
        assert service.feature_names == list(FLOW_FEATURE_NAMES)


# ============================================================================
# Loading
# ============================================================================


class TestLoad:
    """Artifact discovery and the tolerated failure modes."""

    def test_returns_self(self, saved_cascade: Path):
        service = CascadeInferenceService()

        assert service.load(saved_cascade) is service

    def test_a_fresh_service_is_not_loaded(self):
        assert CascadeInferenceService().is_loaded is False

    def test_a_missing_directory_raises_file_not_found(self, tmp_path: Path):
        """
        The API treats this as "no trained models yet" and degrades the flow
        endpoints, exactly as it does for the event models.
        """
        with pytest.raises(FileNotFoundError):
            CascadeInferenceService().load(tmp_path / "nothing-here")

    def test_an_artifact_without_the_contract_file_raises_file_not_found(
        self, saved_cascade: Path, tmp_path: Path
    ):
        import shutil

        stripped = tmp_path / "stripped"
        shutil.copytree(saved_cascade, stripped)
        (stripped / FEATURE_NAMES_FILE).unlink()

        with pytest.raises(FileNotFoundError, match=FEATURE_NAMES_FILE):
            CascadeInferenceService().load(stripped)

    def test_defaults_to_the_configured_cascade_directory(self, saved_cascade: Path):
        """``load()`` with no argument must find ``models/flow/cascade_xgb``."""
        config = _small_config(output_path=saved_cascade.parent)

        service = CascadeInferenceService(config).load()

        assert service.is_loaded is True

    def test_accepts_a_string_path(self, saved_cascade: Path):
        assert CascadeInferenceService().load(str(saved_cascade)).is_loaded is True

    def test_exposes_the_stage1_operating_point(self, service: CascadeInferenceService):
        """
        Persisted at save time so a loaded detector can explain its decisions
        without a validation set at hand.
        """
        operating_point = service.get_operating_point()

        assert {"precision", "recall", "fpr", "fnr", "threshold"} <= set(operating_point)

    def test_the_operating_point_needs_a_loaded_cascade(self):
        with pytest.raises(RuntimeError, match="No flow cascade loaded"):
            CascadeInferenceService().get_operating_point()


# ============================================================================
# Prediction
# ============================================================================


class TestPredict:
    """Single-flow inference."""

    def test_returns_a_full_assessment(self, service: CascadeInferenceService):
        result = service.predict({"features": {"Flow Duration": 1200.0}})

        assert set(result) == {
            "threat_score",
            "severity",
            "attack_type",
            "confidence",
            "is_anomaly",
            "explanation",
            "recommended_actions",
            "details",
        }
        assert 0.0 <= result["threat_score"] <= 1.0

    def test_predicting_without_a_cascade_raises(self):
        with pytest.raises(RuntimeError, match="No flow cascade loaded"):
            CascadeInferenceService().predict({"features": {}})

    def test_reorders_by_name_not_by_position(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        """
        The end-to-end statement of the contract: shuffling the JSON keys must
        not change the verdict.
        """
        vector = synthetic_split["X_test"][0]
        flow = _flow_from_vector(vector)
        shuffled = {
            **flow,
            "features": dict(reversed(list(flow["features"].items()))),
        }

        assert service.predict(flow) == service.predict(shuffled)

    def test_metadata_does_not_change_the_verdict(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        """
        ``src_ip``/``dst_ip``/``protocol`` reach the explanation, never the
        model. Training on them would fit the lab topology.
        """
        flow = _flow_from_vector(synthetic_split["X_test"][0])
        elsewhere = {**flow, "src_ip": "203.0.113.99", "dst_ip": "198.51.100.4"}

        assert service.predict(flow)["threat_score"] == service.predict(elsewhere)["threat_score"]

    def test_separates_benign_from_attack_traffic(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        """
        A floor, not a benchmark. The two clusters are far apart by
        construction; a service that cannot tell them apart is misrouting
        columns, not underfitting.
        """
        flows = [_flow_from_vector(v) for v in synthetic_split["X_test"]]
        predicted_attack = np.array(
            [r["attack_type"] != BENIGN_LABEL for r in service.predict_batch(flows)]
        )
        actual_attack = synthetic_split["y_test"] != BENIGN_LABEL

        assert float((predicted_attack == actual_attack).mean()) > 0.9

    def test_an_empty_payload_still_answers(self, service: CascadeInferenceService):
        """All-zero features are a valid vector; the service must not crash on them."""
        result = service.predict({"features": {}})

        assert 0.0 <= result["threat_score"] <= 1.0

    def test_a_flat_feature_mapping_is_accepted(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        flow = _flow_from_vector(synthetic_split["X_test"][0])

        nested = service.predict(flow)
        flat = service.predict(flow["features"])

        assert nested["threat_score"] == flat["threat_score"]

    def test_benign_verdicts_carry_a_benign_probability(self, service: CascadeInferenceService):
        """
        Flows Stage 1 calls benign never reach Stage 2 — the defining property
        of a cascade — so their classification is synthesised, not predicted.
        """
        result = service.predict({"features": {}})

        assert BENIGN_LABEL in result["details"]["class_probabilities"]


class TestPredictBatch:
    """Batch inference, which is where the vectorisation lives."""

    def test_one_result_per_flow_in_order(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        flows = [_flow_from_vector(v) for v in synthetic_split["X_test"][:10]]

        results = service.predict_batch(flows)

        assert len(results) == 10
        for i, flow in enumerate(flows):
            assert results[i] == service.predict(flow)

    def test_an_empty_batch_returns_an_empty_list(self, service: CascadeInferenceService):
        assert service.predict_batch([]) == []

    def test_a_batch_of_only_benign_flows_skips_stage_two(
        self, service: CascadeInferenceService, monkeypatch: pytest.MonkeyPatch
    ):
        """
        When Stage 1 flags nothing there is nothing for Stage 2 to name, and
        calling it with a zero-row matrix would be both wasteful and, in
        sklearn's view, an error.
        """
        calls: list[int] = []
        classifier = service._pipeline.stage2_classifier
        original = classifier.predict_proba
        monkeypatch.setattr(
            classifier,
            "predict_proba",
            lambda X: (calls.append(X.shape[0]), original(X))[1],
        )
        monkeypatch.setattr(
            service._pipeline.if_model,
            "predict_binary",
            lambda X: np.zeros(X.shape[0], dtype=int),
        )

        results = service.predict_batch([{"features": {}}, {"features": {}}])

        assert calls == []
        assert all(r["attack_type"] == BENIGN_LABEL for r in results)

    def test_stage_two_is_called_once_for_the_whole_batch(
        self,
        service: CascadeInferenceService,
        synthetic_split: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """
        The batch endpoint must not degenerate into one model call per flow —
        that is the whole reason it exists alongside the single-flow route.
        """
        calls: list[int] = []
        classifier = service._pipeline.stage2_classifier
        original = classifier.predict_proba
        monkeypatch.setattr(
            classifier,
            "predict_proba",
            lambda X: (calls.append(X.shape[0]), original(X))[1],
        )
        monkeypatch.setattr(
            service._pipeline.if_model,
            "predict_binary",
            lambda X: np.ones(X.shape[0], dtype=int),
        )

        flows = [_flow_from_vector(v) for v in synthetic_split["X_test"][:12]]
        service.predict_batch(flows)

        assert calls == [12]

    def test_a_mixed_batch_scatters_results_back_to_the_right_rows(
        self, service: CascadeInferenceService, synthetic_split: dict
    ):
        """
        Stage 2 sees only the routed subset, so its answers have to be placed
        back by index. An off-by-one here hands one flow's verdict to another.
        """
        attack_rows = synthetic_split["X_test"][synthetic_split["y_test"] != BENIGN_LABEL][:4]
        benign_rows = synthetic_split["X_test"][synthetic_split["y_test"] == BENIGN_LABEL][:4]
        flows = []
        for attack, benign in zip(attack_rows, benign_rows, strict=True):
            flows.append(_flow_from_vector(benign))
            flows.append(_flow_from_vector(attack))

        results = service.predict_batch(flows)

        for i, result in enumerate(results):
            assert result == service.predict(flows[i]), f"row {i} does not match its own flow"

    def test_batching_without_a_cascade_raises(self):
        with pytest.raises(RuntimeError, match="No flow cascade loaded"):
            CascadeInferenceService().predict_batch([{"features": {}}])


# ============================================================================
# Roundtrip
# ============================================================================


class TestRoundtrip:
    """train → save → load → predict, over the real contract."""

    def test_the_service_reproduces_the_pipeline_it_was_trained_from(
        self, saved_cascade: Path, synthetic_split: dict
    ):
        pipeline = CascadeXGBPipeline(_small_config()).load(saved_cascade)
        expected = pipeline.predict(synthetic_split["X_test"][:20])

        service = CascadeInferenceService().load(saved_cascade)
        flows = [_flow_from_vector(v) for v in synthetic_split["X_test"][:20]]
        served = [r["attack_type"] for r in service.predict_batch(flows)]

        # The service may relabel a confident-but-unexplained anomaly; every
        # other verdict must be the pipeline's, unchanged.
        for got, want in zip(served, expected, strict=True):
            assert got in (want, "unknown_anomaly")

    def test_two_services_loaded_from_one_artifact_agree(
        self, saved_cascade: Path, synthetic_split: dict
    ):
        flow = _flow_from_vector(synthetic_split["X_test"][0])

        first = CascadeInferenceService().load(saved_cascade).predict(flow)
        second = CascadeInferenceService().load(saved_cascade).predict(flow)

        assert first == second


# ============================================================================
# Import hygiene
# ============================================================================


def test_serving_does_not_import_matplotlib():
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale and installs only
    ``requirements.txt``, where matplotlib is deliberately absent. The serving
    path is the one place where that must be beyond doubt.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.flow.serving, sys; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
