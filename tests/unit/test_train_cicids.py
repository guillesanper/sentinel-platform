"""
Tests for the flow track's training CLI.

``ml/training/train_cicids.py`` is excluded from coverage, like every other
training entry point in this repo, so nothing here moves the coverage number.
What these tests buy is regression protection on the three properties that make
the CLI usable at all:

* it does not drag matplotlib into ``ml/``, which ``api/Dockerfile`` copies
  wholesale into the production image;
* it writes a cascade in the layout the serving path expects, and records which
  partitioning protocol produced the metrics — an F1-macro from this track is
  uninterpretable without it;
* what it writes, :class:`~ml.flow.serving.CascadeInferenceService` can load.
  That last one is the only end-to-end evidence that training and inference
  agree on the column order, and a mismatch there produces confident, wrong
  predictions with no exception anywhere.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from ml.flow.config import FlowConfig
from ml.flow.features import FLOW_FEATURE_NAMES
from ml.flow.labels import CLASS_ORDER
from ml.flow.loader import (
    CAPTURE_DAY_COL,
    ORDER_IDX_COL,
    clean_dataset,
    load_cicids2017,
    split_dataset_temporal,
)
from ml.flow.pipelines.base import FEATURE_NAMES_FILE
from ml.flow.serving import CascadeInferenceService
from ml.training.train_cicids import _build_config, _build_matrices, main

# ============================================================================
# Config construction
# ============================================================================


class TestBuildConfig:
    """The CLI options have to survive the trip into FlowConfig unchanged."""

    def _config(self, **overrides) -> FlowConfig:
        kwargs = {
            "data_dir": "api/csvs",
            "output_dir": "models/flow",
            "sample_size": 5000,
            "split": "temporal",
            "temporal_block": "sub-label",
            "beta": 2.0,
            "seed": 7,
        }
        kwargs.update(overrides)
        return _build_config(**kwargs)

    def test_options_reach_the_config(self) -> None:
        config = self._config()

        assert config.data_path == Path("api/csvs")
        assert config.output_path == Path("models/flow")
        assert config.sample_size == 5000
        assert config.split_strategy == "temporal"
        assert config.beta == 2.0
        assert config.random_seed == 7

    def test_the_cascade_paths_are_the_ones_serving_reads(self) -> None:
        config = self._config(output_dir="/tmp/flow")

        assert config.cascade_xgb_path == Path("/tmp/flow/cascade_xgb")
        assert config.cascade_mlp_path == Path("/tmp/flow/cascade_mlp")

    @pytest.mark.parametrize("strategy", ["temporal", "random", "day-holdout"])
    def test_every_advertised_strategy_is_accepted(self, strategy: str) -> None:
        assert self._config(split=strategy).split_strategy == strategy

    def test_an_unknown_strategy_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            self._config(split="chronological")


# ============================================================================
# Feature space
# ============================================================================


class TestBuildMatrices:
    """
    The provenance columns must not reach the model.

    ``capture_day`` and ``order_idx`` are perfectly predictive of the label in
    CICIDS2017 — each class lives on one day — so a model that sees them scores
    beautifully and detects the capture schedule. Nothing raises. The loader
    guards its own boundary; this guards the CLI's.
    """

    def test_provenance_columns_are_not_features(self, cicids_mini_csv: Path) -> None:
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, val_df, test_df = split_dataset_temporal(frame, FlowConfig())

        dataset = _build_matrices(train_df, val_df, test_df)

        assert CAPTURE_DAY_COL not in dataset.feature_names
        assert ORDER_IDX_COL not in dataset.feature_names
        assert "Label" not in dataset.feature_names
        assert dataset.X_train.shape[1] == len(dataset.feature_names)

    def test_labels_follow_the_canonical_order(self, cicids_mini_csv: Path) -> None:
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, val_df, test_df = split_dataset_temporal(frame, FlowConfig())

        dataset = _build_matrices(train_df, val_df, test_df)

        assert set(dataset.labels) == {"BENIGN", "BruteForce", "DDoS", "DoS", "Scan", "WebAttack"}
        # In CLASS_ORDER order, not in the order the frames happened to list
        # them: this list becomes the column layout of the confusion matrix and
        # of the probability matrices the comparison figures read.
        assert dataset.labels == [c for c in CLASS_ORDER if c in set(dataset.labels)]

    def test_the_matrices_are_float32(self, cicids_mini_csv: Path) -> None:
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, val_df, test_df = split_dataset_temporal(frame, FlowConfig())

        dataset = _build_matrices(train_df, val_df, test_df)

        assert dataset.X_train.dtype == np.float32
        assert dataset.X_val.dtype == np.float32
        assert dataset.X_test.dtype == np.float32


# ============================================================================
# Import hygiene
# ============================================================================


def test_train_cicids_does_not_import_matplotlib() -> None:
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale while installing only
    ``requirements.txt``. matplotlib and seaborn live in
    ``requirements-dev.txt``, so a module-level figure import anywhere under
    ``ml/`` breaks the production image. Every ``ml.evaluation.plots`` import in
    the CLI is therefore inside an ``if plots:`` / ``if eda:`` branch, and this
    is what keeps it that way.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.training.train_cicids, sys; assert 'matplotlib' not in sys.modules",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_importing_the_cli_does_not_pull_in_the_experimental_cascade() -> None:
    """
    ``CascadeMLPPipeline`` is the reference implementation — F1-macro 0.7508
    against 0.9516, never served, excluded from coverage — and
    ``ml.flow.pipelines`` deliberately does not re-export it.

    The CLI can build it, so it imports it inside ``_make_pipeline`` rather than
    at module scope. Two things ride on that: a ``--pipeline xgb`` run does not
    pay for 800 lines of hand-written NumPy MLP, and importing the CLI does not
    attach ``cascade_mlp`` to the package that refuses to advertise it —
    which is what
    ``test_flow_pipelines.py::test_pipelines_package_does_not_re_export_the_experimental_cascade``
    asserts, in-process, for the whole session.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.training.train_cicids;"
            "import ml.flow.pipelines as p;"
            "assert 'cascade_mlp' not in p.__dict__",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


# ============================================================================
# End-to-end CLI runs
# ============================================================================

# The real 78-column contract, written to disk so that a cascade trained by the
# CLI is loadable by the serving path. The mini fixture carries three columns,
# which is enough for the loader and the split but cannot exercise the feature
# contract CascadeInferenceService validates.
_ROUNDTRIP_FILES: tuple[tuple[str, str], ...] = (
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "FTP-Patator"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "DoS Hulk"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS"),
)

_ROUNDTRIP_BENIGN_ROWS: int = 110
_ROUNDTRIP_ATTACK_ROWS: int = 220


@pytest.fixture
def cicids_full_width_csv(tmp_path: Path) -> Path:
    """
    A miniature dataset over the genuine 78 columns, in canonical order.

    Written from :data:`FLOW_FEATURE_NAMES` rather than read from
    ``api/csvs/``: the real captures are 844 MB and would turn a unit test into
    a minutes-long one. The header carries both ``Fwd Header Length`` and
    ``Fwd Header Length.1`` — the duplicated CICFlowMeter column as pandas
    renders it — so the frame the loader produces matches the contract exactly.
    """
    data_dir = tmp_path / "cicids-full-width"
    data_dir.mkdir()

    for seed, (filename, raw_label) in enumerate(_ROUNDTRIP_FILES):
        rng = np.random.default_rng(2000 + seed)

        def block(
            n: int, label: str, centre: float, gen: np.random.Generator = rng
        ) -> pd.DataFrame:
            frame = pd.DataFrame(
                gen.normal(centre, 1.0, size=(n, len(FLOW_FEATURE_NAMES))),
                columns=list(FLOW_FEATURE_NAMES),
            )
            frame["Label"] = label
            return frame

        # Benign lead-in, attack window, benign tail: the shape of the real
        # captures, and what gives the temporal split a meaningful "earlier".
        pd.concat(
            [
                block(_ROUNDTRIP_BENIGN_ROWS, "BENIGN", 0.0),
                block(_ROUNDTRIP_ATTACK_ROWS, raw_label, 9.0 + 3.0 * seed),
                block(_ROUNDTRIP_BENIGN_ROWS, "BENIGN", 0.0),
            ],
            ignore_index=True,
        ).to_csv(data_dir / filename, index=False, encoding="latin-1")

    return data_dir


def _invoke(data_dir: Path, output_dir: Path, *extra: str) -> dict:
    """Run the CLI and return the parsed run summary."""
    result = CliRunner().invoke(
        main,
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--pipeline",
            "xgb",
            "--skip-tuning",
            "--no-plots",
            "--no-eda",
            *extra,
        ],
    )
    if result.exit_code != 0:
        raise AssertionError(f"train_cicids failed: {result.output}\n{result.exception}")
    return json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))


@pytest.mark.slow
class TestFlowTrainingCli:
    """A real run of `python -m ml.training.train_cicids`, small enough to be a test."""

    def test_temporal_run_writes_a_complete_cascade(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "flow-temporal"
        summary = _invoke(cicids_mini_csv, output_dir, "--split", "temporal")

        cascade = output_dir / "cascade_xgb"
        assert (cascade / FEATURE_NAMES_FILE).exists()
        assert (cascade / "isolation_forest" / "isolation_forest.joblib").exists()
        assert (cascade / "isolation_forest" / "if_threshold.npy").exists()
        assert (cascade / "isolation_forest" / "if_threshold_info.json").exists()
        assert (cascade / "xgboost" / "xgboost.joblib").exists()
        assert (cascade / "xgboost" / "xgb_label_encoder.joblib").exists()

        assert summary["split"]["strategy"] == "temporal"
        assert summary["model_paths"]["cascade_xgb"] == str(cascade)

    def test_the_summary_records_the_protocol_that_produced_the_metrics(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        The split strategy is the one field without which the metrics mean
        nothing: temporal and random are expected to disagree, and the delta
        between them is itself the result this track reports.
        """
        summary = _invoke(cicids_mini_csv, tmp_path / "flow-random", "--split", "random")

        assert summary["split"]["strategy"] == "random"
        # test_days belongs to day-holdout; reporting it here would read as if
        # a day had been held out.
        assert summary["split"]["test_days"] is None
        assert summary["parameters"]["pipelines"] == ["cascade_xgb"]
        assert summary["mlflow"] == {"requested": False, "logged": False}

        # Only the BENIGN/attack decision is asserted: the fixture draws every
        # attack class from the same distribution, so telling DDoS from
        # PortScan on it is impossible in principle.
        metrics = summary["metrics"]["cascade_xgb"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= summary["stage1_operating_point"]["cascade_xgb"]["test"]["recall"] <= 1.0

    def test_the_summary_separates_the_stage1_selection_point_from_the_test_one(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        Stage 1's threshold is the argmax of F-beta on validation, so the
        operating point measured there is a fit statistic. Recording it as if it
        were held-out performance understated the miss rate of the real run by
        more than an order of magnitude — and Stage 1's recall is the ceiling
        the whole cascade inherits. Both points are kept, and labelled.
        """
        summary = _invoke(cicids_mini_csv, tmp_path / "flow-op")

        operating_point = summary["stage1_operating_point"]["cascade_xgb"]
        assert set(operating_point) == {"selection", "test"}

        for point in operating_point.values():
            assert {"precision", "recall", "fpr", "fnr", "tp", "fp", "tn", "fn"} <= set(point)
            assert 0.0 <= point["recall"] <= 1.0

        # The two are measured on different sets, so their confusion matrices
        # must not be the same object read twice.
        selection, test = operating_point["selection"], operating_point["test"]
        assert selection["threshold"] == test["threshold"]
        assert sum(selection[k] for k in ("tp", "fp", "tn", "fn")) == summary["data"]["val_size"]
        assert sum(test[k] for k in ("tp", "fp", "tn", "fn")) == summary["data"]["test_size"]

        # What `metrics` publishes is the held-out figure, not the selection one.
        assert summary["metrics"]["cascade_xgb"]["if_recall"] == test["recall"]

    def test_day_holdout_records_the_days_it_held_out(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        summary = _invoke(
            cicids_mini_csv,
            tmp_path / "flow-holdout",
            "--split",
            "day-holdout",
            "--test-days",
            "friday",
        )

        assert summary["split"]["strategy"] == "day-holdout"
        assert summary["split"]["test_days"] == ["friday"]

    def test_what_the_cli_writes_the_serving_path_loads(
        self, cicids_full_width_csv: Path, tmp_path: Path
    ) -> None:
        """
        The circle closes here.

        ``CascadeInferenceService.load`` compares the persisted column list
        against ``FLOW_FEATURE_NAMES`` by length, content **and order**, and
        raises ``FeatureContractError`` on any of the three. A mismatch is the
        one failure in this design that would otherwise produce confident,
        plausible, wrong predictions with nothing raised anywhere — so this is
        the cheapest place to catch a column-order regression in the CLI.
        """
        output_dir = tmp_path / "flow-roundtrip"
        _invoke(cicids_full_width_csv, output_dir, "--split", "temporal")

        contract = json.loads(
            (output_dir / "cascade_xgb" / FEATURE_NAMES_FILE).read_text(encoding="utf-8")
        )
        assert contract["feature_names"] == list(FLOW_FEATURE_NAMES)

        service = CascadeInferenceService()
        service.load(output_dir / "cascade_xgb")

        assert service.is_loaded
        assert service.feature_names == list(FLOW_FEATURE_NAMES)

        prediction = service.predict({"features": dict.fromkeys(FLOW_FEATURE_NAMES, 0.0)})
        assert prediction["attack_type"]
        assert 0.0 <= prediction["threat_score"] <= 1.0
