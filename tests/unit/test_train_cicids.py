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
    MIN_MEANINGFUL_BLOCK_ROWS,
    ORDER_IDX_COL,
    clean_dataset,
    load_cicids2017,
    split_dataset_by_strategy,
    split_dataset_temporal,
)
from ml.flow.pipelines.base import FEATURE_NAMES_FILE
from ml.flow.serving import CascadeInferenceService
from ml.training.train_cicids import _build_config, _build_matrices, main
from tests.conftest import CICIDS_RARE_CLASS_LABEL

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
            "class_sample_floor": 0,
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
        assert config.class_sample_floor == 0
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

    def test_the_class_sample_floor_reaches_the_config(self) -> None:
        assert self._config(class_sample_floor=2000).class_sample_floor == 2000

    def test_the_cli_default_leaves_sampling_strictly_proportional(self) -> None:
        """
        The floor is opt-in so that a default invocation reproduces the figures
        in docs/ml/PARTICION-TEMPORAL.md. The CLI default is read off
        ``FlowConfig``, so the two cannot drift apart — this pins the value they
        share.
        """
        floor = next(param for param in main.params if param.name == "class_sample_floor").default
        assert floor == 0 == FlowConfig().class_sample_floor


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

        dataset = _build_matrices(train_df, val_df, test_df, FlowConfig())

        assert CAPTURE_DAY_COL not in dataset.feature_names
        assert ORDER_IDX_COL not in dataset.feature_names
        assert "Label" not in dataset.feature_names
        assert dataset.X_train.shape[1] == len(dataset.feature_names)

    def test_labels_follow_the_canonical_order(self, cicids_mini_csv: Path) -> None:
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, val_df, test_df = split_dataset_temporal(frame, FlowConfig())

        dataset = _build_matrices(train_df, val_df, test_df, FlowConfig())

        assert set(dataset.labels) == {"BENIGN", "BruteForce", "DDoS", "DoS", "Scan", "WebAttack"}
        # In CLASS_ORDER order, not in the order the frames happened to list
        # them: this list becomes the column layout of the confusion matrix and
        # of the probability matrices the comparison figures read.
        assert dataset.labels == [c for c in CLASS_ORDER if c in set(dataset.labels)]

    def test_the_training_matrix_carries_a_temporal_index_aligned_with_it(
        self, cicids_mini_csv: Path
    ) -> None:
        """
        Stage 2's hyperparameter search cuts its folds from this index, and the
        only thing that ties it to ``X_train`` is that both were built from the
        same frame in the same row order.
        """
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        config = FlowConfig()
        train_df, val_df, test_df = split_dataset_temporal(frame, config)

        dataset = _build_matrices(train_df, val_df, test_df, config)

        assert dataset.train_index is not None
        assert len(dataset.train_index) == dataset.X_train.shape[0]
        np.testing.assert_array_equal(dataset.train_index.order, train_df[ORDER_IDX_COL].to_numpy())

    def test_a_random_split_carries_no_temporal_index(self, cicids_mini_csv: Path) -> None:
        """
        A random split shuffles rows across every block, so "the chunk before
        this one" means nothing in it. The folds stay shuffled, deliberately.
        """
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        config = FlowConfig(split_strategy="random")
        train_df, val_df, test_df = split_dataset_by_strategy(frame, config)

        dataset = _build_matrices(train_df, val_df, test_df, config)

        assert dataset.train_index is None

    def test_the_matrices_are_float32(self, cicids_mini_csv: Path) -> None:
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, val_df, test_df = split_dataset_temporal(frame, FlowConfig())

        dataset = _build_matrices(train_df, val_df, test_df, FlowConfig())

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


def _invoke(data_dir: Path, output_dir: Path, *extra: str, pipeline: str = "xgb") -> dict:
    """Run the CLI and return the parsed run summary."""
    result = CliRunner().invoke(
        main,
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--pipeline",
            pipeline,
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

    def test_the_summary_records_the_per_class_f1_and_support_of_every_cascade(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        A macro average hides exactly the failure this track has: one class at
        F1 0.00 costs 1/n of the headline and is invisible in it. The per-class
        block is kept out of ``metrics`` — that one is flat scalars, logged
        value by value — and carries the test support alongside every F1,
        because 0.00 on 24 rows and 0.00 on 24.000 are different results.
        """
        summary = _invoke(cicids_mini_csv, tmp_path / "flow-per-class")

        classes = summary["data"]["classes"]
        per_class = summary["per_class"]["cascade_xgb"]

        assert set(per_class) == {"f1", "support"}
        assert list(per_class["f1"]) == classes
        assert list(per_class["support"]) == classes
        assert sum(per_class["support"].values()) == summary["data"]["test_size"]

        # The headline is the unweighted mean of this table, which is what makes
        # the table an explanation of it rather than a second opinion.
        f1_macro = summary["metrics"]["cascade_xgb"]["f1_macro"]
        assert f1_macro == pytest.approx(float(np.mean(list(per_class["f1"].values()))))

    def test_the_summary_records_the_per_split_support_of_every_class(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        The per-class F1 table is only readable next to the row counts it was
        computed from. ``class_support`` is model-free and sits at the top level
        for that reason: it belongs to the run, not to a cascade.
        """
        summary = _invoke(cicids_mini_csv, tmp_path / "flow-support")

        support = summary["class_support"]
        assert list(support) == summary["data"]["classes"]

        for counts in support.values():
            assert set(counts) == {"train", "val", "test", "total"}
            assert counts["total"] == counts["train"] + counts["val"] + counts["test"]

        assert sum(c["train"] for c in support.values()) == summary["data"]["train_size"]
        assert sum(c["val"] for c in support.values()) == summary["data"]["val_size"]
        assert sum(c["test"] for c in support.values()) == summary["data"]["test_size"]

        # The supports recorded per cascade are the test column of this table:
        # one measurement, not two.
        per_class = summary["per_class"]["cascade_xgb"]["support"]
        assert per_class == {name: counts["test"] for name, counts in support.items()}

    def test_the_summary_records_whether_the_sampling_floor_was_on(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        The floor changes the BENIGN:attack ratio that Stage 1's threshold
        search sees, and that threshold serves every class — so two runs with
        the same protocol and different floors are not comparable, and the file
        has to say which one it was.
        """
        summary = _invoke(cicids_mini_csv, tmp_path / "flow-no-floor")
        assert summary["parameters"]["class_sample_floor"] == 0

    def test_the_floor_keeps_a_rare_class_off_the_floor_of_the_support_table(
        self, cicids_rare_class_csv: Path, tmp_path: Path
    ) -> None:
        """
        End to end on the Botnet-shaped fixture: a proportional draw grinds the
        rare class down, and `--class-sample-floor` is what stops it. Both runs
        are asserted from ``class_support`` alone, which is the point of the
        block — the per-class F1 of a ~10-row class is noise either way.
        """
        rare = CICIDS_RARE_CLASS_LABEL
        common = ["--sample-size", "1000"]

        proportional = _invoke(cicids_rare_class_csv, tmp_path / "flow-prop", *common)
        floored = _invoke(
            cicids_rare_class_csv,
            tmp_path / "flow-floor",
            *common,
            "--class-sample-floor",
            "60",
        )

        assert proportional["parameters"]["class_sample_floor"] == 0
        assert floored["parameters"]["class_sample_floor"] == 60

        assert proportional["class_support"][rare]["total"] < 60
        assert floored["class_support"][rare]["total"] == 60

        # It is a floor, not a rebalance: the classes above it stay where they
        # were, minus the seats the lift cost them.
        assert (
            floored["class_support"]["BENIGN"]["total"]
            < (proportional["class_support"]["BENIGN"]["total"])
        )

    def test_a_class_too_small_to_measure_is_reported_and_kept(
        self, cicids_rare_class_csv: Path, tmp_path: Path
    ) -> None:
        """
        Dropping the class would lift the headline F1-macro by deleting the
        evidence for it, and Botnet is served — ml/flow/fusion.py maps
        credential_stuffing onto it. So it stays in ``data.classes``, in the
        per-class table, and in the macro's denominator.
        """
        summary = _invoke(cicids_rare_class_csv, tmp_path / "flow-rare", "--sample-size", "1000")

        rare = CICIDS_RARE_CLASS_LABEL
        assert summary["class_support"][rare]["total"] < MIN_MEANINGFUL_BLOCK_ROWS
        assert rare in summary["data"]["classes"]
        assert rare in summary["per_class"]["cascade_xgb"]["f1"]

    def test_the_headline_f1_macro_and_the_comparison_f1_macro_are_computed_over_the_same_class_set(
        self, cicids_mini_csv: Path, tmp_path: Path
    ) -> None:
        """
        A ``--pipeline both`` run measures each cascade twice: ``evaluate`` fills
        ``metrics``, and ``compare_models`` fills ``comparison``. They used to
        disagree by construction — only the second passed the canonical class
        list — so the same model carried two F1-macros in one file with nothing
        saying which was the result. Now both average over ``data.classes``.

        Held out on friday so that the run declares six classes while the test
        set carries only friday's three. What that pins here is the denominator:
        an unsupported class still counts. It does not on its own pin the
        ``labels=`` argument — Stage 2 predicts the missing classes *somewhere*
        on this fixture, so the sklearn default union covers all six anyway. The
        arithmetic teeth for that live in
        ``test_flow_pipelines.py::…_scores_a_class_with_no_test_support_…``,
        where the class set is chosen rather than emergent.
        """
        summary = _invoke(
            cicids_mini_csv,
            tmp_path / "flow-both",
            "--split",
            "day-holdout",
            "--test-days",
            "friday",
            pipeline="both",
        )

        assert sorted(summary["metrics"]) == ["cascade_mlp", "cascade_xgb"]

        support = summary["per_class"]["cascade_xgb"]["support"]
        unsupported = [cls for cls, n in support.items() if n == 0]
        assert unsupported, "holding friday out must leave a declared class with no test rows"

        # The macro counts every declared class, so it is strictly below the
        # average over the classes that happen to have test rows — which is the
        # figure an unconstrained f1_score would have published.
        supported = [
            f1 for cls, f1 in summary["per_class"]["cascade_xgb"]["f1"].items() if support[cls]
        ]
        assert summary["metrics"]["cascade_xgb"]["f1_macro"] < float(np.mean(supported))

        # model_a is the XGBoost cascade, model_b the MLP — the order
        # _run_comparison passes them in.
        for name, side in (("cascade_xgb", "model_a"), ("cascade_mlp", "model_b")):
            for metric in ("accuracy", "f1_macro", "f1_weighted"):
                assert summary["metrics"][name][metric] == pytest.approx(
                    summary["comparison"][metric][side]
                ), f"{name}.{metric} disagrees between metrics and comparison"

            assert summary["per_class"][name]["f1"] == pytest.approx(
                {cls: row[side] for cls, row in summary["comparison"]["f1_per_class"].items()}
            )

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
