"""
Unit tests for the CICIDS2017 loader and its temporal partitioning.

The two properties worth protecting here fail *silently* in production:

* ``capture_day`` / ``order_idx`` leaking into the feature matrix produces a
  model that detects the capture schedule and reports near-perfect metrics.
* A random split leaks near-duplicate flows from the same attack burst across
  train and test, which is exactly the optimistic bound the TFA reported.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import ml.flow.loader as loader_module
from ml.flow.config import FlowConfig
from ml.flow.loader import (
    CAPTURE_DAY_COL,
    LABEL_COL,
    META_COLUMNS,
    ORDER_IDX_COL,
    RAW_LABEL_COL,
    clean_dataset,
    filter_minority_classes,
    get_feature_columns,
    load_cicids2017,
    split_dataset,
    split_dataset_by_strategy,
    split_dataset_day_holdout,
    split_dataset_temporal,
    stratified_sample,
)
from tests.conftest import CICIDS_MINI_FEATURES, CICIDS_MINI_MULTI_TOOL_CLASSES

# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def mini_df(cicids_mini_csv: Path) -> pd.DataFrame:
    """The mini dataset, loaded and cleaned — the usual starting point."""
    return clean_dataset(load_cicids2017(cicids_mini_csv))


def _blocks(df: pd.DataFrame, label_col: str = RAW_LABEL_COL) -> set[tuple[str, str]]:
    """
    The set of ``(capture_day, label)`` blocks present in a frame.

    Defaults to the **raw** label, because that is what the default
    ``temporal_block="sub-label"`` protocol actually cuts on. Blocking by the
    canonical class is a coarser, different partition — the one that holds whole
    attack tools out — so the tests that mean *that* one pass ``LABEL_COL``
    explicitly.
    """
    return set(
        map(tuple, df[[CAPTURE_DAY_COL, label_col]].astype(str).drop_duplicates().to_numpy())
    )


def _order_in_block(df: pd.DataFrame, day: str, label: str, label_col: str) -> pd.Series:
    """The ``order_idx`` values of one block within one split."""
    mask = (df[CAPTURE_DAY_COL].astype(str) == day) & (df[label_col].astype(str) == label)
    return df.loc[mask, ORDER_IDX_COL]


class _LogSpy:
    """
    Records structlog events emitted by the loader.

    Neither ``caplog`` nor ``structlog.testing.capture_logs`` is reliable here:
    ``api.middleware.logging`` configures structlog with
    ``cache_logger_on_first_use=True``, so whether the loader's events reach the
    stdlib tree depends on which test ran first. Swapping the module logger is
    order-independent.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def record(event: str, **kwargs) -> None:
            self.events.append((level, event, kwargs))

        return record

    def names(self, level: str | None = None) -> list[str]:
        return [name for lvl, name, _ in self.events if level is None or lvl == level]

    def payloads(self, event: str) -> list[dict]:
        return [kwargs for _, name, kwargs in self.events if name == event]


@pytest.fixture
def log_spy(monkeypatch: pytest.MonkeyPatch) -> _LogSpy:
    """Replace the loader's module logger with a recording stub."""
    spy = _LogSpy()
    monkeypatch.setattr(loader_module, "logger", spy)
    return spy


# ============================================================================
# Loading
# ============================================================================


class TestLoadCicids2017:
    """Reading the CSVs, mapping labels and stamping provenance."""

    def test_loads_every_csv(self, cicids_mini_csv: Path):
        df = load_cicids2017(cicids_mini_csv)
        assert len(df) > 0
        assert LABEL_COL in df.columns

    def test_column_names_are_stripped(self, cicids_mini_csv: Path):
        """The real CSVs ship a leading space in every header."""
        df = load_cicids2017(cicids_mini_csv)
        assert all(col == col.strip() for col in df.columns)
        for feature in CICIDS_MINI_FEATURES:
            assert feature in df.columns

    def test_labels_are_mapped_to_canonical_taxonomy(self, cicids_mini_csv: Path):
        df = load_cicids2017(cicids_mini_csv)
        assert set(df[LABEL_COL].unique()) == {
            "BENIGN",
            "BruteForce",
            "DoS",
            "WebAttack",
            "DDoS",
            "Scan",
        }

    def test_webattack_survives_the_mojibake(self, cicids_mini_csv: Path):
        """Without the latin-1 repair the whole class is discarded in silence."""
        df = load_cicids2017(cicids_mini_csv)
        assert (df[LABEL_COL] == "WebAttack").sum() > 0

    def test_unrecognised_labels_are_discarded(self, cicids_mini_csv: Path):
        df = load_cicids2017(cicids_mini_csv)
        assert "Some Future Attack" not in set(df[LABEL_COL].unique())

    def test_capture_day_comes_from_the_filename(self, cicids_mini_csv: Path):
        df = load_cicids2017(cicids_mini_csv)
        assert set(df[CAPTURE_DAY_COL].unique()) == {
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        }

    def test_order_idx_is_unique_and_monotonic_per_file(self, cicids_mini_csv: Path):
        """
        Ties would make the per-block chronological cut ambiguous: Thursday and
        Friday each span several files.
        """
        df = load_cicids2017(cicids_mini_csv)
        assert df[ORDER_IDX_COL].is_unique
        for _, group in df.groupby(CAPTURE_DAY_COL):
            assert group[ORDER_IDX_COL].is_monotonic_increasing

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_cicids2017(tmp_path / "nope")

    def test_directory_without_csv_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_cicids2017(empty)

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        target = tmp_path / "a-file.csv"
        target.write_text("Label\nBENIGN\n", encoding="latin-1")
        with pytest.raises(NotADirectoryError):
            load_cicids2017(target)

    def test_missing_label_column_raises(self, tmp_path: Path):
        data_dir = tmp_path / "no-label"
        data_dir.mkdir()
        pd.DataFrame({"Flow Duration": [1.0, 2.0]}).to_csv(
            data_dir / "Monday-WorkingHours.pcap_ISCX.csv", index=False
        )
        with pytest.raises(KeyError, match="Label"):
            load_cicids2017(data_dir)


# ============================================================================
# The feature-column guard — risk nº1 of the whole track
# ============================================================================


class TestGetFeatureColumns:
    """
    ``get_feature_columns`` is the single source of truth for the feature space.

    If ``capture_day`` or ``order_idx`` reaches the model, nothing raises: the
    model learns the capture schedule, reports excellent metrics, and is
    worthless on live traffic.
    """

    def test_metadata_columns_are_excluded(self, mini_df: pd.DataFrame):
        feature_cols = get_feature_columns(mini_df)
        assert CAPTURE_DAY_COL not in feature_cols
        assert ORDER_IDX_COL not in feature_cols
        assert LABEL_COL not in feature_cols

    def test_every_meta_column_is_excluded(self, mini_df: pd.DataFrame):
        assert not set(get_feature_columns(mini_df)) & set(META_COLUMNS)

    def test_raw_label_is_not_a_feature(self, mini_df: pd.DataFrame):
        """
        The newest provenance column and the most dangerous of the four:
        ``capture_day`` is merely a near-perfect proxy for the label, while
        ``raw_label`` *is* the label, uncollapsed. A model handed it scores a
        flawless 1.0 and has learned nothing.
        """
        assert RAW_LABEL_COL in mini_df.columns
        assert RAW_LABEL_COL in META_COLUMNS
        assert RAW_LABEL_COL not in get_feature_columns(mini_df)

    def test_real_features_are_kept_in_frame_order(self, mini_df: pd.DataFrame):
        assert get_feature_columns(mini_df) == list(CICIDS_MINI_FEATURES)

    def test_survives_every_stage_of_the_pipeline(self, cicids_mini_csv: Path):
        """The guard must hold after cleaning, filtering, sampling and splitting."""
        df = clean_dataset(load_cicids2017(cicids_mini_csv))
        df, _ = filter_minority_classes(df, FlowConfig(min_class_samples=1, min_class_ratio=0.0))
        df = stratified_sample(df, FlowConfig(sample_size=400))
        train, val, test = split_dataset_temporal(df, FlowConfig())

        for stage in (df, train, val, test):
            assert get_feature_columns(stage) == list(CICIDS_MINI_FEATURES)

    def test_feature_matrix_is_numeric(self, mini_df: pd.DataFrame):
        """A leaked capture_day would blow up here — the loud half of the risk."""
        matrix = mini_df[get_feature_columns(mini_df)].to_numpy(dtype=np.float32)
        assert matrix.shape == (len(mini_df), len(CICIDS_MINI_FEATURES))
        assert np.isfinite(matrix).all()


# ============================================================================
# Cleaning
# ============================================================================


class TestCleanDataset:
    """Numeric artifacts, duplicates and dtype coercion."""

    def test_inf_and_nan_rows_are_dropped(self, cicids_mini_csv: Path):
        raw = load_cicids2017(cicids_mini_csv)
        feature_cols = get_feature_columns(raw)
        assert not np.isfinite(raw[feature_cols].to_numpy(dtype=np.float64)).all()

        cleaned = clean_dataset(raw)
        assert np.isfinite(cleaned[get_feature_columns(cleaned)].to_numpy(dtype=np.float64)).all()

    def test_features_are_cast_to_float32(self, mini_df: pd.DataFrame):
        for col in get_feature_columns(mini_df):
            assert mini_df[col].dtype == np.float32

    def test_provenance_columns_survive(self, mini_df: pd.DataFrame):
        assert CAPTURE_DAY_COL in mini_df.columns
        assert ORDER_IDX_COL in mini_df.columns

    def test_duplicates_are_dropped_ignoring_order_idx(self):
        """
        Regression guard for the subtlest bug this module can have.

        ``order_idx`` is unique by construction, so a plain ``drop_duplicates()``
        over all columns would compare it too and never drop anything — the
        deduplication would become a silent no-op the day it was needed most.
        """
        df = pd.DataFrame(
            {
                "Flow Duration": [1.0, 1.0, 2.0],
                "Total Fwd Packets": [3.0, 3.0, 4.0],
                LABEL_COL: ["BENIGN", "BENIGN", "DoS"],
                CAPTURE_DAY_COL: ["monday", "monday", "monday"],
                ORDER_IDX_COL: [0, 1, 2],
            }
        )
        cleaned = clean_dataset(df)
        assert len(cleaned) == 2

    def test_empty_frame_is_handled(self):
        df = pd.DataFrame(
            {
                "Flow Duration": pd.Series(dtype=float),
                LABEL_COL: pd.Series(dtype=str),
                CAPTURE_DAY_COL: pd.Series(dtype=str),
                ORDER_IDX_COL: pd.Series(dtype="int64"),
            }
        )
        assert len(clean_dataset(df)) == 0

    def test_massive_nan_loss_is_flagged(self, log_spy: _LogSpy):
        """
        Losing most of the dataset to NaN means the CSVs were read with the
        wrong encoding — a silent 50% data loss otherwise just looks like a
        small dataset.
        """
        n = 100
        values = np.arange(n, dtype=float)
        values[: n // 2] = np.nan
        df = pd.DataFrame(
            {
                "Flow Duration": values,
                LABEL_COL: ["BENIGN"] * n,
                CAPTURE_DAY_COL: ["monday"] * n,
                ORDER_IDX_COL: np.arange(n),
            }
        )
        cleaned = clean_dataset(df)
        assert len(cleaned) == n // 2
        assert "nan_inf_drop_ratio_suspicious" in log_spy.names("warning")


# ============================================================================
# Minority filtering and sampling
# ============================================================================


class TestFilterMinorityClasses:
    """Absolute and relative frequency thresholds."""

    def test_nothing_removed_with_permissive_thresholds(self, mini_df: pd.DataFrame):
        config = FlowConfig(min_class_samples=1, min_class_ratio=0.0)
        filtered, removed = filter_minority_classes(mini_df, config)
        assert removed == []
        assert len(filtered) == len(mini_df)

    def test_absolute_threshold_removes_rare_classes(self, mini_df: pd.DataFrame):
        config = FlowConfig(min_class_samples=10_000, min_class_ratio=0.0)
        filtered, removed = filter_minority_classes(mini_df, config)
        assert set(removed) == set(mini_df[LABEL_COL].unique())
        assert len(filtered) == 0

    def test_relative_threshold_removes_rare_classes(self, mini_df: pd.DataFrame):
        config = FlowConfig(min_class_samples=1, min_class_ratio=0.30)
        _, removed = filter_minority_classes(mini_df, config)
        # BENIGN is the majority (5 lead-ins + 5 tails); attacks are ~13% each.
        assert "BENIGN" not in removed
        assert removed

    def test_default_config_is_used_when_none(self, mini_df: pd.DataFrame):
        filtered, _ = filter_minority_classes(mini_df)
        assert isinstance(filtered, pd.DataFrame)


class TestStratifiedSample:
    """Proportional sampling that must not destroy the temporal metadata."""

    def test_skipped_when_dataset_already_fits(self, mini_df: pd.DataFrame):
        out = stratified_sample(mini_df, FlowConfig(sample_size=10_000_000))
        assert len(out) == len(mini_df)

    def test_skipped_when_sample_size_is_zero(self, mini_df: pd.DataFrame):
        out = stratified_sample(mini_df, FlowConfig(sample_size=0))
        assert len(out) == len(mini_df)

    def test_target_size_is_reached(self, mini_df: pd.DataFrame):
        out = stratified_sample(mini_df, FlowConfig(sample_size=500))
        assert len(out) == 500

    def test_order_idx_survives_the_shuffle(self, mini_df: pd.DataFrame):
        """
        The final shuffle reorders rows; the recorded position must not move,
        or the downstream split silently stops being temporal.
        """
        out = stratified_sample(mini_df, FlowConfig(sample_size=500))
        assert ORDER_IDX_COL in out.columns
        assert out[ORDER_IDX_COL].is_unique
        assert set(out[ORDER_IDX_COL]).issubset(set(mini_df[ORDER_IDX_COL]))

    def test_all_classes_survive_sampling(self, mini_df: pd.DataFrame):
        out = stratified_sample(mini_df, FlowConfig(sample_size=500))
        assert set(out[LABEL_COL].unique()) == set(mini_df[LABEL_COL].unique())

    def test_is_reproducible(self, mini_df: pd.DataFrame):
        a = stratified_sample(mini_df, FlowConfig(sample_size=500))
        b = stratified_sample(mini_df, FlowConfig(sample_size=500))
        assert sorted(a[ORDER_IDX_COL]) == sorted(b[ORDER_IDX_COL])

    def test_rounding_shortfall_is_topped_up(self):
        """
        Per-class proportional sampling rounds down; without the top-up the
        caller silently gets fewer rows than the ``sample_size`` it asked for.
        """
        n = 30
        df = pd.DataFrame(
            {
                "Flow Duration": np.arange(n, dtype=float),
                LABEL_COL: ["BENIGN"] * 10 + ["DoS"] * 10 + ["DDoS"] * 10,
                CAPTURE_DAY_COL: ["monday"] * n,
                ORDER_IDX_COL: np.arange(n),
            }
        )
        # 7/30 per class rounds to 2 rows each, i.e. 6 — one short.
        out = stratified_sample(df, FlowConfig(sample_size=7))
        assert len(out) == 7
        assert out[ORDER_IDX_COL].is_unique

    def test_the_top_up_is_stratified_too(self):
        """
        Drawn flat, the rows closing the rounding gap come back at the dataset's
        ~80% BENIGN rate — a small but gratuitous dent in a function whose whole
        job is proportional sampling. The allocation is checked directly because
        through ``stratified_sample`` the top-up is a handful of rows and a flat
        draw would usually pass by luck.
        """
        remaining = pd.DataFrame(
            {
                "Flow Duration": np.arange(100, dtype=float),
                LABEL_COL: ["BENIGN"] * 80 + ["DoS"] * 10 + ["DDoS"] * 10,
                CAPTURE_DAY_COL: ["monday"] * 100,
                ORDER_IDX_COL: np.arange(100),
            }
        )
        topped = loader_module._proportional_topup(remaining, 10, seed=42)

        assert len(topped) == 10
        assert topped[LABEL_COL].value_counts().to_dict() == {"BENIGN": 8, "DoS": 1, "DDoS": 1}

    def test_the_top_up_cannot_ask_for_more_rows_than_exist(self):
        remaining = pd.DataFrame(
            {
                "Flow Duration": [1.0, 2.0],
                LABEL_COL: ["BENIGN", "DoS"],
                CAPTURE_DAY_COL: ["monday"] * 2,
                ORDER_IDX_COL: [0, 1],
            }
        )
        assert len(loader_module._proportional_topup(remaining, 50, seed=42)) == 2


# ============================================================================
# Temporal partitioning — the centre of the plan
# ============================================================================


class TestSplitDatasetTemporal:
    """The 70/15/15 chronological cut inside each (capture_day, class) block."""

    def test_all_classes_present_in_all_three_splits(self, mini_df: pd.DataFrame):
        """
        This is what a naive global positional cut destroys: on the real
        captures the last decile of a file is 100% BENIGN, so test would end up
        without a single attack.
        """
        train, val, test = split_dataset_temporal(mini_df)
        classes = set(mini_df[LABEL_COL].unique())
        for split in (train, val, test):
            assert set(split[LABEL_COL].unique()) == classes

    def test_all_blocks_present_in_all_three_splits(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_temporal(mini_df)
        expected = _blocks(mini_df)
        for split in (train, val, test):
            assert _blocks(split) == expected

    def test_train_strictly_precedes_val_and_test_inside_each_block(self, mini_df: pd.DataFrame):
        """
        No future→past leakage within a scenario. The whole point.

        The guarantee is *per block*, and under the default protocol a block is
        a ``(day, raw_label)`` pair. It deliberately does **not** hold per
        canonical class: Tuesday runs FTP-Patator and then SSH-Patator, both
        ``BruteForce``, so SSH-Patator's training rows come after FTP-Patator's
        test rows. That interleaving is the point of blocking by sub-label —
        every tool is cut on its own timeline — and
        ``test_class_level_order_is_deliberately_not_preserved`` pins it.
        """
        train, val, test = split_dataset_temporal(mini_df)

        for day, label in _blocks(mini_df):

            def _order(split, day=day, label=label):
                return _order_in_block(split, day, label, RAW_LABEL_COL)

            train_idx, val_idx, test_idx = _order(train), _order(val), _order(test)
            assert train_idx.max() < val_idx.min()
            assert val_idx.max() < test_idx.min()
            assert train_idx.max() < test_idx.min()

    def test_ratios_are_respected_within_one_row(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_temporal(mini_df)
        n = len(mini_df)
        assert len(train) + len(val) + len(test) == n

        n_blocks = len(_blocks(mini_df))
        # Per-block flooring can shift each block by at most one row.
        assert abs(len(train) - 0.70 * n) <= n_blocks
        assert abs(len(val) - 0.15 * n) <= n_blocks
        assert abs(len(test) - 0.15 * n) <= n_blocks

    def test_single_block_ratios_are_exact(self):
        """
        With one block and a round row count the cut is exactly 70/15/15. The
        block is deliberately over the small-block warning threshold, so this
        also exercises the quiet path.
        """
        n = 4000
        df = pd.DataFrame(
            {
                "Flow Duration": np.arange(n, dtype=float),
                LABEL_COL: ["DoS"] * n,
                CAPTURE_DAY_COL: ["wednesday"] * n,
                ORDER_IDX_COL: np.arange(n),
            }
        )
        train, val, test = split_dataset_temporal(df)
        assert (len(train), len(val), len(test)) == (2800, 600, 600)

    def test_no_row_is_lost_or_duplicated(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_temporal(mini_df)
        seen = pd.concat([train, val, test])[ORDER_IDX_COL]
        assert seen.is_unique
        assert set(seen) == set(mini_df[ORDER_IDX_COL])

    def test_custom_ratios_are_honoured(self, mini_df: pd.DataFrame):
        config = FlowConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        train, val, test = split_dataset_temporal(mini_df, config)
        n_blocks = len(_blocks(mini_df))
        assert abs(len(train) - 0.50 * len(mini_df)) <= n_blocks
        assert abs(len(val) - 0.25 * len(mini_df)) <= n_blocks

    def test_tiny_block_goes_wholesale_to_train(self):
        """
        Better a class missing from test than a class present *only* in test:
        the second case yields a per-class F1 of zero that looks like a modelling
        failure instead of a data problem.
        """
        df = pd.DataFrame(
            {
                "Flow Duration": [1.0, 2.0],
                LABEL_COL: ["Botnet", "Botnet"],
                CAPTURE_DAY_COL: ["friday", "friday"],
                ORDER_IDX_COL: [0, 1],
            }
        )
        train, val, test = split_dataset_temporal(df)
        assert len(train) == 2
        assert len(val) == 0
        assert len(test) == 0

    def test_three_row_block_yields_one_row_each(self):
        df = pd.DataFrame(
            {
                "Flow Duration": [1.0, 2.0, 3.0],
                LABEL_COL: ["Botnet"] * 3,
                CAPTURE_DAY_COL: ["friday"] * 3,
                ORDER_IDX_COL: [0, 1, 2],
            }
        )
        train, val, test = split_dataset_temporal(df)
        assert (len(train), len(val), len(test)) == (1, 1, 1)

    def test_small_blocks_are_reported(self, mini_df: pd.DataFrame, log_spy: _LogSpy):
        """
        Botnet lives on one day with ~2.000 rows, so its test slice is ~300
        samples. That is a caveat on the per-class F1, and it has to surface in
        the run log rather than as an unexplained number in the report.
        """
        split_dataset_temporal(mini_df)
        warned = log_spy.payloads("temporal_block_small")
        assert warned
        assert all(payload["n_rows"] < payload["threshold"] for payload in warned)

    def test_missing_provenance_columns_raise(self, mini_df: pd.DataFrame):
        stripped = mini_df.drop(columns=[ORDER_IDX_COL])
        with pytest.raises(KeyError, match=ORDER_IDX_COL):
            split_dataset_temporal(stripped)

    def test_is_reproducible(self, mini_df: pd.DataFrame):
        first = split_dataset_temporal(mini_df)
        second = split_dataset_temporal(mini_df)
        for a, b in zip(first, second, strict=True):
            assert list(a[ORDER_IDX_COL]) == list(b[ORDER_IDX_COL])


# ============================================================================
# The block key — what the temporal cut is actually cutting
# ============================================================================


class TestTemporalBlockKey:
    """
    Pins the conceptual error this module shipped, and its fix.

    ``ATTACK_GROUPS`` collapses the raw CICIDS2017 labels into canonical
    classes, and the tools behind one class ran **one after the other** inside
    their capture day. Cutting chronologically on the canonical class therefore
    trains on the early tool and tests on the late one — a novel-attack-tool
    holdout wearing the costume of a leakage fix, and worth 0.47 of F1-macro on
    the real captures.

    The fixture reproduces the mechanism at small scale: Tuesday runs
    FTP-Patator then SSH-Patator (both ``BruteForce``), Wednesday runs DoS Hulk
    then DoS GoldenEye (both ``DoS``). See
    :data:`tests.conftest.CICIDS_MINI_FILES` and docs/ml/PARTICION-TEMPORAL.md.
    """

    # 50/25/25 rather than the default 70/15/15: the fixture splits its attack
    # window evenly between two tools, so a cut at the halfway mark lands
    # exactly on the handover and the two protocols separate completely. Under
    # 70/15/15 the second tool merely *dominates* test instead of owning it,
    # which is the same defect measured less sharply.
    RATIOS: ClassVar[dict[str, float]] = {
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
    }

    def _tools(self, df: pd.DataFrame, canonical_class: str) -> set[str]:
        """The raw attack tools present for one canonical class."""
        return set(df.loc[df[LABEL_COL] == canonical_class, RAW_LABEL_COL].astype(str))

    @pytest.mark.parametrize("canonical_class", CICIDS_MINI_MULTI_TOOL_CLASSES)
    def test_sub_label_blocking_keeps_every_tool_in_all_three_splits(
        self, mini_df: pd.DataFrame, canonical_class: str
    ):
        """The default protocol: each tool is cut on its own timeline."""
        config = FlowConfig(temporal_block="sub-label", **self.RATIOS)
        train, val, test = split_dataset_temporal(mini_df, config)

        expected = self._tools(mini_df, canonical_class)
        assert len(expected) > 1, "the fixture must supply more than one tool to be meaningful"
        for split in (train, val, test):
            assert self._tools(split, canonical_class) == expected

    @pytest.mark.parametrize("canonical_class", CICIDS_MINI_MULTI_TOOL_CLASSES)
    def test_class_blocking_holds_a_whole_tool_out_of_training(
        self, mini_df: pd.DataFrame, canonical_class: str
    ):
        """
        The same rows, the same chronological guarantee, one different key —
        and the test set is now made entirely of a tool that never appeared in
        training. Any per-class F1 measured this way answers "does it generalise
        to an unseen tool?", not "does it detect what it was trained on?".
        """
        config = FlowConfig(temporal_block="class", **self.RATIOS)
        train, _, test = split_dataset_temporal(mini_df, config)

        in_train = self._tools(train, canonical_class)
        in_test = self._tools(test, canonical_class)

        assert in_test, "test must still carry the class, or the comparison is vacuous"
        assert not (in_test & in_train), "the whole point: test's tool is absent from train"
        assert self._tools(mini_df, canonical_class) == in_train | in_test

    def test_the_two_protocols_partition_the_same_rows(self, mini_df: pd.DataFrame):
        """
        Blocking is a choice about *where* the line falls, not about which rows
        take part. Neither protocol drops or invents anything, which is what
        makes the F1-macro gap between them attributable to the key alone.
        """
        fine = split_dataset_temporal(
            mini_df, FlowConfig(temporal_block="sub-label", **self.RATIOS)
        )
        coarse = split_dataset_temporal(mini_df, FlowConfig(temporal_block="class", **self.RATIOS))

        assert set(pd.concat(fine)[ORDER_IDX_COL]) == set(mini_df[ORDER_IDX_COL])
        assert set(pd.concat(coarse)[ORDER_IDX_COL]) == set(mini_df[ORDER_IDX_COL])

    def test_the_block_key_is_recorded_in_the_split_log(
        self, mini_df: pd.DataFrame, log_spy: _LogSpy
    ):
        """
        A run log that does not say which key was used cannot be read back: the
        same F1-macro means two different things under the two protocols.
        """
        split_dataset_temporal(mini_df, FlowConfig(temporal_block="class"))

        payload = log_spy.payloads("dataset_split")[0]
        assert payload["temporal_block"] == "class"
        assert payload["block_cols"] == [CAPTURE_DAY_COL, LABEL_COL]

    def test_sub_label_distribution_is_logged(self, mini_df: pd.DataFrame, log_spy: _LogSpy):
        """
        The breakdown that makes the defect visible from a log alone: a healthy
        looking ``BruteForce`` line hides FTP-Patator test=0 underneath it.
        """
        split_dataset_temporal(mini_df, FlowConfig(temporal_block="class", **self.RATIOS))

        logged = {payload["label"] for payload in log_spy.payloads("split_sublabel_distribution")}
        assert {"FTP-Patator", "SSH-Patator"} <= logged

        starved = [
            payload
            for payload in log_spy.payloads("split_sublabel_distribution")
            if payload["label"] == "FTP-Patator"
        ]
        assert starved and starved[0]["test"] == 0

    def test_a_frame_without_raw_label_falls_back_to_class_blocking(self, log_spy: _LogSpy):
        """
        Frames assembled by hand — as several tests here do — carry no
        ``raw_label``. Refusing to split them would be a gratuitous break, and
        silently splitting them would hide that a coarser key was used, so the
        fallback is taken *and* announced.
        """
        n = 400
        df = pd.DataFrame(
            {
                "Flow Duration": np.arange(n, dtype=float),
                LABEL_COL: ["DoS"] * n,
                CAPTURE_DAY_COL: ["wednesday"] * n,
                ORDER_IDX_COL: np.arange(n),
            }
        )
        train, val, test = split_dataset_temporal(df, FlowConfig(temporal_block="sub-label"))

        assert "temporal_block_fallback_to_class" in log_spy.names("warning")
        assert len(train) + len(val) + len(test) == n

    def test_raw_label_keeps_distinct_tools_out_of_the_deduplication(self):
        """
        Two flows identical in every feature but produced by different tools are
        not the same flow. Collapsing them would quietly thin whichever tool ran
        second — the one the block key exists to protect.
        """
        df = pd.DataFrame(
            {
                "Flow Duration": [1.0, 1.0],
                LABEL_COL: ["DoS", "DoS"],
                RAW_LABEL_COL: ["DoS Hulk", "DoS GoldenEye"],
                CAPTURE_DAY_COL: ["wednesday", "wednesday"],
                ORDER_IDX_COL: [0, 1],
            }
        )
        assert len(clean_dataset(df)) == 2


# ============================================================================
# Random split — kept only to quantify the leakage
# ============================================================================


class TestSplitDatasetRandom:
    """The TFA's original protocol, and the leakage it introduces."""

    def test_returns_the_same_shape_of_tuple(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset(mini_df)
        assert len(train) + len(val) + len(test) == len(mini_df)

    def test_ratios_are_approximately_respected(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset(mini_df)
        n = len(mini_df)
        assert abs(len(train) / n - 0.70) < 0.02
        assert abs(len(val) / n - 0.15) < 0.02
        assert abs(len(test) / n - 0.15) < 0.02

    def test_leaks_future_into_train_unlike_the_temporal_split(self, mini_df: pd.DataFrame):
        """
        Demonstrates the defect being corrected: with a random split, train rows
        routinely come *after* test rows inside the same block.
        """
        train, _, test = split_dataset(mini_df)
        leaks = 0
        for day, label in _blocks(mini_df, LABEL_COL):
            tr = _order_in_block(train, day, label, LABEL_COL)
            te = _order_in_block(test, day, label, LABEL_COL)
            if len(tr) and len(te) and tr.max() > te.min():
                leaks += 1
        assert leaks > 0


# ============================================================================
# Day holdout — secondary protocol
# ============================================================================


class TestSplitDatasetDayHoldout:
    """Train on Mon-Thu, test on Fri: honest, but blind to the Friday classes."""

    def test_test_set_holds_only_the_held_out_days(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_day_holdout(mini_df, test_days=("friday",))
        assert set(test[CAPTURE_DAY_COL].unique()) == {"friday"}
        assert "friday" not in set(train[CAPTURE_DAY_COL].unique())
        assert "friday" not in set(val[CAPTURE_DAY_COL].unique())

    def test_friday_only_classes_have_no_training_samples(self, mini_df: pd.DataFrame):
        """The known, documented weakness of this protocol."""
        train, _, test = split_dataset_day_holdout(mini_df, test_days=("friday",))
        assert {"DDoS", "Scan"}.issubset(set(test[LABEL_COL].unique()))
        assert not {"DDoS", "Scan"} & set(train[LABEL_COL].unique())

    def test_validation_is_carved_out_of_the_training_days(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_day_holdout(mini_df, test_days=("friday",))
        assert len(val) > 0
        assert len(train) + len(val) + len(test) == len(mini_df)

    def test_untrainable_classes_are_logged(self, mini_df: pd.DataFrame, log_spy: _LogSpy):
        """
        The protocol is only honest if the reader knows which classes it could
        not possibly have learned. Reporting a per-class F1 of 0 for DDoS
        without that context is a lie about the model.
        """
        split_dataset_day_holdout(mini_df, test_days=("friday",))
        payloads = log_spy.payloads("day_holdout_classes_absent_from_train")
        assert payloads
        assert {"DDoS", "Scan"}.issubset(set(payloads[0]["labels"]))

    def test_day_matching_is_case_insensitive(self, mini_df: pd.DataFrame):
        _, _, test = split_dataset_day_holdout(mini_df, test_days=("FRIDAY",))
        assert set(test[CAPTURE_DAY_COL].unique()) == {"friday"}

    def test_unknown_day_raises(self, mini_df: pd.DataFrame):
        with pytest.raises(ValueError, match="No rows for test_days"):
            split_dataset_day_holdout(mini_df, test_days=("caturday",))

    def test_holding_out_every_day_raises(self, mini_df: pd.DataFrame):
        every_day = tuple(mini_df[CAPTURE_DAY_COL].unique())
        with pytest.raises(ValueError, match="whole dataset"):
            split_dataset_day_holdout(mini_df, test_days=every_day)


# ============================================================================
# Strategy dispatch
# ============================================================================


class TestSplitDatasetByStrategy:
    """The entry point the training CLI will call."""

    def test_default_strategy_is_temporal(self, mini_df: pd.DataFrame):
        dispatched = split_dataset_by_strategy(mini_df)
        direct = split_dataset_temporal(mini_df)
        for a, b in zip(dispatched, direct, strict=True):
            assert list(a[ORDER_IDX_COL]) == list(b[ORDER_IDX_COL])

    def test_random_strategy_is_dispatched(self, mini_df: pd.DataFrame):
        train, val, test = split_dataset_by_strategy(mini_df, FlowConfig(split_strategy="random"))
        assert len(train) + len(val) + len(test) == len(mini_df)

    def test_day_holdout_strategy_is_dispatched(self, mini_df: pd.DataFrame):
        _, _, test = split_dataset_by_strategy(
            mini_df, FlowConfig(split_strategy="day-holdout"), test_days=("friday",)
        )
        assert set(test[CAPTURE_DAY_COL].unique()) == {"friday"}

    def test_unknown_strategy_raises(self, mini_df: pd.DataFrame):
        config = FlowConfig()
        config.split_strategy = "made-up"  # bypasses __post_init__ validation
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            split_dataset_by_strategy(mini_df, config)


# ============================================================================
# Import hygiene
# ============================================================================


def test_loader_does_not_import_matplotlib():
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale while installing only
    ``requirements.txt``. A module-level matplotlib import anywhere in the chain
    breaks the production image, or bloats it by ~120 MB.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.flow.loader, sys; assert 'matplotlib' not in sys.modules",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
