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
from ml.flow.labels import CLASS_ORDER
from ml.flow.loader import (
    CAPTURE_DAY_COL,
    LABEL_COL,
    META_COLUMNS,
    MIN_MEANINGFUL_BLOCK_ROWS,
    ORDER_IDX_COL,
    RAW_LABEL_COL,
    audit_class_support,
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
from tests.conftest import (
    CICIDS_MINI_FEATURES,
    CICIDS_MINI_MULTI_TOOL_CLASSES,
    CICIDS_RARE_CLASS_ATTACK,
    CICIDS_RARE_CLASS_LABEL,
)

# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def mini_df(cicids_mini_csv: Path) -> pd.DataFrame:
    """The mini dataset, loaded and cleaned — the usual starting point."""
    return clean_dataset(load_cicids2017(cicids_mini_csv))


@pytest.fixture
def rare_class_df(cicids_rare_class_csv: Path) -> pd.DataFrame:
    """
    The mini dataset plus a ~120-row ``Bot`` class, loaded and cleaned.

    The starting point for everything about ``class_sample_floor``: the rare
    class is large enough to clear ``min_class_samples`` and far too small to
    survive a proportional draw, which is Botnet's situation exactly.
    """
    return clean_dataset(load_cicids2017(cicids_rare_class_csv))


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
# Reading order — what makes order_idx chronological rather than alphabetical
# ============================================================================


def _write_capture_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    """
    Write a CICIDS2017-shaped CSV from explicit ``(raw_label, marker)`` rows.

    ``marker`` lands in ``Flow Duration`` and is what lets a test say which file
    a surviving row came from — the loader keeps no provenance beyond the capture
    day, and the whole point here is telling two files of the *same* day apart.
    Markers must be unique across the directory or ``clean_dataset`` would
    deduplicate rows away.
    """
    df = pd.DataFrame(
        {
            "Flow Duration": [marker for _, marker in rows],
            "Total Fwd Packets": [1.0 + i for i in range(len(rows))],
            "Flow Bytes/s": [marker * 10.0 for _, marker in rows],
            "Label": [label for label, _ in rows],
        }
    )
    df.columns = pd.Index([f" {c}" for c in df.columns])
    df.to_csv(path, index=False, encoding="latin-1")


class TestCaptureOrder:
    """
    ``order_idx`` counts rows in the order the files are read, so the reading
    order *is* the chronology. Sorting the filenames does not give it: Friday's
    two afternoon captures sort before its morning one, and Thursday inverts the
    same way. Attack blocks live in a single file each and never notice; the
    BENIGN blocks of Thursday and Friday span files and do — and BENIGN is the
    population Stage 1 is fitted on.
    """

    # Rank order, which for these five is Tue → Wed → Thu-AM → Fri-PM-DDos →
    # Fri-PM-PortScan. Alphabetically the two Friday files would come first.
    EXPECTED_MINI_ORDER: ClassVar[list[str]] = [
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ]

    def test_files_are_read_in_capture_order_not_alphabetical_order(
        self, cicids_mini_csv: Path, log_spy: _LogSpy
    ):
        load_cicids2017(cicids_mini_csv)

        payload = log_spy.payloads("cicids_load_started")[0]
        assert payload["file_order"] == self.EXPECTED_MINI_ORDER
        assert payload["file_order"] != sorted(
            self.EXPECTED_MINI_ORDER
        ), "the fixture must be able to tell the two orders apart"

    def test_order_idx_follows_the_capture_week(self, cicids_mini_csv: Path):
        """The days themselves must come out Tue < Wed < Thu < Fri."""
        df = load_cicids2017(cicids_mini_csv)
        first_seen = df.groupby(CAPTURE_DAY_COL, observed=True)[ORDER_IDX_COL].min()

        assert list(first_seen.sort_values().index) == [
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        ]

    def test_an_unrecognised_filename_sorts_last_without_breaking_the_load(
        self, tmp_path: Path, log_spy: _LogSpy
    ):
        """
        A directory the canonical ranking says nothing about still loads, and
        still loads deterministically: unknown names share the last rank and are
        tie-broken by name. Here the extra file sorts *first* alphabetically, so
        only the ranking can put Monday ahead of it.
        """
        data_dir = tmp_path / "extra-file"
        data_dir.mkdir()
        _write_capture_csv(
            data_dir / "Monday-WorkingHours.pcap_ISCX.csv",
            [("BENIGN", float(i)) for i in range(10)],
        )
        _write_capture_csv(
            data_dir / "Aardvark-WorkingHours.csv",
            [("BENIGN", 100.0 + i) for i in range(10)],
        )

        df = load_cicids2017(data_dir)

        assert log_spy.payloads("cicids_load_started")[0]["file_order"] == [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Aardvark-WorkingHours.csv",
        ]
        assert len(df) == 20
        monday = df.loc[df[CAPTURE_DAY_COL] == "monday", ORDER_IDX_COL]
        unknown = df.loc[df[CAPTURE_DAY_COL] == "unknown", ORDER_IDX_COL]
        assert monday.max() < unknown.min()

    def test_the_friday_benign_block_is_cut_with_the_morning_capture_first(self, tmp_path: Path):
        """
        The regression this fix exists for. ``(friday, BENIGN)`` spans three
        files; under the alphabetical glob the afternoon captures were numbered
        *before* the morning one, so the block's chronological cut ran backwards
        and Stage 1 was fitted on Friday's last hours to be tested on its first.

        Two files are enough to show it, and the assertion is directional: the
        morning rows belong in train, the late afternoon ones in test.
        """
        data_dir = tmp_path / "friday"
        data_dir.mkdir()
        # Markers below 500 are morning, above are afternoon.
        _write_capture_csv(
            data_dir / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            [("BENIGN", float(i)) for i in range(20)] + [("Bot", 200.0 + i) for i in range(10)],
        )
        _write_capture_csv(
            data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            [("BENIGN", 500.0 + i) for i in range(20)] + [("DDoS", 700.0 + i) for i in range(10)],
        )

        df = clean_dataset(load_cicids2017(data_dir))
        train, _, test = split_dataset_temporal(df)

        def benign_markers(split: pd.DataFrame) -> set[float]:
            return set(split.loc[split[LABEL_COL] == "BENIGN", "Flow Duration"].astype(float))

        morning = {float(i) for i in range(20)}
        assert morning <= benign_markers(train), "the morning capture is the earliest, so it trains"
        assert not (benign_markers(test) & morning), "no morning flow may land in test"
        assert benign_markers(test), "test must still carry BENIGN, or the check is vacuous"


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
# The per-class sampling floor — opt-in, and off by default on purpose
# ============================================================================


class TestClassSampleFloor:
    """
    ``class_sample_floor`` protects a rare class from proportional sampling.

    The class it exists for is Botnet: one sub-label, one day, ~1.966 rows out
    of ~2.8M, of which ``--sample-size 200000`` keeps ~155 and the split leaves
    ~24 in test. Its published F1 of 0.00 is measured on those 24 rows.
    """

    def test_the_floor_is_off_by_default(self):
        """
        Every figure in docs/ml/PARTICION-TEMPORAL.md was measured under strict
        proportionality. A floor switched on by default would move them without
        a line of the diff saying so.
        """
        assert FlowConfig().class_sample_floor == 0

    def test_the_default_config_never_reaches_the_floored_path(
        self, mini_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Stronger than comparing outputs: the published runs must go through the
        *same code* they always did, not through a floored path that happens to
        agree at floor 0.
        """

        def explode(*args, **kwargs):
            raise AssertionError("the floored path ran under the default config")

        monkeypatch.setattr(loader_module, "_floored_sample", explode)
        assert len(stratified_sample(mini_df, FlowConfig(sample_size=500))) == 500

    def test_a_negative_floor_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="class_sample_floor must be >= 0"):
            FlowConfig(class_sample_floor=-1)

    def test_a_rare_class_is_lifted_to_the_floor_instead_of_sampled_to_noise(
        self, rare_class_df: pd.DataFrame
    ):
        proportional = stratified_sample(rare_class_df, FlowConfig(sample_size=1_000))
        floored = stratified_sample(
            rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=60)
        )

        rare = CICIDS_RARE_CLASS_LABEL
        drawn = int((proportional[LABEL_COL] == rare).sum())
        lifted = int((floored[LABEL_COL] == rare).sum())

        assert drawn < 60 <= lifted
        assert lifted == 60

    def test_the_floor_is_capped_by_the_rows_the_class_actually_has(
        self, rare_class_df: pd.DataFrame
    ):
        """
        A floor is a request, not an invention: asking for more rows than exist
        must hand back every row of the class, not resample it.
        """
        available = int((rare_class_df[LABEL_COL] == CICIDS_RARE_CLASS_LABEL).sum())
        assert available == CICIDS_RARE_CLASS_ATTACK

        floored = stratified_sample(
            rare_class_df,
            FlowConfig(sample_size=1_000, class_sample_floor=available + 5),
        )
        assert int((floored[LABEL_COL] == CICIDS_RARE_CLASS_LABEL).sum()) == available

    def test_the_remaining_budget_is_still_split_proportionally(self, rare_class_df: pd.DataFrame):
        """
        Lifting one class costs the others seats, but it must not change how
        those seats are shared out among them.
        """
        floored = stratified_sample(
            rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=60)
        )

        source = rare_class_df[LABEL_COL].value_counts()
        drawn = floored[LABEL_COL].value_counts()
        others = [c for c in source.index if c != CICIDS_RARE_CLASS_LABEL]

        # The mini fixture gives every attack class the same number of rows, so
        # "proportional" is checkable directly: they must come back within a
        # seat of each other, and BENIGN must keep its multiple of them.
        attacks = [c for c in others if c != "BENIGN"]
        assert max(drawn[attacks]) - min(drawn[attacks]) <= 1
        assert drawn["BENIGN"] / drawn[attacks].mean() == pytest.approx(
            source["BENIGN"] / source[attacks].mean(), rel=0.02
        )

    def test_the_total_is_exactly_the_sample_size(self, rare_class_df: pd.DataFrame):
        """
        The quotas are integers computed before any row is drawn, so — unlike
        the proportional path, which rounds per class and then tops up — there
        is nothing left to reconcile.
        """
        floored = stratified_sample(
            rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=60)
        )
        assert len(floored) == 1_000
        assert floored[ORDER_IDX_COL].is_unique

    def test_order_idx_survives_the_floored_draw(self, rare_class_df: pd.DataFrame):
        floored = stratified_sample(
            rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=60)
        )
        assert set(floored[ORDER_IDX_COL]).issubset(set(rare_class_df[ORDER_IDX_COL]))

    def test_the_floored_draw_is_reproducible(self, rare_class_df: pd.DataFrame):
        config = FlowConfig(sample_size=1_000, class_sample_floor=60)
        a = stratified_sample(rare_class_df, config)
        b = stratified_sample(rare_class_df, config)
        assert sorted(a[ORDER_IDX_COL]) == sorted(b[ORDER_IDX_COL])

    def test_a_floor_the_budget_cannot_pay_for_is_reported_not_overshot(
        self, rare_class_df: pd.DataFrame, log_spy: _LogSpy
    ):
        """
        Seven classes at a floor of 200 cost more than a 1.000-row budget. There
        is no allocation that honours both, so the floor is dropped for the run
        and said so — returning 1.320 rows from a `--sample-size 1000` would
        break the promise every caller downstream relies on.
        """
        floored = stratified_sample(
            rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=200)
        )

        assert len(floored) == 1_000
        assert "class_sample_floor_unaffordable" in log_spy.names("warning")

        payload = log_spy.payloads("class_sample_floor_unaffordable")[0]
        assert payload["sample_size"] == 1_000
        assert payload["required"] > 1_000

        # Degraded means proportional, not "floor applied to whoever fits".
        # The rows themselves differ — exact quotas draw with `n=`, the
        # proportional path with `frac=` — but the allocation is the same.
        proportional = stratified_sample(rare_class_df, FlowConfig(sample_size=1_000))
        assert (
            floored[LABEL_COL].value_counts().to_dict()
            == proportional[LABEL_COL].value_counts().to_dict()
        )

    def test_the_quotas_never_ask_a_class_for_more_rows_than_it_has(self):
        """
        The capacity cap is unreachable through ``stratified_sample`` — it only
        floors a sample smaller than the dataset, and reserving a floor can only
        shrink the seats-per-row ratio of the classes still free. It is checked
        directly so the helper stays total if a caller ever asks for more seats
        than there are rows.
        """
        counts = pd.Series({"BENIGN": 100, "Botnet": 4})
        quotas = loader_module._class_quotas(counts, sample_size=104, floor=2)

        assert quotas.to_dict() == {"BENIGN": 100, "Botnet": 4}

    def test_the_quotas_lift_only_the_classes_below_the_floor(self):
        counts = pd.Series({"BENIGN": 100_000, "DoS": 5_000, "Botnet": 1_966})
        quotas = loader_module._class_quotas(counts, sample_size=20_000, floor=2_000)

        assert int(quotas.sum()) == 20_000
        # Botnet has fewer rows than the floor asks for, so it comes back whole.
        assert quotas["Botnet"] == 1_966
        # DoS is above the floor in absolute terms but its proportional quota
        # (~916) is not, so the floor binds there too.
        assert quotas["DoS"] == 2_000
        assert quotas["BENIGN"] == 20_000 - 1_966 - 2_000

    def test_the_quotas_hold_their_three_invariants_across_random_shapes(self):
        """
        The allocation is a fixed point of a loop, not a formula, so the three
        properties it promises are checked over shapes rather than argued
        about: the quotas sum to the budget, no class is asked for more rows
        than it has, and — whenever the floors are affordable at all — every
        class reaches its floor.
        """
        rng = np.random.default_rng(0)

        for _ in range(300):
            n_classes = int(rng.integers(1, 9))
            counts = pd.Series(
                rng.integers(1, 5_000, n_classes),
                index=[f"class_{i}" for i in range(n_classes)],
            )
            available = int(counts.sum())
            sample_size = int(rng.integers(1, available + 1))
            floor = int(rng.integers(1, 3_000))

            quotas = loader_module._class_quotas(counts, sample_size, floor)

            assert int(quotas.sum()) == sample_size
            assert (quotas <= counts).all()
            assert (quotas >= 0).all()

            floors = counts.clip(upper=floor)
            if int(floors.sum()) <= sample_size:
                assert (quotas >= floors).all()

    def test_the_floored_draw_names_the_classes_it_lifted(
        self, rare_class_df: pd.DataFrame, log_spy: _LogSpy
    ):
        """
        The lifted classes are exactly the ones whose support in the published
        tables is no longer proportional, so the run log has to name them.
        """
        stratified_sample(rare_class_df, FlowConfig(sample_size=1_000, class_sample_floor=60))

        payload = log_spy.payloads("stratified_sample_drawn_with_floor")[0]
        assert payload["lifted_classes"] == [CICIDS_RARE_CLASS_LABEL]
        assert payload["floor"] == 60
        assert sum(payload["quotas"].values()) == 1_000


# ============================================================================
# Per-split support audit
# ============================================================================

_Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class TestClassSupportAudit:
    """
    ``audit_class_support`` reports how many rows each class was measured on.

    It runs after the partition because that is where the number that matters
    appears: sampling sets how many rows a class has, the split sets how many
    are left in test, and a per-class F1 is computed from the second.
    """

    @pytest.fixture
    def splits(self, rare_class_df: pd.DataFrame) -> _Splits:
        sampled = stratified_sample(rare_class_df, FlowConfig(sample_size=1_000))
        return split_dataset_temporal(sampled, FlowConfig(sample_size=1_000))

    def test_every_class_gets_its_three_counts_and_a_total(self, splits: _Splits):
        support = audit_class_support(*splits, FlowConfig())

        present = set(splits[0][LABEL_COL]) | set(splits[1][LABEL_COL]) | set(splits[2][LABEL_COL])
        assert set(support) == {str(c) for c in present}
        for counts in support.values():
            assert set(counts) == {"train", "val", "test", "total"}
            assert counts["total"] == counts["train"] + counts["val"] + counts["test"]

    def test_the_counts_are_the_ones_the_splits_actually_carry(self, splits: _Splits):
        train, val, test = splits
        support = audit_class_support(train, val, test, FlowConfig())

        rare = support[CICIDS_RARE_CLASS_LABEL]
        assert rare["train"] == int((train[LABEL_COL] == CICIDS_RARE_CLASS_LABEL).sum())
        assert rare["val"] == int((val[LABEL_COL] == CICIDS_RARE_CLASS_LABEL).sum())
        assert rare["test"] == int((test[LABEL_COL] == CICIDS_RARE_CLASS_LABEL).sum())

    def test_a_class_under_the_floor_is_reported_with_its_three_counts(
        self, splits: _Splits, log_spy: _LogSpy
    ):
        audit_class_support(*splits, FlowConfig())

        reported = {p["label"]: p for p in log_spy.payloads("class_support_below_floor")}
        assert CICIDS_RARE_CLASS_LABEL in reported

        payload = reported[CICIDS_RARE_CLASS_LABEL]
        assert {"train", "val", "test", "total"} <= set(payload)
        assert payload["threshold"] == MIN_MEANINGFUL_BLOCK_ROWS
        assert payload["total"] < MIN_MEANINGFUL_BLOCK_ROWS

    def test_it_reports_but_never_drops(self, splits: _Splits, log_spy: _LogSpy):
        """
        Dropping the class it warns about would raise the headline F1-macro by
        ~0.11 — one of seven classes going from 0.00 to absent — which is
        improving the number by deleting the evidence for it. Botnet is also a
        *served* class: ml/flow/fusion.py maps credential_stuffing onto it.
        """
        train, val, test = splits
        before = [len(train), len(val), len(test)]

        support = audit_class_support(train, val, test, FlowConfig())

        assert [len(train), len(val), len(test)] == before
        assert CICIDS_RARE_CLASS_LABEL in support
        assert log_spy.names("warning")  # it did warn about it

    def test_the_classes_come_back_in_canonical_order(self, splits: _Splits):
        support = audit_class_support(*splits, FlowConfig())
        canonical = [c for c in CLASS_ORDER if c in support]
        assert list(support) == canonical

    def test_the_floor_setting_travels_with_the_warning(self, splits: _Splits, log_spy: _LogSpy):
        """
        A class below the threshold means one thing under strict proportionality
        and another under a floor that already tried to save it. The log line
        carries which one it was.
        """
        audit_class_support(*splits, FlowConfig(class_sample_floor=2_000))

        payloads = log_spy.payloads("class_support_below_floor")
        assert payloads and all(p["class_sample_floor"] == 2_000 for p in payloads)

    def test_it_defaults_to_a_plain_config(self, splits: _Splits):
        assert audit_class_support(*splits) == audit_class_support(*splits, FlowConfig())


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
