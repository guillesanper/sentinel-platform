"""
CICIDS2017 loading, cleaning and partitioning for the flow-based track.

Ported from the TFA's ``utils/loader.py``, with the methodological fix the TFA
itself flags in its §10.1: the original code partitions the dataset with a
**row-wise stratified random split**. CICIDS2017 is a continuous capture, so
near-identical flows from the same attack burst (same host, same scenario,
millisecond-apart timings) end up split across train and test.
``drop_duplicates()`` removes exact copies but not near-duplicates, and the
reported metrics are therefore an optimistic bound.

Two obvious fixes do not work on this distribution:

* A **global positional cut** (first 70% train, last 15% test) is useless: in
  ``Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`` the last decile is 100%
  BENIGN, so the test set would contain no attacks at all.
* A **day holdout** ("reserve Friday for test") starves the classifier: every
  class lives on exactly one day, so holding out Friday leaves DDoS, Scan and
  Botnet with zero training samples.

What this module implements instead is a **per-block temporal partition**:
inside each ``(capture_day, class)`` group, rows are ordered by their position
in the original capture and cut 70/15/15 chronologically. No test flow precedes
a train flow within its own scenario, every class survives in all three splits,
and near-duplicates from a single burst stay contiguous — which is where the
honesty of the metric is recovered.

Two provenance columns make that possible, and they are the single biggest
footgun in this file: ``capture_day`` and ``order_idx`` are *not* features.
:func:`get_feature_columns` is the only sanctioned way to select the feature
matrix; leaking either column into training produces a model that "detects
Friday" with spectacular metrics and zero value, and nothing raises an error.
"""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from ml.flow.config import FlowConfig
from ml.flow.labels import map_capture_day, map_label

logger = structlog.get_logger()


# ============================================================================
# Column contract
# ============================================================================

# Ground-truth column, as named in the CICIDS2017 CSVs.
LABEL_COL: Final[str] = "Label"

# Capture day derived from the source filename (the MachineLearningCVE CSVs
# carry no Timestamp column, so the filename is the only carrier of time).
CAPTURE_DAY_COL: Final[str] = "capture_day"

# Monotonic position of the row in load order. See _assign_order_idx.
ORDER_IDX_COL: Final[str] = "order_idx"

# Everything that must never reach the model. Consumed by get_feature_columns.
META_COLUMNS: Final[tuple[str, ...]] = (LABEL_COL, CAPTURE_DAY_COL, ORDER_IDX_COL)

# Default held-out day for the day-holdout protocol.
DEFAULT_TEST_DAYS: Final[tuple[str, ...]] = ("friday",)


# ============================================================================
# Tuning constants
# ============================================================================

# Below this many rows a (day, class) block cannot be cut into three usable
# parts, so it goes wholesale to train and a warning is emitted.
_MIN_SPLITTABLE_ROWS: Final[int] = 3

# Botnet contributes ~2.000 rows on a single day; at 15% that is ~300 test
# samples, which is the practical floor for a meaningful per-class F1. Blocks
# under this size are reported so the weakness shows up in the run log rather
# than as an unexplained metric.
_SMALL_BLOCK_ROWS: Final[int] = 2_000

# CICIDS2017 loses <5% of rows to NaN/Inf. Much more than that means the CSVs
# were read with the wrong encoding or are truncated.
_NAN_DROP_WARN_PCT: Final[float] = 20.0


# ============================================================================
# Loading
# ============================================================================


def load_cicids2017(data_path: Path | str) -> pd.DataFrame:
    """
    Load and merge every CICIDS2017 CSV found in ``data_path``.

    Column names are stripped of surrounding whitespace, labels are mapped to
    the 7-class canonical taxonomy (see :mod:`ml.flow.labels`), and rows with
    unrecognised labels are discarded with a warning.

    Two provenance columns are added, without which the temporal partition is
    impossible: ``capture_day`` (from the filename) and ``order_idx``.

    Args:
        data_path: Directory containing the CICIDS2017 CSV files.

    Returns:
        Concatenated DataFrame with normalised column names, canonical labels
        and the two provenance columns.

    Raises:
        FileNotFoundError: The directory does not exist or holds no CSV file.
        NotADirectoryError: ``data_path`` points at a file.
        KeyError: The concatenated frame has no ``Label`` column.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path.resolve()}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path.resolve()}")

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    logger.info("cicids_load_started", n_files=len(csv_files), data_path=str(data_path))

    frames: list[pd.DataFrame] = []
    offset = 0
    for csv_file in csv_files:
        # latin-1, not utf-8: the "Web Attack" labels carry bytes that are not
        # valid UTF-8. ml.flow.labels.map_label repairs the resulting mojibake.
        frame = pd.read_csv(csv_file, low_memory=False, encoding="latin-1")
        frame.columns = pd.Index([str(c).strip() for c in frame.columns])

        capture_day = map_capture_day(csv_file.name)
        frame[CAPTURE_DAY_COL] = capture_day
        frame[ORDER_IDX_COL] = _assign_order_idx(len(frame), offset)
        offset += len(frame)

        frames.append(frame)
        logger.debug(
            "cicids_file_loaded",
            filename=csv_file.name,
            capture_day=capture_day,
            n_rows=len(frame),
            n_cols=frame.shape[1],
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "cicids_files_merged",
        n_rows=len(combined),
        n_cols=combined.shape[1],
        capture_days=sorted(combined[CAPTURE_DAY_COL].unique().tolist()),
    )

    return _apply_label_mapping(combined)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the feature columns of ``df``: everything that is not metadata.

    Single source of truth for the training/inference feature space. Nothing in
    the codebase should build the list by hand — ``capture_day`` and
    ``order_idx`` silently turn into extremely predictive features if they slip
    through, and the resulting model detects the capture schedule, not attacks.

    Args:
        df: Any frame produced by this module.

    Returns:
        Column names in frame order, excluding :data:`META_COLUMNS`.
    """
    return [c for c in df.columns if c not in META_COLUMNS]


# ============================================================================
# Cleaning and sampling
# ============================================================================


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove numeric artifacts and duplicate rows.

    Applied in order:
        1. ±Inf → NaN in the feature columns.
        2. Drop rows carrying NaN in any feature column.
        3. Drop duplicate rows, compared on **features + label only** — the
           provenance columns are excluded on purpose, otherwise ``order_idx``
           (unique by construction) would make every row unique and the
           deduplication would silently become a no-op.
        4. Cast the feature columns to float32.

    Args:
        df: Frame as returned by :func:`load_cicids2017`.

    Returns:
        Cleaned frame with a fresh RangeIndex. ``capture_day`` and
        ``order_idx`` are preserved.
    """
    n0 = len(df)
    feature_cols = get_feature_columns(df)

    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    n_after_na = len(df)
    if n0 - n_after_na > 0:
        logger.info("rows_dropped_nan_inf", n_dropped=n0 - n_after_na, n_before=n0)

    drop_pct = 100.0 * (n0 - n_after_na) / n0 if n0 > 0 else 0.0
    if drop_pct > _NAN_DROP_WARN_PCT:
        logger.warning(
            "nan_inf_drop_ratio_suspicious",
            drop_pct=round(drop_pct, 2),
            expected_max_pct=5.0,
            hint="check the CSV encoding; CICIDS2017 normally loses <5% of rows",
        )

    df = df.drop_duplicates(subset=[*feature_cols, LABEL_COL])
    n_after_dup = len(df)
    if n_after_na - n_after_dup > 0:
        logger.info("rows_dropped_duplicates", n_dropped=n_after_na - n_after_dup)

    df[feature_cols] = df[feature_cols].astype(np.float32)

    logger.info("dataset_cleaned", n_rows=len(df), n_removed=n0 - len(df))
    return df.reset_index(drop=True)


def filter_minority_classes(
    df: pd.DataFrame, config: FlowConfig | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop classes below the absolute or relative frequency thresholds.

    A class is removed when **either** condition holds:
    ``count < config.min_class_samples`` or
    ``count / total < config.min_class_ratio``.

    Runs before stratified sampling, so the sample cannot inherit residual
    minority classes whose per-class metrics would be pure noise.

    Args:
        df: Cleaned frame with a ``Label`` column.
        config: Track configuration. Defaults to ``FlowConfig()``.

    Returns:
        ``(filtered_df, removed_class_names)``.
    """
    config = config or FlowConfig()

    total = len(df)
    counts = df[LABEL_COL].value_counts()
    removed: list[str] = []

    for cls, count in counts.items():
        ratio = count / total if total else 0.0
        if count < config.min_class_samples or ratio < config.min_class_ratio:
            logger.warning(
                "minority_class_removed",
                label=str(cls),
                n_samples=int(count),
                ratio=round(float(ratio), 6),
                min_samples=config.min_class_samples,
                min_ratio=config.min_class_ratio,
            )
            removed.append(str(cls))

    if removed:
        df = df[~df[LABEL_COL].isin(removed)].reset_index(drop=True)
        logger.info("minority_classes_filtered", n_removed=len(removed), labels=removed)
    else:
        logger.info("minority_classes_none")

    return df, removed


def stratified_sample(df: pd.DataFrame, config: FlowConfig | None = None) -> pd.DataFrame:
    """
    Draw a stratified sample of ``config.sample_size`` rows.

    Sampling is proportional to each class's share of the dataset. The frame is
    shuffled at the end; ``order_idx`` survives as a column, so a temporal split
    downstream is still exact — the shuffle only touches row order, never the
    recorded position in the capture.

    Args:
        df: Filtered frame with a ``Label`` column.
        config: Track configuration. Defaults to ``FlowConfig()``.

    Returns:
        The sampled frame, or ``df`` unchanged when ``sample_size`` is 0 or the
        dataset already fits.
    """
    config = config or FlowConfig()

    n = len(df)
    if config.sample_size == 0 or n <= config.sample_size:
        logger.info("stratified_sample_skipped", n_rows=n, sample_size=config.sample_size)
        return df

    frac = config.sample_size / n
    # groupby().sample() rather than groupby().apply(lambda g: g.sample(...)):
    # the apply form is deprecated in pandas 2.2 (the grouping columns are
    # passed to the callable) and warns on every run.
    sampled = df.groupby(LABEL_COL, group_keys=False).sample(
        frac=frac, random_state=config.random_seed
    )

    # Per-class rounding leaves a shortfall; top it up from the leftovers.
    shortfall = config.sample_size - len(sampled)
    if shortfall > 0:
        remaining = df.drop(sampled.index)
        if len(remaining) > 0:
            extra = remaining.sample(
                n=min(shortfall, len(remaining)), random_state=config.random_seed
            )
            sampled = pd.concat([sampled, extra])

    sampled = sampled.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    logger.info("stratified_sample_drawn", n_before=n, n_after=len(sampled))
    return sampled


# ============================================================================
# Partitioning
# ============================================================================


def split_dataset_temporal(
    df: pd.DataFrame, config: FlowConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition chronologically inside each ``(capture_day, class)`` block.

    The default protocol. Rows of a block are ordered by ``order_idx`` and cut
    ``train_ratio``/``val_ratio``/``test_ratio`` by position, so within any one
    attack scenario no test flow precedes a train flow, while every class keeps
    samples in all three splits.

    Args:
        df: Frame carrying ``Label``, ``capture_day`` and ``order_idx``.
        config: Track configuration. Defaults to ``FlowConfig()``.

    Returns:
        ``(train_df, val_df, test_df)`` — same shape of return as
        :func:`split_dataset`.

    Raises:
        KeyError: A provenance column is missing (the frame did not come from
            :func:`load_cicids2017`).
    """
    config = config or FlowConfig()
    _require_meta_columns(df)

    train, val, test = _temporal_cut(
        df,
        (config.train_ratio, config.val_ratio, config.test_ratio),
        seed=config.random_seed,
    )

    logger.info(
        "dataset_split",
        strategy="temporal",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
    )
    _log_split_distribution(train, val, test)
    return train, val, test


def split_dataset_day_holdout(
    df: pd.DataFrame,
    config: FlowConfig | None = None,
    test_days: tuple[str, ...] | list[str] = DEFAULT_TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold entire capture days out as the test set.

    Secondary protocol: it is the honest measure of generalisation to an unseen
    day, but in CICIDS2017 each attack class lives on a single day, so the held
    out classes have **no training samples at all**. Those classes are logged
    explicitly and the result must be reported separately — never as the
    headline metric.

    Validation is carved out of the remaining days with the same per-block
    temporal cut used by :func:`split_dataset_temporal`.

    Args:
        df: Frame carrying ``Label``, ``capture_day`` and ``order_idx``.
        config: Track configuration. Defaults to ``FlowConfig()``.
        test_days: Capture days reserved for test, matched case-insensitively.

    Returns:
        ``(train_df, val_df, test_df)``.

    Raises:
        KeyError: A provenance column is missing.
        ValueError: ``test_days`` matches no row, or matches every row.
    """
    config = config or FlowConfig()
    _require_meta_columns(df)

    wanted = {str(day).strip().lower() for day in test_days}
    in_test = df[CAPTURE_DAY_COL].astype(str).str.lower().isin(wanted)

    if not in_test.any():
        available = sorted(df[CAPTURE_DAY_COL].unique().tolist())
        raise ValueError(f"No rows for test_days={sorted(wanted)}. Available days: {available}")
    if in_test.all():
        raise ValueError(
            f"test_days={sorted(wanted)} covers the whole dataset; nothing to train on"
        )

    test = df[in_test].reset_index(drop=True)
    train_pool = df[~in_test]

    # Renormalise train/val to the retained days.
    val_frac = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = _temporal_cut(train_pool, (1.0 - val_frac, val_frac), seed=config.random_seed)

    train_classes = set(train[LABEL_COL].unique())
    test_classes = set(test[LABEL_COL].unique())
    unseen_in_train = sorted(test_classes - train_classes)
    unseen_in_test = sorted(train_classes - test_classes)

    if unseen_in_train:
        logger.warning(
            "day_holdout_classes_absent_from_train",
            labels=unseen_in_train,
            test_days=sorted(wanted),
            hint="these classes cannot be learned; exclude them when reporting",
        )
    if unseen_in_test:
        logger.warning(
            "day_holdout_classes_absent_from_test",
            labels=unseen_in_test,
            test_days=sorted(wanted),
        )

    logger.info(
        "dataset_split",
        strategy="day-holdout",
        test_days=sorted(wanted),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
    )
    _log_split_distribution(train, val, test)
    return train, val, test


def split_dataset(
    df: pd.DataFrame, config: FlowConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Row-wise stratified random split — the TFA's original protocol.

    Kept **only** to quantify the leakage: training once with this and once with
    :func:`split_dataset_temporal` and reporting the delta in F1-macro is a
    result in itself. Do not use it as the reference protocol.

    Args:
        df: Sampled, cleaned frame with a ``Label`` column.
        config: Track configuration. Defaults to ``FlowConfig()``.

    Returns:
        ``(train_df, val_df, test_df)``.
    """
    config = config or FlowConfig()

    train_val, test = train_test_split(
        df,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        stratify=df[LABEL_COL],
    )
    val_ratio_adj = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adj,
        random_state=config.random_seed,
        stratify=train_val[LABEL_COL],
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(
        "dataset_split",
        strategy="random",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
    )
    _log_split_distribution(train, val, test)
    return train, val, test


def split_dataset_by_strategy(
    df: pd.DataFrame,
    config: FlowConfig | None = None,
    test_days: tuple[str, ...] | list[str] = DEFAULT_TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dispatch to the partitioning protocol named by ``config.split_strategy``.

    Args:
        df: Frame to partition.
        config: Track configuration. Defaults to ``FlowConfig()`` (``temporal``).
        test_days: Forwarded to :func:`split_dataset_day_holdout`; ignored by
            the other strategies.

    Returns:
        ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: ``config.split_strategy`` is not a known strategy. This
            cannot normally happen — ``FlowConfig`` validates it on init.
    """
    config = config or FlowConfig()

    if config.split_strategy == "temporal":
        return split_dataset_temporal(df, config)
    if config.split_strategy == "random":
        return split_dataset(df, config)
    if config.split_strategy == "day-holdout":
        return split_dataset_day_holdout(df, config, test_days)
    raise ValueError(f"Unknown split_strategy: {config.split_strategy!r}")


# ============================================================================
# Private helpers
# ============================================================================


def _assign_order_idx(n_rows: int, offset: int) -> np.ndarray:
    """
    Build the ``order_idx`` values for one CSV.

    The index is the row's position within its own file, shifted by the total
    number of rows loaded before it. Two properties matter and both are load
    bearing: it is **monotonic within a file**, which is what makes the cut
    chronological; and it is **globally unique**, which keeps the cut
    unambiguous when several files map to the same capture day (Thursday and
    Friday both span three files). A plain per-file position would tie those
    files together and interleave them arbitrarily.
    """
    return np.arange(offset, offset + n_rows, dtype=np.int64)


def _apply_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw labels to the canonical taxonomy, dropping unrecognised rows."""
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Column {LABEL_COL!r} not found. Available columns: {list(df.columns[:10])}..."
        )

    mapped = df[LABEL_COL].map(map_label)
    unknown = mapped.isna()

    if unknown.any():
        raw_unknown = sorted({str(v) for v in df.loc[unknown, LABEL_COL].unique()})
        logger.warning(
            "labels_unrecognised_discarded",
            n_labels=len(raw_unknown),
            n_rows=int(unknown.sum()),
            labels=raw_unknown,
        )

    df = df.loc[~unknown].copy()
    df[LABEL_COL] = mapped.loc[~unknown]

    logger.info(
        "labels_mapped",
        n_classes=int(df[LABEL_COL].nunique()),
        labels=sorted(df[LABEL_COL].unique().tolist()),
    )
    return df.reset_index(drop=True)


def _require_meta_columns(df: pd.DataFrame) -> None:
    """Raise when the provenance columns needed for a temporal split are absent."""
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s) {missing} required for temporal partitioning. "
            "The frame must come from load_cicids2017()."
        )


def _block_boundaries(n: int, ratios: tuple[float, ...]) -> list[int]:
    """
    Positional cut points for a block of ``n`` rows.

    Returns ``len(ratios) + 1`` boundaries. When the block has at least one row
    per part, the boundaries are forced strictly increasing so that no part
    comes back empty because of rounding; when it does not, the whole block goes
    to the first part (train) rather than leaving a class present only in test.
    """
    n_parts = len(ratios)
    if n < n_parts:
        return [0] + [n] * n_parts

    cumulative = np.cumsum(ratios)
    bounds = [0] + [int(n * c) for c in cumulative[:-1]] + [n]

    # Each part i must start at least i rows in ...
    for i in range(1, n_parts):
        bounds[i] = max(bounds[i], i)
    # ... and leave at least one row for every part after it.
    for i in range(n_parts - 1, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)

    return bounds


def _temporal_cut(df: pd.DataFrame, ratios: tuple[float, ...], *, seed: int) -> list[pd.DataFrame]:
    """
    Cut ``df`` into ``len(ratios)`` chronological parts inside each block.

    A block is a ``(capture_day, Label)`` group: the finest partition of the
    capture in which "earlier" is a meaningful statement. Rows are ordered by
    ``order_idx`` with a stable sort, cut by position, and each resulting part
    is shuffled — shuffling row order is irrelevant to the guarantee, which is
    carried by the ``order_idx`` values, not by their arrangement.
    """
    n_parts = len(ratios)
    parts: list[list[pd.DataFrame]] = [[] for _ in range(n_parts)]

    for (day, label), block in df.groupby([CAPTURE_DAY_COL, LABEL_COL], sort=True):
        ordered = block.sort_values(ORDER_IDX_COL, kind="mergesort")
        n = len(ordered)
        bounds = _block_boundaries(n, ratios)
        sizes = [bounds[i + 1] - bounds[i] for i in range(n_parts)]

        for i in range(n_parts):
            if sizes[i] > 0:
                parts[i].append(ordered.iloc[bounds[i] : bounds[i + 1]])

        if n < _MIN_SPLITTABLE_ROWS:
            logger.warning(
                "temporal_block_not_splittable",
                capture_day=str(day),
                label=str(label),
                n_rows=n,
                hint="whole block assigned to the first split",
            )
        elif n < _SMALL_BLOCK_ROWS:
            logger.warning(
                "temporal_block_small",
                capture_day=str(day),
                label=str(label),
                n_rows=n,
                sizes=sizes,
                threshold=_SMALL_BLOCK_ROWS,
            )
        else:
            logger.debug(
                "temporal_block_split",
                capture_day=str(day),
                label=str(label),
                n_rows=n,
                sizes=sizes,
            )

    empty = df.iloc[0:0]
    return [
        (
            pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
            if chunks
            else empty.copy()
        )
        for chunks in parts
    ]


def _log_split_distribution(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Emit one structured event per class with its count in each split."""
    splits = {"train": train, "val": val, "test": test}
    all_classes = sorted(
        set(train[LABEL_COL].unique())
        | set(val[LABEL_COL].unique())
        | set(test[LABEL_COL].unique())
    )

    for cls in all_classes:
        counts = {name: int((split[LABEL_COL] == cls).sum()) for name, split in splits.items()}
        event = logger.warning if min(counts.values()) == 0 else logger.info
        event("split_class_distribution", label=str(cls), **counts)
