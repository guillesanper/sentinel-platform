"""
Unit tests for the chronological cross-validation folds of the flow track.

The property that carries this module is easy to state and easy to break:
**the folds are cut on ``order_idx`` values, never on row positions**.
``ml.flow.loader._temporal_cut`` shuffles every part it returns, so the rows of
a training frame arrive in arbitrary order and only the ``order_idx`` column
still knows what came first. A fold builder that trusted row order would look
correct on any synthetic frame written in chronological order — which is what
every fixture here would be, if one of them did not deliberately arrive
backwards.

The rest of the tests pin the guarantees the scheme exists for: a fold's
validation chunk comes after its training chunks *inside every block*, the
training half only ever grows, no row lands on both sides, and a block too small
to cut trains everywhere and validates nowhere — the same rule, from the same
:func:`ml.flow.loader.block_boundaries`, that the train/val/test split applies.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ml.flow import loader
from ml.flow.config import FlowConfig
from ml.flow.cv import TemporalIndex, temporal_block_folds
from ml.flow.loader import (
    CAPTURE_DAY_COL,
    LABEL_COL,
    ORDER_IDX_COL,
    RAW_LABEL_COL,
    build_temporal_index,
    clean_dataset,
    load_cicids2017,
    split_dataset_temporal,
)

# ============================================================================
# Helpers
# ============================================================================


def _index(blocks: list[int], orders: list[int]) -> TemporalIndex:
    """A TemporalIndex from two plain lists, one entry per row."""
    return TemporalIndex(
        order=np.asarray(orders, dtype=np.int64),
        block=np.asarray(blocks, dtype=np.int32),
    )


def _uniform_blocks(n_blocks: int, per_block: int) -> TemporalIndex:
    """``n_blocks`` blocks of ``per_block`` rows each, laid out in row order."""
    blocks = [b for b in range(n_blocks) for _ in range(per_block)]
    return _index(blocks, list(range(len(blocks))))


# ============================================================================
# TemporalIndex
# ============================================================================


class TestTemporalIndex:
    """The alignment contract, which nothing downstream can check for itself."""

    def test_length_is_the_row_count(self):
        assert len(_uniform_blocks(3, 8)) == 24

    def test_arrays_of_different_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="must describe the same rows"):
            TemporalIndex(order=np.arange(5, dtype=np.int64), block=np.zeros(4, dtype=np.int32))

    def test_a_two_dimensional_array_is_rejected(self):
        with pytest.raises(ValueError, match="1-D arrays"):
            TemporalIndex(
                order=np.zeros((4, 2), dtype=np.int64),
                block=np.zeros(4, dtype=np.int32),
            )

    def test_mask_selects_the_same_rows_as_the_matrix_would(self):
        index = _index([0, 0, 1, 1], [10, 11, 12, 13])
        keep = np.array([True, False, True, False])

        narrowed = index.mask(keep)

        assert len(narrowed) == 2
        np.testing.assert_array_equal(narrowed.order, [10, 12])
        np.testing.assert_array_equal(narrowed.block, [0, 1])

    def test_a_mask_of_the_wrong_length_raises_instead_of_truncating(self):
        """
        The whole failure mode this class guards against: an index narrowed by a
        mask other than the one that selected the rows still has a plausible
        length and produces plausible, meaningless folds.
        """
        index = _uniform_blocks(2, 4)

        with pytest.raises(ValueError, match="describes 8 rows"):
            index.mask(np.ones(5, dtype=bool))

    def test_an_integer_mask_is_rejected(self):
        index = _uniform_blocks(2, 4)

        with pytest.raises(ValueError, match="boolean array"):
            index.mask(np.array([0, 1, 2]))


# ============================================================================
# temporal_block_folds
# ============================================================================


class TestTemporalBlockFolds:
    """The expanding chronological window, block by block."""

    def test_folds_index_positions_not_order_idx_values_so_a_shuffled_frame_is_handled(self):
        """
        The trap this module exists to avoid.

        ``_temporal_cut`` shuffles each part before returning it, so a training
        frame's row order is meaningless and only ``order_idx`` is chronological.
        Here the rows arrive in exactly reverse order: the first fold must
        therefore train on the *last* row positions, and the returned arrays must
        index the matrix, not repeat ``order_idx`` values.
        """
        # Row 0 is the newest flow, row 7 the oldest.
        index = _index([0] * 8, [700, 600, 500, 400, 300, 200, 100, 0])

        folds = temporal_block_folds(index, n_splits=3)
        first_train, first_val = folds[0]

        # The two oldest flows sit at the *end* of the matrix. Returned
        # ascending, since a fold is a set of rows and not a sequence.
        assert first_train.tolist() == [6, 7]
        assert first_val.tolist() == [4, 5]
        # Positions, not order_idx values — those are all multiples of 100.
        assert first_train.max() < len(index)

    def test_every_fold_validates_strictly_after_its_train_inside_every_block(self):
        index = _uniform_blocks(n_blocks=3, per_block=12)

        for train, val in temporal_block_folds(index, n_splits=3):
            for code in np.unique(index.block):
                in_block = index.block == code
                train_orders = index.order[train][in_block[train]]
                val_orders = index.order[val][in_block[val]]
                if train_orders.size and val_orders.size:
                    assert train_orders.max() < val_orders.min()

    def test_the_training_half_only_grows(self):
        """
        Expanding window, not K-fold: fold k+1 trains on everything fold k
        trained on, plus fold k's validation chunk.
        """
        index = _uniform_blocks(n_blocks=2, per_block=16)

        folds = temporal_block_folds(index, n_splits=3)

        for (train, val), (next_train, _) in pairwise(folds):
            assert set(train.tolist()) < set(next_train.tolist())
            assert set(next_train.tolist()) == set(train.tolist()) | set(val.tolist())

    def test_no_row_is_in_both_halves_of_a_fold(self):
        index = _uniform_blocks(n_blocks=4, per_block=9)

        for train, val in temporal_block_folds(index, n_splits=3):
            assert not set(train.tolist()) & set(val.tolist())

    def test_the_number_of_folds_is_the_number_of_splits(self):
        index = _uniform_blocks(n_blocks=2, per_block=20)

        assert len(temporal_block_folds(index, n_splits=2)) == 2
        assert len(temporal_block_folds(index, n_splits=4)) == 4

    def test_no_fold_validates_a_class_its_own_train_never_saw(self):
        """
        The reason the folds are per block rather than one global chronological
        cut: the days of CICIDS2017 are disjoint by class, so a global cut would
        validate every fold on classes absent from its training half.
        """
        # One class per block, each large enough to reach every chunk.
        classes = np.array(["DoS"] * 12 + ["Scan"] * 12 + ["BruteForce"] * 12)
        index = _uniform_blocks(n_blocks=3, per_block=12)

        for train, val in temporal_block_folds(index, n_splits=3):
            assert set(classes[val]) <= set(classes[train])

    def test_a_block_smaller_than_the_chunk_count_trains_everywhere_and_validates_nowhere(self):
        """
        Same rule as ``test_tiny_block_goes_wholesale_to_train`` in the split: a
        block that cannot be cut goes to the first chunk entire, which puts it in
        every fold's training half and in no fold's validation half.
        """
        # Block 1 holds 3 rows and the folds ask for 4 chunks.
        index = _index([0] * 12 + [1, 1, 1], list(range(15)))
        tiny = [12, 13, 14]

        folds = temporal_block_folds(index, n_splits=3)

        for train, val in folds:
            assert set(tiny) <= set(train.tolist())
            assert not set(tiny) & set(val.tolist())

    def test_the_folds_are_reproducible(self):
        index = _uniform_blocks(n_blocks=3, per_block=11)

        first = temporal_block_folds(index, n_splits=3)
        second = temporal_block_folds(index, n_splits=3)

        for (t1, v1), (t2, v2) in zip(first, second, strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_blocks_are_cut_with_the_loaders_own_boundary_rule(self, monkeypatch):
        """
        One implementation of "strictly increasing, everything to the first part
        when the block is too small", shared with the train/val/test split. Two
        copies would drift, and the drift would read as folds disagreeing with
        the split about which rows are early.
        """
        calls: list[tuple[int, tuple[float, ...]]] = []
        real = loader.block_boundaries

        def spy(n: int, ratios: tuple[float, ...]) -> list[int]:
            calls.append((n, ratios))
            return real(n, ratios)

        monkeypatch.setattr("ml.flow.cv.block_boundaries", spy)
        temporal_block_folds(_uniform_blocks(n_blocks=2, per_block=10), n_splits=3)

        assert [n for n, _ in calls] == [10, 10]
        # n_splits + 1 equal chunks.
        assert all(len(ratios) == 4 for _, ratios in calls)
        assert all(ratios == pytest.approx((0.25, 0.25, 0.25, 0.25)) for _, ratios in calls)

    def test_an_empty_index_yields_empty_folds_rather_than_raising(self):
        """
        Degenerate but reachable — the caller decides what to do about it, and
        :meth:`FlowAttackClassifier._build_cv` falls back when fewer than two
        folds survive.
        """
        folds = temporal_block_folds(_index([], []), n_splits=3)

        assert len(folds) == 3
        assert all(train.size == 0 and val.size == 0 for train, val in folds)

    @pytest.mark.parametrize("n_splits", [0, -1])
    def test_fewer_than_one_split_is_rejected(self, n_splits: int):
        with pytest.raises(ValueError, match="at least 1"):
            temporal_block_folds(_uniform_blocks(2, 8), n_splits)


# ============================================================================
# build_temporal_index — over the real loading path
# ============================================================================


class TestBuildTemporalIndex:
    """The bridge from a partitioned frame to the folds."""

    def test_the_index_describes_the_frame_row_for_row(self, cicids_mini_csv: Path):
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, _, _ = split_dataset_temporal(frame, FlowConfig())

        index = build_temporal_index(train_df, FlowConfig())

        assert len(index) == len(train_df)
        np.testing.assert_array_equal(index.order, train_df[ORDER_IDX_COL].to_numpy())

    def test_the_block_key_follows_the_configured_protocol(self, cicids_mini_csv: Path):
        """
        ``sub-label`` splits Tuesday's two patators into two blocks; ``class``
        merges them into one. Cutting the folds on a different key than the split
        used would validate each fold on a tool its train never saw — silently.
        """
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, _, _ = split_dataset_temporal(frame, FlowConfig())

        by_sublabel = build_temporal_index(train_df, FlowConfig(temporal_block="sub-label"))
        by_class = build_temporal_index(train_df, FlowConfig(temporal_block="class"))

        assert np.unique(by_sublabel.block).size == len(
            train_df.groupby([CAPTURE_DAY_COL, RAW_LABEL_COL], observed=True)
        )
        assert np.unique(by_class.block).size == len(
            train_df.groupby([CAPTURE_DAY_COL, LABEL_COL], observed=True)
        )
        assert np.unique(by_sublabel.block).size > np.unique(by_class.block).size

    def test_rows_of_one_block_share_one_code(self, cicids_mini_csv: Path):
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, _, _ = split_dataset_temporal(frame, FlowConfig())

        index = build_temporal_index(train_df, FlowConfig())

        pairs = list(zip(train_df[CAPTURE_DAY_COL], train_df[RAW_LABEL_COL], strict=True))
        for code in np.unique(index.block):
            assert len({pairs[i] for i in np.flatnonzero(index.block == code)}) == 1

    def test_a_frame_without_the_provenance_columns_is_rejected(self, cicids_mini_csv: Path):
        frame = clean_dataset(load_cicids2017(cicids_mini_csv)).drop(columns=[ORDER_IDX_COL])

        with pytest.raises(KeyError, match="required for temporal partitioning"):
            build_temporal_index(frame, FlowConfig())

    def test_the_folds_of_a_real_split_respect_order_idx_not_row_order(self, cicids_mini_csv: Path):
        """
        End to end over the shuffling the loader really does: the training frame
        comes back with its rows permuted, and the folds still come out
        chronological inside every block.
        """
        frame = clean_dataset(load_cicids2017(cicids_mini_csv))
        train_df, _, _ = split_dataset_temporal(frame, FlowConfig())
        index = build_temporal_index(train_df, FlowConfig())

        # The premise: row order is not order_idx order.
        assert not np.all(np.diff(index.order) > 0)

        for train, val in temporal_block_folds(index, n_splits=3):
            for code in np.unique(index.block):
                in_block = index.block == code
                train_orders = index.order[train][in_block[train]]
                val_orders = index.order[val][in_block[val]]
                if train_orders.size and val_orders.size:
                    assert train_orders.max() < val_orders.min()
