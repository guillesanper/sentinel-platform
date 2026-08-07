"""
Unit tests for the event-track training pipeline.

The file exists for one property above all others: **the string labels stay
aligned with their feature rows across both cuts of the split.**

That alignment used to be broken. ``train.py`` split once carrying the labels,
split again *without* them, and then recovered the validation labels by slicing
the first split's array positionally — an order the second (shuffling) split had
already destroyed. Nothing raised. The anomaly threshold was tuned against a
permutation of the ground truth, and the recall printed at the end of every run
was a number about noise.

A bug of that shape survives any smoke test that only asserts "training
finished", so the tests here assert the correspondence row by row.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ml.models.classifier import LABEL_TO_IDX
from ml.training.train import main, split_train_val_test, to_anomaly_labels

# ============================================================================
# A dataset where position and label are perfectly correlated
# ============================================================================

_N_ROWS = 400
_N_FEATURES = 17
# Rows below this index are normal, rows at or above it are attacks. Making the
# label a function of the *original position* is what turns a misalignment into
# a detectable one: under the old positional slice the validation labels came
# out looking like a random draw from the base rate.
_ATTACK_FROM = 300

_ATTACK_LABEL = "brute_force"

# Column 0 carries the original row index, so every row can be traced back to
# the label it started with no matter how many times it is shuffled.
_POSITION_COL = 0


def _positional_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build ``(X, y, labels)`` where the label is a function of the row index."""
    rng = np.random.default_rng(11)

    X = rng.normal(0.0, 1.0, size=(_N_ROWS, _N_FEATURES)).astype(np.float32)
    X[:, _POSITION_COL] = np.arange(_N_ROWS, dtype=np.float32)

    labels = ["normal" if i < _ATTACK_FROM else _ATTACK_LABEL for i in range(_N_ROWS)]
    y = np.array([LABEL_TO_IDX[label] for label in labels], dtype=np.int32)

    return X, y, labels


def _expected_labels(X_split: np.ndarray) -> list[str]:
    """The label each row of `X_split` must carry, read off its original index."""
    return [
        "normal" if int(round(float(row[_POSITION_COL]))) < _ATTACK_FROM else _ATTACK_LABEL
        for row in X_split
    ]


@pytest.fixture
def positional_split():
    """`split_train_val_test` applied to the positional dataset."""
    X, y, labels = _positional_dataset()
    return split_train_val_test(X, y, labels, test_size=0.2, val_size=0.25, seed=42)


# ============================================================================
# The regression: labels aligned with features after BOTH cuts
# ============================================================================


class TestSplitKeepsLabelsAligned:
    """The property whose absence silently poisoned threshold optimisation."""

    def test_validation_labels_match_their_rows(self, positional_split) -> None:
        """
        Row by row, not set-wise: a permuted `labels_val` has exactly the same
        class counts as the correct one, so any assertion weaker than this
        passes with the bug present.
        """
        _, X_val, _, _, _, _, _, labels_val, _ = positional_split

        assert labels_val == _expected_labels(X_val)

    def test_validation_anomaly_labels_match_their_rows(self, positional_split) -> None:
        """The binary view is what the detector actually optimises against."""
        _, X_val, _, _, _, _, _, labels_val, _ = positional_split

        y_val_anomaly = to_anomaly_labels(labels_val)
        expected = np.array(
            [1 if row[_POSITION_COL] >= _ATTACK_FROM else 0 for row in X_val],
            dtype=int,
        )

        np.testing.assert_array_equal(y_val_anomaly, expected)

    def test_train_labels_match_their_rows(self, positional_split) -> None:
        X_train, _, _, _, _, _, labels_train, _, _ = positional_split

        assert labels_train == _expected_labels(X_train)

    def test_test_labels_match_their_rows(self, positional_split) -> None:
        """`labels_test` feeds the reported metrics; a shift there fakes them."""
        _, _, X_test, _, _, _, _, _, labels_test = positional_split

        assert labels_test == _expected_labels(X_test)

    def test_the_second_cut_really_shuffles(self, positional_split) -> None:
        """
        Non-vacuity guard. If the second `train_test_split` preserved order, a
        positional slice would happen to work and the tests above would hold for
        the wrong reason — and would stop protecting anything the day sklearn's
        default changed back.
        """
        _, X_val, _, _, _, _, _, _, _ = positional_split

        positions = X_val[:, _POSITION_COL]
        assert not np.all(np.diff(positions) > 0)

    def test_a_positional_slice_would_disagree(self) -> None:
        """
        The bug, reproduced. Splitting once with the labels and then recovering
        the validation slice by position — exactly what `train.py` used to do —
        yields labels that do not match the rows they are paired with.
        """
        from sklearn.model_selection import train_test_split

        X, y, labels = _positional_dataset()

        X_train_full, _, y_train_full, _, labels_train_full, _ = train_test_split(
            X, y, np.asarray(labels, dtype=object), test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, _, _ = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )

        sliced = list(labels_train_full[len(X_train) : len(X_train) + len(X_val)])

        assert sliced != _expected_labels(X_val)


class TestSplitShapes:
    """The mundane guarantees the split must also keep."""

    def test_sizes_add_up(self, positional_split) -> None:
        X_train, X_val, X_test, *_ = positional_split

        assert len(X_train) + len(X_val) + len(X_test) == _N_ROWS

    def test_every_part_gets_labels_of_its_own_length(self, positional_split) -> None:
        X_train, X_val, X_test, _, _, _, labels_train, labels_val, labels_test = positional_split

        assert len(labels_train) == len(X_train)
        assert len(labels_val) == len(X_val)
        assert len(labels_test) == len(X_test)

    def test_encoded_and_string_labels_agree(self, positional_split) -> None:
        """`y_*` and `labels_*` describe the same rows or the metrics are fiction."""
        _, _, _, y_train, y_val, y_test, labels_train, labels_val, labels_test = positional_split

        for encoded, strings in (
            (y_train, labels_train),
            (y_val, labels_val),
            (y_test, labels_test),
        ):
            expected = np.array([LABEL_TO_IDX[label] for label in strings], dtype=encoded.dtype)
            np.testing.assert_array_equal(encoded, expected)

    def test_the_splits_are_disjoint(self, positional_split) -> None:
        X_train, X_val, X_test, *_ = positional_split

        parts = [
            set(part[:, _POSITION_COL].astype(int).tolist()) for part in (X_train, X_val, X_test)
        ]
        assert parts[0] & parts[1] == set()
        assert parts[0] & parts[2] == set()
        assert parts[1] & parts[2] == set()


# ============================================================================
# to_anomaly_labels
# ============================================================================


class TestToAnomalyLabels:
    """The one place the event track's "normal" convention is applied."""

    def test_normal_maps_to_zero_and_everything_else_to_one(self) -> None:
        result = to_anomaly_labels(["normal", "brute_force", "normal", "data_exfiltration"])

        np.testing.assert_array_equal(result, np.array([0, 1, 0, 1]))

    def test_empty_input_gives_an_empty_array(self) -> None:
        assert to_anomaly_labels([]).shape == (0,)


# ============================================================================
# The CLI, end to end
# ============================================================================


@pytest.mark.slow
class TestTrainingCli:
    """A real run of `python -m ml.training.train`, small enough to be a test."""

    def _run(self, tmp_path: Path, *extra: str):
        result = CliRunner().invoke(
            main,
            [
                "--n-samples",
                "1200",
                "--attack-ratio",
                "0.3",
                "--n-estimators",
                "20",
                "--no-mlflow",
                "--no-window-features",
                "--output-dir",
                str(tmp_path),
                *extra,
            ],
        )
        if result.exit_code != 0:
            raise AssertionError(f"training failed: {result.output}\n{result.exception}")
        return json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))

    def test_fbeta_strategy_records_its_operating_point(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, "--threshold-strategy", "fbeta", "--beta", "2.0")

        threshold = summary["threshold"]
        assert threshold["strategy"] == "fbeta"
        assert threshold["beta"] == 2.0
        assert threshold["operating_point"]["beta"] == 2.0
        assert 0.0 <= threshold["operating_point"]["recall"] <= 1.0

    def test_recall_strategy_reaches_its_target_on_held_out_data(self, tmp_path: Path) -> None:
        """
        A threshold tuned on the validation set has to still mean something on
        the test set — recall near the target, precision clearly above the 0.3
        base rate.

        This is a smoke, not the regression test: a threshold fitted against
        permuted labels does not reliably *lower* recall. Measured on this
        generator it did the opposite, sliding to "flag everything" — recall
        1.00 at precision 0.20 for an operator who asked for 0.80. What pins the
        bug itself is the row-by-row alignment above; what this pins is that the
        end-to-end run produces an operating point that is not degenerate.
        """
        summary = self._run(tmp_path, "--threshold-strategy", "recall", "--target-recall", "0.8")

        assert summary["threshold"]["strategy"] == "recall"
        assert summary["metrics"]["anomaly"]["recall"] >= 0.7
        assert summary["metrics"]["anomaly"]["precision"] > 0.4

    def test_none_strategy_keeps_the_contamination_percentile(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, "--no-optimize-threshold")

        threshold = summary["threshold"]
        assert threshold["strategy"] == "none"
        assert threshold["operating_point"] is None
