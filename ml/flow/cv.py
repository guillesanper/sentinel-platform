"""
Chronological cross-validation folds for the flow-based track.

``ml.flow.loader`` partitions CICIDS2017 so that no test flow precedes a train
flow inside its own scenario. Until this module existed, the Stage 2
hyperparameter search then threw that away: ``RandomizedSearchCV`` cut the
training anomalies with a shuffled ``StratifiedKFold``, which puts
near-duplicates from a single attack burst on both sides of a fold boundary —
the exact leakage the partition exists to remove — the gap between the random
and the temporal protocol is 0.299 of F1-macro where it has been measured
(``docs/ml/PARTICION-TEMPORAL.md`` §4). The damage was bounded
— it biases *which* hyperparameters win, never a published metric, all of which
come from the temporal test set — but it meant the search was answering a
different question from the one the run reports.

**Why the folds are per block and not one global chronological cut.** The days
of CICIDS2017 are disjoint by class: Tuesday is BruteForce, Wednesday DoS,
Friday Botnet/Scan/DDoS. A global cut on ``order_idx`` would therefore hand each
fold a validation set full of classes its own training half never contains, and
score models on their ability to name an attack they have not seen. The finest
partition in which "earlier" is a meaningful statement is the block
``(capture_day, raw_label)`` — the same unit :func:`ml.flow.loader._temporal_cut`
cuts — so that is the unit the folds are built from.

**Expanding window, not K-fold over blocks.** With ``n_splits=K`` every block is
cut into ``K+1`` chronological chunks and fold *k* trains on chunks ``0..k-1``
and validates on chunk *k*. A K-fold over blocks would put chunks *later* than
the validation chunk into the training half, reinstating the future→past
relationship in a different costume. Two consequences worth stating rather than
discovering:

* The folds train on 1/(K+1), 2/(K+1) ... K/(K+1) of the rows, never on all of
  them. That is ``TimeSeriesSplit``'s conservative bias, and it favours simpler
  models — a known property of the scheme, not a defect of this implementation.
* It is *cheaper* than the shuffled K-fold it replaces: (1+2+3)/4 = 1.5n of
  fitting against 2.0n for ``StratifiedKFold(3)``, ~25% less.

A block smaller than the number of chunks goes wholesale into the first chunk,
so it trains in every fold and validates in none — the same rule, from the same
function, that sends a tiny block wholesale to train in the split itself.
"""

from dataclasses import dataclass

import numpy as np
import structlog

from ml.flow.loader import block_boundaries

logger = structlog.get_logger()


@dataclass(frozen=True, eq=False)
class TemporalIndex:
    """
    Chronological position and block membership of every row of a matrix.

    Lives here rather than in ``ml.training.train_cicids`` so that the pipelines
    and the Stage 2 classifier can type against it without importing the CLI.
    Build one with :func:`ml.flow.loader.build_temporal_index`, which is where
    the block key and the provenance columns are known.

    The two arrays are positional: ``order[i]`` and ``block[i]`` describe row
    *i* of the feature matrix. That correspondence is the whole contract, and
    nothing checks it beyond the length — an index carried across a row
    selection its matrix did not undergo produces plausible folds and useless
    hyperparameters, silently. Use :meth:`mask` with the very same boolean array
    that selected the rows.

    Attributes:
        order: ``order_idx`` value of each row, int64. Only the ordering
            matters; the values are never compared across blocks.
        block: Integer code of the row's ``(capture_day, label)`` block, int32.
            Codes, not concatenated strings: a ``"|".join`` key collides the day
            a label contains the separator.

    Note:
        ``eq=False``: the generated ``__eq__`` would compare two NumPy arrays and
        raise "truth value of an array is ambiguous" at the first ``==``.
    """

    order: np.ndarray
    block: np.ndarray

    def __post_init__(self) -> None:
        """
        Raise when the two arrays cannot describe the same rows.

        Raises:
            ValueError: The arrays are not one-dimensional or disagree in length.
        """
        if self.order.ndim != 1 or self.block.ndim != 1:
            raise ValueError(
                f"TemporalIndex takes two 1-D arrays; got order.ndim={self.order.ndim} "
                f"and block.ndim={self.block.ndim}."
            )
        if self.order.shape[0] != self.block.shape[0]:
            raise ValueError(
                f"order has {self.order.shape[0]} entries but block has "
                f"{self.block.shape[0]}; they must describe the same rows."
            )

    def __len__(self) -> int:
        """Number of rows described."""
        return int(self.order.shape[0])

    def mask(self, mask: np.ndarray) -> "TemporalIndex":
        """
        Select the same rows a boolean mask selects from the matrix.

        Args:
            mask: Boolean array of length ``len(self)``.

        Returns:
            A new index over the selected rows, in the same order.

        Raises:
            ValueError: ``mask`` is not boolean or has the wrong length. Both are
                programming errors — an index that no longer lines up with its
                matrix is undetectable downstream — so they fail here rather than
                producing a shorter index that still looks usable.
        """
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(f"mask must be a boolean array, got dtype {mask.dtype}.")
        if mask.shape != (len(self),):
            raise ValueError(
                f"mask has shape {mask.shape} but this index describes {len(self)} rows."
            )
        return TemporalIndex(order=self.order[mask], block=self.block[mask])


def temporal_block_folds(
    index: TemporalIndex,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window cross-validation folds, chronological within a block.

    Every block is sorted by ``order_idx`` and cut into ``n_splits + 1``
    chronological chunks with :func:`ml.flow.loader.block_boundaries` — the same
    function, and therefore the same rounding and the same tiny-block rule, that
    the train/val/test partition uses. Fold *k* then trains on the union of
    chunks ``0..k-1`` across all blocks and validates on chunk *k*.

    Args:
        index: Positions and block codes of the rows of the matrix to be split.
        n_splits: Number of folds, i.e. one less than the number of chunks.

    Returns:
        ``n_splits`` pairs of **row positions** — ``(train_positions,
        val_positions)`` — each sorted ascending. Positions, not ``order_idx``
        values: :func:`ml.flow.loader._temporal_cut` shuffles every part it
        returns, so the rows of a training frame are *not* in chronological
        order. Only the ``order_idx`` values are, and this function is the only
        thing that reads them.

    Raises:
        ValueError: ``n_splits`` is below 1.

    Note:
        A fold can come back with an empty validation half — one block, cut into
        four, contributes nothing to chunk 3 if it holds three rows. Callers
        decide what to do about that; the pairs are returned as computed so the
        decision is not buried here. :meth:`FlowAttackClassifier._build_cv
        <ml.flow.models.xgboost_classifier.FlowAttackClassifier>` drops such
        folds and falls back entirely when fewer than two survive.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}.")

    n_chunks = n_splits + 1
    ratios = tuple([1.0 / n_chunks] * n_chunks)
    chunks: list[list[np.ndarray]] = [[] for _ in range(n_chunks)]

    for code in np.unique(index.block):
        positions = np.flatnonzero(index.block == code)
        # Stable, so rows sharing an order_idx keep their relative arrangement
        # and the folds are reproducible run to run.
        positions = positions[np.argsort(index.order[positions], kind="stable")]

        bounds = block_boundaries(len(positions), ratios)
        for i in range(n_chunks):
            piece = positions[bounds[i] : bounds[i + 1]]
            if piece.size:
                chunks[i].append(piece)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_so_far: list[np.ndarray] = []
    for k in range(1, n_chunks):
        train_so_far.extend(chunks[k - 1])
        folds.append((_positions(train_so_far), _positions(chunks[k])))

    logger.debug(
        "temporal_block_folds_built",
        n_splits=n_splits,
        n_blocks=int(np.unique(index.block).size),
        train_sizes=[int(train.size) for train, _ in folds],
        val_sizes=[int(val.size) for _, val in folds],
    )
    return folds


def _positions(pieces: list[np.ndarray]) -> np.ndarray:
    """Flatten block chunks into one ascending array of row positions."""
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(pieces)).astype(np.int64)
