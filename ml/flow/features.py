"""
The training↔inference feature contract of the flow-based track.

:data:`FLOW_FEATURE_NAMES` is the canonical column order of the 78 CICFlowMeter
features, frozen here and derived from the real CICIDS2017 headers — not
reconstructed from documentation. It is the same list
:func:`ml.flow.loader.get_feature_columns` produces on a loaded frame, and the
same list every saved cascade carries in its ``feature_names.json``. The three
are asserted equal by the test suite, because they are three views of one
contract.

Why this module exists at all: a cascade trained on columns in one order and
served data in another raises nothing, logs nothing and scores perfectly well on
its own test set. It simply predicts confident nonsense in production. So
inference never trusts the order of an incoming payload — it reorders **by
name**, through :class:`FlowFeatureExtractor`.

Note on ``Fwd Header Length.1``: that is not a typo and not a duplicate to be
"fixed". The raw CSVs carry ``Fwd Header Length`` twice (positions 34 and 55), a
known CICFlowMeter artifact documented in ``docs/ml/memoria-cicids2017.md``; the
``.1`` suffix is what pandas appends on read. Since a JSON object cannot carry
the same key twice, the pandas spelling is the canonical wire name, with a
dot-free alias accepted for clients that dislike it (see
:data:`FLOW_FEATURE_ALIASES`).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import structlog

logger = structlog.get_logger()


# ============================================================================
# The contract
# ============================================================================

# The 78 CICFlowMeter features, in the order get_feature_columns() returns them
# on a frame loaded by ml.flow.loader. Derived from the headers of the eight
# real CICIDS2017 MachineLearningCVE CSVs, which are identical to one another.
# Do not reorder, do not deduplicate, do not "clean up" the .1 suffix.
FLOW_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)

# Number of features a cascade of this track is trained on.
N_FLOW_FEATURES: Final[int] = len(FLOW_FEATURE_NAMES)

# Accepted spellings that are not the canonical name. Kept deliberately small:
# every alias is a second way to say the same thing, and each one is a chance
# for a client to believe it sent a feature it did not send.
#
# The only entry exists because the canonical name for the duplicated
# CICFlowMeter column is a pandas artifact (``.1``), which is awkward to type in
# a JSON client and easy to mistake for a version suffix.
FLOW_FEATURE_ALIASES: Final[dict[str, str]] = {
    "Fwd Header Length 1": "Fwd Header Length.1",
}

# Value substituted for a feature the payload omits, or whose value is not a
# finite number. Matches what clean_dataset() leaves behind at training time —
# with one asymmetry that is unavoidable and worth stating: training *drops*
# rows carrying NaN/Inf, while serving has to answer, so it zero-fills instead.
MISSING_VALUE: Final[float] = 0.0

# Key under which the wire format nests the model input. Everything outside it
# — src_ip, dst_ip, dst_port, protocol — is metadata that reaches the log and
# the explanation but never the model. The separation is structural rather than
# a denylist, so a new metadata field cannot accidentally become a feature.
FEATURES_KEY: Final[str] = "features"


def feature_mapping(flow: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Extract the model input from a flow in either accepted shape.

    Args:
        flow: Either the wire shape (``{"features": {...}, "src_ip": ...}``) or
            a flat feature mapping.

    Returns:
        The feature mapping. A flow carrying a ``features`` key that is not a
        mapping is treated as flat, so a client sending a feature literally
        named ``features`` still gets a sane answer rather than a crash.
    """
    nested = flow.get(FEATURES_KEY)
    if isinstance(nested, Mapping):
        return nested
    return flow


class FlowFeatureExtractor:
    """
    Turns a network-flow mapping into the model's input vector.

    Reordering is by name and only by name. Keys the contract does not know are
    ignored; features the payload does not carry become :data:`MISSING_VALUE`.
    Non-finite values (``inf``, ``-inf``, ``NaN``) are replaced the same way,
    which mirrors :func:`ml.flow.loader.clean_dataset`.

    Attributes:
        feature_names: The column order this extractor produces.

    Example:
        >>> extractor = FlowFeatureExtractor()
        >>> vector = extractor.transform({"Flow Duration": 1200, "SYN Flag Count": 1})
        >>> vector.shape
        (78,)
        >>> vector.dtype
        dtype('float32')
    """

    def __init__(self, feature_names: Sequence[str] | None = None) -> None:
        """
        Args:
            feature_names: Column order to produce. Defaults to
                :data:`FLOW_FEATURE_NAMES`. Override it only to exercise the
                extractor against a non-CICIDS column set in tests — serving
                validates the persisted order against the frozen contract, so an
                override never reaches production.

        Raises:
            ValueError: ``feature_names`` is empty or carries a duplicate. A
                repeated name would make one column unreachable by name, which
                is precisely the failure this class exists to prevent.
        """
        names = list(FLOW_FEATURE_NAMES if feature_names is None else feature_names)
        if not names:
            raise ValueError("feature_names must not be empty.")

        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"feature_names carries duplicate entries {duplicates}; a name must "
                "identify exactly one column for reordering-by-name to be well defined."
            )

        self._feature_names: list[str] = names
        # name → column position, plus the accepted aliases. Built once: the
        # per-flow path is a dict lookup per supplied key, not a scan of 78.
        self._index: dict[str, int] = {name: i for i, name in enumerate(names)}
        for alias, canonical in FLOW_FEATURE_ALIASES.items():
            if canonical in self._index and alias not in self._index:
                self._index[alias] = self._index[canonical]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """The column order produced by :meth:`transform`. A copy."""
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        """Width of the vectors produced by :meth:`transform`."""
        return len(self._feature_names)

    def transform(self, flow: Mapping[str, Any]) -> np.ndarray:
        """
        Build the feature vector of a single flow.

        Args:
            flow: Feature name → value. Order is irrelevant; unknown keys are
                ignored and absent features are zero-filled.

        Returns:
            ``float32`` array of shape ``(n_features,)``.
        """
        vector: np.ndarray = self.transform_batch([flow])[0]
        return vector

    def transform_batch(self, flows: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """
        Build the feature matrix of a batch of flows.

        Args:
            flows: Iterable of feature mappings.

        Returns:
            ``float32`` array of shape ``(n_flows, n_features)``. An empty batch
            yields a well-formed ``(0, n_features)`` matrix rather than an
            error — the serving layer is allowed to hand down an empty request.
        """
        rows = list(flows)
        matrix = np.full((len(rows), self.n_features), MISSING_VALUE, dtype=np.float64)

        for i, flow in enumerate(rows):
            matched = 0
            for key, value in flow.items():
                column = self._index.get(key)
                if column is None:
                    continue
                matrix[i, column] = _to_float(value)
                matched += 1

            if matched == 0:
                # An all-zero vector is a perfectly plausible-looking input that
                # the model will happily score. Saying so is the only way the
                # caller finds out its payload never reached the model.
                logger.warning(
                    "flow_features_no_known_names",
                    n_keys=len(flow),
                    keys=sorted(str(k) for k in flow)[:10],
                    hint="payload shares no key with FLOW_FEATURE_NAMES; vector is all zeros",
                )

        # inf/-inf/NaN in one pass over the whole batch, same substitution
        # clean_dataset applies at training time.
        np.nan_to_num(
            matrix, copy=False, nan=MISSING_VALUE, posinf=MISSING_VALUE, neginf=MISSING_VALUE
        )

        result: np.ndarray = matrix.astype(np.float32)
        return result

    def known_names(self, flow: Mapping[str, Any]) -> list[str]:
        """
        Return the keys of ``flow`` this extractor recognises, aliases included.

        Args:
            flow: Feature mapping to inspect.

        Returns:
            The recognised keys, in the order the contract lists their columns.
        """
        return sorted((k for k in flow if k in self._index), key=lambda k: self._index[k])


def _to_float(value: Any) -> float:
    """
    Coerce a payload value to float, falling back to :data:`MISSING_VALUE`.

    Booleans, numeric strings and numpy scalars all convert; anything else — a
    nested object, ``None``, a non-numeric string — becomes the missing value
    rather than aborting the whole request over one bad field.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return MISSING_VALUE
