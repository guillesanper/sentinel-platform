"""
Unit tests for the training↔inference feature contract of the flow track.

This file guards the most expensive failure the flow design can have. Feeding a
model's columns in the wrong order raises nothing, logs nothing and degrades no
offline metric — the cascade simply answers confidently from features it has
mixed up. Every assertion below exists so that failure has no path to
production.

The centrepiece is :class:`TestContractMatchesTheLoader`, which pins
``FLOW_FEATURE_NAMES`` against the **verbatim header of a real CICIDS2017 CSV**
rather than against anything derived from the constant itself. A circular test
here would pass forever while the constant drifted away from the data.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.flow.features import (
    FEATURES_KEY,
    FLOW_FEATURE_ALIASES,
    FLOW_FEATURE_NAMES,
    MISSING_VALUE,
    N_FLOW_FEATURES,
    FlowFeatureExtractor,
    feature_mapping,
)
from ml.flow.loader import get_feature_columns, load_cicids2017

# ============================================================================
# The real header, copied verbatim
# ============================================================================

# First line of api/csvs/Monday-WorkingHours.pcap_ISCX.csv, byte for byte,
# including the erratic leading spaces of the original. All eight CICIDS2017
# MachineLearningCVE files carry this same header.
#
# It is pasted rather than derived so that the contract test is not circular:
# reconstructing this line from FLOW_FEATURE_NAMES would assert only that the
# constant equals itself. Note that "Fwd Header Length" appears TWICE (once
# among the header fields, once again near "Avg Bwd Segment Size") — that is the
# known CICFlowMeter artifact, and the ".1" of the canonical name is the suffix
# pandas adds on read, not something present in the file.
REAL_CICIDS_HEADER: str = (
    " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
    "Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max,"
    " Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,"
    "Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean,"
    " Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std,"
    " Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max,"
    " Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,"
    "Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length,"
    " Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length,"
    " Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,"
    "FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count,"
    " URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio,"
    " Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size,"
    " Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate,"
    " Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets,"
    " Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward,"
    " Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean,"
    " Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
)

# 78 features + Label.
REAL_CICIDS_COLUMN_COUNT: int = 79


@pytest.fixture
def real_header_csv(tmp_path: Path) -> Path:
    """
    A one-row CSV carrying the genuine CICIDS2017 header, and its directory.

    Enough to exercise the whole loading path — column stripping, label mapping,
    provenance columns — without touching the 844 MB of real captures, which no
    unit test may read.
    """
    data_dir = tmp_path / "cicids-header"
    data_dir.mkdir()

    n_features = REAL_CICIDS_COLUMN_COUNT - 1
    row = ",".join(str(float(i)) for i in range(n_features)) + ",BENIGN"
    (data_dir / "Monday-WorkingHours.pcap_ISCX.csv").write_text(
        f"{REAL_CICIDS_HEADER}\n{row}\n", encoding="latin-1"
    )
    return data_dir


# ============================================================================
# Shape of the constant
# ============================================================================


class TestFlowFeatureNames:
    """Properties the frozen contract must hold on its own."""

    def test_has_78_features(self):
        assert len(FLOW_FEATURE_NAMES) == 78
        assert N_FLOW_FEATURES == 78

    def test_has_no_duplicates(self):
        """
        ``Fwd Header Length`` and ``Fwd Header Length.1`` are two *distinct*
        names for the two columns the raw CSV spells identically. Collapsing
        them would drop a real feature and shift every column after it.
        """
        assert len(set(FLOW_FEATURE_NAMES)) == len(FLOW_FEATURE_NAMES)

    def test_carries_both_spellings_of_the_duplicated_column(self):
        assert "Fwd Header Length" in FLOW_FEATURE_NAMES
        assert "Fwd Header Length.1" in FLOW_FEATURE_NAMES

    def test_is_immutable(self):
        """A tuple, not a list: the contract must not be editable in place."""
        assert isinstance(FLOW_FEATURE_NAMES, tuple)

    def test_names_are_stripped(self):
        """
        The raw header carries stray leading spaces; the loader strips them, so
        the contract must be stored stripped or every lookup misses by a space.
        """
        assert all(name == name.strip() for name in FLOW_FEATURE_NAMES)

    def test_excludes_the_provenance_columns(self):
        """
        ``capture_day`` and ``order_idx`` are the risk this whole track is built
        around: as features they are near-perfect predictors of the capture
        schedule and worthless as detectors.
        """
        assert "capture_day" not in FLOW_FEATURE_NAMES
        assert "order_idx" not in FLOW_FEATURE_NAMES
        assert "Label" not in FLOW_FEATURE_NAMES


# ============================================================================
# The contract against the loader — the assertion that matters most
# ============================================================================


class TestContractMatchesTheLoader:
    """
    ``FLOW_FEATURE_NAMES`` and ``get_feature_columns()`` are two views of one
    contract: the first is what inference builds, the second is what training
    consumes. If they ever disagree, every served prediction is wrong and
    nothing anywhere raises.
    """

    def test_matches_get_feature_columns_on_a_real_header(self, real_header_csv: Path):
        df = load_cicids2017(real_header_csv)

        # Equality of lists, not sets: order *is* the contract.
        assert get_feature_columns(df) == list(FLOW_FEATURE_NAMES)

    def test_the_pinned_header_really_holds_79_columns(self):
        assert len(REAL_CICIDS_HEADER.split(",")) == REAL_CICIDS_COLUMN_COUNT

    def test_the_raw_header_spells_fwd_header_length_twice(self):
        """
        The premise behind the ``.1`` suffix, asserted directly. If a future
        CICIDS distribution stops duplicating the column, this test fails and
        the canonical name has to be revisited — rather than the mismatch
        surfacing as silently shifted columns.
        """
        raw_names = [c.strip() for c in REAL_CICIDS_HEADER.split(",")]

        assert raw_names.count("Fwd Header Length") == 2

    def test_pandas_is_what_adds_the_dot_one_suffix(self, real_header_csv: Path):
        frame = pd.read_csv(
            real_header_csv / "Monday-WorkingHours.pcap_ISCX.csv",
            encoding="latin-1",
            low_memory=False,
        )
        columns = [str(c).strip() for c in frame.columns]

        assert "Fwd Header Length.1" in columns
        assert columns.count("Fwd Header Length") == 1


# ============================================================================
# FlowFeatureExtractor
# ============================================================================


class TestTransform:
    """Single-flow vectorisation."""

    @pytest.fixture
    def extractor(self) -> FlowFeatureExtractor:
        return FlowFeatureExtractor()

    def test_returns_one_float32_value_per_contract_column(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({"Flow Duration": 1200.0})

        assert vector.shape == (78,)
        assert vector.dtype == np.float32

    def test_places_each_value_under_its_own_name(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({"Destination Port": 443.0, "Idle Min": 7.0})

        assert vector[FLOW_FEATURE_NAMES.index("Destination Port")] == 443.0
        assert vector[FLOW_FEATURE_NAMES.index("Idle Min")] == 7.0

    def test_permuted_keys_give_an_identical_vector(self, extractor: FlowFeatureExtractor):
        """
        The single most important behaviour in this module. A client is free to
        serialise its JSON object in any order; the vector must not move.
        """
        flow = {name: float(i) for i, name in enumerate(FLOW_FEATURE_NAMES)}
        permuted = dict(reversed(list(flow.items())))

        np.testing.assert_array_equal(
            extractor.transform(flow),
            extractor.transform(permuted),
        )

    def test_a_full_flow_round_trips_in_contract_order(self, extractor: FlowFeatureExtractor):
        """Reordering by name must reproduce the canonical order exactly."""
        flow = {name: float(i) for i, name in enumerate(FLOW_FEATURE_NAMES)}
        permuted = dict(sorted(flow.items()))

        np.testing.assert_array_equal(
            extractor.transform(permuted),
            np.arange(78, dtype=np.float32),
        )

    def test_missing_features_become_zero(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({"Flow Duration": 5.0})

        assert vector[FLOW_FEATURE_NAMES.index("Flow Duration")] == 5.0
        assert vector.sum() == 5.0

    def test_an_empty_flow_is_all_zeros(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({})

        assert vector.shape == (78,)
        assert not vector.any()

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_values_become_zero(self, extractor: FlowFeatureExtractor, bad: float):
        """
        Same substitution ``clean_dataset`` implies at training time. Training
        drops such rows; serving has to answer, so it zero-fills.
        """
        vector = extractor.transform({"Flow Bytes/s": bad, "Flow Duration": 3.0})

        assert np.isfinite(vector).all()
        assert vector[FLOW_FEATURE_NAMES.index("Flow Bytes/s")] == MISSING_VALUE
        assert vector[FLOW_FEATURE_NAMES.index("Flow Duration")] == 3.0

    @pytest.mark.parametrize("bad", [None, "not a number", [1, 2], {"a": 1}])
    def test_unparseable_values_become_zero_without_failing_the_request(
        self, extractor: FlowFeatureExtractor, bad: object
    ):
        vector = extractor.transform({"Flow Bytes/s": bad, "Flow Duration": 3.0})

        assert vector[FLOW_FEATURE_NAMES.index("Flow Bytes/s")] == MISSING_VALUE
        assert vector[FLOW_FEATURE_NAMES.index("Flow Duration")] == 3.0

    def test_numeric_strings_are_accepted(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({"Flow Duration": "1200.5"})

        assert vector[FLOW_FEATURE_NAMES.index("Flow Duration")] == pytest.approx(1200.5)

    def test_unknown_keys_are_ignored(self, extractor: FlowFeatureExtractor):
        vector = extractor.transform({"capture_day": 4.0, "Label": 1.0, "Flow Duration": 2.0})

        assert vector.sum() == 2.0

    def test_aliases_land_on_their_canonical_column(self, extractor: FlowFeatureExtractor):
        for alias, canonical in FLOW_FEATURE_ALIASES.items():
            vector = extractor.transform({alias: 42.0})

            assert vector[FLOW_FEATURE_NAMES.index(canonical)] == 42.0

    def test_a_payload_with_no_known_names_warns(
        self, extractor: FlowFeatureExtractor, monkeypatch: pytest.MonkeyPatch
    ):
        """
        An all-zero vector scores like any other input. Saying so out loud is
        the only way a client discovers its payload never reached the model.

        ``caplog`` and ``structlog.testing.capture_logs`` are unreliable in this
        suite (the logger is cached on first use), so the module logger is
        patched directly.
        """
        events: list[str] = []

        class _Spy:
            def warning(self, event, **kwargs):
                events.append(event)

            def __getattr__(self, _name):
                return lambda *a, **k: None

        monkeypatch.setattr("ml.flow.features.logger", _Spy())
        extractor.transform({"totally": 1, "wrong": 2})

        assert "flow_features_no_known_names" in events


class TestTransformBatch:
    """Batch vectorisation."""

    @pytest.fixture
    def extractor(self) -> FlowFeatureExtractor:
        return FlowFeatureExtractor()

    def test_shape_and_dtype(self, extractor: FlowFeatureExtractor):
        matrix = extractor.transform_batch([{"Flow Duration": 1.0}, {"Flow Duration": 2.0}])

        assert matrix.shape == (2, 78)
        assert matrix.dtype == np.float32

    def test_rows_match_transform_one_by_one(self, extractor: FlowFeatureExtractor):
        flows = [
            {"Flow Duration": 1.0, "Idle Max": 9.0},
            {"Destination Port": 22.0},
            {},
        ]

        matrix = extractor.transform_batch(flows)

        for i, flow in enumerate(flows):
            np.testing.assert_array_equal(matrix[i], extractor.transform(flow))

    def test_an_empty_batch_yields_a_well_formed_empty_matrix(
        self, extractor: FlowFeatureExtractor
    ):
        """A serving layer is allowed to hand down nothing; that is not an error."""
        matrix = extractor.transform_batch([])

        assert matrix.shape == (0, 78)

    def test_accepts_a_generator(self, extractor: FlowFeatureExtractor):
        matrix = extractor.transform_batch({"Flow Duration": float(i)} for i in range(3))

        assert matrix.shape == (3, 78)

    def test_rows_do_not_leak_into_one_another(self, extractor: FlowFeatureExtractor):
        matrix = extractor.transform_batch([{"Flow Duration": 5.0}, {}])

        assert matrix[0].sum() == 5.0
        assert matrix[1].sum() == 0.0


class TestExtractorConstruction:
    """The extractor's own invariants."""

    def test_defaults_to_the_frozen_contract(self):
        assert FlowFeatureExtractor().feature_names == list(FLOW_FEATURE_NAMES)

    def test_feature_names_hands_out_a_copy(self):
        extractor = FlowFeatureExtractor()
        extractor.feature_names.append("injected")

        assert extractor.feature_names == list(FLOW_FEATURE_NAMES)

    def test_accepts_an_explicit_column_set(self):
        extractor = FlowFeatureExtractor(["b", "a"])

        np.testing.assert_array_equal(extractor.transform({"a": 1.0, "b": 2.0}), [2.0, 1.0])

    def test_rejects_duplicate_names(self):
        """
        A repeated name makes one column unreachable by name — precisely the
        failure this class exists to prevent, so it must not be constructible.
        """
        with pytest.raises(ValueError, match="duplicate"):
            FlowFeatureExtractor(["a", "b", "a"])

    def test_rejects_an_empty_column_set(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FlowFeatureExtractor([])

    def test_known_names_reports_recognised_keys_in_contract_order(self):
        extractor = FlowFeatureExtractor()

        known = extractor.known_names({"Idle Min": 1, "Destination Port": 2, "nonsense": 3})

        assert known == ["Destination Port", "Idle Min"]


# ============================================================================
# Wire shape
# ============================================================================


class TestFeatureMapping:
    """
    Metadata is separated from features structurally, not by a denylist, so a
    new metadata field can never accidentally become a model input.
    """

    def test_reads_the_nested_features_object(self):
        flow = {FEATURES_KEY: {"Flow Duration": 1.0}, "src_ip": "10.0.0.1"}

        assert feature_mapping(flow) == {"Flow Duration": 1.0}

    def test_metadata_never_reaches_the_vector(self):
        flow = {
            FEATURES_KEY: {"Flow Duration": 1.0},
            "src_ip": "10.0.0.1",
            "dst_port": 443,
            "protocol": "TCP",
        }

        vector = FlowFeatureExtractor().transform(feature_mapping(flow))

        assert vector.sum() == 1.0

    def test_a_flat_mapping_is_passed_through(self):
        flow = {"Flow Duration": 1.0}

        assert feature_mapping(flow) is flow

    def test_a_non_mapping_features_key_is_treated_as_flat(self):
        """A client sending a feature literally named ``features`` gets an
        answer, not a crash."""
        flow = {FEATURES_KEY: 3.0, "Flow Duration": 1.0}

        assert feature_mapping(flow) == flow


# ============================================================================
# Import hygiene
# ============================================================================


def test_features_module_does_not_import_matplotlib():
    """
    ``api/Dockerfile`` copies ``ml/`` wholesale and installs only
    ``requirements.txt``, where matplotlib is deliberately absent. A
    module-level plotting import on the per-request path would either break the
    image or add ~120 MB and its CVE surface to a security product.

    (pandas is not asserted against: ``ml/__init__.py`` imports the event-based
    classifier, which pulls xgboost and therefore pandas, long before anything
    under ``ml.flow`` is reached. That is the event track's dependency, not this
    module's — ``features.py`` itself imports only numpy and structlog.)
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml.flow.features, sys; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
