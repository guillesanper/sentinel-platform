"""
Unit tests for the CICIDS2017 label taxonomy (flow-based track).

The load-bearing test here is the mojibake one: if the latin-1 repair is lost,
the three "Web Attack" labels stop resolving, their rows are silently dropped
by the loader, and the WebAttack class disappears from training without a
single error being raised.
"""

import pytest

from ml.flow.labels import (
    ATTACK_GROUPS,
    BENIGN_LABEL,
    CAPTURE_DAY_ORDER,
    CAPTURE_DAYS,
    CAPTURE_FILE_ORDER,
    CLASS_ORDER,
    UNKNOWN_CAPTURE_DAY,
    UNKNOWN_CAPTURE_FILE_RANK,
    capture_file_rank,
    map_capture_day,
    map_label,
)

# ============================================================================
# Taxonomy invariants
# ============================================================================


class TestTaxonomy:
    """Structural invariants of the 8-class taxonomy."""

    def test_class_order_has_eight_classes(self):
        assert len(CLASS_ORDER) == 8

    def test_infiltration_is_its_own_class(self):
        """
        It used to be folded into ``Scan``, which merged 36 flows of a
        host-compromise scenario into 158.930 of port sweeping *and* smuggled
        them past ``filter_minority_classes`` — the filter runs after the
        collapse, so ``min_class_samples`` never saw them. Separate, the filter
        drops the class on its own, which is what it is for.
        """
        assert "Infiltration" in CLASS_ORDER
        assert ATTACK_GROUPS["Infiltration"] == "Infiltration"
        assert ATTACK_GROUPS["PortScan"] == "Scan"

    def test_class_order_has_no_duplicates(self):
        assert len(set(CLASS_ORDER)) == len(CLASS_ORDER)

    def test_benign_is_first_in_class_order(self):
        assert CLASS_ORDER[0] == BENIGN_LABEL

    def test_attack_groups_only_target_canonical_classes(self):
        assert set(ATTACK_GROUPS.values()) == set(CLASS_ORDER)

    def test_attack_groups_covers_all_fifteen_raw_labels(self):
        assert len(ATTACK_GROUPS) == 15

    def test_benign_maps_to_itself(self):
        assert ATTACK_GROUPS[BENIGN_LABEL] == BENIGN_LABEL

    def test_disjoint_from_event_track_taxonomy(self):
        """Flow and event taxonomies must not overlap — mixing them is silent."""
        from ml.models.classifier import ATTACK_TYPES

        assert not set(CLASS_ORDER) & set(ATTACK_TYPES)


# ============================================================================
# map_label
# ============================================================================


class TestMapLabel:
    """Raw CICIDS2017 label → canonical class."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("BENIGN", "BENIGN"),
            ("DoS Hulk", "DoS"),
            ("DoS GoldenEye", "DoS"),
            ("DoS slowloris", "DoS"),
            ("DoS Slowhttptest", "DoS"),
            ("Heartbleed", "DoS"),
            ("DDoS", "DDoS"),
            ("PortScan", "Scan"),
            ("Infiltration", "Infiltration"),
            ("FTP-Patator", "BruteForce"),
            ("SSH-Patator", "BruteForce"),
            ("Bot", "Botnet"),
        ],
    )
    def test_maps_ascii_labels(self, raw, expected):
        assert map_label(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Web Attack \u2013 Brute Force", "WebAttack"),
            ("Web Attack \u2013 XSS", "WebAttack"),
            ("Web Attack \u2013 Sql Injection", "WebAttack"),
        ],
    )
    def test_maps_web_attack_labels_with_real_en_dash(self, raw, expected):
        assert map_label(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Web Attack ï¿½ Brute Force", "WebAttack"),
            ("Web Attack ï¿½ XSS", "WebAttack"),
            ("Web Attack ï¿½ Sql Injection", "WebAttack"),
        ],
    )
    def test_repairs_latin1_mojibake(self, raw, expected):
        """
        The three characters 'ï¿½' are the UTF-8 bytes of U+FFFD read back as
        latin-1. Without this repair the WebAttack class silently vanishes.
        """
        assert map_label(raw) == expected

    def test_mojibake_case_from_the_plan(self):
        assert map_label("Web Attack ï¿½ XSS") == "WebAttack"

    @pytest.mark.parametrize(
        "raw",
        ["  BENIGN", "BENIGN  ", "\tDoS Hulk\n", " Web Attack ï¿½ XSS "],
    )
    def test_strips_surrounding_whitespace(self, raw):
        assert map_label(raw) is not None

    @pytest.mark.parametrize(
        "raw",
        ["Not An Attack", "", "benign", "DOS HULK", "Web Attack - XSS"],
    )
    def test_returns_none_for_unknown_labels(self, raw):
        assert map_label(raw) is None

    def test_returns_none_for_none(self):
        assert map_label(None) is None

    @pytest.mark.parametrize("raw", [float("nan"), 42, 3.14])
    def test_returns_none_for_non_string_values(self, raw):
        assert map_label(raw) is None

    def test_every_attack_group_key_round_trips(self):
        for raw, expected in ATTACK_GROUPS.items():
            assert map_label(raw) == expected


# ============================================================================
# map_capture_day
# ============================================================================


class TestMapCaptureDay:
    """CSV filename → capture day."""

    def test_all_canonical_filenames_resolve(self):
        for filename, expected in CAPTURE_DAYS.items():
            assert map_capture_day(filename) == expected

    def test_capture_days_only_yield_known_weekdays(self):
        assert set(CAPTURE_DAYS.values()) <= set(CAPTURE_DAY_ORDER)

    def test_all_five_weekdays_are_covered(self):
        assert set(CAPTURE_DAYS.values()) == set(CAPTURE_DAY_ORDER)

    def test_friday_spans_three_files(self):
        friday_files = [f for f, d in CAPTURE_DAYS.items() if d == "friday"]
        assert len(friday_files) == 3

    def test_thursday_spans_two_files(self):
        thursday_files = [f for f, d in CAPTURE_DAYS.items() if d == "thursday"]
        assert len(thursday_files) == 2

    def test_accepts_a_full_posix_path(self):
        path = "/data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        assert map_capture_day(path) == "friday"

    def test_accepts_a_full_windows_path(self):
        path = r"C:\data\cicids2017\Monday-WorkingHours.pcap_ISCX.csv"
        assert map_capture_day(path) == "monday"

    def test_is_case_insensitive(self):
        assert map_capture_day("MONDAY-WORKINGHOURS.PCAP_ISCX.CSV") == "monday"

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("Tuesday-mini.csv", "tuesday"),
            ("wednesday-WorkingHours-sample.csv", "wednesday"),
            ("Friday-generated-20000rows.csv", "friday"),
        ],
    )
    def test_falls_back_to_the_leading_weekday_token(self, filename, expected):
        """Generated sample CSVs keep the pattern but not the exact names."""
        assert map_capture_day(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        ["random.csv", "data.csv", "Saturday-WorkingHours.csv", ""],
    )
    def test_returns_unknown_for_unrecognised_filenames(self, filename):
        assert map_capture_day(filename) == UNKNOWN_CAPTURE_DAY

    def test_unknown_sentinel_is_not_a_weekday(self):
        assert UNKNOWN_CAPTURE_DAY not in CAPTURE_DAY_ORDER


# ============================================================================
# capture_file_rank
# ============================================================================


class TestCaptureFileRank:
    """
    CSV filename → position in the capture week.

    Exists because the filenames do not sort chronologically, and ``order_idx``
    is a counter over the loader's reading order.
    """

    def test_capture_file_order_covers_exactly_the_known_filenames(self):
        """
        A file present in one table and missing from the other is the failure
        mode: it would keep a capture day but lose its place in the week.
        """
        assert set(CAPTURE_FILE_ORDER) == set(CAPTURE_DAYS)
        assert len(CAPTURE_FILE_ORDER) == len(CAPTURE_DAYS)

    def test_ranks_are_a_permutation_of_range(self):
        ranks = [capture_file_rank(name) for name in CAPTURE_FILE_ORDER]
        assert ranks == list(range(len(CAPTURE_FILE_ORDER)))

    def test_the_days_are_ranked_in_weekday_order(self):
        days = [map_capture_day(name) for name in CAPTURE_FILE_ORDER]
        assert days == sorted(days, key=CAPTURE_DAY_ORDER.index)

    def test_morning_captures_rank_before_afternoon_ones(self):
        """
        The whole point: alphabetically 'Afternoon' precedes 'Morning', so a
        sorted glob inverts both Thursday and Friday.
        """
        assert capture_file_rank(
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
        ) < capture_file_rank("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        assert capture_file_rank("Friday-WorkingHours-Morning.pcap_ISCX.csv") < capture_file_rank(
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        )

    def test_the_two_friday_afternoon_captures_keep_their_recorded_order(self):
        assert capture_file_rank(
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ) < capture_file_rank("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

    def test_is_case_insensitive(self):
        assert capture_file_rank("MONDAY-WORKINGHOURS.PCAP_ISCX.CSV") == capture_file_rank(
            "Monday-WorkingHours.pcap_ISCX.csv"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/data/cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
            r"C:\data\cicids2017\Friday-WorkingHours-Morning.pcap_ISCX.csv",
        ],
    )
    def test_accepts_a_full_path(self, path):
        assert capture_file_rank(path) == capture_file_rank(
            "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        )

    @pytest.mark.parametrize(
        "filename",
        ["random.csv", "Friday-generated-20000rows.csv", "Saturday-WorkingHours.csv", ""],
    )
    def test_unrecognised_filenames_rank_last(self, filename):
        assert capture_file_rank(filename) == UNKNOWN_CAPTURE_FILE_RANK
        assert all(capture_file_rank(known) < capture_file_rank(filename) for known in CAPTURE_DAYS)
