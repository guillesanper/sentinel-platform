"""
CICIDS2017 label taxonomy for the flow-based track.

Single source of truth for:

* ``ATTACK_GROUPS`` — the 15 raw CICIDS2017 labels collapsed into 8 canonical
  classes.
* ``map_label`` — raw label (as read from the CSVs) → canonical class, or None
  when the label is not recognised. Carries the latin-1 mojibake repair that
  used to be buried in ``utils/loader.py::_map_labels``.
* ``normalize_raw_label`` — the same repair *without* the collapse, so that the
  attack tool behind a class stays visible. The temporal split blocks on it.
* ``CAPTURE_DAYS`` / ``map_capture_day`` — CSV filename → capture day. The
  MachineLearningCVE distribution of CICIDS2017 has no ``Timestamp`` column, so
  the filename is the only carrier of time.
* ``CAPTURE_FILE_ORDER`` / ``capture_file_rank`` — the same filename → its
  position in the capture week. Needed because the filenames do *not* sort
  chronologically, and ``ml.flow.loader`` derives ``order_idx`` from the order in
  which it reads them.

Note that the collapse is lossy in a way that matters for partitioning, not just
for reporting: the tools merged into one class ran **sequentially** inside their
capture day, so a chronological cut on the collapsed class holds whole tools out
of training. That is why ``ml.flow.loader`` keeps the raw label around and can
block on it. See ``docs/ml/PARTICION-TEMPORAL.md``.

This taxonomy is intentionally disjoint from ``ml.models.classifier.ATTACK_TYPES``
(the event-based track): mixing them yields a model that silently predicts
``normal`` for everything.
"""

from typing import Final

import structlog

logger = structlog.get_logger()


# ============================================================================
# Mojibake repair
# ============================================================================

# The CICIDS2017 "Web Attack" labels carry an EN DASH (U+2013). In the
# MachineLearningCVE CSVs those bytes survive as the UTF-8 encoding of U+FFFD
# (EF BF BD), which, read back with the latin-1 codec the loader must use,
# renders as the three characters 'ï¿½'. Restoring them to U+2013 before the
# ATTACK_GROUPS lookup is what keeps the WebAttack class alive: without it the
# three Web Attack labels are unrecognised, their rows are discarded, and the
# class vanishes from training with no error raised anywhere.
_MOJIBAKE_EN_DASH: Final[str] = "ï¿½"
# The EN DASH below is load-bearing, not a typo for a hyphen: it is the
# character the three "Web Attack" labels actually contain. test_flow_labels.py
# pins it with an explicit U+2013 escape, so a bad encoding round-trip on this
# file fails the suite instead of silently dropping the WebAttack class.
_EN_DASH: Final[str] = "–"  # noqa: RUF001


# ============================================================================
# Class taxonomy
# ============================================================================

# Raw CICIDS2017 label → canonical 8-class taxonomy.
ATTACK_GROUPS: Final[dict[str, str]] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Scan",
    # Its own class, not a `Scan`. The TFA folded Infiltration's 36 flows into
    # the 158.930 of PortScan, which merges a host-compromise-and-exfiltration
    # scenario into port sweeping on the strength of one shared phase. It also
    # defeated the minority-class filter: `filter_minority_classes` runs *after*
    # the collapse, so `min_class_samples=100` never saw the 36 rows. Kept
    # separate, the filter drops it on its own — which is what the filter is for.
    "Infiltration": "Infiltration",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    f"Web Attack {_EN_DASH} Brute Force": "WebAttack",
    f"Web Attack {_EN_DASH} XSS": "WebAttack",
    f"Web Attack {_EN_DASH} Sql Injection": "WebAttack",
    "Bot": "Botnet",
}

# Canonical class representing legitimate traffic.
BENIGN_LABEL: Final[str] = "BENIGN"

# Canonical class order (BENIGN first, attacks alphabetical).
#
# `Infiltration` is listed even though CICIDS2017 carries only 36 such flows and
# `filter_minority_classes` therefore drops it from every realistic run. The
# taxonomy states what the labels mean; whether a class has enough support to be
# modelled is a separate decision, taken later and logged when it is taken.
CLASS_ORDER: Final[tuple[str, ...]] = (
    "BENIGN",
    "Botnet",
    "BruteForce",
    "DDoS",
    "DoS",
    "Infiltration",
    "Scan",
    "WebAttack",
)


# ============================================================================
# Capture days
# ============================================================================

# Sentinel value for CSVs whose filename carries no recognisable weekday.
UNKNOWN_CAPTURE_DAY: Final[str] = "unknown"

# CICIDS2017 MachineLearningCVE filename → capture day. Thursday and Friday
# each span several files; they collapse to a single day because the day is the
# unit of the day-holdout protocol, and because within a day the attack classes
# are disjoint anyway (Thu-AM WebAttack / Thu-PM Infiltration; Fri-AM Botnet /
# Fri-PM DDoS / Fri-PM PortScan).
CAPTURE_DAYS: Final[dict[str, str]] = {
    "Monday-WorkingHours.pcap_ISCX.csv": "monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "thursday",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "thursday",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "friday",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "friday",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "friday",
}

# Chronological order of the capture week — useful for day-holdout selection.
CAPTURE_DAY_ORDER: Final[tuple[str, ...]] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
)

# The eight capture files in the order they were recorded.
#
# Sorting the filenames does **not** produce this order, and the difference is
# not cosmetic: alphabetically `Friday-…-Afternoon-DDos` < `Friday-…-Afternoon-
# PortScan` < `Friday-…-Morning`, so the two afternoon captures would come before
# the morning one; Thursday inverts the same way. `ml.flow.loader` numbers rows
# by the order in which it reads the files, so a sorted glob makes `order_idx`
# claim an afternoon flow happened before a morning one.
#
# Attack blocks survive that inversion untouched — every attack label lives in
# exactly one file — but `(friday, BENIGN)` spans three files and
# `(thursday, BENIGN)` two, and BENIGN is precisely the population Stage 1 is
# fitted on. See `ml.flow.loader.load_cicids2017`.
CAPTURE_FILE_ORDER: Final[tuple[str, ...]] = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
)

# Rank handed to filenames that are not part of the canonical distribution: they
# sort after every known capture, and the caller is expected to break the
# resulting tie by name so the load order stays deterministic.
UNKNOWN_CAPTURE_FILE_RANK: Final[int] = len(CAPTURE_FILE_ORDER)

# Case-insensitive lookup tables, built once at import time.
_CAPTURE_DAYS_CI: Final[dict[str, str]] = {name.lower(): day for name, day in CAPTURE_DAYS.items()}
_WEEKDAYS_CI: Final[frozenset[str]] = frozenset(CAPTURE_DAY_ORDER)
_CAPTURE_FILE_RANKS_CI: Final[dict[str, int]] = {
    name.lower(): rank for rank, name in enumerate(CAPTURE_FILE_ORDER)
}


# ============================================================================
# Public API
# ============================================================================


def _basename(filename: str) -> str:
    """Last path component of ``filename``, for both separators."""
    return filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def normalize_raw_label(raw: object) -> str:
    """
    Clean a raw CICIDS2017 label without collapsing it into its canonical class.

    Same repair :func:`map_label` applies — latin-1 mojibake, then whitespace —
    but it stops before the ``ATTACK_GROUPS`` lookup, so ``DoS Hulk`` and
    ``DoS GoldenEye`` stay distinct instead of both becoming ``DoS``. That
    distinction is what the ``sub-label`` temporal blocking partitions on; see
    :data:`ml.flow.loader.RAW_LABEL_COL`.

    Args:
        raw: Label as read from the CSV.

    Returns:
        The repaired label. Unrecognised values come back cleaned rather than
        as None — this function does not judge, it only normalises.

    Example:
        >>> normalize_raw_label("  DoS Hulk ")
        'DoS Hulk'
        >>> normalize_raw_label("Web Attack \\u00ef\\u00bf\\u00bd XSS") == "Web Attack \\u2013 XSS"
        True
    """
    return str(raw).replace(_MOJIBAKE_EN_DASH, _EN_DASH).strip()


def map_label(raw: object) -> str | None:
    """
    Map a raw CICIDS2017 label to its canonical class.

    Applies, in order: latin-1 mojibake repair, whitespace stripping, and the
    ``ATTACK_GROUPS`` lookup.

    Args:
        raw: Label as read from the CSV. Non-string values are coerced with
            ``str()``, so NaN and numeric junk simply come back as None.

    Returns:
        Canonical class name, or None when the label is not recognised. Callers
        are expected to drop the corresponding rows.

    Example:
        >>> map_label("  DoS Hulk ")
        'DoS'
        >>> map_label("Web Attack \\u00ef\\u00bf\\u00bd XSS")
        'WebAttack'
        >>> map_label("Not An Attack") is None
        True
    """
    if raw is None:
        return None
    return ATTACK_GROUPS.get(normalize_raw_label(raw))


def map_capture_day(filename: str) -> str:
    """
    Derive the capture day from a CICIDS2017 CSV filename.

    Resolution order: exact match in ``CAPTURE_DAYS``, then case-insensitive
    match, then the leading weekday token of the filename (which is what makes
    generated sample CSVs work as long as they keep the real naming pattern).

    Args:
        filename: File name (a full path is accepted; only the last component
            is inspected).

    Returns:
        Lowercase weekday, or ``UNKNOWN_CAPTURE_DAY`` when the filename carries
        no recognisable weekday.

    Example:
        >>> map_capture_day("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        'friday'
        >>> map_capture_day("random.csv")
        'unknown'
    """
    name = _basename(filename)

    if name in CAPTURE_DAYS:
        return CAPTURE_DAYS[name]

    lowered = name.lower()
    if lowered in _CAPTURE_DAYS_CI:
        return _CAPTURE_DAYS_CI[lowered]

    leading_token = lowered.split("-", maxsplit=1)[0]
    if leading_token in _WEEKDAYS_CI:
        return leading_token

    logger.warning("capture_day_unrecognised", filename=name)
    return UNKNOWN_CAPTURE_DAY


def capture_file_rank(filename: str) -> int:
    """
    Position of a CICIDS2017 CSV in the capture week.

    The sort key ``ml.flow.loader.load_cicids2017`` reads the CSVs with, and the
    reason it does not simply call :func:`sorted`. See
    :data:`CAPTURE_FILE_ORDER` for what alphabetical order gets wrong.

    Deliberately *not* derived from :func:`map_capture_day`: the inversion this
    fixes happens **inside** a day, between the morning and afternoon captures of
    Thursday and Friday, so day granularity cannot express it.

    Args:
        filename: File name (a full path is accepted; only the last component is
            inspected). Matched case-insensitively, like :func:`map_capture_day`.

    Returns:
        A 0-based rank for the eight canonical filenames, or
        :data:`UNKNOWN_CAPTURE_FILE_RANK` for anything else — including the
        generated sample CSVs, which all share that rank and therefore keep
        whatever order the caller's tie-break gives them.

    Example:
        >>> capture_file_rank("Friday-WorkingHours-Morning.pcap_ISCX.csv")
        5
        >>> capture_file_rank("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        6
        >>> capture_file_rank("random.csv") == UNKNOWN_CAPTURE_FILE_RANK
        True
    """
    return _CAPTURE_FILE_RANKS_CI.get(_basename(filename).lower(), UNKNOWN_CAPTURE_FILE_RANK)
