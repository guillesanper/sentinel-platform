"""
CICIDS2017 label taxonomy for the flow-based track.

Single source of truth for:

* ``ATTACK_GROUPS`` — the 15 raw CICIDS2017 labels collapsed into 7 canonical
  classes.
* ``map_label`` — raw label (as read from the CSVs) → canonical class, or None
  when the label is not recognised. Carries the latin-1 mojibake repair that
  used to be buried in ``utils/loader.py::_map_labels``.
* ``CAPTURE_DAYS`` / ``map_capture_day`` — CSV filename → capture day. The
  MachineLearningCVE distribution of CICIDS2017 has no ``Timestamp`` column, so
  the filename is the only carrier of time. The temporal split partitions by
  ``(capture_day, class)``.

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

# Raw CICIDS2017 label → canonical 7-class taxonomy.
ATTACK_GROUPS: Final[dict[str, str]] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Scan",
    "Infiltration": "Scan",
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
CLASS_ORDER: Final[tuple[str, ...]] = (
    "BENIGN",
    "Botnet",
    "BruteForce",
    "DDoS",
    "DoS",
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

# Case-insensitive lookup tables, built once at import time.
_CAPTURE_DAYS_CI: Final[dict[str, str]] = {name.lower(): day for name, day in CAPTURE_DAYS.items()}
_WEEKDAYS_CI: Final[frozenset[str]] = frozenset(CAPTURE_DAY_ORDER)


# ============================================================================
# Public API
# ============================================================================


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
    cleaned = str(raw).replace(_MOJIBAKE_EN_DASH, _EN_DASH).strip()
    return ATTACK_GROUPS.get(cleaned)


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
    name = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]

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
