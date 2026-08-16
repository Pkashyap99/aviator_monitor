"""Audit collected Aviatrix multiplier rows for model-input quality.

This module does not make predictions. It checks whether the stored round data
looks clean enough to use for analysis: duplicates, timestamp gaps, stale data,
source mixing, and unusual capture clumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"

TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
STALE_SECONDS = 180
RECENT_ROWS = 500
MAX_DETAIL_ROWS = 8


def parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value or ""), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def clean_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number


def compact_float(value, digits=4):
    if value is None:
        return None

    return round(float(value), digits)


def percentile(values: list[float], percent: float):
    if not values:
        return None

    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def normalize_rows(rows: Iterable[dict]) -> list[dict]:
    normalized = []

    for line_number, row in enumerate(rows, start=2):
        multiplier = clean_float(row.get("multiplier"))
        timestamp = str(row.get("timestamp", "") or "")
        timestamp_dt = parse_time(timestamp)

        normalized.append(
            {
                "line_number": line_number,
                "timestamp": timestamp,
                "timestamp_dt": timestamp_dt,
                "multiplier": multiplier,
                "round_id": str(row.get("round_id", "") or ""),
                "source": str(row.get("source", "") or "unknown"),
            }
        )

    return normalized


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with Path(path).open("r", newline="", encoding="utf-8", errors="ignore") as f:
        return normalize_rows(csv.DictReader(f))


def interval_summary(valid_rows: list[dict]) -> dict:
    intervals = []
    zero_or_negative = 0
    out_of_order = 0
    largest_gaps = []

    previous = None

    for row in valid_rows:
        current = row.get("timestamp_dt")

        if current is None:
            continue

        if previous is not None:
            seconds = (current - previous["timestamp_dt"]).total_seconds()

            if seconds < 0:
                out_of_order += 1
            elif seconds == 0:
                zero_or_negative += 1
            else:
                intervals.append(seconds)
                largest_gaps.append(
                    {
                        "seconds": compact_float(seconds, 1),
                        "from": previous.get("timestamp"),
                        "to": row.get("timestamp"),
                        "from_multiplier": previous.get("multiplier"),
                        "to_multiplier": row.get("multiplier"),
                    }
                )

        previous = row

    median_seconds = statistics.median(intervals) if intervals else None
    p95_seconds = percentile(intervals, 0.95)
    max_seconds = max(intervals) if intervals else None
    gap_threshold = max(
        180,
        (median_seconds or 0) * 8,
    )
    possible_capture_gaps = [
        item
        for item in largest_gaps
        if item["seconds"] is not None
        and item["seconds"] >= gap_threshold
    ]

    possible_capture_gaps.sort(
        key=lambda item: item["seconds"] or 0,
        reverse=True,
    )

    return {
        "median_seconds": compact_float(median_seconds, 1),
        "p95_seconds": compact_float(p95_seconds, 1),
        "max_seconds": compact_float(max_seconds, 1),
        "gap_threshold_seconds": compact_float(gap_threshold, 1),
        "possible_capture_gap_count": len(possible_capture_gaps),
        "largest_gaps": possible_capture_gaps[:MAX_DETAIL_ROWS],
        "zero_second_interval_count": zero_or_negative,
        "out_of_order_count": out_of_order,
    }


def target_rates(valid_rows: list[dict]) -> dict:
    multipliers = [
        row["multiplier"]
        for row in valid_rows
        if row.get("multiplier") is not None
    ]

    if not multipliers:
        return {}

    return {
        f"{target:.2f}": compact_float(
            sum(1 for value in multipliers if value >= target) / len(multipliers),
            6,
        )
        for target in TARGETS
    }


def audit_rows(rows: Iterable[dict], now=None) -> dict:
    normalized = normalize_rows(rows)
    now = now or datetime.now()

    invalid_multiplier_rows = [
        row
        for row in normalized
        if row.get("multiplier") is None
        or row.get("multiplier") < 1
    ]
    invalid_timestamp_rows = [
        row
        for row in normalized
        if row.get("timestamp_dt") is None
    ]
    valid_rows = [
        row
        for row in normalized
        if row.get("multiplier") is not None
        and row.get("multiplier") >= 1
    ]
    exact_keys = [
        (
            row.get("round_id"),
            row.get("source"),
        )
        for row in valid_rows
        if row.get("round_id")
    ]
    exact_counts = Counter(exact_keys)
    duplicate_exact_count = sum(
        count - 1
        for count in exact_counts.values()
        if count > 1
    )
    possible_no_id_keys = [
        (
            row.get("timestamp"),
            row.get("multiplier"),
            row.get("source"),
        )
        for row in valid_rows
        if not row.get("round_id")
    ]
    possible_no_id_counts = Counter(possible_no_id_keys)
    possible_no_id_duplicate_count = sum(
        count - 1
        for count in possible_no_id_counts.values()
        if count > 1
    )

    timestamp_counts = Counter(
        row.get("timestamp")
        for row in valid_rows
        if row.get("timestamp")
    )
    same_timestamp_groups = [
        {
            "timestamp": timestamp,
            "count": count,
        }
        for timestamp, count in timestamp_counts.items()
        if count > 1
    ]
    same_timestamp_groups.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    intervals = interval_summary(
        valid_rows
    )
    recent_intervals = interval_summary(
        valid_rows[-RECENT_ROWS:]
    )
    source_counts = Counter(
        row.get("source") or "unknown"
        for row in valid_rows
    )
    last_row = valid_rows[-1] if valid_rows else None
    last_age_seconds = None

    if last_row and last_row.get("timestamp_dt") is not None:
        last_age_seconds = max(
            0,
            int(
                (now - last_row["timestamp_dt"]).total_seconds()
            ),
        )

    issues = []
    score = 100

    if not valid_rows:
        issues.append(
            {
                "severity": "bad",
                "label": "No valid rows",
                "detail": "rounds.csv has no usable multiplier rows.",
            }
        )
        score = 0

    if last_age_seconds is not None and last_age_seconds > STALE_SECONDS:
        issues.append(
            {
                "severity": "bad",
                "label": "CSV stale",
                "detail": f"Last stored round is {last_age_seconds}s old.",
            }
        )
        score -= 30

    if duplicate_exact_count:
        issues.append(
            {
                "severity": "warn",
                "label": "Duplicate rows",
                "detail": f"{duplicate_exact_count} duplicate round ids were found.",
            }
        )
        score -= min(20, duplicate_exact_count)

    if possible_no_id_duplicate_count:
        issues.append(
            {
                "severity": "info",
                "label": "No-id repeats",
                "detail": (
                    f"{possible_no_id_duplicate_count} same timestamp/multiplier rows "
                    "have no round id, so they are not treated as confirmed duplicates."
                ),
            }
        )

    if invalid_multiplier_rows:
        issues.append(
            {
                "severity": "bad",
                "label": "Invalid multipliers",
                "detail": f"{len(invalid_multiplier_rows)} rows have missing or invalid multiplier values.",
            }
        )
        score -= 25

    if invalid_timestamp_rows:
        issues.append(
            {
                "severity": "warn",
                "label": "Timestamp issues",
                "detail": f"{len(invalid_timestamp_rows)} rows have invalid timestamps.",
            }
        )
        score -= min(20, len(invalid_timestamp_rows))

    if recent_intervals["out_of_order_count"]:
        issues.append(
            {
                "severity": "bad",
                "label": "Recent order issue",
                "detail": f"{recent_intervals['out_of_order_count']} recent timestamp reversals found.",
            }
        )
        score -= 25
    elif intervals["out_of_order_count"]:
        issues.append(
            {
                "severity": "warn",
                "label": "Old order issue",
                "detail": (
                    f"{intervals['out_of_order_count']} historical timestamp reversals "
                    "found from earlier recovery runs."
                ),
            }
        )
        score -= 12

    if recent_intervals["possible_capture_gap_count"]:
        issues.append(
            {
                "severity": "warn",
                "label": "Recent capture gap",
                "detail": (
                    f"{recent_intervals['possible_capture_gap_count']} recent gaps "
                    f">= {recent_intervals['gap_threshold_seconds']}s."
                ),
            }
        )
        score -= 15
    elif intervals["possible_capture_gap_count"]:
        issues.append(
            {
                "severity": "info",
                "label": "Old capture gaps",
                "detail": f"{intervals['possible_capture_gap_count']} historical long gaps were detected.",
            }
        )
        score -= 5

    if same_timestamp_groups and same_timestamp_groups[0]["count"] >= 8:
        issues.append(
            {
                "severity": "info",
                "label": "Recovery clumps",
                "detail": (
                    f"Up to {same_timestamp_groups[0]['count']} rows share one timestamp; "
                    "this usually means startup recovery appended missed visible history."
                ),
            }
        )
        score -= 3

    if len(source_counts) > 1:
        issues.append(
            {
                "severity": "info",
                "label": "Mixed sources",
                "detail": ", ".join(
                    f"{source}:{count}"
                    for source, count in sorted(source_counts.items())
                ),
            }
        )

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    if score >= 90:
        status = "good"
        headline = "Data looks clean"
    elif score >= 70:
        status = "watch"
        headline = "Data needs watching"
    else:
        status = "bad"
        headline = "Data quality issue"

    if not issues and valid_rows:
        issues.append(
            {
                "severity": "good",
                "label": "No blocking issues",
                "detail": "No duplicates, stale rows, or recent capture gaps found.",
            }
        )

    return {
        "available": bool(
            normalized
        ),
        "status": status,
        "headline": headline,
        "score": score,
        "total_rows": len(
            normalized
        ),
        "valid_rows": len(
            valid_rows
        ),
        "invalid_multiplier_rows": len(
            invalid_multiplier_rows
        ),
        "invalid_timestamp_rows": len(
            invalid_timestamp_rows
        ),
        "duplicate_exact_count": duplicate_exact_count,
        "possible_no_id_duplicate_count": possible_no_id_duplicate_count,
        "same_timestamp_group_count": len(
            same_timestamp_groups
        ),
        "largest_same_timestamp_groups": same_timestamp_groups[:MAX_DETAIL_ROWS],
        "source_counts": dict(
            sorted(
                source_counts.items()
            )
        ),
        "first_timestamp": valid_rows[0].get(
            "timestamp"
        ) if valid_rows else None,
        "last_timestamp": last_row.get(
            "timestamp"
        ) if last_row else None,
        "last_round_age_seconds": last_age_seconds,
        "intervals": intervals,
        "recent_intervals": recent_intervals,
        "target_rates": target_rates(
            valid_rows
        ),
        "issues": issues[:MAX_DETAIL_ROWS],
    }


def audit_csv(path: Path = DEFAULT_CSV_PATH) -> dict:
    return audit_rows(
        read_rows(
            path
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--csv",
        default=str(
            DEFAULT_CSV_PATH
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report.",
    )
    args = parser.parse_args()
    report = audit_csv(
        Path(
            args.csv
        )
    )

    if args.json:
        print(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"{report['headline']} ({report['score']}/100)"
    )
    print(
        f"Rows: {report['valid_rows']} valid / {report['total_rows']} total"
    )
    print(
        "Interval median/p95/max: "
        f"{report['intervals']['median_seconds']}s / "
        f"{report['intervals']['p95_seconds']}s / "
        f"{report['intervals']['max_seconds']}s"
    )
    print("Issues:")

    for issue in report["issues"]:
        print(
            f"- [{issue['severity']}] {issue['label']}: {issue['detail']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
