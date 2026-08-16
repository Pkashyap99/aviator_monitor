"""Search for honest statistical edges in collected Aviatrix rounds.

This does not use hidden seeds, tokens, or private game internals. It looks for
repeatable public-history patterns and validates them chronologically on later
rounds so we do not fool ourselves by overfitting old data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "edge_audit.json"

TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
TRANSITION_THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
LOW_STREAK_THRESHOLDS = [1.2, 1.5, 2.0]
LOW_STREAK_LENGTHS = [2, 3, 5, 8]
SINCE_THRESHOLDS = [5.0, 10.0, 20.0, 50.0]
SINCE_BINS = [
    (1, 1, "1 round ago"),
    (2, 3, "2-3 rounds ago"),
    (4, 7, "4-7 rounds ago"),
    (8, 15, "8-15 rounds ago"),
    (16, 31, "16-31 rounds ago"),
    (32, 10_000, "32+ rounds ago"),
]
WATCH_Z_THRESHOLD = 2.0
STRONG_Z_THRESHOLD = 3.0
FDR_ALPHA = 0.05
CONFIDENCE_Z = 1.96
DEFAULT_WALK_FORWARD_FOLDS = 6
WALK_FORWARD_MIN_VALID_FOLDS = 4
WALK_FORWARD_MIN_POSITIVE_SHARE = 0.75
WALK_FORWARD_MIN_SIGNIFICANT_FOLDS = 2


@dataclass(frozen=True)
class Round:
    timestamp: str
    timestamp_dt: Optional[datetime]
    multiplier: float
    round_id: str
    source: str


def parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def read_rounds(path: Path) -> list[Round]:
    rows: list[tuple[int, Round]] = []

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for original_index, row in enumerate(csv.DictReader(f)):
            try:
                multiplier = float(row.get("multiplier", ""))
            except (TypeError, ValueError):
                continue

            if multiplier < 1:
                continue

            timestamp = row.get("timestamp", "")
            rows.append(
                (
                    original_index,
                    Round(
                        timestamp=timestamp,
                        timestamp_dt=parse_time(timestamp),
                        multiplier=multiplier,
                        round_id=row.get("round_id", ""),
                        source=row.get("source", ""),
                    ),
                )
            )

    rows.sort(
        key=lambda item: (
            item[1].timestamp_dt is None,
            item[1].timestamp_dt or datetime.max,
            item[0],
        )
    )

    sorted_rounds = []
    seen_round_ids = set()

    for _, round_data in rows:
        if round_data.round_id:
            if round_data.round_id in seen_round_ids:
                continue

            seen_round_ids.add(round_data.round_id)

        sorted_rounds.append(round_data)

    return sorted_rounds


def bucket(value: float) -> str:
    if value < 1.2:
        return "<1.20"
    if value < 1.5:
        return "1.20-1.49"
    if value < 2.0:
        return "1.50-1.99"
    if value < 3.0:
        return "2.00-2.99"
    if value < 5.0:
        return "3.00-4.99"
    if value < 10.0:
        return "5.00-9.99"
    if value < 20.0:
        return "10.00-19.99"
    if value < 50.0:
        return "20.00-49.99"
    return "50.00+"


def binomial_z(successes: int, checked: int, baseline: float) -> float:
    if checked <= 0 or baseline <= 0 or baseline >= 1:
        return 0.0

    expected_std = math.sqrt(checked * baseline * (1 - baseline))
    if expected_std <= 0:
        return 0.0

    return (successes - checked * baseline) / expected_std


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2))


def clean_float(value: Optional[float], digits: int = 6):
    if value is None:
        return None

    return round(float(value), digits)


def target_counts(rounds: list[Round], indexes: Iterable[int], target: float) -> dict:
    checked = 0
    successes = 0

    for index in indexes:
        checked += 1

        if rounds[index].multiplier >= target:
            successes += 1

    rate = None if checked == 0 else successes / checked

    return {
        "checked": checked,
        "successes": successes,
        "rate": rate,
    }


def lift_confidence_interval(
    successes: int,
    checked: int,
    baseline_successes: int,
    baseline_checked: int,
    z_value: float = CONFIDENCE_Z,
) -> tuple[Optional[float], Optional[float]]:
    if checked <= 0 or baseline_checked <= 0:
        return None, None

    rate = successes / checked
    baseline = baseline_successes / baseline_checked
    lift = rate - baseline
    standard_error = math.sqrt(
        (rate * (1 - rate) / checked)
        + (baseline * (1 - baseline) / baseline_checked)
    )

    return (
        lift - z_value * standard_error,
        lift + z_value * standard_error,
    )


def apply_bh_fdr(items: list[dict], alpha: float = FDR_ALPHA) -> None:
    tested = []

    for index, item in enumerate(items):
        try:
            p_value = float(
                item.get("holdout", {}).get("p_value")
            )
        except (TypeError, ValueError):
            continue

        if not math.isfinite(p_value):
            continue

        tested.append(
            (
                max(
                    0.0,
                    min(
                        1.0,
                        p_value,
                    ),
                ),
                index,
            )
        )

    test_count = len(tested)

    for item in items:
        holdout = item.setdefault(
            "holdout",
            {}
        )
        holdout["q_value"] = None
        holdout["fdr_significant"] = False

    if not tested:
        return

    ordered = sorted(
        tested,
        key=lambda pair: pair[0],
    )
    adjusted = [1.0] * test_count
    running_min = 1.0

    for rank_from_end, (p_value, _index) in enumerate(
        reversed(
            ordered
        ),
        start=1,
    ):
        rank = test_count - rank_from_end + 1
        running_min = min(
            running_min,
            p_value * test_count / rank,
        )
        adjusted[rank - 1] = min(
            1.0,
            running_min,
        )

    for (_p_value, item_index), q_value in zip(
        ordered,
        adjusted,
    ):
        holdout = items[item_index]["holdout"]
        holdout["q_value"] = clean_float(
            q_value,
            8,
        )
        holdout["fdr_significant"] = bool(
            q_value <= alpha
        )


def base_rates(rounds: list[Round], indexes: Iterable[int]) -> dict[str, float]:
    selected = [rounds[index].multiplier for index in indexes]
    if not selected:
        return {f"{target:.2f}": 0.0 for target in TARGETS}

    return {
        f"{target:.2f}": sum(1 for value in selected if value >= target) / len(selected)
        for target in TARGETS
    }


def previous_low_streak(rounds: list[Round], index: int, threshold: float) -> int:
    count = 0
    cursor = index - 1

    while cursor >= 0 and rounds[cursor].multiplier < threshold:
        count += 1
        cursor -= 1

    return count


def rounds_since_previous_ge(rounds: list[Round], index: int, threshold: float) -> int:
    for offset, cursor in enumerate(range(index - 1, -1, -1), start=1):
        if rounds[cursor].multiplier >= threshold:
            return offset

    return 10_000


def condition_label(spec: dict) -> str:
    kind = spec["kind"]

    if kind == "prev_bucket":
        return f"previous was {spec['bucket']}"

    if kind == "prev_ge":
        return f"previous >= {spec['threshold']:.2f}x"

    if kind == "prev_lt":
        return f"previous < {spec['threshold']:.2f}x"

    if kind == "low_streak":
        return (
            f"{spec['length']}+ rounds below "
            f"{spec['threshold']:.2f}x"
        )

    if kind == "since_ge":
        return (
            f"last {spec['threshold']:.2f}x+ was "
            f"{spec['label']}"
        )

    if kind == "weekday":
        return f"previous weekday {spec['weekday']}"

    if kind == "hour":
        return f"previous hour {spec['hour']:02d}:00"

    return kind


def condition_matches(rounds: list[Round], index: int, spec: dict) -> bool:
    if index <= 0:
        return False

    previous = rounds[index - 1]
    kind = spec["kind"]

    if kind == "prev_bucket":
        return bucket(previous.multiplier) == spec["bucket"]

    if kind == "prev_ge":
        return previous.multiplier >= spec["threshold"]

    if kind == "prev_lt":
        return previous.multiplier < spec["threshold"]

    if kind == "low_streak":
        return previous_low_streak(rounds, index, spec["threshold"]) >= spec["length"]

    if kind == "since_ge":
        since = rounds_since_previous_ge(rounds, index, spec["threshold"])
        return spec["minimum"] <= since <= spec["maximum"]

    if kind == "weekday":
        return (
            previous.timestamp_dt is not None
            and previous.timestamp_dt.weekday() == spec["weekday"]
        )

    if kind == "hour":
        return (
            previous.timestamp_dt is not None
            and previous.timestamp_dt.hour == spec["hour"]
        )

    return False


def evaluate_condition(
    rounds: list[Round],
    indexes: list[int],
    spec: dict,
    target: float,
    baseline: float,
    baseline_checked: int,
    baseline_successes: int,
) -> dict:
    checked = 0
    successes = 0

    for index in indexes:
        if not condition_matches(rounds, index, spec):
            continue

        checked += 1
        if rounds[index].multiplier >= target:
            successes += 1

    rate = None if checked == 0 else successes / checked
    lift = None if rate is None else rate - baseline
    z_value = binomial_z(successes, checked, baseline)
    lift_ci_low, lift_ci_high = lift_confidence_interval(
        successes,
        checked,
        baseline_successes,
        baseline_checked,
    )

    return {
        "condition": condition_label(spec),
        "spec": spec,
        "target": f"{target:.2f}",
        "checked": checked,
        "successes": successes,
        "baseline_checked": baseline_checked,
        "baseline_successes": baseline_successes,
        "rate": clean_float(rate),
        "baseline": clean_float(baseline),
        "lift": clean_float(lift),
        "lift_ci_low": clean_float(lift_ci_low),
        "lift_ci_high": clean_float(lift_ci_high),
        "lift_ci_excludes_zero": bool(
            lift_ci_low is not None
            and lift_ci_low > 0
        ),
        "z": clean_float(z_value, 4),
        "p_value": clean_float(normal_two_sided_p(z_value), 8),
    }


def candidate_specs(rounds: list[Round]) -> list[dict]:
    bucket_names = sorted({bucket(item.multiplier) for item in rounds})
    specs = [{"kind": "prev_bucket", "bucket": item} for item in bucket_names]

    specs.extend({"kind": "prev_ge", "threshold": item} for item in TRANSITION_THRESHOLDS)
    specs.extend({"kind": "prev_lt", "threshold": item} for item in LOW_STREAK_THRESHOLDS)

    for threshold in LOW_STREAK_THRESHOLDS:
        for length in LOW_STREAK_LENGTHS:
            specs.append(
                {
                    "kind": "low_streak",
                    "threshold": threshold,
                    "length": length,
                }
            )

    for threshold in SINCE_THRESHOLDS:
        for minimum, maximum, label in SINCE_BINS:
            specs.append(
                {
                    "kind": "since_ge",
                    "threshold": threshold,
                    "minimum": minimum,
                    "maximum": maximum,
                    "label": label,
                }
            )

    specs.extend({"kind": "weekday", "weekday": item} for item in range(7))
    specs.extend({"kind": "hour", "hour": item} for item in range(24))
    return specs


def split_indexes(rounds: list[Round], holdout_fraction: float) -> tuple[list[int], list[int]]:
    indexes = list(range(1, len(rounds)))
    split_at = int(len(indexes) * (1 - holdout_fraction))
    split_at = max(1, min(split_at, len(indexes) - 1))
    return indexes[:split_at], indexes[split_at:]


def split_contiguous_blocks(indexes: list[int], block_count: int) -> list[tuple[int, int, list[int]]]:
    if block_count <= 0:
        return []

    size = len(indexes)
    base_size = size // block_count
    remainder = size % block_count
    blocks = []
    cursor = 0

    for block_index in range(block_count):
        block_size = base_size + (1 if block_index < remainder else 0)
        start = cursor
        end = min(
            size,
            start + block_size,
        )
        blocks.append(
            (
                start,
                end,
                indexes[start:end],
            )
        )
        cursor = end

    return blocks


def walk_forward_condition(
    rounds: list[Round],
    indexes: list[int],
    spec: dict,
    target: float,
    min_sample: int,
    fold_count: int,
) -> dict:
    blocks = split_contiguous_blocks(
        indexes,
        fold_count + 1,
    )
    folds = []

    for fold_number, (start, _end, test_indexes) in enumerate(
        blocks[1:],
        start=1,
    ):
        train_indexes = indexes[:start]

        if not train_indexes or not test_indexes:
            continue

        baseline_counts = target_counts(
            rounds,
            train_indexes,
            target,
        )
        baseline = baseline_counts.get(
            "rate"
        )

        if baseline is None:
            continue

        result = evaluate_condition(
            rounds,
            test_indexes,
            spec,
            target,
            baseline,
            baseline_counts["checked"],
            baseline_counts["successes"],
        )
        lift = result.get(
            "lift"
        )
        p_value = result.get(
            "p_value"
        )
        valid = bool(
            result["checked"] >= min_sample
        )
        positive = bool(
            valid
            and lift is not None
            and lift > 0
        )
        significant_positive = bool(
            positive
            and p_value is not None
            and p_value <= 0.05
        )

        folds.append(
            {
                "fold": fold_number,
                "train_rounds": len(
                    train_indexes
                ),
                "test_rounds": len(
                    test_indexes
                ),
                "start_timestamp": rounds[test_indexes[0]].timestamp,
                "end_timestamp": rounds[test_indexes[-1]].timestamp,
                "valid": valid,
                "positive": positive,
                "significant_positive": significant_positive,
                **{
                    key: result[key]
                    for key in (
                        "checked",
                        "successes",
                        "baseline_checked",
                        "baseline_successes",
                        "rate",
                        "baseline",
                        "lift",
                        "lift_ci_low",
                        "lift_ci_high",
                        "lift_ci_excludes_zero",
                        "z",
                        "p_value",
                    )
                },
            }
        )

    valid_folds = [
        item
        for item in folds
        if item["valid"]
    ]
    positive_folds = [
        item
        for item in valid_folds
        if item["positive"]
    ]
    significant_folds = [
        item
        for item in valid_folds
        if item["significant_positive"]
    ]
    average_lift = (
        sum(
            item["lift"]
            for item in valid_folds
            if item["lift"] is not None
        )
        / len(valid_folds)
        if valid_folds
        else None
    )
    positive_share = (
        len(positive_folds) / len(valid_folds)
        if valid_folds
        else 0
    )
    stable = bool(
        len(valid_folds) >= WALK_FORWARD_MIN_VALID_FOLDS
        and positive_share >= WALK_FORWARD_MIN_POSITIVE_SHARE
        and len(significant_folds) >= WALK_FORWARD_MIN_SIGNIFICANT_FOLDS
        and average_lift is not None
        and average_lift > 0
    )

    if stable:
        verdict = "stable_positive"
    elif not valid_folds:
        verdict = "insufficient_sample"
    elif positive_share >= WALK_FORWARD_MIN_POSITIVE_SHARE:
        verdict = "positive_but_weak"
    else:
        verdict = "not_stable"

    return {
        "fold_count": fold_count,
        "valid_folds": len(
            valid_folds
        ),
        "positive_folds": len(
            positive_folds
        ),
        "significant_positive_folds": len(
            significant_folds
        ),
        "positive_fold_share": clean_float(
            positive_share,
            4,
        ),
        "average_lift": clean_float(
            average_lift,
        ),
        "stable": stable,
        "verdict": verdict,
        "folds": folds,
    }


def gap_summary(rounds: list[Round], target: float) -> dict:
    hit_indexes = [
        index
        for index, item in enumerate(rounds)
        if item.multiplier >= target
    ]
    gaps = [
        current - previous
        for previous, current in zip(hit_indexes, hit_indexes[1:])
    ]
    current_gap = (
        len(rounds) - 1 - hit_indexes[-1]
        if hit_indexes
        else len(rounds)
    )

    if not gaps:
        return {
            "target": f"{target:.2f}",
            "hits": len(hit_indexes),
            "rate": clean_float(len(hit_indexes) / len(rounds) if rounds else 0),
            "current_gap": current_gap,
            "average_gap": None,
            "median_gap": None,
            "max_gap": None,
        }

    sorted_gaps = sorted(gaps)
    return {
        "target": f"{target:.2f}",
        "hits": len(hit_indexes),
        "rate": clean_float(len(hit_indexes) / len(rounds)),
        "current_gap": current_gap,
        "average_gap": clean_float(sum(gaps) / len(gaps), 2),
        "median_gap": sorted_gaps[len(sorted_gaps) // 2],
        "max_gap": max(gaps),
    }


def audit(
    rounds: list[Round],
    min_sample: int,
    holdout_fraction: float,
    top: int,
    walk_forward_folds: int = DEFAULT_WALK_FORWARD_FOLDS,
) -> dict:
    train_indexes, holdout_indexes = split_indexes(rounds, holdout_fraction)
    train_rates = base_rates(rounds, train_indexes)
    holdout_rates = base_rates(rounds, holdout_indexes)
    train_counts = {
        f"{target:.2f}": target_counts(
            rounds,
            train_indexes,
            target,
        )
        for target in TARGETS
    }
    holdout_counts = {
        f"{target:.2f}": target_counts(
            rounds,
            holdout_indexes,
            target,
        )
        for target in TARGETS
    }
    specs = candidate_specs(rounds)
    train_results = []
    patterns_tested = len(specs) * len(TARGETS)

    for spec in specs:
        for target in TARGETS:
            target_key = f"{target:.2f}"
            result = evaluate_condition(
                rounds,
                train_indexes,
                spec,
                target,
                train_rates[target_key],
                train_counts[target_key]["checked"],
                train_counts[target_key]["successes"],
            )

            if result["checked"] >= min_sample:
                train_results.append(result)

    train_results.sort(
        key=lambda item: (
            item["z"],
            item["lift"] or -1,
            item["checked"],
        ),
        reverse=True,
    )

    validated = []
    for train_result in train_results:
        target = float(train_result["target"])
        target_key = train_result["target"]
        holdout_result = evaluate_condition(
            rounds,
            holdout_indexes,
            train_result["spec"],
            target,
            holdout_rates[target_key],
            holdout_counts[target_key]["checked"],
            holdout_counts[target_key]["successes"],
        )

        if holdout_result["checked"] < min_sample:
            continue

        validated.append(
            {
                "condition": train_result["condition"],
                "spec": train_result["spec"],
                "target": train_result["target"],
                "train": {
                    key: train_result[key]
                    for key in (
                        "checked",
                        "successes",
                        "baseline_checked",
                        "baseline_successes",
                        "rate",
                        "baseline",
                        "lift",
                        "lift_ci_low",
                        "lift_ci_high",
                        "lift_ci_excludes_zero",
                        "z",
                        "p_value",
                    )
                },
                "holdout": {
                    key: holdout_result[key]
                    for key in (
                        "checked",
                        "successes",
                        "baseline_checked",
                        "baseline_successes",
                        "rate",
                        "baseline",
                        "lift",
                        "lift_ci_low",
                        "lift_ci_high",
                        "lift_ci_excludes_zero",
                        "z",
                        "p_value",
                    )
                },
                "watch_candidate": False,
                "fdr_confirmed": False,
                "strong_edge": False,
                "status": "tested",
            }
        )

    apply_bh_fdr(
        validated,
        FDR_ALPHA,
    )

    for item in validated:
        train = item["train"]
        holdout = item["holdout"]
        target = float(
            item["target"]
        )
        walk_forward = walk_forward_condition(
            rounds,
            list(
                range(
                    1,
                    len(rounds),
                )
            ),
            item["spec"],
            target,
            min_sample,
            walk_forward_folds,
        )
        replicated_positive_lift = (
            (train.get("lift") or 0) > 0
            and (holdout.get("lift") or 0) > 0
        )
        raw_watch = (
            replicated_positive_lift
            and (holdout.get("z") or 0) >= WATCH_Z_THRESHOLD
        )
        fdr_confirmed = (
            raw_watch
            and bool(
                holdout.get(
                    "fdr_significant"
                )
            )
            and bool(
                holdout.get(
                    "lift_ci_excludes_zero"
                )
            )
        )
        walk_forward_stable = bool(
            fdr_confirmed
            and walk_forward.get(
                "stable"
            )
        )
        strong_edge = (
            walk_forward_stable
            and (train.get("z") or 0) >= STRONG_Z_THRESHOLD
            and (holdout.get("z") or 0) >= STRONG_Z_THRESHOLD
        )

        item["walk_forward"] = walk_forward
        item["watch_candidate"] = bool(
            raw_watch
        )
        item["fdr_confirmed"] = bool(
            fdr_confirmed
        )
        item["walk_forward_stable"] = bool(
            walk_forward_stable
        )
        item["strong_edge"] = bool(
            strong_edge
        )
        item["status"] = (
            "strong_edge"
            if strong_edge
            else "walk_forward_stable"
            if walk_forward_stable
            else "fdr_confirmed"
            if fdr_confirmed
            else "unconfirmed_watch"
            if raw_watch
            else "tested"
        )

    validated.sort(
        key=lambda item: (
            item["strong_edge"],
            item["walk_forward_stable"],
            item["fdr_confirmed"],
            item["watch_candidate"],
            item["holdout"].get("fdr_significant", False),
            item["walk_forward"].get("positive_fold_share", 0),
            item["holdout"]["z"],
            item["holdout"]["lift"] or -1,
        ),
        reverse=True,
    )

    strong_edges = [
        item
        for item in validated
        if item["strong_edge"]
    ]
    watch_candidates = [
        item
        for item in validated
        if item["watch_candidate"]
    ]
    fdr_confirmed = [
        item
        for item in validated
        if item["fdr_confirmed"]
    ]
    walk_forward_stable = [
        item
        for item in validated
        if item["walk_forward_stable"]
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": len(rounds),
        "train_rounds": len(train_indexes),
        "holdout_rounds": len(holdout_indexes),
        "min_sample": min_sample,
        "holdout_fraction": holdout_fraction,
        "patterns_tested": patterns_tested,
        "train_eligible_count": len(train_results),
        "validated_test_count": len(validated),
        "fdr_alpha": FDR_ALPHA,
        "walk_forward_folds": walk_forward_folds,
        "walk_forward_min_valid_folds": WALK_FORWARD_MIN_VALID_FOLDS,
        "walk_forward_min_positive_share": WALK_FORWARD_MIN_POSITIVE_SHARE,
        "walk_forward_min_significant_folds": WALK_FORWARD_MIN_SIGNIFICANT_FOLDS,
        "base_rates": {
            "all": base_rates(rounds, range(len(rounds))),
            "train": train_rates,
            "holdout": holdout_rates,
        },
        "top_train_patterns": train_results[:top],
        "validated_patterns": validated[:top],
        "strong_edge_count": len(strong_edges),
        "watch_candidate_count": len(watch_candidates),
        "fdr_confirmed_count": len(fdr_confirmed),
        "walk_forward_stable_count": len(walk_forward_stable),
        "big_multiplier_gaps": [
            gap_summary(rounds, target)
            for target in [10.0, 20.0, 50.0, 100.0]
        ],
        "conclusion": (
            "STRONG STATISTICAL EDGE FOUND - review with caution and keep live tracking."
            if strong_edges
            else "WALK-FORWARD-STABLE WATCH PATTERNS FOUND - still not a guarantee."
            if walk_forward_stable
            else "FDR-CONFIRMED WATCH PATTERNS FOUND, BUT NOT WALK-FORWARD STABLE."
            if fdr_confirmed
            else "UNCONFIRMED WATCH PATTERNS FOUND - raw lifts did not survive correction."
            if watch_candidates
            else "NO VALIDATED LOOPHOLE FOUND - patterns did not repeat strongly on holdout."
        ),
        "warning": (
            "This is a statistical audit only. Raw watch patterns can be false "
            "positives after many tests; FDR correction and confidence intervals "
            "are used before calling anything confirmed."
        ),
    }


def print_summary(report: dict) -> None:
    print(f"Rounds checked: {report['rounds']}")
    print(f"Train/Holdout: {report['train_rounds']} / {report['holdout_rounds']}")
    print(
        "Patterns tested: "
        f"{report.get('patterns_tested', 0)} "
        f"(validated {report.get('validated_test_count', 0)}, "
        f"FDR alpha {report.get('fdr_alpha', FDR_ALPHA):.2f})"
    )
    print(
        "Walk-forward: "
        f"{report.get('walk_forward_folds', DEFAULT_WALK_FORWARD_FOLDS)} folds, "
        f"stable patterns {report.get('walk_forward_stable_count', 0)}"
    )
    print(report["conclusion"])
    print()
    print("Top validated patterns:")

    if not report["validated_patterns"]:
        print("- none")
    else:
        for item in report["validated_patterns"][:8]:
            holdout = item["holdout"]
            marker = (
                "EDGE"
                if item["strong_edge"]
                else "stable"
                if item.get("walk_forward_stable")
                else "confirmed"
                if item.get("fdr_confirmed")
                else "candidate"
                if item["watch_candidate"]
                else "watch"
            )
            q_value = holdout.get("q_value")
            q_text = "n/a" if q_value is None else f"{q_value:.4f}"
            ci_low = holdout.get("lift_ci_low")
            ci_high = holdout.get("lift_ci_high")
            ci_text = (
                "n/a"
                if ci_low is None or ci_high is None
                else f"{ci_low:.3f}..{ci_high:.3f}"
            )
            walk = item.get(
                "walk_forward",
                {}
            )
            walk_text = (
                f"WF {walk.get('positive_folds', 0)}/"
                f"{walk.get('valid_folds', 0)}+"
                f", sig {walk.get('significant_positive_folds', 0)}"
            )
            print(
                "- "
                f"[{marker}] {item['condition']} -> >= {item['target']}x | "
                f"holdout {holdout['rate']:.3f} vs {holdout['baseline']:.3f} "
                f"(lift {holdout['lift']:.3f}, z {holdout['z']:.2f}, "
                f"q {q_text}, lift CI {ci_text}, n {holdout['checked']}, "
                f"{walk_text})"
            )

    print()
    print("Big multiplier gaps:")
    for item in report["big_multiplier_gaps"]:
        print(
            "- "
            f">= {item['target']}x: hits {item['hits']}, "
            f"rate {item['rate']:.4f}, current gap {item['current_gap']}, "
            f"avg gap {item['average_gap']}, max gap {item['max_gap']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-sample", type=int, default=80)
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=DEFAULT_WALK_FORWARD_FOLDS,
    )
    args = parser.parse_args()

    rounds = read_rounds(Path(args.csv))
    if len(rounds) < 300:
        raise SystemExit("Need at least 300 valid rounds for an edge audit.")

    report = audit(
        rounds,
        min_sample=args.min_sample,
        holdout_fraction=args.holdout_fraction,
        top=args.top,
        walk_forward_folds=args.walk_forward_folds,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_summary(report)
    print()
    print(f"Saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
