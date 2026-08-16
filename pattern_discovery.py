"""Discover human-readable multiplier patterns in saved Aviatrix rounds.

This script looks for the manual patterns a user might notice:
- several small multipliers followed by a big multiplier
- medium multiplier clusters around 2x-6x
- big multipliers after cooldown gaps
- fixed hour and weekday/hour effects

It does not tune the prediction model. Results are checked on a chronological
holdout and corrected for many tests, so weak historical coincidences are not
promoted as reliable signals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from aviator_analyzer import load_rounds, select_prediction_rounds


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "pattern_discovery.json"

HOLDOUT_FRACTION = 0.30
MIN_HOLDOUT_MATCHES = 60
MIN_TRAIN_MATCHES = 100
FDR_ALPHA = 0.05
WATCH_Z = 1.6
STRONG_Z = 2.0
WALK_FORWARD_FOLDS = 6
WALK_FORWARD_MIN_MATCHES = 25
TOP_PER_GROUP = 8

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEQUENCE_BUCKETS = [
    ("tiny", "<1.20x"),
    ("small", "1.20x-1.99x"),
    ("medium", "2.00x-5.99x"),
    ("high", "6.00x-19.99x"),
    ("very_high", "20.00x+"),
]


@dataclass(frozen=True)
class Outcome:
    key: str
    label: str
    predicate: Callable[[float], bool]


@dataclass(frozen=True)
class Pattern:
    key: str
    label: str
    group: str
    predicate: Callable[[list[dict], int], bool]


OUTCOMES = [
    Outcome("ge_5", "next >= 5x", lambda value: value >= 5.0),
    Outcome("ge_10", "next >= 10x", lambda value: value >= 10.0),
    Outcome("ge_20", "next >= 20x", lambda value: value >= 20.0),
    Outcome("ge_50", "next >= 50x", lambda value: value >= 50.0),
    Outcome("ge_100", "next >= 100x", lambda value: value >= 100.0),
    Outcome("medium_2_6", "next 2x-6x", lambda value: 2.0 <= value < 6.0),
    Outcome("lt_2", "next < 2x", lambda value: value < 2.0),
    Outcome("lt_3", "next < 3x", lambda value: value < 3.0),
]


def parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value or ""), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def compact(value, digits=6):
    if value is None:
        return None

    return round(float(value), digits)


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2))


def binomial_z(successes: int, checked: int, baseline: float) -> float:
    if checked <= 0 or baseline <= 0 or baseline >= 1:
        return 0.0

    standard_error = math.sqrt(checked * baseline * (1 - baseline))
    if standard_error <= 0:
        return 0.0

    return (successes - checked * baseline) / standard_error


def lift_ci(
    successes: int,
    checked: int,
    baseline_successes: int,
    baseline_checked: int,
    confidence_z: float = 1.96,
):
    if checked <= 0 or baseline_checked <= 0:
        return None, None

    rate = successes / checked
    baseline = baseline_successes / baseline_checked
    standard_error = math.sqrt(
        (rate * (1 - rate) / checked)
        + (baseline * (1 - baseline) / baseline_checked)
    )
    lift = rate - baseline
    return lift - confidence_z * standard_error, lift + confidence_z * standard_error


def all_previous_below(rounds: list[dict], index: int, length: int, threshold: float) -> bool:
    if index < length:
        return False

    return all(
        float(rounds[cursor]["multiplier"]) < threshold
        for cursor in range(index - length, index)
    )


def all_previous_between(
    rounds: list[dict],
    index: int,
    length: int,
    minimum: float,
    maximum: float,
) -> bool:
    if index < length:
        return False

    return all(
        minimum <= float(rounds[cursor]["multiplier"]) < maximum
        for cursor in range(index - length, index)
    )


def count_previous_between(
    rounds: list[dict],
    index: int,
    window: int,
    minimum: float,
    maximum: float,
) -> int:
    if index < window:
        return 0

    return sum(
        1
        for cursor in range(index - window, index)
        if minimum <= float(rounds[cursor]["multiplier"]) < maximum
    )


def rounds_since_ge(rounds: list[dict], index: int, threshold: float) -> int:
    for distance, cursor in enumerate(range(index - 1, -1, -1), start=1):
        if float(rounds[cursor]["multiplier"]) >= threshold:
            return distance

    return 100_000


def previous_timestamp(rounds: list[dict], index: int) -> Optional[datetime]:
    if index <= 0:
        return None

    return rounds[index - 1].get("_timestamp_dt")


def with_cached_timestamps(rounds: list[dict]) -> list[dict]:
    cached = []

    for row in rounds:
        item = dict(row)
        item["_timestamp_dt"] = parse_time(
            item.get(
                "timestamp",
                "",
            )
        )
        cached.append(item)

    return cached


def sequence_bucket(value: float) -> str:
    if value < 1.2:
        return "tiny"
    if value < 2.0:
        return "small"
    if value < 6.0:
        return "medium"
    if value < 20.0:
        return "high"
    return "very_high"


def sequence_label(keys: tuple[str, ...]) -> str:
    label_map = dict(SEQUENCE_BUCKETS)
    return " -> ".join(label_map.get(key, key) for key in keys)


def previous_sequence_matches(
    rounds: list[dict],
    index: int,
    expected: tuple[str, ...],
) -> bool:
    length = len(expected)

    if index < length:
        return False

    actual = tuple(
        sequence_bucket(float(rounds[cursor]["multiplier"]))
        for cursor in range(index - length, index)
    )
    return actual == expected


def make_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    for threshold, label in (
        (1.5, "very small"),
        (2.0, "small"),
        (3.0, "below 3x"),
    ):
        for length in (2, 3, 4, 5, 8):
            patterns.append(
                Pattern(
                    key=f"{length}_below_{threshold:.1f}",
                    label=f"last {length} were {label} (<{threshold:.1f}x)",
                    group="small_streak_then_big",
                    predicate=lambda rows, index, length=length, threshold=threshold: all_previous_below(
                        rows,
                        index,
                        length,
                        threshold,
                    ),
                )
            )

    for length in (2, 3, 4, 5):
        patterns.append(
            Pattern(
                key=f"{length}_medium_2_6",
                label=f"last {length} were medium (2x-6x)",
                group="medium_cluster",
                predicate=lambda rows, index, length=length: all_previous_between(
                    rows,
                    index,
                    length,
                    2.0,
                    6.0,
                ),
            )
        )

    for window, needed in (
        (3, 2),
        (5, 3),
        (8, 5),
    ):
        patterns.append(
            Pattern(
                key=f"{needed}_of_{window}_medium_2_6",
                label=f"{needed} of last {window} were medium (2x-6x)",
                group="medium_cluster",
                predicate=lambda rows, index, window=window, needed=needed: count_previous_between(
                    rows,
                    index,
                    window,
                    2.0,
                    6.0,
                )
                >= needed,
            )
        )

    sequence_keys = [key for key, _label in SEQUENCE_BUCKETS]
    for length in (2, 3, 4):
        for sequence in itertools.product(
            sequence_keys,
            repeat=length,
        ):
            patterns.append(
                Pattern(
                    key=f"seq_{length}_{'_'.join(sequence)}",
                    label=f"last {length} sequence: {sequence_label(sequence)}",
                    group="bucket_sequence",
                    predicate=lambda rows, index, sequence=sequence: previous_sequence_matches(
                        rows,
                        index,
                        sequence,
                    ),
                )
            )

    cooldown_bins = [
        (1, 3, "1-3 rounds ago"),
        (4, 7, "4-7 rounds ago"),
        (8, 15, "8-15 rounds ago"),
        (16, 31, "16-31 rounds ago"),
        (32, 63, "32-63 rounds ago"),
        (64, 10_000, "64+ rounds ago"),
    ]
    for threshold in (10.0, 20.0, 50.0, 100.0):
        for minimum, maximum, label in cooldown_bins:
            patterns.append(
                Pattern(
                    key=f"since_{threshold:.0f}_{minimum}_{maximum}",
                    label=f"last {threshold:.0f}x+ was {label}",
                    group="big_cooldown",
                    predicate=lambda rows, index, threshold=threshold, minimum=minimum, maximum=maximum: (
                        minimum <= rounds_since_ge(rows, index, threshold) <= maximum
                    ),
                )
            )

    for hour in range(24):
        patterns.append(
            Pattern(
                key=f"hour_{hour:02d}",
                label=f"previous round hour {hour:02d}:00",
                group="fixed_time",
                predicate=lambda rows, index, hour=hour: (
                    (previous_timestamp(rows, index) is not None)
                    and previous_timestamp(rows, index).hour == hour
                ),
            )
        )

    for weekday in range(7):
        patterns.append(
            Pattern(
                key=f"weekday_{weekday}",
                label=f"previous round weekday {WEEKDAYS[weekday]}",
                group="fixed_time",
                predicate=lambda rows, index, weekday=weekday: (
                    (previous_timestamp(rows, index) is not None)
                    and previous_timestamp(rows, index).weekday() == weekday
                ),
            )
        )

    return patterns


def outcome_counts(rounds: list[dict], indexes: Iterable[int], outcome: Outcome) -> dict:
    checked = 0
    successes = 0

    for index in indexes:
        checked += 1
        if outcome.predicate(float(rounds[index]["multiplier"])):
            successes += 1

    return {
        "checked": checked,
        "successes": successes,
        "rate": None if checked == 0 else successes / checked,
    }


def evaluate_pattern(
    rounds: list[dict],
    indexes: list[int],
    pattern: Pattern,
    outcome: Outcome,
    baseline_counts: dict,
) -> dict:
    checked = 0
    successes = 0

    for index in indexes:
        if not pattern.predicate(rounds, index):
            continue

        checked += 1
        if outcome.predicate(float(rounds[index]["multiplier"])):
            successes += 1

    baseline = baseline_counts["rate"] or 0.0
    rate = None if checked == 0 else successes / checked
    lift = None if rate is None else rate - baseline
    z_value = binomial_z(successes, checked, baseline)
    ci_low, ci_high = lift_ci(
        successes,
        checked,
        baseline_counts["successes"],
        baseline_counts["checked"],
    )

    return {
        "checked": checked,
        "successes": successes,
        "rate": compact(rate),
        "baseline": compact(baseline),
        "lift": compact(lift),
        "lift_ci_low": compact(ci_low),
        "lift_ci_high": compact(ci_high),
        "lift_ci_excludes_zero": bool(ci_low is not None and ci_low > 0),
        "z": compact(z_value, 4),
        "p_value": compact(normal_two_sided_p(z_value), 8),
    }


def evaluate_match_indexes(
    match_indexes: list[int],
    success_indexes: set[int],
    baseline_counts: dict,
) -> dict:
    checked = len(match_indexes)
    successes = sum(1 for index in match_indexes if index in success_indexes)
    baseline = baseline_counts["rate"] or 0.0
    rate = None if checked == 0 else successes / checked
    lift = None if rate is None else rate - baseline
    z_value = binomial_z(successes, checked, baseline)
    ci_low, ci_high = lift_ci(
        successes,
        checked,
        baseline_counts["successes"],
        baseline_counts["checked"],
    )

    return {
        "checked": checked,
        "successes": successes,
        "rate": compact(rate),
        "baseline": compact(baseline),
        "lift": compact(lift),
        "lift_ci_low": compact(ci_low),
        "lift_ci_high": compact(ci_high),
        "lift_ci_excludes_zero": bool(ci_low is not None and ci_low > 0),
        "z": compact(z_value, 4),
        "p_value": compact(normal_two_sided_p(z_value), 8),
    }


def split_indexes(total_rounds: int, holdout_fraction: float) -> tuple[list[int], list[int]]:
    indexes = list(range(1, total_rounds))
    split_at = int(len(indexes) * (1 - holdout_fraction))
    split_at = max(1, min(split_at, len(indexes) - 1))
    return indexes[:split_at], indexes[split_at:]


def split_blocks(indexes: list[int], count: int) -> list[list[int]]:
    if count <= 0:
        return []

    blocks = []
    size = len(indexes)
    cursor = 0

    for block_index in range(count):
        block_size = size // count + (1 if block_index < size % count else 0)
        blocks.append(indexes[cursor : cursor + block_size])
        cursor += block_size

    return blocks


def walk_forward(
    rounds: list[dict],
    indexes: list[int],
    pattern: Pattern,
    outcome: Outcome,
    min_matches: int,
    folds: int,
) -> dict:
    blocks = split_blocks(indexes, folds + 1)
    results = []

    for fold_number, test_indexes in enumerate(blocks[1:], start=1):
        previous_blocks = blocks[:fold_number]
        train_indexes = [
            index
            for block in previous_blocks
            for index in block
        ]

        if not train_indexes or not test_indexes:
            continue

        baseline = outcome_counts(rounds, train_indexes, outcome)
        result = evaluate_pattern(
            rounds,
            test_indexes,
            pattern,
            outcome,
            baseline,
        )
        result["fold"] = fold_number
        result["valid"] = result["checked"] >= min_matches
        result["positive"] = bool(
            result["valid"]
            and result["lift"] is not None
            and result["lift"] > 0
        )
        result["significant_positive"] = bool(
            result["positive"]
            and result["p_value"] is not None
            and result["p_value"] <= 0.05
        )
        results.append(result)

    valid = [item for item in results if item["valid"]]
    positive = [item for item in valid if item["positive"]]
    significant = [item for item in valid if item["significant_positive"]]

    return {
        "valid_folds": len(valid),
        "positive_folds": len(positive),
        "significant_positive_folds": len(significant),
        "positive_fold_share": compact(len(positive) / len(valid) if valid else None),
        "average_lift": compact(
            sum(item["lift"] for item in valid if item["lift"] is not None) / len(valid)
            if valid
            else None
        ),
    }


def walk_forward_matches(
    match_indexes: list[int],
    outcome_key: str,
    success_indexes: set[int],
    fold_contexts: list[dict],
    min_matches: int,
) -> dict:
    results = []

    for context in fold_contexts:
        test_matches = [
            index
            for index in match_indexes
            if index in context["test_set"]
        ]
        result = evaluate_match_indexes(
            test_matches,
            success_indexes,
            context["baselines"][outcome_key],
        )
        result["fold"] = context["fold"]
        result["valid"] = result["checked"] >= min_matches
        result["positive"] = bool(
            result["valid"]
            and result["lift"] is not None
            and result["lift"] > 0
        )
        result["significant_positive"] = bool(
            result["positive"]
            and result["p_value"] is not None
            and result["p_value"] <= 0.05
        )
        results.append(result)

    valid = [item for item in results if item["valid"]]
    positive = [item for item in valid if item["positive"]]
    significant = [item for item in valid if item["significant_positive"]]

    return {
        "valid_folds": len(valid),
        "positive_folds": len(positive),
        "significant_positive_folds": len(significant),
        "positive_fold_share": compact(len(positive) / len(valid) if valid else None),
        "average_lift": compact(
            sum(item["lift"] for item in valid if item["lift"] is not None) / len(valid)
            if valid
            else None
        ),
    }


def apply_fdr(items: list[dict], alpha: float) -> None:
    candidates = []

    for index, item in enumerate(items):
        p_value = item["holdout"].get("p_value")
        if p_value is None:
            continue

        candidates.append((float(p_value), index))

    for item in items:
        item["holdout"]["q_value"] = None
        item["holdout"]["fdr_significant"] = False

    if not candidates:
        return

    ordered = sorted(candidates, key=lambda pair: pair[0])
    count = len(ordered)
    adjusted = [1.0] * count
    running = 1.0

    for rank_from_end, (p_value, _index) in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_value * count / rank)
        adjusted[rank - 1] = min(1.0, running)

    for (_p_value, item_index), q_value in zip(ordered, adjusted):
        items[item_index]["holdout"]["q_value"] = compact(q_value, 8)
        items[item_index]["holdout"]["fdr_significant"] = q_value <= alpha


def classify_item(item: dict) -> str:
    train = item["train"]
    holdout = item["holdout"]
    wf = item["walk_forward"]
    holdout_lift = holdout.get("lift") or 0.0
    train_lift = train.get("lift") or 0.0
    z_value = holdout.get("z") or 0.0
    fdr_confirmed = bool(holdout.get("fdr_significant"))
    stable = bool(
        wf.get("valid_folds", 0) >= 4
        and wf.get("positive_fold_share", 0) >= 0.75
        and wf.get("significant_positive_folds", 0) >= 2
    )

    if (
        fdr_confirmed
        and stable
        and holdout_lift > 0
        and train_lift > 0
        and holdout.get("lift_ci_excludes_zero")
    ):
        return "confirmed"

    if (
        holdout_lift > 0
        and train_lift >= 0
        and z_value >= STRONG_Z
        and stable
    ):
        return "watch_strong"

    if holdout_lift > 0 and train_lift >= 0 and z_value >= WATCH_Z:
        return "watch"

    if holdout_lift > 0 and train_lift >= 0:
        return "weak_positive"

    return "not_useful"


def discover_patterns(
    rounds: list[dict],
    min_train_matches: int,
    min_holdout_matches: int,
    holdout_fraction: float,
    fdr_alpha: float,
    walk_forward_folds: int,
) -> dict:
    train_indexes, holdout_indexes = split_indexes(len(rounds), holdout_fraction)
    all_indexes = train_indexes + holdout_indexes
    train_set = set(train_indexes)
    holdout_set = set(holdout_indexes)
    patterns = make_patterns()
    rows = []
    success_indexes = {
        outcome.key: {
            index
            for index in all_indexes
            if outcome.predicate(float(rounds[index]["multiplier"]))
        }
        for outcome in OUTCOMES
    }
    baseline_train = {
        outcome.key: outcome_counts(rounds, train_indexes, outcome)
        for outcome in OUTCOMES
    }
    baseline_holdout = {
        outcome.key: outcome_counts(rounds, holdout_indexes, outcome)
        for outcome in OUTCOMES
    }
    fold_blocks = split_blocks(all_indexes, walk_forward_folds + 1)
    fold_contexts = []

    for fold_number, test_indexes in enumerate(fold_blocks[1:], start=1):
        train_fold_indexes = [
            index
            for block in fold_blocks[:fold_number]
            for index in block
        ]
        fold_contexts.append(
            {
                "fold": fold_number,
                "test_set": set(test_indexes),
                "baselines": {
                    outcome.key: outcome_counts(
                        rounds,
                        train_fold_indexes,
                        outcome,
                    )
                    for outcome in OUTCOMES
                },
            }
        )

    for pattern in patterns:
        matched_indexes = [
            index
            for index in all_indexes
            if pattern.predicate(rounds, index)
        ]
        train_matches = [
            index
            for index in matched_indexes
            if index in train_set
        ]
        holdout_matches = [
            index
            for index in matched_indexes
            if index in holdout_set
        ]

        for outcome in OUTCOMES:
            train = evaluate_match_indexes(
                train_matches,
                success_indexes[outcome.key],
                baseline_train[outcome.key],
            )
            holdout = evaluate_match_indexes(
                holdout_matches,
                success_indexes[outcome.key],
                baseline_holdout[outcome.key],
            )

            if (
                train["checked"] < min_train_matches
                or holdout["checked"] < min_holdout_matches
            ):
                continue

            item = {
                "pattern": pattern.label,
                "pattern_key": pattern.key,
                "group": pattern.group,
                "outcome": outcome.label,
                "outcome_key": outcome.key,
                "train": train,
                "holdout": holdout,
                "walk_forward": walk_forward_matches(
                    matched_indexes,
                    outcome.key,
                    success_indexes[outcome.key],
                    fold_contexts,
                    WALK_FORWARD_MIN_MATCHES,
                ),
            }
            rows.append(item)

    apply_fdr(rows, fdr_alpha)

    for item in rows:
        item["status"] = classify_item(item)

    rows.sort(
        key=lambda item: (
            item["status"] == "confirmed",
            item["status"] == "watch_strong",
            item["status"] == "watch",
            item["holdout"].get("z") or 0,
            item["holdout"].get("lift") or 0,
            item["holdout"].get("checked") or 0,
        ),
        reverse=True,
    )

    by_group = {}
    for group in sorted({item["group"] for item in rows}):
        group_items = [item for item in rows if item["group"] == group]
        by_group[group] = group_items[:TOP_PER_GROUP]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": len(rounds),
        "train_rounds": len(train_indexes),
        "holdout_rounds": len(holdout_indexes),
        "patterns_tested": len(rows),
        "min_train_matches": min_train_matches,
        "min_holdout_matches": min_holdout_matches,
        "fdr_alpha": fdr_alpha,
        "confirmed_count": sum(1 for item in rows if item["status"] == "confirmed"),
        "watch_count": sum(
            1 for item in rows if item["status"] in {"watch", "watch_strong"}
        ),
        "baseline_train": baseline_train,
        "baseline_holdout": baseline_holdout,
        "top": rows[:30],
        "groups": by_group,
    }


def load_prediction_rounds(path: Path) -> tuple[list[dict], dict]:
    all_rounds = load_rounds(path)
    rounds, selection = select_prediction_rounds(all_rounds)
    return with_cached_timestamps(rounds), selection


def status_prefix(status: str) -> str:
    return {
        "confirmed": "confirmed",
        "watch_strong": "watch",
        "watch": "watch",
        "weak_positive": "weak",
        "not_useful": "no edge",
    }.get(status, status)


def print_item(item: dict) -> None:
    holdout = item["holdout"]
    wf = item["walk_forward"]
    print(
        "- "
        f"[{status_prefix(item['status'])}] "
        f"{item['pattern']} -> {item['outcome']} | "
        f"{(holdout.get('rate') or 0) * 100:.1f}% vs "
        f"{(holdout.get('baseline') or 0) * 100:.1f}% "
        f"(lift {(holdout.get('lift') or 0) * 100:+.1f} pp, "
        f"z {holdout.get('z')}, q {holdout.get('q_value')}, "
        f"n {holdout.get('checked')}, "
        f"WF {wf.get('positive_folds')}/{wf.get('valid_folds')}+)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN_MATCHES)
    parser.add_argument("--min-holdout", type=int, default=MIN_HOLDOUT_MATCHES)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rounds, selection = load_prediction_rounds(Path(args.csv))
    report = discover_patterns(
        rounds,
        min_train_matches=args.min_train,
        min_holdout_matches=args.min_holdout,
        holdout_fraction=args.holdout_fraction,
        fdr_alpha=FDR_ALPHA,
        walk_forward_folds=WALK_FORWARD_FOLDS,
    )
    report["data_selection"] = selection

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"Rounds checked: {report['rounds']}")
    print(
        f"Train/Holdout: {report['train_rounds']} / {report['holdout_rounds']} "
        f"| pattern tests kept: {report['patterns_tested']}"
    )
    print(
        f"Confirmed: {report['confirmed_count']} | watch candidates: {report['watch_count']}"
    )
    print("Holdout baselines:")
    for outcome in OUTCOMES:
        baseline = report["baseline_holdout"][outcome.key]["rate"]
        print(f"- {outcome.label}: {(baseline or 0) * 100:.1f}%")

    for group, label in (
        ("small_streak_then_big", "Small streak then big"),
        ("medium_cluster", "Medium 2x-6x cluster"),
        ("bucket_sequence", "Exact bucket sequences"),
        ("big_cooldown", "Big multiplier cooldown"),
        ("fixed_time", "Fixed time"),
    ):
        print(f"\n{label}:")
        rows = report["groups"].get(group, [])
        if not rows:
            print("- no enough-sample candidates")
            continue

        for item in rows[:5]:
            print_item(item)

    print(f"\nSaved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
