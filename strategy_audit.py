"""Evaluate whether saved rounds contain a usable cashout strategy.

The audit intentionally chooses strategies using older rounds, then reports the
result on later holdout rounds. That keeps the report from tuning itself to the
same data it is judging.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from aviator_analyzer import load_rounds


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"
DEFAULT_OUT_PATH = DATA_DIR / "strategy_audit.json"

CASHOUT_TARGETS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
HOUR_MIN_BETS = 80
DAY_MIN_BETS = 120
PATTERN_MIN_BETS = 80
HOLDOUT_FRACTION = 0.30
GOOD_HOLDOUT_ROI = 0.01


@dataclass(frozen=True)
class Round:
    index: int
    timestamp: str
    timestamp_dt: datetime | None
    multiplier: float


def parse_time(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[Round]:
    rows = []

    for index, row in enumerate(load_rounds(path)):
        try:
            multiplier = float(row["multiplier"])
        except (TypeError, ValueError):
            continue

        rows.append(
            Round(
                index=index,
                timestamp=row.get("timestamp", ""),
                timestamp_dt=parse_time(row.get("timestamp", "")),
                multiplier=multiplier,
            )
        )

    return rows


def clean_float(value, digits=6):
    if value is None:
        return None

    return round(float(value), digits)


def strategy_stats(rows: list[Round], indexes: Iterable[int], cashout: float):
    selected = [
        rows[index]
        for index in indexes
        if 0 <= index < len(rows)
    ]
    bets = len(selected)

    if not bets:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": None,
            "profit_units": 0,
            "roi": None,
        }

    wins = sum(
        1
        for row in selected
        if row.multiplier >= cashout
    )
    profit_units = wins * cashout - bets

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets,
        "profit_units": profit_units,
        "roi": profit_units / bets,
    }


def compact_stats(stats):
    return {
        "bets": stats["bets"],
        "wins": stats["wins"],
        "win_rate": clean_float(stats["win_rate"]),
        "profit_units": clean_float(stats["profit_units"]),
        "roi": clean_float(stats["roi"]),
    }


def candidate(
    rows,
    family,
    label,
    cashout,
    train_indexes,
    holdout_indexes,
):
    return {
        "family": family,
        "label": label,
        "cashout": cashout,
        "train": compact_stats(
            strategy_stats(
                rows,
                train_indexes,
                cashout,
            )
        ),
        "holdout": compact_stats(
            strategy_stats(
                rows,
                holdout_indexes,
                cashout,
            )
        ),
    }


def roi_value(item, section="train"):
    value = item.get(section, {}).get("roi")
    return float(value) if value is not None else -999


def enough_bets(item, section, minimum):
    return int(
        item.get(section, {}).get("bets", 0)
    ) >= minimum


def weekday_name(value):
    return [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ][value]


def indexes_by_condition(rows, start, end, predicate):
    indexes = []

    for index in range(start, end):
        row = rows[index]

        if predicate(index, row):
            indexes.append(index)

    return indexes


def small_streak_predicate(rows, threshold, length):
    def predicate(index, _row):
        if index < length:
            return False

        return all(
            rows[previous].multiplier < threshold
            for previous in range(index - length, index)
        )

    return predicate


def gap_since_big_predicate(rows, threshold, minimum_gap):
    last_big_index = None
    gaps = {}

    for index, row in enumerate(rows):
        if last_big_index is not None:
            gaps[index] = index - last_big_index

        if row.multiplier >= threshold:
            last_big_index = index

    def predicate(index, _row):
        return gaps.get(index, 0) >= minimum_gap

    return predicate


def build_candidates(rows: list[Round], split_index: int):
    train_all = list(
        range(
            0,
            split_index,
        )
    )
    holdout_all = list(
        range(
            split_index,
            len(rows),
        )
    )
    candidates = []

    for cashout in CASHOUT_TARGETS:
        candidates.append(
            candidate(
                rows,
                "fixed_cashout",
                f"Every round, cash out at {cashout:g}x",
                cashout,
                train_all,
                holdout_all,
            )
        )

    for cashout in CASHOUT_TARGETS:
        for hour in range(24):
            train_indexes = indexes_by_condition(
                rows,
                0,
                split_index,
                lambda _index, row, hour=hour: (
                    row.timestamp_dt is not None
                    and row.timestamp_dt.hour == hour
                ),
            )
            holdout_indexes = indexes_by_condition(
                rows,
                split_index,
                len(rows),
                lambda _index, row, hour=hour: (
                    row.timestamp_dt is not None
                    and row.timestamp_dt.hour == hour
                ),
            )
            candidates.append(
                candidate(
                    rows,
                    "hour",
                    f"Only hour {hour:02d}:00, cash out at {cashout:g}x",
                    cashout,
                    train_indexes,
                    holdout_indexes,
                )
            )

    for cashout in CASHOUT_TARGETS:
        for weekday in range(7):
            train_indexes = indexes_by_condition(
                rows,
                0,
                split_index,
                lambda _index, row, weekday=weekday: (
                    row.timestamp_dt is not None
                    and row.timestamp_dt.weekday() == weekday
                ),
            )
            holdout_indexes = indexes_by_condition(
                rows,
                split_index,
                len(rows),
                lambda _index, row, weekday=weekday: (
                    row.timestamp_dt is not None
                    and row.timestamp_dt.weekday() == weekday
                ),
            )
            candidates.append(
                candidate(
                    rows,
                    "weekday",
                    (
                        f"Only {weekday_name(weekday)}, "
                        f"cash out at {cashout:g}x"
                    ),
                    cashout,
                    train_indexes,
                    holdout_indexes,
                )
            )

    for cashout in CASHOUT_TARGETS:
        for threshold in [1.5, 2.0, 3.0]:
            for length in [2, 3, 4, 5, 8]:
                predicate = small_streak_predicate(
                    rows,
                    threshold,
                    length,
                )
                candidates.append(
                    candidate(
                        rows,
                        "small_streak",
                        (
                            f"After {length} rounds below {threshold:g}x, "
                            f"cash out at {cashout:g}x"
                        ),
                        cashout,
                        indexes_by_condition(rows, 0, split_index, predicate),
                        indexes_by_condition(rows, split_index, len(rows), predicate),
                    )
                )

    for cashout in CASHOUT_TARGETS:
        for threshold in [10.0, 20.0, 50.0, 100.0]:
            for minimum_gap in [4, 8, 16, 32, 64]:
                predicate = gap_since_big_predicate(
                    rows,
                    threshold,
                    minimum_gap,
                )
                candidates.append(
                    candidate(
                        rows,
                        "big_gap",
                        (
                            f"When {threshold:g}x+ is {minimum_gap}+ rounds old, "
                            f"cash out at {cashout:g}x"
                        ),
                        cashout,
                        indexes_by_condition(rows, 0, split_index, predicate),
                        indexes_by_condition(rows, split_index, len(rows), predicate),
                    )
                )

    return candidates


def best_by_family(candidates):
    minimums = {
        "fixed_cashout": 200,
        "hour": HOUR_MIN_BETS,
        "weekday": DAY_MIN_BETS,
        "small_streak": PATTERN_MIN_BETS,
        "big_gap": PATTERN_MIN_BETS,
    }
    grouped = {}

    for item in candidates:
        minimum = minimums.get(
            item["family"],
            PATTERN_MIN_BETS,
        )

        if not enough_bets(
            item,
            "train",
            minimum,
        ):
            continue

        grouped.setdefault(
            item["family"],
            [],
        ).append(item)

    winners = []

    for family, items in grouped.items():
        winners.append(
            max(
                items,
                key=lambda item: roi_value(
                    item,
                    "train",
                ),
            )
        )

    return sorted(
        winners,
        key=lambda item: roi_value(
            item,
            "train",
        ),
        reverse=True,
    )


def classify_report(family_winners):
    train_positive = [
        item
        for item in family_winners
        if roi_value(item, "train") > 0
    ]
    holdout_positive = [
        item
        for item in train_positive
        if (
            roi_value(item, "holdout") > GOOD_HOLDOUT_ROI
            and enough_bets(item, "holdout", PATTERN_MIN_BETS)
        )
    ]

    if holdout_positive:
        return {
            "status": "candidate",
            "headline": "Candidate strategy needs more live proof",
            "message": (
                "One older-data strategy stayed positive on holdout. "
                "Treat it as research only until it repeats in future rounds."
            ),
        }

    if train_positive:
        return {
            "status": "no_edge",
            "headline": "Past winners failed later",
            "message": (
                "Some strategies looked profitable in older rounds, "
                "but did not stay profitable on later holdout rounds."
            ),
        }

    return {
        "status": "no_edge",
        "headline": "No profitable strategy found",
        "message": (
            "The tested strategies did not beat the game on older data "
            "or later holdout data."
        ),
    }


def build_report(csv_path: Path, holdout_fraction: float = HOLDOUT_FRACTION):
    rows = read_rows(csv_path)

    if len(rows) < 1000:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "available": False,
            "status": "collecting",
            "headline": "Collecting more rounds",
            "message": "Need at least 1000 rounds for a useful strategy audit.",
            "rounds": len(rows),
        }

    split_index = int(
        len(rows) * (1 - holdout_fraction)
    )
    candidates = build_candidates(
        rows,
        split_index,
    )
    family_winners = best_by_family(
        candidates
    )
    verdict = classify_report(
        family_winners
    )
    best_train = family_winners[0] if family_winners else None
    positive_both = [
        item
        for item in family_winners
        if (
            roi_value(item, "train") > 0
            and roi_value(item, "holdout") > GOOD_HOLDOUT_ROI
            and enough_bets(item, "holdout", PATTERN_MIN_BETS)
        )
    ]
    positive_both.sort(
        key=lambda item: roi_value(
            item,
            "holdout",
        ),
        reverse=True,
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available": True,
        "rounds": len(rows),
        "train_rounds": split_index,
        "holdout_rounds": len(rows) - split_index,
        "holdout_fraction": holdout_fraction,
        "families_tested": len(
            {
                item["family"]
                for item in candidates
            }
        ),
        "candidates_tested": len(candidates),
        "status": verdict["status"],
        "headline": verdict["headline"],
        "message": verdict["message"],
        "best_train_strategy": best_train,
        "best_forward_candidate": positive_both[0] if positive_both else None,
        "positive_both_count": len(positive_both),
        "positive_both": positive_both,
        "family_winners": family_winners,
        "note": (
            "Strategies are selected on older rounds and judged on later rounds. "
            "A good training ROI with weak holdout ROI is treated as overfitting."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether saved Aviatrix rounds contain a profitable cashout strategy."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_PATH),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=HOLDOUT_FRACTION,
    )
    args = parser.parse_args()

    report = build_report(
        Path(args.csv),
        holdout_fraction=args.holdout_fraction,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{report.get('headline')} | rounds {report.get('rounds')} | "
        f"saved {out_path}"
    )

    best = report.get("best_train_strategy")

    if best:
        print(
            f"Best old strategy: {best['label']} | "
            f"train ROI {best['train']['roi']:.2%} | "
            f"holdout ROI {best['holdout']['roi']:.2%}"
        )


if __name__ == "__main__":
    main()
