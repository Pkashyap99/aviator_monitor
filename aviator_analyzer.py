import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "analysis.json"


DEFAULT_TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]
RECENT_WINDOW = 80
MULTI_PATTERN_LOOKBACKS = [1, 2, 3, 4]
MULTI_PATTERN_MATCH_LIMIT = 320
CONFIDENCE_RANGE_TARGET = 0.81
CONFIDENCE_RANGE_NEAR_TARGET = 0.79
CONFIDENCE_RANGE_MIN_EDGE = 0.015
CONFIDENCE_RANGE_MAXIMUM = 5.0
CONFIDENCE_RANGE_MAX_BUCKETS = 10
ACTIONABLE_RANGE_MAXIMUM = 3.0
ACTIONABLE_RANGE_MAX_BUCKETS = 6
DEDUPLICATE_WINDOW_SECONDS = 6
MIN_REAL_ROUNDS_FOR_PREDICTION = 300
LEGACY_SOURCES = {"", "unlabeled", "legacy"}
LIVE_GAME_SOURCES = {"real", "demo"}
EXCLUDED_PREDICTION_SOURCES = set()


def make_range_label(minimum, maximum):
    if maximum is None:
        return (
            f"MORE THAN {minimum:.2f}x",
            f"{minimum:.2f}x+",
        )

    return (
        f"{minimum:.2f}x TO {maximum:.2f}x",
        f"{minimum:.2f}x-{maximum:.2f}x",
    )


def make_range_bucket(minimum, maximum):
    label, short = make_range_label(minimum, maximum)
    return {
        "label": label,
        "short": short,
        "minimum": minimum,
        "maximum": maximum,
    }


RANGE_BUCKETS = [
    make_range_bucket(1.0, 1.1),
    make_range_bucket(1.1, 1.2),
    make_range_bucket(1.2, 1.3),
    make_range_bucket(1.3, 1.5),
    make_range_bucket(1.5, 1.7),
    make_range_bucket(1.7, 2.0),
    make_range_bucket(2.0, 2.5),
    make_range_bucket(2.5, 3.0),
    make_range_bucket(3.0, 4.0),
    make_range_bucket(4.0, 5.0),
    make_range_bucket(5.0, 7.0),
    make_range_bucket(7.0, 10.0),
    make_range_bucket(10.0, 15.0),
    make_range_bucket(15.0, 25.0),
    make_range_bucket(25.0, 50.0),
    make_range_bucket(50.0, 100.0),
    make_range_bucket(100.0, None),
]


def bucket_labels_for_range(minimum, maximum):
    upper_limit = math.inf if maximum is None else maximum
    labels = []

    for bucket in RANGE_BUCKETS:
        bucket_maximum = (
            math.inf
            if bucket["maximum"] is None
            else bucket["maximum"]
        )

        if bucket["minimum"] >= minimum and bucket_maximum <= upper_limit:
            labels.append(bucket["label"])

    return labels


def make_adaptive_candidate(minimum, maximum):
    label, short = make_range_label(minimum, maximum)
    return {
        "label": label,
        "short": short,
        "minimum": minimum,
        "maximum": maximum,
        "bucket_labels": bucket_labels_for_range(minimum, maximum),
    }


ADAPTIVE_RANGE_CANDIDATES = [
    make_adaptive_candidate(1.0, 1.2),
    make_adaptive_candidate(1.0, 1.3),
    make_adaptive_candidate(1.0, 1.5),
    make_adaptive_candidate(1.0, 1.7),
    make_adaptive_candidate(1.0, 2.0),
    make_adaptive_candidate(1.1, 1.5),
    make_adaptive_candidate(1.2, 1.7),
    make_adaptive_candidate(1.2, 2.0),
    make_adaptive_candidate(1.3, 2.0),
    make_adaptive_candidate(1.5, 2.5),
    make_adaptive_candidate(1.5, 3.0),
    make_adaptive_candidate(1.7, 3.0),
    make_adaptive_candidate(2.0, 3.0),
    make_adaptive_candidate(2.0, 4.0),
    make_adaptive_candidate(2.5, 5.0),
    make_adaptive_candidate(3.0, 5.0),
    make_adaptive_candidate(5.0, 10.0),
    make_adaptive_candidate(10.0, None),
]
PROFILE_WEIGHTS = {
    "balanced": {
        "overall": 0.30,
        "recent": 0.22,
        "pattern": 0.18,
        "multi_pattern": 0.18,
        "streak": 0.15,
    },
    "defensive": {
        "overall": 0.58,
        "recent": 0.15,
        "pattern": 0.10,
        "multi_pattern": 0.10,
        "streak": 0.10,
    },
    "recent_heavy": {
        "overall": 0.26,
        "recent": 0.44,
        "pattern": 0.10,
        "multi_pattern": 0.10,
        "streak": 0.10,
    },
    "pattern_heavy": {
        "overall": 0.25,
        "recent": 0.15,
        "pattern": 0.32,
        "multi_pattern": 0.18,
        "streak": 0.10,
    },
    "streak_heavy": {
        "overall": 0.26,
        "recent": 0.15,
        "pattern": 0.10,
        "multi_pattern": 0.10,
        "streak": 0.45,
    },
}

MIN_HIGH_PROBABILITY_BY_TARGET = {
    1.5: 0.58,
    2.0: 0.50,
    3.0: 0.38,
    5.0: 0.28,
    10.0: 0.18,
    25.0: 0.10,
    50.0: 0.07,
    100.0: 0.05,
}


def parse_round_time(value):
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        return None


def load_rounds(path):
    rounds = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                multiplier = float(row["multiplier"])
            except (KeyError, TypeError, ValueError):
                continue

            if multiplier < 1:
                continue

            timestamp = row.get("timestamp", "")
            current_time = parse_round_time(timestamp)

            if rounds:
                previous = rounds[-1]
                previous_time = parse_round_time(
                    previous.get("timestamp", "")
                )

                if (
                    round(float(previous["multiplier"]), 2)
                    == round(multiplier, 2)
                    and current_time is not None
                    and previous_time is not None
                    and abs((current_time - previous_time).total_seconds())
                    <= DEDUPLICATE_WINDOW_SECONDS
                ):
                    continue

            rounds.append(
                {
                    "timestamp": timestamp,
                    "multiplier": multiplier,
                    "round_id": row.get(
                        "round_id",
                        ""
                    ),
                    "source": row.get(
                        "source",
                        ""
                    ),
                }
            )

    return rounds


def source_label(source):
    if source in LEGACY_SOURCES:
        return "unlabeled"

    return source or "unlabeled"


def source_counts(rounds):
    counts = Counter()

    for round_data in rounds:
        counts[source_label(round_data.get("source", ""))] += 1

    return dict(counts)


def select_prediction_rounds(rounds, preferred_source="real", minimum_source_rounds=None):
    minimum = (
        MIN_REAL_ROUNDS_FOR_PREDICTION
        if minimum_source_rounds is None
        else minimum_source_rounds
    )
    source_rounds = [
        round_data
        for round_data in rounds
        if source_label(round_data.get("source", "")) in LIVE_GAME_SOURCES
    ]
    trusted_rounds = [
        round_data
        for round_data in rounds
        if (
            source_label(round_data.get("source", "")) in LIVE_GAME_SOURCES
            or round_data.get("source", "") in LEGACY_SOURCES
        )
    ]
    legacy_round_count = sum(
        1
        for round_data in rounds
        if round_data.get("source", "") in LEGACY_SOURCES
    )
    excluded_round_count = sum(
        1
        for round_data in rounds
        if source_label(round_data.get("source", "")) in EXCLUDED_PREDICTION_SOURCES
    )

    if len(trusted_rounds) >= minimum and legacy_round_count:
        return trusted_rounds, {
            "mode": "trusted",
            "preferred_source": preferred_source,
            "combined_sources": sorted(LIVE_GAME_SOURCES),
            "minimum_source_rounds": minimum,
            "using_source_only": False,
            "using_trusted_sources": True,
            "source_rounds": len(source_rounds),
            "demo_rounds": sum(
                1
                for round_data in rounds
                if source_label(round_data.get("source", "")) == "demo"
            ),
            "real_rounds": sum(
                1
                for round_data in rounds
                if source_label(round_data.get("source", "")) == "real"
            ),
            "legacy_rounds": legacy_round_count,
            "trusted_rounds": len(trusted_rounds),
            "excluded_rounds": excluded_round_count,
            "total_rounds": len(rounds),
            "counts": source_counts(rounds),
        }

    if len(source_rounds) >= minimum:
        return source_rounds, {
            "mode": "game",
            "preferred_source": preferred_source,
            "combined_sources": sorted(LIVE_GAME_SOURCES),
            "minimum_source_rounds": minimum,
            "using_source_only": True,
            "using_trusted_sources": False,
            "source_rounds": len(source_rounds),
            "demo_rounds": sum(
                1
                for round_data in rounds
                if source_label(round_data.get("source", "")) == "demo"
            ),
            "real_rounds": sum(
                1
                for round_data in rounds
                if source_label(round_data.get("source", "")) == "real"
            ),
            "legacy_rounds": legacy_round_count,
            "trusted_rounds": len(source_rounds),
            "excluded_rounds": excluded_round_count,
            "total_rounds": len(rounds),
            "counts": source_counts(rounds),
        }

    return rounds, {
        "mode": "all",
        "preferred_source": preferred_source,
        "combined_sources": sorted(LIVE_GAME_SOURCES),
        "minimum_source_rounds": minimum,
        "using_source_only": False,
        "using_trusted_sources": False,
        "source_rounds": len(source_rounds),
        "demo_rounds": sum(
            1
            for round_data in rounds
            if source_label(round_data.get("source", "")) == "demo"
        ),
        "real_rounds": sum(
            1
            for round_data in rounds
            if source_label(round_data.get("source", "")) == "real"
        ),
        "legacy_rounds": legacy_round_count,
        "trusted_rounds": len(trusted_rounds),
        "excluded_rounds": 0,
        "total_rounds": len(rounds),
        "counts": source_counts(rounds),
    }


def bucket_multiplier(multiplier):
    if multiplier < 1.2:
        return "<1.20"
    if multiplier < 1.5:
        return "1.20-1.49"
    if multiplier < 2.0:
        return "1.50-1.99"
    if multiplier < 3.0:
        return "2.00-2.99"
    if multiplier < 5.0:
        return "3.00-4.99"
    if multiplier < 10.0:
        return "5.00-9.99"
    return "10.00+"


def range_bucket(multiplier):
    for bucket in RANGE_BUCKETS:
        maximum = bucket["maximum"]

        if maximum is None:
            return bucket

        if multiplier < maximum:
            return bucket

    return RANGE_BUCKETS[-1]


def percentile(sorted_values, pct):
    if not sorted_values:
        return None

    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def smoothed_bucket_probabilities(sample, baseline_counts, prior_weight):
    total = len(sample) + prior_weight
    probabilities = {}

    if total <= 0:
        return {
            bucket["label"]: 0
            for bucket in RANGE_BUCKETS
        }

    sample_counts = Counter(
        range_bucket(value)["label"]
        for value in sample
    )

    for bucket in RANGE_BUCKETS:
        label = bucket["label"]
        prior = baseline_counts.get(label, 0)
        probabilities[label] = (
            sample_counts.get(label, 0)
            + prior * prior_weight
        ) / total

    return probabilities


def weighted_bucket_probabilities(
    values,
    pattern_sample,
    recent_sample,
    multi_pattern_sample=None,
):
    if not values:
        return []

    multi_pattern_sample = multi_pattern_sample or []
    baseline_counts = Counter(
        range_bucket(value)["label"]
        for value in values
    )
    baseline_probabilities = {
        label: count / len(values)
        for label, count in baseline_counts.items()
    }
    recent_probabilities = smoothed_bucket_probabilities(
        recent_sample,
        baseline_probabilities,
        prior_weight=12,
    )
    pattern_probabilities = smoothed_bucket_probabilities(
        pattern_sample,
        baseline_probabilities,
        prior_weight=8,
    )
    multi_pattern_probabilities = smoothed_bucket_probabilities(
        multi_pattern_sample,
        baseline_probabilities,
        prior_weight=10,
    )
    probabilities = []

    for bucket in RANGE_BUCKETS:
        label = bucket["label"]
        if multi_pattern_sample:
            probability_value = (
                baseline_probabilities.get(label, 0) * 0.24
                + recent_probabilities.get(label, 0) * 0.28
                + pattern_probabilities.get(label, 0) * 0.18
                + multi_pattern_probabilities.get(label, 0) * 0.30
            )
        else:
            probability_value = (
                baseline_probabilities.get(label, 0) * 0.35
                + recent_probabilities.get(label, 0) * 0.35
                + pattern_probabilities.get(label, 0) * 0.30
            )
        probabilities.append(
            {
                **bucket,
                "probability": probability_value,
                "baseline_probability": baseline_probabilities.get(label, 0),
            }
        )

    return probabilities


def model_bucket_probabilities(values, sample, prior_weight, model_name):
    if not values:
        return []

    baseline_counts = Counter(
        range_bucket(value)["label"]
        for value in values
    )
    baseline_probabilities = {
        label: count / len(values)
        for label, count in baseline_counts.items()
    }

    if model_name == "baseline":
        model_probabilities = baseline_probabilities
    else:
        model_probabilities = smoothed_bucket_probabilities(
            sample,
            baseline_probabilities,
            prior_weight=prior_weight,
        )

    return [
        {
            **bucket,
            "probability": model_probabilities.get(bucket["label"], 0),
            "baseline_probability": baseline_probabilities.get(bucket["label"], 0),
        }
        for bucket in RANGE_BUCKETS
    ]


def range_quality(best_bucket, runner_up_bucket, evidence_score):
    probability_value = best_bucket["probability"]
    runner_up_probability = (
        runner_up_bucket["probability"]
        if runner_up_bucket
        else 0
    )
    edge = probability_value - runner_up_probability

    if (
        probability_value >= 0.34
        and edge >= 0.10
        and evidence_score >= 0.55
    ):
        return True, "high", "clear range", edge

    if (
        probability_value >= 0.28
        and edge >= 0.06
        and evidence_score >= 0.45
    ):
        return True, "medium", "clear range", edge

    if probability_value < 0.24:
        return False, "low", "top range probability too low", edge

    if edge < 0.04:
        return False, "low", "top ranges too close", edge

    return False, "low", "range not strong enough", edge


def adaptive_range_quality(candidate, evidence_score):
    probability_value = candidate["probability"]
    edge = candidate["probability"] - candidate["baseline_probability"]
    bucket_count = len(candidate["bucket_labels"])
    maximum = candidate.get("maximum")
    width = confidence_range_width(
        candidate.get("minimum", 1.0),
        maximum,
    )

    if maximum is None or maximum > ACTIONABLE_RANGE_MAXIMUM:
        return False, "low", "range too broad for main call", edge

    if bucket_count > ACTIONABLE_RANGE_MAX_BUCKETS or width > 1.15:
        return False, "low", "range too broad for main call", edge

    if (
        probability_value >= 0.50
        and edge >= 0.06
        and evidence_score >= 0.55
    ):
        return True, "high", "clear compact range", edge

    if (
        probability_value >= 0.42
        and edge >= 0.04
        and evidence_score >= 0.50
    ):
        return True, "medium", "clear compact range", edge

    if probability_value < 0.32:
        return False, "low", "range probability too low", edge

    if edge < 0.025:
        return False, "low", "range has no edge", edge

    return False, "low", "wider range not strong enough", edge


def adaptive_range_estimate(bucket_probabilities, evidence_score):
    probability_by_label = {
        item["label"]: item["probability"]
        for item in bucket_probabilities
    }
    baseline_by_label = {
        item["label"]: item["baseline_probability"]
        for item in bucket_probabilities
    }
    candidates = []

    for candidate in ADAPTIVE_RANGE_CANDIDATES:
        labels = candidate["bucket_labels"]
        probability_value = sum(
            probability_by_label.get(label, 0)
            for label in labels
        )
        baseline_probability = sum(
            baseline_by_label.get(label, 0)
            for label in labels
        )
        edge = probability_value - baseline_probability
        bucket_count = len(labels)
        width = confidence_range_width(
            candidate.get("minimum", 1.0),
            candidate.get("maximum"),
        )
        score = (
            probability_value
            + max(edge, 0) * 1.5
            - width * 0.16
            - max(bucket_count - 3, 0) * 0.02
        )
        clear_signal, confidence, clear_reason, quality_edge = adaptive_range_quality(
            {
                **candidate,
                "probability": probability_value,
                "baseline_probability": baseline_probability,
            },
            evidence_score,
        )
        candidates.append(
            {
                **candidate,
                "probability": probability_value,
                "baseline_probability": baseline_probability,
                "score": score,
                "clear_signal": clear_signal,
                "confidence": confidence,
                "clear_reason": clear_reason,
                "edge": quality_edge,
            }
        )

    return max(
        candidates,
        key=lambda item: item["score"],
    )


def confidence_range_label(minimum, maximum):
    if maximum is None:
        return (
            f"MORE THAN {minimum:.2f}x",
            f"{minimum:.2f}x+",
        )

    return (
        f"{minimum:.2f}x TO {maximum:.2f}x",
        f"{minimum:.2f}x-{maximum:.2f}x",
    )


def confidence_range_width(minimum, maximum):
    if maximum is None:
        return 12

    return math.log(
        max(maximum / max(minimum, 1), 1.01),
        2,
    )


def confidence_range_quality(candidate, evidence_score):
    probability_value = candidate["probability"]
    edge = probability_value - candidate["baseline_probability"]
    bucket_count = len(candidate["bucket_labels"])
    maximum = candidate.get("maximum")

    if maximum is None or maximum > ACTIONABLE_RANGE_MAXIMUM:
        return False, "low", "coverage range too broad for main call", edge

    if bucket_count > ACTIONABLE_RANGE_MAX_BUCKETS:
        return False, "low", "coverage range too broad for main call", edge

    if (
        probability_value >= CONFIDENCE_RANGE_TARGET
        and edge >= CONFIDENCE_RANGE_MIN_EDGE + 0.01
        and evidence_score >= 0.55
        and maximum is not None
        and maximum <= ACTIONABLE_RANGE_MAXIMUM
        and bucket_count <= ACTIONABLE_RANGE_MAX_BUCKETS
    ):
        return True, "high", "useful confidence range", edge

    if (
        probability_value >= CONFIDENCE_RANGE_TARGET
        and edge >= CONFIDENCE_RANGE_MIN_EDGE
        and evidence_score >= 0.45
        and maximum is not None
        and maximum <= ACTIONABLE_RANGE_MAXIMUM
        and bucket_count <= ACTIONABLE_RANGE_MAX_BUCKETS
    ):
        return True, "medium", "useful confidence range", edge

    if probability_value < CONFIDENCE_RANGE_TARGET:
        return False, "low", "below confidence target", edge

    if maximum is None or maximum > CONFIDENCE_RANGE_MAXIMUM:
        return False, "low", "confidence range too wide", edge

    if edge < CONFIDENCE_RANGE_MIN_EDGE:
        return False, "low", "confidence range not beating baseline", edge

    if evidence_score < 0.45:
        return False, "low", "not enough evidence for confidence range", edge

    if bucket_count > CONFIDENCE_RANGE_MAX_BUCKETS:
        return False, "low", "confidence range too broad", edge

    return False, "low", "confidence range not strong enough", edge


def recommended_cashout_for_range(candidate, sorted_sample):
    maximum = candidate.get("maximum")
    minimum = candidate.get("minimum", 1.0)
    median = percentile(sorted_sample, 0.50)
    p40 = percentile(sorted_sample, 0.40)
    base = p40 if p40 is not None else median

    if base is None:
        base = minimum

    target = max(1.05, float(base) * 0.9)

    if maximum is not None:
        target = min(target, maximum * 0.78)

    if minimum > 1.0:
        target = max(target, minimum * 0.95)

    return round(max(1.01, target), 2)


def confidence_range_estimate(bucket_probabilities, evidence_score, sorted_sample):
    candidates = []

    for start_index in range(len(bucket_probabilities)):
        for end_index in range(start_index, len(bucket_probabilities)):
            selected_buckets = bucket_probabilities[start_index:end_index + 1]
            minimum = selected_buckets[0]["minimum"]
            maximum = selected_buckets[-1]["maximum"]
            labels = [
                bucket["label"]
                for bucket in selected_buckets
            ]
            probability_value = sum(
                bucket["probability"]
                for bucket in selected_buckets
            )
            baseline_probability = sum(
                bucket["baseline_probability"]
                for bucket in selected_buckets
            )
            label, short = confidence_range_label(
                minimum,
                maximum,
            )
            width = confidence_range_width(
                minimum,
                maximum,
            )
            target_gap = abs(
                probability_value - CONFIDENCE_RANGE_TARGET
            )
            edge = probability_value - baseline_probability
            bucket_count = len(selected_buckets)
            score = (
                target_gap * 4.0
                + width * 0.22
                + bucket_count * 0.035
                - max(edge, 0) * 1.4
            )
            candidate = {
                "label": label,
                "short": short,
                "minimum": minimum,
                "maximum": maximum,
                "bucket_labels": labels,
                "probability": probability_value,
                "baseline_probability": baseline_probability,
                "score": score,
                "coverage_gap": probability_value - CONFIDENCE_RANGE_TARGET,
                "target_confidence": CONFIDENCE_RANGE_TARGET,
                "bucket_count": bucket_count,
            }
            clear_signal, confidence, clear_reason, quality_edge = confidence_range_quality(
                candidate,
                evidence_score,
            )
            candidate.update(
                {
                    "clear_signal": clear_signal,
                    "confidence": confidence,
                    "clear_reason": clear_reason,
                    "edge": quality_edge,
                }
            )
            candidate["cashout_target"] = recommended_cashout_for_range(
                candidate,
                sorted_sample,
            )
            candidates.append(
                candidate
            )

    allowed_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["maximum"] is not None
            and candidate["maximum"] <= CONFIDENCE_RANGE_MAXIMUM
            and candidate["bucket_count"] <= CONFIDENCE_RANGE_MAX_BUCKETS
        )
    ]
    near_target_candidates = [
        candidate
        for candidate in allowed_candidates
        if candidate["probability"] >= CONFIDENCE_RANGE_NEAR_TARGET
    ]
    pool = near_target_candidates or allowed_candidates or candidates

    return min(
        pool,
        key=lambda item: item["score"],
    )


def compact_coverage_range(candidate):
    if not candidate:
        return None

    return {
        "label": candidate["label"],
        "short": candidate["short"],
        "minimum": candidate["minimum"],
        "maximum": candidate["maximum"],
        "probability": candidate["probability"],
        "baseline_probability": candidate["baseline_probability"],
        "range_type": "confidence_80",
        "confidence": candidate["confidence"],
        "clear_signal": candidate["clear_signal"],
        "clear_reason": candidate["clear_reason"],
        "edge": candidate["edge"],
        "target_confidence": candidate.get("target_confidence"),
        "cashout_target": candidate.get("cashout_target"),
        "coverage_gap": candidate.get("coverage_gap"),
        "bucket_count": candidate.get("bucket_count"),
    }


def compact_model_candidate(candidate):
    if not candidate:
        return None

    keys = (
        "candidate_model",
        "label",
        "short",
        "minimum",
        "maximum",
        "probability",
        "baseline_probability",
        "confidence",
        "source",
        "range_type",
        "clear_signal",
        "clear_reason",
        "edge",
        "target_confidence",
        "cashout_target",
        "coverage_gap",
        "bucket_count",
        "sample_size",
    )
    return {
        key: candidate.get(key)
        for key in keys
    }


def range_estimate_from_bucket_probabilities(
    bucket_probabilities,
    range_sample,
    fallback_values,
    evidence_score,
    source,
    candidate_model,
):
    best_bucket = max(
        bucket_probabilities,
        key=lambda item: item["probability"],
    )
    ordered_buckets = sorted(
        bucket_probabilities,
        key=lambda item: item["probability"],
        reverse=True,
    )
    runner_up_bucket = ordered_buckets[1] if len(ordered_buckets) > 1 else None
    sorted_sample = sorted(range_sample or fallback_values)
    clear_signal, confidence, clear_reason, edge = range_quality(
        best_bucket,
        runner_up_bucket,
        evidence_score,
    )
    adaptive = adaptive_range_estimate(
        bucket_probabilities,
        evidence_score,
    )
    confidence_range = confidence_range_estimate(
        bucket_probabilities,
        evidence_score,
        sorted_sample,
    )

    if not clear_signal and adaptive["probability"] >= best_bucket["probability"] + 0.15:
        best_bucket = adaptive
        clear_signal = adaptive["clear_signal"]
        confidence = adaptive["confidence"]
        clear_reason = adaptive["clear_reason"]
        edge = adaptive["edge"]

    coverage_range = compact_coverage_range(
        confidence_range
    )

    if confidence_range and confidence_range["clear_signal"]:
        best_bucket = confidence_range
        clear_signal = confidence_range["clear_signal"]
        confidence = confidence_range["confidence"]
        clear_reason = confidence_range["clear_reason"]
        edge = confidence_range["edge"]

    return {
        "label": best_bucket["label"],
        "short": best_bucket["short"],
        "minimum": best_bucket["minimum"],
        "maximum": best_bucket["maximum"],
        "probability": best_bucket["probability"],
        "baseline_probability": best_bucket["baseline_probability"],
        "low": percentile(sorted_sample, 0.25),
        "median": percentile(sorted_sample, 0.50),
        "high": percentile(sorted_sample, 0.75),
        "sample_size": len(range_sample or fallback_values),
        "source": source,
        "candidate_model": candidate_model,
        "range_type": (
            "confidence_80"
            if best_bucket.get("target_confidence")
            else "adaptive"
            if "bucket_labels" in best_bucket
            else "narrow"
        ),
        "confidence": confidence,
        "clear_signal": clear_signal,
        "clear_reason": clear_reason,
        "edge": edge,
        "target_confidence": best_bucket.get("target_confidence"),
        "cashout_target": best_bucket.get("cashout_target"),
        "coverage_gap": best_bucket.get("coverage_gap"),
        "bucket_count": best_bucket.get("bucket_count"),
        "coverage_range": coverage_range,
        "runner_up_label": runner_up_bucket["label"] if runner_up_bucket else "",
        "runner_up_probability": (
            runner_up_bucket["probability"]
            if runner_up_bucket
            else 0
        ),
        "buckets": bucket_probabilities,
    }


def range_candidate_evidence(bucket_probabilities, sample_size, source):
    best_probability = max(
        (
            item["probability"]
            for item in bucket_probabilities
        ),
        default=0,
    )

    if source == "pattern":
        return (
            0.70 * clamp(sample_size / 40, 0, 1)
            + 0.30 * best_probability
        )

    if source == "multi_pattern":
        return (
            0.68 * clamp(sample_size / MULTI_PATTERN_MATCH_LIMIT, 0, 1)
            + 0.32 * best_probability
        )

    if source == "recent":
        return (
            0.65 * clamp(sample_size / RECENT_WINDOW, 0, 1)
            + 0.35 * best_probability
        )

    if source == "baseline":
        return (
            0.45 * clamp(sample_size / 500, 0, 1)
            + 0.15 * best_probability
        )

    return (
        0.55 * clamp(sample_size / 25, 0, 1)
        + 0.25
        + 0.20 * best_probability
    )


def next_range_estimate(
    values,
    pattern_sample,
    recent_sample,
    multi_pattern_sample=None,
):
    if not values:
        return None

    multi_pattern_sample = multi_pattern_sample or []

    if len(pattern_sample) >= 8:
        range_sample = pattern_sample
        source = "pattern"
    elif len(multi_pattern_sample) >= 18:
        range_sample = multi_pattern_sample
        source = "multi_pattern"
    else:
        range_sample = recent_sample
        source = "recent"

    bucket_probabilities = weighted_bucket_probabilities(
        values,
        pattern_sample,
        recent_sample,
        multi_pattern_sample,
    )
    confidence_score = range_candidate_evidence(
        bucket_probabilities,
        len(range_sample),
        source,
    )
    estimate = range_estimate_from_bucket_probabilities(
        bucket_probabilities,
        range_sample,
        values,
        confidence_score,
        source,
        "ensemble",
    )
    candidate_specs = [
        (
            "baseline",
            model_bucket_probabilities(
                values,
                values,
                0,
                "baseline",
            ),
            values,
            "baseline",
        ),
        (
            "recent",
            model_bucket_probabilities(
                values,
                recent_sample,
                12,
                "recent",
            ),
            recent_sample,
            "recent",
        ),
        (
            "pattern",
            model_bucket_probabilities(
                values,
                pattern_sample,
                8,
                "pattern",
            ),
            pattern_sample,
            "pattern",
        ),
        (
            "multi_pattern",
            model_bucket_probabilities(
                values,
                multi_pattern_sample,
                10,
                "multi_pattern",
            ),
            multi_pattern_sample,
            "multi_pattern",
        ),
        (
            "ensemble",
            bucket_probabilities,
            range_sample,
            source,
        ),
    ]
    model_candidates = []

    for candidate_model, probabilities, sample, candidate_source in candidate_specs:
        if not probabilities:
            continue

        candidate_evidence = range_candidate_evidence(
            probabilities,
            len(sample),
            candidate_source,
        )
        candidate = range_estimate_from_bucket_probabilities(
            probabilities,
            sample,
            values,
            candidate_evidence,
            candidate_source,
            candidate_model,
        )
        model_candidates.append(
            compact_model_candidate(candidate)
        )

    estimate["model_candidates"] = model_candidates
    return estimate


def summarize(values):
    sorted_values = sorted(values)
    bucket_counts = Counter(bucket_multiplier(value) for value in values)

    return {
        "rounds": len(values),
        "latest_multiplier": values[-1] if values else None,
        "average": sum(values) / len(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "p25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.50),
        "p75": percentile(sorted_values, 0.75),
        "p90": percentile(sorted_values, 0.90),
        "buckets": dict(bucket_counts),
    }


def target_probabilities(values, targets):
    probabilities = {}

    for target in targets:
        hits = sum(1 for value in values if value >= target)
        probabilities[f">={target:.2f}x"] = {
            "hits": hits,
            "total": len(values),
            "probability": hits / len(values) if values else 0,
        }

    return probabilities


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def probability(values, target):
    if not values:
        return 0

    return sum(1 for value in values if value >= target) / len(values)


def smoothed_probability(values, target, prior, prior_weight):
    if not values:
        return prior

    hits = sum(1 for value in values if value >= target)
    return (hits + prior * prior_weight) / (len(values) + prior_weight)


def pattern_matches(values, lookback):
    if len(values) <= lookback:
        return {
            "pattern": [],
            "matches": [],
        }

    buckets = [bucket_multiplier(value) for value in values]
    latest_pattern = buckets[-lookback:]
    matches = []

    # The current ending pattern has no known next result, so it is not included
    # in this loop.
    for end_index in range(lookback, len(values)):
        pattern = buckets[end_index - lookback:end_index]

        if pattern != latest_pattern:
            continue

        if end_index >= len(values):
            continue

        matches.append(values[end_index])

    return {
        "pattern": latest_pattern,
        "matches": matches,
    }


def multi_lookback_matches(values, active_lookback):
    lookbacks = sorted(
        {
            lookback
            for lookback in (
                [active_lookback]
                + MULTI_PATTERN_LOOKBACKS
                + [active_lookback + 1]
            )
            if lookback >= 1
        }
    )
    matches = []
    details = []

    for lookback in lookbacks:
        match_data = pattern_matches(values, lookback)
        lookback_matches = match_data["matches"][-MULTI_PATTERN_MATCH_LIMIT:]

        if lookback_matches:
            matches.extend(lookback_matches)

        details.append(
            {
                "lookback": lookback,
                "pattern": match_data["pattern"],
                "matches": len(match_data["matches"]),
            }
        )

    return {
        "lookbacks": lookbacks,
        "details": details,
        "matches": matches[-MULTI_PATTERN_MATCH_LIMIT:],
    }


def current_below_streak(values, target):
    streak = 0

    for value in reversed(values):
        if value >= target:
            break

        streak += 1

    return streak


def streak_matches(values, target):
    if len(values) < 3:
        return {
            "streak": 0,
            "matches": [],
        }

    current_streak = current_below_streak(values, target)
    capped_streak = min(current_streak, 8)
    matches = []
    streak_before_round = 1 if values[0] < target else 0

    for round_index in range(1, len(values)):
        historical_streak = min(streak_before_round, 8)

        if historical_streak == capped_streak:
            matches.append(values[round_index])

        if values[round_index] >= target:
            streak_before_round = 0
        else:
            streak_before_round += 1

    return {
        "streak": current_streak,
        "matches": matches[:-1] if matches else [],
    }


def confidence_label(score):
    if score >= 0.72:
        return "high"
    if score >= 0.42:
        return "medium"
    return "low"


def signal_label(probability_value, baseline_probability, confidence, decision_margin=0):
    edge = probability_value - baseline_probability

    if confidence == "low":
        return "WAIT"
    if edge >= decision_margin + 0.08:
        return "FAVOR"
    if edge <= decision_margin - 0.08:
        return "AVOID"
    return "NEUTRAL"


def min_high_probability(target):
    return MIN_HIGH_PROBABILITY_BY_TARGET.get(
        float(target),
        0.5,
    )


def predicted_high_label(probability_value, baseline_probability, decision_margin, target):
    threshold = max(
        baseline_probability + max(decision_margin, 0),
        min_high_probability(target),
    )

    return probability_value >= threshold


def clear_signal_label(
    edge,
    confidence,
    signal,
    evidence_score,
    evidence,
    probability_value,
    baseline_probability,
    predicted_high,
    target,
    decision_margin,
):
    if confidence == "low" or signal in ("WAIT", "NEUTRAL"):
        return False, "weak signal"

    if evidence_score < 0.42:
        return False, "not enough evidence"

    if (
        evidence["pattern_matches"] < 8
        and evidence.get("multi_pattern_matches", 0) < 24
        and evidence["streak_matches"] < 20
        and evidence["recent_rounds"] < 60
    ):
        return False, "small sample"

    if abs(edge) < 0.03:
        return False, "edge too small"

    if predicted_high and probability_value < min_high_probability(target):
        return False, "high call probability too low"

    if predicted_high and probability_value < baseline_probability + max(decision_margin, 0):
        return False, "not above target threshold"

    return True, "clear signal"


def target_key(target):
    return f"{target:.2f}"


def calibrated_probability(raw_probability, baseline, calibration):
    checked = int(calibration.get("checked", 0))
    recent_accuracy = calibration.get("recent_accuracy")
    recent_balanced_accuracy = calibration.get("recent_balanced_accuracy")
    profile_skill = calibration.get("profile_skill")
    quality = (
        recent_balanced_accuracy
        if recent_balanced_accuracy is not None
        else recent_accuracy
    )

    if checked < 12 or quality is None:
        edge_weight = 0.65
    elif quality < 0.42:
        edge_weight = 0.25
    elif quality < 0.48:
        edge_weight = 0.35
    elif quality < 0.54:
        edge_weight = 0.60
    elif quality > 0.62:
        edge_weight = 1.10
    else:
        edge_weight = 0.85

    if (
        checked >= 50
        and profile_skill is not None
        and profile_skill < 0.02
    ):
        edge_weight = min(
            edge_weight,
            0.10 if profile_skill < 0 else 0.20,
        )

    adjusted = baseline + (raw_probability - baseline) * edge_weight
    return clamp(adjusted, 0, 1), edge_weight


def blend_components(components, profile_name):
    weights = PROFILE_WEIGHTS.get(
        profile_name,
        PROFILE_WEIGHTS["balanced"],
    )
    total = sum(weights.values())

    if total <= 0:
        return components["overall"]

    baseline = components.get("overall", 0)
    return sum(
        components.get(name, baseline) * weight
        for name, weight in weights.items()
    ) / total


def next_round_prediction(values, lookback, targets, calibration=None):
    calibration = calibration or {}
    match_data = pattern_matches(values, lookback)
    pattern_sample = match_data["matches"]
    multi_match_data = multi_lookback_matches(values, lookback)
    multi_pattern_sample = multi_match_data["matches"]
    recent_sample = values[-RECENT_WINDOW:]

    predictions = []

    for target in targets:
        baseline = probability(values, target)
        target_calibration = calibration.get(target_key(target), {})
        profile_name = target_calibration.get(
            "profile",
            "balanced",
        )
        decision_margin = float(
            target_calibration.get(
                "decision_margin",
                0,
            )
        )
        recent = smoothed_probability(
            recent_sample,
            target,
            baseline,
            prior_weight=12,
        )
        pattern = smoothed_probability(
            pattern_sample,
            target,
            baseline,
            prior_weight=8,
        )
        multi_pattern = smoothed_probability(
            multi_pattern_sample,
            target,
            baseline,
            prior_weight=10,
        )
        streak_data = streak_matches(values, target)
        streak_sample = streak_data["matches"]
        streak = smoothed_probability(
            streak_sample,
            target,
            baseline,
            prior_weight=10,
        )

        components = {
            "overall": baseline,
            "recent": recent,
            "pattern": pattern,
            "multi_pattern": multi_pattern,
            "streak": streak,
        }
        raw_ensemble = blend_components(
            components,
            profile_name,
        )
        ensemble, calibration_weight = calibrated_probability(
            raw_ensemble,
            baseline,
            target_calibration,
        )

        evidence_score = (
            0.30 * clamp(len(pattern_sample) / 25, 0, 1)
            + 0.18 * clamp(len(multi_pattern_sample) / 80, 0, 1)
            + 0.30 * clamp(len(streak_sample) / 35, 0, 1)
            + 0.22 * clamp(len(values) / 500, 0, 1)
        )
        confidence = confidence_label(evidence_score)
        signal = signal_label(
            ensemble,
            baseline,
            confidence,
            decision_margin=decision_margin,
        )
        predicted_high = predicted_high_label(
            ensemble,
            baseline,
            decision_margin,
            target,
        )
        if (
            (predicted_high and signal == "AVOID")
            or (not predicted_high and signal == "FAVOR")
        ):
            signal = "NEUTRAL"
        evidence = {
            "recent_rounds": len(recent_sample),
            "pattern_matches": len(pattern_sample),
            "multi_pattern_matches": len(multi_pattern_sample),
            "multi_pattern_lookbacks": multi_match_data["details"],
            "below_target_streak": streak_data["streak"],
            "streak_matches": len(streak_sample),
        }
        clear_signal, clear_reason = clear_signal_label(
            ensemble - baseline,
            confidence,
            signal,
            evidence_score,
            evidence,
            ensemble,
            baseline,
            predicted_high,
            target,
            decision_margin,
        )
        recent_balanced_accuracy = target_calibration.get(
            "recent_balanced_accuracy"
        )
        profile_skill = target_calibration.get("profile_skill")

        if (
            recent_balanced_accuracy is not None
            and int(target_calibration.get("checked", 0)) >= 20
            and recent_balanced_accuracy < 0.5
        ):
            clear_signal = False
            clear_reason = "recently unreliable"

        if (
            profile_skill is not None
            and int(target_calibration.get("checked", 0)) >= 50
            and profile_skill < 0.02
        ):
            signal = "NEUTRAL"
            clear_signal = False
            clear_reason = "no edge over baseline"

        predictions.append(
            {
                "target": target,
                "probability": ensemble,
                "raw_probability": raw_ensemble,
                "baseline_probability": baseline,
                "edge": ensemble - baseline,
                "decision_margin": decision_margin,
                "predicted_high": predicted_high,
                "confidence": confidence,
                "signal": signal,
                "clear_signal": clear_signal,
                "clear_reason": clear_reason,
                "sample_size": len(values),
                "source": "ensemble",
                "profile": profile_name,
                "components": components,
                "calibration": {
                    "edge_weight": calibration_weight,
                    **target_calibration,
                },
                "evidence": evidence,
            }
        )

    return {
        "lookback": lookback,
        "latest_pattern": match_data["pattern"],
        "pattern_match_count": len(pattern_sample),
        "range_estimate": next_range_estimate(
            values,
            pattern_sample,
            recent_sample,
            multi_pattern_sample,
        ),
        "predictions": predictions,
    }


def ensemble_prediction_for_target(values, lookback, target):
    return next_round_prediction(
        values,
        lookback,
        [target],
    )["predictions"][0]


def backtest(values, lookback, target, min_matches):
    if len(values) <= lookback + 1:
        return {
            "target": target,
            "lookback": lookback,
            "tested_rounds": 0,
            "accuracy": None,
            "baseline_accuracy": None,
            "skill": None,
            "coverage": 0,
        }

    baseline_probability = sum(1 for value in values if value >= target) / len(values)
    correct = 0
    tested = 0
    high_predictions = 0
    actual_high_count = 0
    start_index = max(lookback + 1, len(values) - 500)

    for round_index in range(start_index, len(values)):
        training_values = values[:round_index]
        prediction = ensemble_prediction_for_target(
            training_values,
            lookback,
            target,
        )
        evidence = prediction["evidence"]

        if (
            evidence["pattern_matches"] < min_matches
            and evidence["streak_matches"] < min_matches
        ):
            continue

        predicted_high = prediction["predicted_high"]
        actual_high = values[round_index] >= target
        actual_high_count += int(actual_high)

        if predicted_high:
            high_predictions += 1

        if predicted_high == actual_high:
            correct += 1

        tested += 1

    if tested:
        baseline_correct = max(
            actual_high_count,
            tested - actual_high_count,
        )
        baseline_accuracy = baseline_correct / tested
        accuracy = correct / tested
    else:
        baseline_accuracy = None
        accuracy = None

    return {
        "target": target,
        "lookback": lookback,
        "min_matches": min_matches,
        "method": "ensemble",
        "tested_rounds": tested,
        "coverage": tested / max(len(values) - start_index, 1),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "skill": (
            accuracy - baseline_accuracy
            if accuracy is not None and baseline_accuracy is not None
            else None
        ),
        "baseline_probability": baseline_probability,
        "high_predictions": high_predictions,
    }


def build_report(rounds, lookback, targets, min_matches, calibration=None, data_selection=None):
    values = [round_data["multiplier"] for round_data in rounds]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv_rounds": len(rounds),
        "data_selection": data_selection or {},
        "summary": summarize(values),
        "overall_probabilities": target_probabilities(values, targets),
        "next_round": next_round_prediction(
            values,
            lookback,
            targets,
            calibration=calibration,
        ),
        "backtests": [
            backtest(values, lookback, target, min_matches)
            for target in targets
        ],
        "warning": (
            "Crash-game multipliers are normally random. This report estimates "
            "historical frequencies and pattern-conditioned probabilities; it "
            "does not guarantee future results."
        ),
    }


def format_probability(value):
    return f"{value * 100:.1f}%"


def print_report(report):
    summary = report["summary"]

    print("Aviator multiplier analysis")
    print("=" * 28)
    print(report["warning"])
    print()
    data_selection = report.get("data_selection") or {}

    if data_selection:
        if data_selection.get("using_trusted_sources"):
            data_mode = (
                "trusted "
                f"({data_selection.get('trusted_rounds', 0)} game+legacy / "
                f"{data_selection.get('total_rounds', summary['rounds'])} total; "
                f"excluded {data_selection.get('excluded_rounds', 0)})"
            )
        else:
            data_mode = (
                f"{data_selection.get('mode', 'all')} "
                f"({data_selection.get('source_rounds', 0)} game / "
                f"{data_selection.get('total_rounds', summary['rounds'])} total; "
                f"switch at {data_selection.get('minimum_source_rounds', 0)} game)"
            )

        print(
            f"Data mode: {data_mode}"
        )

    print(f"Rounds analyzed: {summary['rounds']}")
    print(f"Latest multiplier: {summary['latest_multiplier']:.2f}x")
    print(f"Average: {summary['average']:.2f}x")
    print(f"Median: {summary['median']:.2f}x")
    print(f"P90: {summary['p90']:.2f}x")
    print(f"Max: {summary['maximum']:.2f}x")
    print()

    print("Overall probability")
    for label, item in report["overall_probabilities"].items():
        print(
            f"  {label}: {format_probability(item['probability'])} "
            f"({item['hits']}/{item['total']})"
        )

    next_round = report["next_round"]
    pattern = " -> ".join(next_round["latest_pattern"]) or "not enough data"
    print()
    print(f"Latest pattern ({next_round['lookback']} rounds): {pattern}")
    print(f"Historical pattern matches: {next_round['pattern_match_count']}")
    print()
    print("Next-round probability estimate")

    for prediction in next_round["predictions"]:
        print(
            f"  >={prediction['target']:.2f}x: "
            f"{format_probability(prediction['probability'])} "
            f"{prediction['signal']} "
            f"confidence={prediction['confidence']} "
            f"edge={format_probability(prediction['edge'])}"
        )

    print()
    print("Backtest accuracy")

    for result in report["backtests"]:
        accuracy = (
            format_probability(result["accuracy"])
            if result["accuracy"] is not None
            else "not enough data"
        )
        baseline_accuracy = (
            format_probability(result["baseline_accuracy"])
            if result.get("baseline_accuracy") is not None
            else "n/a"
        )
        skill = (
            format_probability(result["skill"])
            if result.get("skill") is not None
            else "n/a"
        )

        print(
            f"  >={result['target']:.2f}x: {accuracy}, "
            f"coverage {format_probability(result['coverage'])}, "
            f"majority baseline {baseline_accuracy}, "
            f"skill {skill}, "
            f"target frequency {format_probability(result['baseline_probability'])}"
        )


def parse_targets(raw_targets):
    if not raw_targets:
        return DEFAULT_TARGETS

    targets = []

    for raw_target in raw_targets.split(","):
        raw_target = raw_target.strip()

        if not raw_target:
            continue

        targets.append(float(raw_target))

    return targets or DEFAULT_TARGETS


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stored Aviator multipliers and estimate next-round probabilities."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Path to rounds CSV file.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=2,
        help="Number of latest rounds to use as the pattern.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(str(target) for target in DEFAULT_TARGETS),
        help="Comma-separated multiplier targets, for example: 1.5,2,3,5,10",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=5,
        help="Minimum historical pattern matches required during backtesting.",
    )
    parser.add_argument(
        "--json-out",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the JSON analysis report.",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    json_path = Path(args.json_out)
    targets = parse_targets(args.targets)

    if args.lookback < 1:
        raise SystemExit("--lookback must be at least 1")

    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    all_rounds = load_rounds(csv_path)
    rounds, data_selection = select_prediction_rounds(
        all_rounds
    )

    if not all_rounds:
        raise SystemExit("No valid rounds found in CSV.")

    report = build_report(
        rounds=rounds,
        lookback=args.lookback,
        targets=targets,
        min_matches=args.min_matches,
        data_selection=data_selection,
    )

    json_path.parent.mkdir(exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print()
    print(f"JSON report written to: {json_path}")


if __name__ == "__main__":
    main()
