import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "rounds.csv"
DEFAULT_REPORT_PATH = DATA_DIR / "analysis.json"


DEFAULT_TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0]
RECENT_WINDOW = 80
DEDUPLICATE_WINDOW_SECONDS = 6
MIN_REAL_ROUNDS_FOR_PREDICTION = 300
RANGE_BUCKETS = [
    {
        "label": "LESS THAN 1.20x",
        "short": "<1.20x",
        "minimum": 1.0,
        "maximum": 1.2,
    },
    {
        "label": "1.20x TO 1.50x",
        "short": "1.20x-1.50x",
        "minimum": 1.2,
        "maximum": 1.5,
    },
    {
        "label": "1.50x TO 2.00x",
        "short": "1.50x-2.00x",
        "minimum": 1.5,
        "maximum": 2.0,
    },
    {
        "label": "2.00x TO 3.00x",
        "short": "2.00x-3.00x",
        "minimum": 2.0,
        "maximum": 3.0,
    },
    {
        "label": "3.00x TO 5.00x",
        "short": "3.00x-5.00x",
        "minimum": 3.0,
        "maximum": 5.0,
    },
    {
        "label": "5.00x TO 10.00x",
        "short": "5.00x-10.00x",
        "minimum": 5.0,
        "maximum": 10.0,
    },
    {
        "label": "10.00x TO 25.00x",
        "short": "10.00x-25.00x",
        "minimum": 10.0,
        "maximum": 25.0,
    },
    {
        "label": "25.00x TO 50.00x",
        "short": "25.00x-50.00x",
        "minimum": 25.0,
        "maximum": 50.0,
    },
    {
        "label": "50.00x TO 100.00x",
        "short": "50.00x-100.00x",
        "minimum": 50.0,
        "maximum": 100.0,
    },
    {
        "label": "MORE THAN 100.00x",
        "short": "100.00x+",
        "minimum": 100.0,
        "maximum": None,
    },
]
ADAPTIVE_RANGE_CANDIDATES = [
    {
        "label": "1.00x TO 1.50x",
        "short": "1.00x-1.50x",
        "minimum": 1.0,
        "maximum": 1.5,
        "bucket_labels": [
            "LESS THAN 1.20x",
            "1.20x TO 1.50x",
        ],
    },
    {
        "label": "1.00x TO 2.00x",
        "short": "1.00x-2.00x",
        "minimum": 1.0,
        "maximum": 2.0,
        "bucket_labels": [
            "LESS THAN 1.20x",
            "1.20x TO 1.50x",
            "1.50x TO 2.00x",
        ],
    },
    {
        "label": "1.20x TO 2.00x",
        "short": "1.20x-2.00x",
        "minimum": 1.2,
        "maximum": 2.0,
        "bucket_labels": [
            "1.20x TO 1.50x",
            "1.50x TO 2.00x",
        ],
    },
    {
        "label": "1.20x TO 3.00x",
        "short": "1.20x-3.00x",
        "minimum": 1.2,
        "maximum": 3.0,
        "bucket_labels": [
            "1.20x TO 1.50x",
            "1.50x TO 2.00x",
            "2.00x TO 3.00x",
        ],
    },
    {
        "label": "1.50x TO 3.00x",
        "short": "1.50x-3.00x",
        "minimum": 1.5,
        "maximum": 3.0,
        "bucket_labels": [
            "1.50x TO 2.00x",
            "2.00x TO 3.00x",
        ],
    },
    {
        "label": "1.50x TO 5.00x",
        "short": "1.50x-5.00x",
        "minimum": 1.5,
        "maximum": 5.0,
        "bucket_labels": [
            "1.50x TO 2.00x",
            "2.00x TO 3.00x",
            "3.00x TO 5.00x",
        ],
    },
    {
        "label": "2.00x TO 5.00x",
        "short": "2.00x-5.00x",
        "minimum": 2.0,
        "maximum": 5.0,
        "bucket_labels": [
            "2.00x TO 3.00x",
            "3.00x TO 5.00x",
        ],
    },
    {
        "label": "2.00x TO 10.00x",
        "short": "2.00x-10.00x",
        "minimum": 2.0,
        "maximum": 10.0,
        "bucket_labels": [
            "2.00x TO 3.00x",
            "3.00x TO 5.00x",
            "5.00x TO 10.00x",
        ],
    },
    {
        "label": "3.00x TO 10.00x",
        "short": "3.00x-10.00x",
        "minimum": 3.0,
        "maximum": 10.0,
        "bucket_labels": [
            "3.00x TO 5.00x",
            "5.00x TO 10.00x",
        ],
    },
    {
        "label": "5.00x TO 25.00x",
        "short": "5.00x-25.00x",
        "minimum": 5.0,
        "maximum": 25.0,
        "bucket_labels": [
            "5.00x TO 10.00x",
            "10.00x TO 25.00x",
        ],
    },
    {
        "label": "MORE THAN 10.00x",
        "short": "10.00x+",
        "minimum": 10.0,
        "maximum": None,
        "bucket_labels": [
            "10.00x TO 25.00x",
            "25.00x TO 50.00x",
            "50.00x TO 100.00x",
            "MORE THAN 100.00x",
        ],
    },
]
PROFILE_WEIGHTS = {
    "balanced": {
        "overall": 0.35,
        "recent": 0.25,
        "pattern": 0.25,
        "streak": 0.15,
    },
    "defensive": {
        "overall": 0.65,
        "recent": 0.15,
        "pattern": 0.10,
        "streak": 0.10,
    },
    "recent_heavy": {
        "overall": 0.30,
        "recent": 0.50,
        "pattern": 0.10,
        "streak": 0.10,
    },
    "pattern_heavy": {
        "overall": 0.30,
        "recent": 0.15,
        "pattern": 0.45,
        "streak": 0.10,
    },
    "streak_heavy": {
        "overall": 0.30,
        "recent": 0.15,
        "pattern": 0.10,
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


def source_counts(rounds):
    counts = Counter()

    for round_data in rounds:
        source = round_data.get("source") or "unlabeled"
        counts[source] += 1

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
        if round_data.get("source") == preferred_source
    ]

    if len(source_rounds) >= minimum:
        return source_rounds, {
            "mode": preferred_source,
            "preferred_source": preferred_source,
            "minimum_source_rounds": minimum,
            "using_source_only": True,
            "source_rounds": len(source_rounds),
            "total_rounds": len(rounds),
            "counts": source_counts(rounds),
        }

    return rounds, {
        "mode": "all",
        "preferred_source": preferred_source,
        "minimum_source_rounds": minimum,
        "using_source_only": False,
        "source_rounds": len(source_rounds),
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


def weighted_bucket_probabilities(values, pattern_sample, recent_sample):
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
    probabilities = []

    for bucket in RANGE_BUCKETS:
        label = bucket["label"]
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

    if (
        probability_value >= 0.62
        and edge >= 0.025
        and evidence_score >= 0.38
        and bucket_count <= 3
    ):
        return True, "high", "clear wider range", edge

    if (
        probability_value >= 0.54
        and edge >= 0.025
        and evidence_score >= 0.35
        and bucket_count <= 3
    ):
        return True, "medium", "clear wider range", edge

    if probability_value < 0.50:
        return False, "low", "wider range probability too low", edge

    if edge < 0.02:
        return False, "low", "wider range has no edge", edge

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
        score = (
            probability_value
            + max(edge, 0) * 1.5
            - max(bucket_count - 2, 0) * 0.035
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


def next_range_estimate(values, pattern_sample, recent_sample):
    if not values:
        return None

    range_sample = pattern_sample if len(pattern_sample) >= 8 else recent_sample
    sorted_sample = sorted(range_sample or values)
    bucket_probabilities = weighted_bucket_probabilities(
        values,
        pattern_sample,
        recent_sample,
    )
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
    confidence_score = (
        0.55 * clamp(len(pattern_sample) / 25, 0, 1)
        + 0.25 * clamp(len(recent_sample) / RECENT_WINDOW, 0, 1)
        + 0.20 * best_bucket["probability"]
    )
    clear_signal, confidence, clear_reason, edge = range_quality(
        best_bucket,
        runner_up_bucket,
        confidence_score,
    )
    adaptive = adaptive_range_estimate(
        bucket_probabilities,
        confidence_score,
    )

    if not clear_signal and adaptive["probability"] >= best_bucket["probability"] + 0.15:
        best_bucket = adaptive
        clear_signal = adaptive["clear_signal"]
        confidence = adaptive["confidence"]
        clear_reason = adaptive["clear_reason"]
        edge = adaptive["edge"]

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
        "sample_size": len(range_sample or values),
        "source": "pattern" if len(pattern_sample) >= 8 else "recent",
        "range_type": "adaptive" if "bucket_labels" in best_bucket else "narrow",
        "confidence": confidence,
        "clear_signal": clear_signal,
        "clear_reason": clear_reason,
        "edge": edge,
        "runner_up_label": runner_up_bucket["label"] if runner_up_bucket else "",
        "runner_up_probability": (
            runner_up_bucket["probability"]
            if runner_up_bucket
            else 0
        ),
        "buckets": bucket_probabilities,
    }


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

    return sum(
        components[name] * weight
        for name, weight in weights.items()
    ) / total


def next_round_prediction(values, lookback, targets, calibration=None):
    calibration = calibration or {}
    match_data = pattern_matches(values, lookback)
    pattern_sample = match_data["matches"]
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
            0.45 * clamp(len(pattern_sample) / 25, 0, 1)
            + 0.30 * clamp(len(streak_sample) / 35, 0, 1)
            + 0.25 * clamp(len(values) / 500, 0, 1)
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
        evidence = {
            "recent_rounds": len(recent_sample),
            "pattern_matches": len(pattern_sample),
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

        if (
            recent_balanced_accuracy is not None
            and int(target_calibration.get("checked", 0)) >= 20
            and recent_balanced_accuracy < 0.5
        ):
            clear_signal = False
            clear_reason = "recently unreliable"

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
            "coverage": 0,
        }

    baseline_probability = sum(1 for value in values if value >= target) / len(values)
    correct = 0
    tested = 0
    high_predictions = 0
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

        if predicted_high:
            high_predictions += 1

        if predicted_high == actual_high:
            correct += 1

        tested += 1

    return {
        "target": target,
        "lookback": lookback,
        "min_matches": min_matches,
        "method": "ensemble",
        "tested_rounds": tested,
        "coverage": tested / max(len(values) - start_index, 1),
        "accuracy": correct / tested if tested else None,
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
        print(
            "Data mode: "
            f"{data_selection.get('mode', 'all')} "
            f"({data_selection.get('source_rounds', 0)} real / "
            f"{data_selection.get('total_rounds', summary['rounds'])} total; "
            f"switch at {data_selection.get('minimum_source_rounds', 0)} real)"
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

        print(
            f"  >={result['target']:.2f}x: {accuracy}, "
            f"coverage {format_probability(result['coverage'])}, "
            f"baseline {format_probability(result['baseline_probability'])}"
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
