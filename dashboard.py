import argparse
import copy
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from aviator_analyzer import (
    DEFAULT_TARGETS,
    PROFILE_WEIGHTS,
    backtest,
    load_rounds,
    next_round_prediction,
    select_prediction_rounds,
    summarize,
    target_probabilities,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "rounds.csv"
PREDICTION_STATE_PATH = DATA_DIR / "prediction_state.json"
PREDICTION_HISTORY_PATH = DATA_DIR / "prediction_history.csv"
RANGE_PREDICTION_HISTORY_PATH = DATA_DIR / "range_prediction_history.csv"
ROUND_CONTEXT_PATH = DATA_DIR / "round_context.csv"
RANGE_MODEL_VERSION = "adaptive-v3"
DASHBOARD_DIR = ROOT / "dashboard"
TRACKED_TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0]
TRACKING_LOCK = threading.Lock()
BACKTEST_LOCK = threading.Lock()
DASHBOARD_CACHE = {}
CACHE_MAX_ITEMS = 12
ACCURACY_SUMMARY_CACHE = {
    "signature": None,
    "summary": None,
}
BACKTEST_REFRESH_SECONDS = 20
PARTICIPANT_CONTEXT_LIVE_SECONDS = 5
BACKTEST_CACHE = {
    "key": None,
    "generated_at": 0,
    "in_progress": False,
    "items": [],
}
WARM_CACHE_QUERIES = [
    {
        "lookback": ["2"],
        "min_matches": ["5"],
    },
]
WARM_CACHE_POLL_SECONDS = 0.1


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def parse_int(value, default, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(maximum, parsed))


def round_float(value):
    if value is None:
        return None

    return round(value, 4)


def parse_round_time(value):
    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def ingest_status(rounds):
    if not rounds:
        return {
            "last_round_timestamp": None,
            "last_round_age_seconds": None,
            "is_stale": True,
        }

    last_timestamp = rounds[-1].get("timestamp", "")
    parsed = parse_round_time(last_timestamp)

    if parsed is None:
        return {
            "last_round_timestamp": last_timestamp,
            "last_round_age_seconds": None,
            "is_stale": False,
        }

    age_seconds = max(0, int((datetime.now() - parsed).total_seconds()))

    return {
        "last_round_timestamp": last_timestamp,
        "last_round_age_seconds": age_seconds,
        "is_stale": age_seconds > 180,
    }


def refresh_cached_ingest(payload):
    payload = copy.deepcopy(payload)
    recent_rounds = payload.get("recent_rounds", [])

    if recent_rounds:
        last_timestamp = recent_rounds[0].get("timestamp", "")
        parsed = parse_round_time(last_timestamp)
        age_seconds = (
            max(0, int((datetime.now() - parsed).total_seconds()))
            if parsed
            else None
        )
        payload["ingest"] = {
            "last_round_timestamp": last_timestamp,
            "last_round_age_seconds": age_seconds,
            "is_stale": bool(age_seconds is not None and age_seconds > 180),
        }

    if payload.get("round_context"):
        update_round_context_age(
            payload["round_context"]
        )

        for key in ("radar", "participants"):
            if payload["round_context"].get(key):
                update_round_context_age(
                    payload["round_context"][key]
                )

    payload["generated_at"] = now_string()
    payload["cache_age_ms"] = int(
        (time.monotonic() - payload.get("_cached_at", time.monotonic()))
        * 1000
    )
    payload.pop("_cached_at", None)
    return payload


def csv_signature():
    if not CSV_PATH.exists():
        return None

    stat = CSV_PATH.stat()
    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


def file_signature(path):
    if not path.exists():
        return None

    stat = path.stat()
    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


def parse_context_float(value):
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_context_int(value):
    number = parse_context_float(value)

    if number is None:
        return None

    return int(number)


def update_round_context_age(context):
    if not context or not context.get("observed_at"):
        return context

    parsed = parse_round_time(
        context.get("observed_at")
    )

    if parsed is None:
        context["age_seconds"] = None
        return context

    context["age_seconds"] = max(
        0,
        int(
            (datetime.now() - parsed).total_seconds()
        )
    )

    return context


def context_from_row(row):
    if not row:
        return None

    context = {
        "available": True,
        "path": str(ROUND_CONTEXT_PATH),
        "observed_at": row.get("observed_at", ""),
        "round_id": row.get("round_id", ""),
        "source": row.get("source", ""),
        "game_source": row.get("game_source", ""),
        "player_count": parse_context_int(
            row.get("player_count")
        ),
        "bet_count": parse_context_int(
            row.get("bet_count")
        ),
        "total_bet": parse_context_float(
            row.get("total_bet")
        ),
        "avg_bet": parse_context_float(
            row.get("avg_bet")
        ),
        "max_bet": parse_context_float(
            row.get("max_bet")
        ),
        "cashed_out_count": parse_context_int(
            row.get("cashed_out_count")
        ),
        "avg_cashout": parse_context_float(
            row.get("avg_cashout")
        ),
        "max_cashout": parse_context_float(
            row.get("max_cashout")
        ),
        "total_win": parse_context_float(
            row.get("total_win")
        ),
        "max_win": parse_context_float(
            row.get("max_win")
        ),
        "payload_records": parse_context_int(
            row.get("payload_records")
        ),
    }

    return update_round_context_age(
        context
    )


def latest_round_context():
    empty = {
        "available": False,
        "path": str(ROUND_CONTEXT_PATH),
        "radar": None,
        "participants": None,
    }

    if not ROUND_CONTEXT_PATH.exists():
        return empty

    latest = None
    by_source = {}

    try:
        with ROUND_CONTEXT_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if (
                    row.get("source") == "flight_radar_dom"
                    and not parse_context_int(
                        row.get("player_count")
                    )
                ):
                    continue

                latest = row
                by_source[row.get("source", "")] = row

    except OSError:
        return empty

    if not latest:
        return empty

    context = context_from_row(
        latest
    )
    context["radar"] = context_from_row(
        by_source.get(
            "flight_radar_dom"
        )
    )
    participants = context_from_row(
        by_source.get(
            "participants_dom"
        )
    )
    context["last_participants"] = participants

    if (
        participants
        and participants.get("age_seconds") is not None
        and participants.get("age_seconds") > PARTICIPANT_CONTEXT_LIVE_SECONDS
    ):
        participants = None

    context["participants"] = participants

    return context


def build_live_report(
    rounds,
    lookback,
    targets,
    min_matches,
    calibration=None,
    data_selection=None,
):
    values = [
        round_data["multiplier"]
        for round_data in rounds
    ]
    backtest_key = (
        len(rounds),
        lookback,
        min_matches,
    )

    maybe_refresh_backtests(
        values,
        lookback,
        targets,
        min_matches,
        backtest_key,
    )

    with BACKTEST_LOCK:
        backtests = list(BACKTEST_CACHE["items"])

    return {
        "generated_at": now_string(),
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
        "backtests": backtests,
        "warning": (
            "Crash-game multipliers are normally random. This report estimates "
            "historical frequencies and pattern-conditioned probabilities; it "
            "does not guarantee future results."
        ),
    }


def maybe_refresh_backtests(values, lookback, targets, min_matches, cache_key):
    now = time.monotonic()

    with BACKTEST_LOCK:
        if BACKTEST_CACHE["in_progress"]:
            return

        if (
            BACKTEST_CACHE["key"] == cache_key
            and now - BACKTEST_CACHE["generated_at"] < BACKTEST_REFRESH_SECONDS
        ):
            return

        BACKTEST_CACHE["in_progress"] = True

    values_snapshot = list(values)

    def worker():
        try:
            items = [
                backtest(
                    values_snapshot,
                    lookback,
                    target,
                    min_matches,
                )
                for target in targets
            ]

            with BACKTEST_LOCK:
                BACKTEST_CACHE["key"] = cache_key
                BACKTEST_CACHE["generated_at"] = time.monotonic()
                BACKTEST_CACHE["items"] = items
                BACKTEST_CACHE["in_progress"] = False

        except Exception:
            with BACKTEST_LOCK:
                BACKTEST_CACHE["in_progress"] = False

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


def prediction_target_key(target):
    return f"{target:.2f}"


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_prediction_state():
    if not PREDICTION_STATE_PATH.exists():
        return {
            "pending": None,
            "metrics": {},
            "range_metrics": {},
        }

    try:
        with PREDICTION_STATE_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {
            "pending": None,
            "metrics": {},
            "range_metrics": {},
        }

    state.setdefault("pending", None)
    state.setdefault("metrics", {})
    state.setdefault("range_metrics", {})
    return state


def save_prediction_state(state):
    DATA_DIR.mkdir(exist_ok=True)

    with PREDICTION_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def ensure_prediction_history():
    DATA_DIR.mkdir(exist_ok=True)

    headers = [
        "checked_at",
        "predicted_at",
        "predicted_after_round_count",
        "actual_round_count",
        "actual_timestamp",
        "target",
        "probability",
        "baseline_probability",
        "edge",
        "decision_margin",
        "profile",
        "signal",
        "confidence",
        "clear_signal",
        "clear_reason",
        "predicted_high",
        "actual_multiplier",
        "actual_high",
        "correct",
    ]

    if PREDICTION_HISTORY_PATH.exists():
        with PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if not fieldnames:
                return

            has_decision_margin = "decision_margin" in fieldnames
            has_profile = "profile" in fieldnames
            has_clear_signal = "clear_signal" in fieldnames
            has_clear_reason = "clear_reason" in fieldnames

            if (
                has_decision_margin
                and has_profile
                and has_clear_signal
                and has_clear_reason
            ):
                return

            rows = list(reader)

        if not rows:
            return

        with PREDICTION_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for row in rows:
                if not has_decision_margin:
                    row["decision_margin"] = "0.000000"
                if not has_profile:
                    row["profile"] = "balanced"
                if not has_clear_signal:
                    row["clear_signal"] = "0"
                if not has_clear_reason:
                    row["clear_reason"] = ""
                writer.writerow(
                    {
                        header: row.get(header, "")
                        for header in headers
                    }
                )

        return

    with PREDICTION_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_prediction_result(row):
    ensure_prediction_history()

    with PREDICTION_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def ensure_range_prediction_history():
    DATA_DIR.mkdir(exist_ok=True)
    headers = [
        "checked_at",
        "predicted_at",
        "predicted_after_round_count",
        "actual_round_count",
        "actual_timestamp",
        "model_version",
        "predicted_label",
        "minimum",
        "maximum",
        "probability",
        "confidence",
        "source",
        "range_type",
        "clear_signal",
        "clear_reason",
        "scored",
        "actual_multiplier",
        "correct",
    ]

    if RANGE_PREDICTION_HISTORY_PATH.exists():
        with RANGE_PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if all(header in fieldnames for header in headers):
                return

            rows = list(reader)

        with RANGE_PREDICTION_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        "checked_at": row.get("checked_at", ""),
                        "predicted_at": row.get("predicted_at", ""),
                        "predicted_after_round_count": row.get(
                            "predicted_after_round_count",
                            "",
                        ),
                        "actual_round_count": row.get("actual_round_count", ""),
                        "actual_timestamp": row.get("actual_timestamp", ""),
                        "model_version": row.get("model_version", "range-v1"),
                        "predicted_label": row.get("predicted_label", ""),
                        "minimum": row.get("minimum", ""),
                        "maximum": row.get("maximum", ""),
                        "probability": row.get("probability", ""),
                        "confidence": row.get("confidence", ""),
                        "source": row.get("source", ""),
                        "range_type": row.get("range_type", ""),
                        "clear_signal": row.get("clear_signal", "1"),
                        "clear_reason": row.get("clear_reason", ""),
                        "scored": row.get("scored", "1"),
                        "actual_multiplier": row.get("actual_multiplier", ""),
                        "correct": row.get("correct", ""),
                    }
                )

        return

    with RANGE_PREDICTION_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_range_prediction_result(row):
    ensure_range_prediction_history()

    with RANGE_PREDICTION_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def rolling_accuracy(results):
    if not results:
        return None

    return sum(1 for result in results if result) / len(results)


def parse_bool_int(value):
    return str(value).strip() in ("1", "true", "True")


def prediction_history_rows():
    ensure_prediction_history()

    try:
        with PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def range_prediction_history_rows():
    ensure_range_prediction_history()

    try:
        with RANGE_PREDICTION_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def accuracy_for_rows(rows):
    if not rows:
        return {
            "checked": 0,
            "correct": 0,
            "accuracy": None,
        }

    correct = sum(
        1
        for row in rows
        if parse_bool_int(row.get("correct", "0"))
    )

    return {
        "checked": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
    }


def build_accuracy_summary():
    rows = prediction_history_rows()
    range_rows = range_prediction_history_rows()
    current_range_rows = [
        row
        for row in range_rows
        if row.get("model_version") == RANGE_MODEL_VERSION
    ]
    scored_range_rows = [
        row
        for row in current_range_rows
        if parse_bool_int(row.get("scored", "1"))
    ]
    windows = {}

    for size in (100, 300, 1000):
        windows[str(size)] = accuracy_for_rows(
            rows[-size:]
        )

    clear_rows = [
        row
        for row in rows
        if parse_bool_int(row.get("clear_signal", "0"))
    ]
    weak_rows = [
        row
        for row in rows
        if not parse_bool_int(row.get("clear_signal", "0"))
    ]
    target_accuracy = {}

    for target in TRACKED_TARGETS:
        key = f"{target:.2f}"
        target_rows = [
            row
            for row in rows
            if row.get("target") == key
        ]
        target_accuracy[key] = accuracy_for_rows(
            target_rows[-100:]
        )

    return {
        "windows": windows,
        "range": accuracy_for_rows(scored_range_rows[-300:]),
        "range_skipped": len(current_range_rows) - len(scored_range_rows),
        "clear": accuracy_for_rows(clear_rows[-300:]),
        "weak": accuracy_for_rows(weak_rows[-300:]),
        "targets": target_accuracy,
    }


def cached_accuracy_summary():
    signature = (
        file_signature(PREDICTION_HISTORY_PATH),
        file_signature(RANGE_PREDICTION_HISTORY_PATH),
    )

    if (
        ACCURACY_SUMMARY_CACHE["signature"] == signature
        and ACCURACY_SUMMARY_CACHE["summary"] is not None
    ):
        return copy.deepcopy(ACCURACY_SUMMARY_CACHE["summary"])

    summary = build_accuracy_summary()
    ACCURACY_SUMMARY_CACHE["signature"] = signature
    ACCURACY_SUMMARY_CACHE["summary"] = copy.deepcopy(summary)
    return summary


def prediction_quality(calls, profile_name, margin):
    if not calls:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "brier": None,
        }

    correct = 0
    positives = 0
    negatives = 0
    true_positives = 0
    true_negatives = 0
    brier_total = 0

    for call in calls:
        baseline = float(call.get("baseline_probability", 0))
        estimate = profile_probability(call, profile_name)
        predicted_high = (estimate - baseline) >= margin
        actual_high = bool(call["actual_high"])

        if actual_high:
            positives += 1
        else:
            negatives += 1

        if predicted_high == actual_high:
            correct += 1

        if predicted_high and actual_high:
            true_positives += 1

        if not predicted_high and not actual_high:
            true_negatives += 1

        actual_probability = 1 if actual_high else 0
        brier_total += (estimate - actual_probability) ** 2

    accuracy = correct / len(calls)

    if positives and negatives:
        sensitivity = true_positives / positives
        specificity = true_negatives / negatives
        balanced_accuracy = (sensitivity + specificity) / 2
    else:
        balanced_accuracy = accuracy

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "brier": brier_total / len(calls),
    }


def learn_decision_margin(calls):
    if len(calls) < 6:
        return 0

    best_margin = 0
    best_accuracy = -1

    for step in range(-10, 11):
        margin = step / 100
        correct = 0

        for call in calls:
            predicted_high = float(call["edge"]) >= margin
            actual_high = bool(call["actual_high"])

            if predicted_high == actual_high:
                correct += 1

        accuracy = correct / len(calls)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_margin = margin

    return best_margin


def profile_probability(call, profile_name):
    components = call.get("components") or {}
    weights = PROFILE_WEIGHTS.get(profile_name, PROFILE_WEIGHTS["balanced"])
    total = sum(weights.values())

    if total <= 0:
        return float(call.get("baseline_probability", 0))

    return sum(
        float(components.get(name, call.get("baseline_probability", 0)))
        * weight
        for name, weight in weights.items()
    ) / total


def learn_profile_and_margin(calls):
    component_calls = [
        call
        for call in calls
        if call.get("components")
    ]

    if len(component_calls) < 12:
        return {
            "profile": "balanced",
            "decision_margin": learn_decision_margin(calls),
            "profile_accuracy": None,
        }

    best = {
        "profile": "balanced",
        "decision_margin": 0,
        "profile_accuracy": -1,
        "balanced_accuracy": -1,
        "brier": 1,
    }

    for profile_name in PROFILE_WEIGHTS:
        estimates = [
            (
                profile_probability(call, profile_name),
                float(call.get("baseline_probability", 0)),
                bool(call["actual_high"]),
            )
            for call in component_calls
        ]

        for step in range(-12, 13):
            margin = step / 100
            correct = 0
            positives = 0
            negatives = 0
            true_positives = 0
            true_negatives = 0
            brier_total = 0

            for estimate, baseline, actual_high in estimates:
                predicted_high = (estimate - baseline) >= margin

                if predicted_high == actual_high:
                    correct += 1

                if actual_high:
                    positives += 1

                    if predicted_high:
                        true_positives += 1

                else:
                    negatives += 1

                    if not predicted_high:
                        true_negatives += 1

                actual_probability = 1 if actual_high else 0
                brier_total += (estimate - actual_probability) ** 2

            accuracy = correct / len(estimates)

            if positives and negatives:
                sensitivity = true_positives / positives
                specificity = true_negatives / negatives
                balanced_accuracy = (sensitivity + specificity) / 2
            else:
                balanced_accuracy = accuracy

            brier = brier_total / len(estimates)

            if (
                balanced_accuracy > best["balanced_accuracy"]
                or (
                    balanced_accuracy == best["balanced_accuracy"]
                    and brier < best["brier"]
                )
            ):
                best = {
                    "profile": profile_name,
                    "decision_margin": margin,
                    "profile_accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "brier": brier,
                }

    return best


def build_calibration(metrics):
    calibration = {}

    for key, metric in metrics.items():
        recent_results = metric.get("recent_results", [])
        recent_clear_results = metric.get("recent_clear_results", [])
        recent_calls = metric.get("recent_calls", [])
        strategy = learn_profile_and_margin(recent_calls)
        clear_checked = int(metric.get("clear_checked", 0))
        clear_correct = int(metric.get("clear_correct", 0))
        calibration[key] = {
            "checked": int(metric.get("checked", 0)),
            "correct": int(metric.get("correct", 0)),
            "accuracy": (
                metric.get("correct", 0) / metric.get("checked", 1)
                if metric.get("checked", 0)
                else None
            ),
            "recent_accuracy": rolling_accuracy(recent_results),
            "recent_checked": len(recent_results),
            "clear_checked": clear_checked,
            "clear_correct": clear_correct,
            "clear_accuracy": (
                clear_correct / clear_checked
                if clear_checked
                else None
            ),
            "recent_clear_accuracy": rolling_accuracy(recent_clear_results),
            "recent_clear_checked": len(recent_clear_results),
            "decision_margin": strategy["decision_margin"],
            "profile": strategy["profile"],
            "profile_accuracy": strategy["profile_accuracy"],
            "recent_balanced_accuracy": strategy.get("balanced_accuracy"),
            "recent_brier": strategy.get("brier"),
        }

    return calibration


def compact_prediction(prediction):
    return {
        "target": prediction["target"],
        "probability": prediction["probability"],
        "baseline_probability": prediction["baseline_probability"],
        "edge": prediction["edge"],
        "decision_margin": prediction.get("decision_margin", 0),
        "predicted_high": prediction.get("predicted_high", False),
        "clear_signal": prediction.get("clear_signal", False),
        "clear_reason": prediction.get("clear_reason", "weak signal"),
        "profile": prediction.get("profile", "balanced"),
        "components": prediction.get("components", {}),
        "signal": prediction["signal"],
        "confidence": prediction["confidence"],
    }


def compact_range_estimate(range_estimate):
    if not range_estimate:
        return None

    return {
        "label": range_estimate.get("label", ""),
        "short": range_estimate.get("short", ""),
        "minimum": range_estimate.get("minimum"),
        "maximum": range_estimate.get("maximum"),
        "probability": range_estimate.get("probability"),
        "confidence": range_estimate.get("confidence", ""),
        "source": range_estimate.get("source", ""),
        "range_type": range_estimate.get("range_type", ""),
        "model_version": RANGE_MODEL_VERSION,
        "clear_signal": range_estimate.get("clear_signal", False),
        "clear_reason": range_estimate.get("clear_reason", ""),
        "edge": range_estimate.get("edge", 0),
        "runner_up_label": range_estimate.get("runner_up_label", ""),
        "runner_up_probability": range_estimate.get("runner_up_probability", 0),
    }


def format_range_label(range_prediction):
    if not range_prediction:
        return "range unavailable"

    if range_prediction.get("maximum") is None:
        return f"above {float(range_prediction['minimum']):.2f}x"

    return (
        f"{float(range_prediction['minimum']):.2f}x to "
        f"{float(range_prediction['maximum']):.2f}x"
    )


def score_range_prediction(range_prediction, actual_multiplier):
    if not range_prediction:
        return None

    minimum = range_prediction.get("minimum")
    maximum = range_prediction.get("maximum")

    if minimum is None:
        return None

    minimum = float(minimum)
    maximum_value = None if maximum is None else float(maximum)
    actual = float(actual_multiplier)
    correct = actual >= minimum and (
        maximum_value is None
        or actual < maximum_value
    )
    clear_signal = bool(range_prediction.get("clear_signal", False))

    return {
        "label": range_prediction.get("label", ""),
        "short": range_prediction.get("short", ""),
        "display": format_range_label(range_prediction),
        "minimum": minimum,
        "maximum": maximum_value,
        "probability": range_prediction.get("probability"),
        "confidence": range_prediction.get("confidence", ""),
        "source": range_prediction.get("source", ""),
        "range_type": range_prediction.get("range_type", ""),
        "model_version": range_prediction.get("model_version", RANGE_MODEL_VERSION),
        "clear_signal": clear_signal,
        "clear_reason": range_prediction.get("clear_reason", ""),
        "scored": clear_signal,
        "correct": correct if clear_signal else None,
    }


def score_pending_prediction(state, rounds):
    pending = state.get("pending")

    if not pending:
        return None

    pending_range = pending.get("range_prediction") or {}

    if (
        pending_range
        and pending_range.get("model_version") != RANGE_MODEL_VERSION
    ):
        state["pending"] = None
        return state.get("last_result")

    after_round_count = int(pending.get("after_round_count", 0))

    if len(rounds) <= after_round_count:
        return None

    actual_round = rounds[after_round_count]
    actual_multiplier = float(actual_round["multiplier"])
    actual_round_count = after_round_count + 1
    score_id = f"{after_round_count}:{actual_round_count}:{actual_round.get('timestamp', '')}"

    if score_id in state.get("scored_ids", []):
        state["pending"] = None
        return state.get("last_result")

    scored = []
    range_result = score_range_prediction(
        pending.get("range_prediction"),
        actual_multiplier,
    )

    if range_result:
        range_metric = state.setdefault(
            "range_metrics",
            {
                "checked": 0,
                "correct": 0,
                "skipped": 0,
                "recent_results": [],
            },
        )

        if range_result["scored"]:
            range_metric["checked"] = int(range_metric.get("checked", 0)) + 1

            if range_result["correct"]:
                range_metric["correct"] = int(range_metric.get("correct", 0)) + 1

            range_metric.setdefault("recent_results", [])
            range_metric["recent_results"].append(range_result["correct"])
            range_metric["recent_results"] = range_metric["recent_results"][-100:]

        else:
            range_metric["skipped"] = int(range_metric.get("skipped", 0)) + 1

        append_range_prediction_result(
            [
                now_string(),
                pending.get("predicted_at", ""),
                after_round_count,
                actual_round_count,
                actual_round.get("timestamp", ""),
                range_result.get("model_version", RANGE_MODEL_VERSION),
                range_result.get("label", ""),
                f"{float(range_result['minimum']):.2f}",
                (
                    ""
                    if range_result.get("maximum") is None
                    else f"{float(range_result['maximum']):.2f}"
                ),
                (
                    ""
                    if range_result.get("probability") is None
                    else f"{float(range_result['probability']):.6f}"
                ),
                range_result.get("confidence", ""),
                range_result.get("source", ""),
                range_result.get("range_type", ""),
                int(range_result.get("clear_signal", False)),
                range_result.get("clear_reason", ""),
                int(range_result.get("scored", False)),
                f"{actual_multiplier:.2f}",
                (
                    ""
                    if range_result.get("correct") is None
                    else int(range_result["correct"])
                ),
            ]
        )

    for prediction in pending.get("predictions", []):
        target = float(prediction["target"])
        key = prediction_target_key(target)
        predicted_high = bool(
            prediction.get(
                "predicted_high",
                float(prediction["edge"])
                >= float(prediction.get("decision_margin", 0)),
            )
        )
        actual_high = actual_multiplier >= target
        correct = predicted_high == actual_high
        clear_signal = bool(
            prediction.get(
                "clear_signal",
                False,
            )
        )

        metric = state["metrics"].setdefault(
            key,
            {
                "checked": 0,
                "correct": 0,
                "clear_checked": 0,
                "clear_correct": 0,
                "recent_results": [],
                "recent_clear_results": [],
                "recent_calls": [],
            },
        )
        metric["checked"] += 1

        if correct:
            metric["correct"] += 1

        metric["recent_results"].append(correct)
        metric["recent_results"] = metric["recent_results"][-100:]

        if clear_signal:
            metric["clear_checked"] = int(metric.get("clear_checked", 0)) + 1

            if correct:
                metric["clear_correct"] = int(metric.get("clear_correct", 0)) + 1

            metric.setdefault("recent_clear_results", [])
            metric["recent_clear_results"].append(correct)
            metric["recent_clear_results"] = metric["recent_clear_results"][-100:]

        metric.setdefault("recent_calls", [])
        metric["recent_calls"].append(
            {
                "edge": float(prediction["edge"]),
                "baseline_probability": float(
                    prediction["baseline_probability"]
                ),
                "components": prediction.get("components", {}),
                "actual_high": actual_high,
            }
        )
        metric["recent_calls"] = metric["recent_calls"][-100:]

        append_prediction_result(
            [
                now_string(),
                pending.get("predicted_at", ""),
                after_round_count,
                actual_round_count,
                actual_round.get("timestamp", ""),
                f"{target:.2f}",
                f"{float(prediction['probability']):.6f}",
                f"{float(prediction['baseline_probability']):.6f}",
                f"{float(prediction['edge']):.6f}",
                f"{float(prediction.get('decision_margin', 0)):.6f}",
                prediction.get("profile", "balanced"),
                prediction.get("signal", ""),
                prediction.get("confidence", ""),
                int(clear_signal),
                prediction.get("clear_reason", ""),
                int(predicted_high),
                f"{actual_multiplier:.2f}",
                int(actual_high),
                int(correct),
            ]
        )

        scored.append(
            {
                "target": target,
                "predicted_high": predicted_high,
                "actual_high": actual_high,
                "correct": correct,
                "probability": prediction["probability"],
                "baseline_probability": prediction["baseline_probability"],
            }
        )

    state["last_result"] = {
        "score_id": score_id,
        "checked_at": now_string(),
        "actual_round_count": actual_round_count,
        "actual_timestamp": actual_round.get("timestamp", ""),
        "actual_multiplier": actual_multiplier,
        "range_result": range_result,
        "results": scored,
    }
    scored_ids = state.setdefault("scored_ids", [])
    scored_ids.append(score_id)
    state["scored_ids"] = scored_ids[-500:]
    state["pending"] = None
    return state["last_result"]


def update_prediction_tracking(rounds, report, state=None, calibration=None):
    state = state or load_prediction_state()
    score_pending_prediction(state, rounds)

    current_count = len(rounds)
    pending = state.get("pending")

    if (
        not pending
        or int(pending.get("after_round_count", 0)) != current_count
        or not pending.get("range_prediction")
        or "clear_signal" not in pending.get("range_prediction", {})
        or pending.get("range_prediction", {}).get("model_version") != RANGE_MODEL_VERSION
    ):
        predictions = [
            compact_prediction(prediction)
            for prediction in report["next_round"]["predictions"]
            if prediction["target"] in TRACKED_TARGETS
        ]
        state["pending"] = {
            "predicted_at": now_string(),
            "after_round_count": current_count,
            "latest_multiplier": rounds[-1]["multiplier"] if rounds else None,
            "range_prediction": compact_range_estimate(
                report["next_round"].get("range_estimate")
            ),
            "predictions": predictions,
        }

    state["calibration"] = (
        calibration
        if calibration is not None
        else build_calibration(state.get("metrics", {}))
    )
    save_prediction_state(state)

    return {
        "pending": state.get("pending"),
        "metrics": state.get("calibration", {}),
        "range_metrics": state.get("range_metrics", {}),
        "last_result": state.get("last_result"),
        "history_path": str(PREDICTION_HISTORY_PATH),
        "range_history_path": str(RANGE_PREDICTION_HISTORY_PATH),
    }


def compact_report(report, rounds):
    summary = report["summary"]
    recent = rounds[-60:]

    return {
        "generated_at": report["generated_at"],
        "warning": report["warning"],
        "data_selection": report.get("data_selection", {}),
        "ingest": ingest_status(rounds),
        "round_context": latest_round_context(),
        "summary": {
            "rounds": summary["rounds"],
            "latest_multiplier": round_float(summary["latest_multiplier"]),
            "average": round_float(summary["average"]),
            "median": round_float(summary["median"]),
            "p90": round_float(summary["p90"]),
            "maximum": round_float(summary["maximum"]),
            "buckets": summary["buckets"],
        },
        "overall_probabilities": report["overall_probabilities"],
        "next_round": report["next_round"],
        "backtests": report["backtests"],
        "recent_rounds": [
            {
                "timestamp": item["timestamp"],
                "multiplier": item["multiplier"],
                "round_id": item.get("round_id", ""),
                "source": item.get("source", ""),
            }
            for item in reversed(recent)
        ],
        "chart_rounds": [
            {
                "timestamp": item["timestamp"],
                "multiplier": item["multiplier"],
                "round_id": item.get("round_id", ""),
                "source": item.get("source", ""),
            }
            for item in recent
        ],
    }


def build_dashboard_payload(query):
    with TRACKING_LOCK:
        lookback = parse_int(
            query.get("lookback", ["2"])[0],
            default=2,
            minimum=1,
            maximum=8,
        )
        min_matches = parse_int(
            query.get("min_matches", ["5"])[0],
            default=5,
            minimum=1,
            maximum=50,
        )

        current_csv_signature = csv_signature()
        current_context_signature = file_signature(
            ROUND_CONTEXT_PATH
        )

        cache_key = (
            lookback,
            min_matches,
            current_csv_signature,
            current_context_signature,
        )
        cached = DASHBOARD_CACHE.get(cache_key)

        if cached:
            return refresh_cached_ingest(cached)

        all_rounds = load_rounds(CSV_PATH) if CSV_PATH.exists() else []
        rounds, data_selection = select_prediction_rounds(
            all_rounds
        )
        ensure_prediction_history()

        if not all_rounds:
            return {
                "generated_at": None,
            "warning": "No rounds have been collected yet.",
            "ingest": {
                "last_round_timestamp": None,
                "last_round_age_seconds": None,
                "is_stale": True,
            },
            "summary": {
                    "rounds": 0,
                    "latest_multiplier": None,
                    "average": None,
                    "median": None,
                    "p90": None,
                    "maximum": None,
                    "buckets": {},
                },
                "overall_probabilities": {},
                "next_round": {
                    "lookback": lookback,
                    "latest_pattern": [],
                    "pattern_match_count": 0,
                    "predictions": [],
                },
                "backtests": [],
                "recent_rounds": [],
                "chart_rounds": [],
                "round_context": latest_round_context(),
                "data_selection": data_selection,
                "tracking": {
                    "pending": None,
                    "metrics": {},
                    "range_metrics": {},
                    "last_result": None,
                    "history_path": str(PREDICTION_HISTORY_PATH),
                    "range_history_path": str(RANGE_PREDICTION_HISTORY_PATH),
                },
                "accuracy_summary": cached_accuracy_summary(),
            }

        prediction_state = load_prediction_state()
        score_pending_prediction(prediction_state, rounds)
        calibration = build_calibration(prediction_state.get("metrics", {}))

        report = build_live_report(
            rounds=rounds,
            lookback=lookback,
            targets=DEFAULT_TARGETS,
            min_matches=min_matches,
            calibration=calibration,
            data_selection=data_selection,
        )
        tracking = update_prediction_tracking(
            rounds,
            report,
            state=prediction_state,
            calibration=calibration,
        )
        payload = compact_report(report, rounds)
        payload["tracking"] = tracking
        payload["accuracy_summary"] = cached_accuracy_summary()
        payload["_cached_at"] = time.monotonic()

        if len(DASHBOARD_CACHE) >= CACHE_MAX_ITEMS:
            DASHBOARD_CACHE.clear()

        DASHBOARD_CACHE[cache_key] = copy.deepcopy(payload)

        return refresh_cached_ingest(payload)


def warm_dashboard_cache_once():
    for query in WARM_CACHE_QUERIES:
        build_dashboard_payload(query)


def start_cache_warmer():
    def worker():
        last_signature = object()

        while True:
            try:
                current_signature = (
                    csv_signature(),
                    file_signature(
                        ROUND_CONTEXT_PATH
                    ),
                )

                if current_signature != last_signature:
                    warm_dashboard_cache_once()
                    last_signature = current_signature

            except Exception:
                pass

            time.sleep(WARM_CACHE_POLL_SECONDS)

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )
    thread.start()
    return thread


class DashboardHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/summary":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        if parsed.path in ("", "/"):
            self.send_file_headers(DASHBOARD_DIR / "index.html")
            return

        requested = parsed.path.lstrip("/")
        path = (DASHBOARD_DIR / requested).resolve()

        if DASHBOARD_DIR.resolve() not in path.parents:
            self.send_error(404)
            return

        self.send_file_headers(path)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/summary":
            self.send_json(
                build_dashboard_payload(parse_qs(parsed.query))
            )
            return

        if parsed.path in ("", "/"):
            self.send_file(DASHBOARD_DIR / "index.html")
            return

        requested = parsed.path.lstrip("/")
        path = (DASHBOARD_DIR / requested).resolve()

        if DASHBOARD_DIR.resolve() not in path.parents:
            self.send_error(404)
            return

        self.send_file(path)

    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def send_file(self, path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        body = path.read_bytes()
        content_type = CONTENT_TYPES.get(
            path.suffix,
            "application/octet-stream",
        )

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def send_file_headers(self, path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        content_type = CONTENT_TYPES.get(
            path.suffix,
            "application/octet-stream",
        )

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(
        description="Serve the Aviator realtime dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        DashboardHandler,
    )
    start_cache_warmer()

    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
