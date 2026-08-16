import argparse
import copy
import csv
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timedelta
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
CONFIG_PATH = ROOT / "config.json"
PREDICTION_STATE_PATH = DATA_DIR / "prediction_state.json"
PREDICTION_HISTORY_PATH = DATA_DIR / "prediction_history.csv"
RANGE_PREDICTION_HISTORY_PATH = DATA_DIR / "range_prediction_history.csv"
RANGE_MODEL_HISTORY_PATH = DATA_DIR / "range_model_history.csv"
ROUND_CONTEXT_PATH = DATA_DIR / "round_context.csv"
COLLECTOR_STATE_PATH = DATA_DIR / "state.json"
EDGE_AUDIT_PATH = DATA_DIR / "edge_audit.json"
PATTERN_DISCOVERY_PATH = DATA_DIR / "pattern_discovery.json"
STRATEGY_AUDIT_PATH = DATA_DIR / "strategy_audit.json"
ML_PREDICTIONS_PATH = DATA_DIR / "ml_predictions.json"
ML_REPORT_PATH = DATA_DIR / "ml_report.json"
ML_MANIFEST_PATH = ROOT / "models" / "manifest.json"
ML_CHAMPION_METADATA_PATH = ROOT / "models" / "champion.json"
ML_RETRAIN_STATE_PATH = DATA_DIR / "ml_retrain_state.json"
RANGE_MODEL_VERSION = "adaptive-v8-fine-multilookback"
DASHBOARD_DIR = ROOT / "dashboard"
TRACKED_TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]
BIG_MULTIPLIER_TARGETS = [10.0, 20.0, 50.0, 100.0]
BIG_MULTIPLIER_RECENT_LIMIT = 8
TIMING_TARGETS = [2.0, 5.0, 10.0]
TIMING_MIN_ROUNDS = 500
TIMING_MIN_BUCKET_ROUNDS = 50
TIMING_TOP_LIMIT = 5
WEEKDAY_LABELS = [
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
]
TRACKING_LOCK = threading.Lock()
BACKTEST_LOCK = threading.Lock()
DISPLAY_MONEY_LOCK = threading.Lock()
ML_PREDICTION_LOCK = threading.Lock()
ML_RETRAIN_LOCK = threading.Lock()
EDGE_AUDIT_LOCK = threading.Lock()
DASHBOARD_CACHE = {}
CACHE_MAX_ITEMS = 12
ACCURACY_SUMMARY_CACHE = {
    "signature": None,
    "summary": None,
}
BACKTEST_REFRESH_SECONDS = 20
PARTICIPANT_CONTEXT_LIVE_SECONDS = 5
RECENT_CONTEXT_MAX_ROWS = 2500
RECENT_CONTEXT_MAX_BYTES = 1024 * 1024
ACTIONABLE_LEADERBOARD_MAXIMUM = 3.0
MIN_LEADERBOARD_CHECKED = 30
MIN_LEADERBOARD_ACCURACY = 0.56
MIN_LEADERBOARD_LONG_CHECKED = 150
MIN_LEADERBOARD_LONG_ACCURACY = 0.56
MIN_ACTIONABLE_RANGE_CHECKED = 100
MIN_ACTIONABLE_RANGE_ACCURACY = 0.56
DEFENSIVE_LOW_TARGETS = [2.0, 3.0]
DEFENSIVE_LOW_MIN_PROBABILITY = {
    2.0: 0.55,
    3.0: 0.64,
}
DEFENSIVE_LOW_MIN_CHECKED = 100
DEFENSIVE_LOW_MIN_ACCURACY = 0.60
BACKTEST_CACHE = {
    "key": None,
    "generated_at": 0,
    "in_progress": False,
    "items": [],
}
DISPLAY_MONEY_CACHE = {
    "signature": None,
    "settings": None,
    "checked_at": 0,
}
ML_PREDICTION_CACHE = {
    "signature": None,
    "payload": None,
}
EDGE_AUDIT_SETTINGS = {
    "enabled": True,
    "every_rounds": 250,
    "check_seconds": 60,
    "min_sample": 80,
    "top": 20,
    "walk_forward_folds": 6,
}
EDGE_AUDIT_STATE = {
    "thread": None,
    "status": "idle",
    "last_checked_at": None,
    "last_started_at": None,
    "last_finished_at": None,
    "last_success_at": None,
    "last_error": None,
    "last_returncode": None,
    "last_output_tail": None,
    "last_audited_rounds": None,
}
SEQUENCE_WATCH_BUCKETS = [
    ("tiny", "<1.20x"),
    ("small", "1.20x-1.99x"),
    ("medium", "2.00x-5.99x"),
    ("high", "6.00x-19.99x"),
    ("very_high", "20.00x+"),
]
ML_RETRAIN_STATE = {
    "thread": None,
}
ML_RETRAIN_SETTINGS = {
    "enabled": True,
    "min_new_rounds": 500,
    "check_seconds": 30,
    "minimum_training_rounds": 3000,
    "include_context": False,
    "promotion_min_skill_improvement": 0.005,
}
DISPLAY_MONEY_REFRESH_SECONDS = 2
DISPLAY_MONEY_FIELDS = {
    "total_bet": "display_total_bet",
    "avg_bet": "display_avg_bet",
    "max_bet": "display_max_bet",
    "total_win": "display_total_win",
    "max_win": "display_max_win",
    "net_result": "display_net_result",
}
DEFAULT_DISPLAY_CURRENCY = "INR"
DEFAULT_EUR_TO_INR_RATE = 100.0
DEFAULT_DOM_DISPLAY_CURRENCY = "INR"
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


def data_quality_snapshot(rounds):
    try:
        from data_quality_audit import audit_rows

        return audit_rows(
            rounds
        )
    except Exception as exc:
        return {
            "available": False,
            "status": "bad",
            "headline": "Data audit unavailable",
            "score": 0,
            "issues": [
                {
                    "severity": "bad",
                    "label": "Audit error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }


def compact_round_event(round_data, round_number, total_rounds, threshold=None):
    return {
        "timestamp": round_data.get("timestamp", ""),
        "multiplier": round_float(round_data.get("multiplier")),
        "round_id": round_data.get("round_id", ""),
        "source": round_data.get("source", ""),
        "round_number": round_number,
        "rounds_ago": max(0, total_rounds - round_number),
        "threshold": threshold,
    }


def big_round_watch(rounds):
    total_rounds = len(rounds)
    thresholds = []
    recent_events = []

    for target in BIG_MULTIPLIER_TARGETS:
        hits = []

        for index, round_data in enumerate(rounds):
            multiplier = round_data.get("multiplier")

            if multiplier is not None and multiplier >= target:
                hits.append((index + 1, round_data))

        last_hit = hits[-1] if hits else None
        thresholds.append(
            {
                "target": target,
                "count": len(hits),
                "rate": (len(hits) / total_rounds) if total_rounds else None,
                "last": (
                    compact_round_event(
                        last_hit[1],
                        last_hit[0],
                        total_rounds,
                        threshold=target,
                    )
                    if last_hit
                    else None
                ),
            }
        )

    for index, round_data in enumerate(rounds):
        multiplier = round_data.get("multiplier")

        if multiplier is None or multiplier < BIG_MULTIPLIER_TARGETS[0]:
            continue

        threshold = max(
            target for target in BIG_MULTIPLIER_TARGETS if multiplier >= target
        )
        recent_events.append(
            compact_round_event(
                round_data,
                index + 1,
                total_rounds,
                threshold=threshold,
            )
        )

    recent_events = recent_events[-BIG_MULTIPLIER_RECENT_LIMIT:]
    recent_events.reverse()
    latest = recent_events[0] if recent_events else None

    return {
        "targets": BIG_MULTIPLIER_TARGETS,
        "latest": latest,
        "current_round_big": bool(latest and latest.get("rounds_ago") == 0),
        "recent": recent_events,
        "thresholds": thresholds,
    }


def bucket_rate(values, target):
    if not values:
        return None

    return len(
        [
            value
            for value in values
            if value >= target
        ]
    ) / len(values)


def timing_bucket_score(rates, baselines):
    score = 0
    weights = {
        2.0: 0.35,
        5.0: 0.35,
        10.0: 0.30,
    }

    for target, weight in weights.items():
        rate = rates.get(
            f"{target:.2f}"
        )
        baseline = baselines.get(
            f"{target:.2f}"
        )

        if rate is None or baseline is None:
            continue

        score += (
            rate - baseline
        ) * weight

    return score


def compact_timing_bucket(label, key, values, baselines, bucket_type):
    sorted_values = sorted(
        values
    )
    rates = {
        f"{target:.2f}": bucket_rate(
            values,
            target
        )
        for target in TIMING_TARGETS
    }

    score = timing_bucket_score(
        rates,
        baselines,
    )

    return {
        "type": bucket_type,
        "key": key,
        "label": label,
        "rounds": len(values),
        "average": round_float(
            sum(values) / len(values)
        ),
        "median": round_float(
            sorted_values[len(sorted_values) // 2]
        ),
        "maximum": round_float(
            max(values)
        ),
        "rates": rates,
        "score": round_float(
            score
        ),
        "edge_2x": round_float(
            rates["2.00"] - baselines["2.00"]
        ),
        "edge_5x": round_float(
            rates["5.00"] - baselines["5.00"]
        ),
        "edge_10x": round_float(
            rates["10.00"] - baselines["10.00"]
        ),
    }


def compact_window_stats(values):
    if not values:
        return {
            "rounds": 0,
            "average": None,
            "median": None,
            "maximum": None,
            "rates": {},
        }

    sorted_values = sorted(
        values
    )

    return {
        "rounds": len(values),
        "average": round_float(
            sum(values) / len(values)
        ),
        "median": round_float(
            sorted_values[len(sorted_values) // 2]
        ),
        "maximum": round_float(
            max(values)
        ),
        "rates": {
            f"{target:.2f}": bucket_rate(
                values,
                target,
            )
            for target in [
                2.0,
                5.0,
                10.0,
                20.0,
                50.0,
                100.0,
            ]
        },
    }


def window_values(parsed, start, end):
    return [
        multiplier
        for timestamp, multiplier in parsed
        if start <= timestamp < end
    ]


def same_time_last_week_comparison(parsed):
    if not parsed:
        return {
            "available": False,
            "message": "Same-time comparison needs saved rounds.",
            "windows": [],
        }

    anchor = parsed[-1][0]
    previous_anchor = anchor - timedelta(
        days=7
    )
    earliest = parsed[0][0]
    windows = []

    for label, hours in (
        ("1h", 1),
        ("6h", 6),
        ("24h", 24),
    ):
        current_start = anchor - timedelta(
            hours=hours
        )
        previous_start = previous_anchor - timedelta(
            hours=hours
        )
        current_values = window_values(
            parsed,
            current_start,
            anchor,
        )
        previous_values = window_values(
            parsed,
            previous_start,
            previous_anchor,
        )
        current_stats = compact_window_stats(
            current_values
        )
        previous_stats = compact_window_stats(
            previous_values
        )
        deltas = {}

        for target in ("2.00", "5.00", "10.00", "20.00", "50.00", "100.00"):
            current_rate = current_stats["rates"].get(
                target
            )
            previous_rate = previous_stats["rates"].get(
                target
            )
            deltas[target] = (
                round_float(
                    current_rate - previous_rate
                )
                if current_rate is not None and previous_rate is not None
                else None
            )

        windows.append(
            {
                "label": label,
                "hours": hours,
                "current": current_stats,
                "last_week": previous_stats,
                "deltas": deltas,
            }
        )

    has_history = previous_anchor >= earliest
    return {
        "available": has_history and any(
            window["last_week"]["rounds"] > 0
            for window in windows
        ),
        "anchor_timestamp": anchor.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "last_week_timestamp": previous_anchor.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "message": (
            "Compared with the same time last week."
            if has_history
            else "Need at least 7 days of matching history."
        ),
        "windows": windows,
        "note": "Same-time history is only a comparison, not a guarantee.",
    }


def timing_insights(rounds):
    parsed = []

    for round_data in rounds:
        timestamp = parse_round_time(
            round_data.get(
                "timestamp",
                ""
            )
        )
        multiplier = round_data.get(
            "multiplier"
        )

        if timestamp is None or multiplier is None:
            continue

        parsed.append(
            (
                timestamp,
                float(multiplier),
            )
        )

    values = [
        multiplier
        for _, multiplier in parsed
    ]

    if len(values) < TIMING_MIN_ROUNDS:
        return {
            "available": False,
            "rounds": len(values),
            "minimum_rounds": TIMING_MIN_ROUNDS,
            "message": "Timing check needs more saved rounds.",
            "top_windows": [],
            "top_weekdays": [],
            "top_hours": [],
        }

    baselines = {
        f"{target:.2f}": bucket_rate(
            values,
            target
        )
        for target in TIMING_TARGETS
    }
    hour_buckets = {}
    weekday_buckets = {}
    window_buckets = {}

    for timestamp, multiplier in parsed:
        weekday = timestamp.weekday()
        hour = timestamp.hour

        hour_buckets.setdefault(
            hour,
            []
        ).append(
            multiplier
        )
        weekday_buckets.setdefault(
            weekday,
            []
        ).append(
            multiplier
        )
        window_buckets.setdefault(
            (
                weekday,
                hour,
            ),
            []
        ).append(
            multiplier
        )

    def ranked(items, labeler, bucket_type):
        buckets = []

        for key, bucket_values in items.items():
            if len(bucket_values) < TIMING_MIN_BUCKET_ROUNDS:
                continue

            buckets.append(
                compact_timing_bucket(
                    labeler(
                        key
                    ),
                    key,
                    bucket_values,
                    baselines,
                    bucket_type,
                )
            )

        buckets.sort(
            key=lambda item: (
                item["score"],
                item["rounds"],
            ),
            reverse=True,
        )

        return buckets[:TIMING_TOP_LIMIT]

    top_windows = ranked(
        window_buckets,
        lambda key: f"{WEEKDAY_LABELS[key[0]]} {key[1]:02d}:00",
        "weekday_hour",
    )
    top_weekdays = ranked(
        weekday_buckets,
        lambda key: WEEKDAY_LABELS[key],
        "weekday",
    )
    top_hours = ranked(
        hour_buckets,
        lambda key: f"{key:02d}:00",
        "hour",
    )
    current_time = datetime.now()
    current_key = (
        current_time.weekday(),
        current_time.hour,
    )
    current_values = window_buckets.get(
        current_key,
        []
    )
    current_window = None

    if len(current_values) >= TIMING_MIN_BUCKET_ROUNDS:
        current_window = compact_timing_bucket(
            f"{WEEKDAY_LABELS[current_key[0]]} {current_key[1]:02d}:00",
            current_key,
            current_values,
            baselines,
            "current_weekday_hour",
        )

    best = top_windows[0] if top_windows else None
    message = (
        f"Historically strongest window: {best['label']}"
        if best and best.get("score", 0) > 0
        else "No clear timing edge above average."
    )

    return {
        "available": True,
        "rounds": len(values),
        "minimum_bucket_rounds": TIMING_MIN_BUCKET_ROUNDS,
        "baselines": baselines,
        "message": message,
        "current_window": current_window,
        "top_windows": top_windows,
        "top_weekdays": top_weekdays,
        "top_hours": top_hours,
        "same_time_last_week": same_time_last_week_comparison(
            parsed
        ),
        "note": "Historical timing only; it is not a profit guarantee.",
    }


def safe_number(value, default=0):
    try:
        if value is None:
            return default

        return float(
            value
        )
    except (TypeError, ValueError):
        return default


def holdout_has_edge(item):
    status = str(
        item.get(
            "holdout_status",
            item.get(
                "validation_status",
                "",
            ),
        )
        or ""
    ).upper()
    model = str(
        item.get(
            "model",
            "",
        )
        or ""
    )

    if model == "historical_frequency":
        return False

    if "NO PREDICTIVE EDGE" in status:
        return False

    return safe_number(
        item.get(
            "edge"
        )
    ) >= 0.03


def best_ml_edge(ml_prediction):
    predictions = (
        ml_prediction or {}
    ).get(
        "predictions",
        {}
    )
    candidates = []

    for target, item in predictions.items():
        edge = safe_number(
            item.get(
                "edge"
            )
        )

        if not holdout_has_edge(
            item
        ):
            continue

        candidates.append(
            {
                "target": target,
                "edge": edge,
                "probability": item.get(
                    "probability"
                ),
                "model": item.get(
                    "model"
                ),
                "holdout_status": item.get(
                    "holdout_status"
                ),
            }
        )

    return max(
        candidates,
        key=lambda item: item["edge"],
        default=None,
    )


def best_direct_edge(predictions):
    candidates = []

    for prediction in predictions or []:
        edge = safe_number(
            prediction.get(
                "edge"
            )
        )
        target = safe_number(
            prediction.get(
                "target"
            )
        )

        if (
            not prediction.get(
                "clear_signal"
            )
            or not prediction.get(
                "predicted_high"
            )
            or target < 1.5
            or edge < 0.03
        ):
            continue

        candidates.append(
            {
                "target": target,
                "edge": edge,
                "probability": prediction.get(
                    "probability"
                ),
                "confidence": prediction.get(
                    "confidence",
                    "low",
                ),
                "signal": prediction.get(
                    "signal",
                    "",
                ),
            }
        )

    return max(
        candidates,
        key=lambda item: (
            item["edge"],
            item["target"],
        ),
        default=None,
    )


def compact_signal_reason(label, detail, tone="neutral"):
    return {
        "label": label,
        "detail": detail,
        "tone": tone,
    }


def format_signal_multiplier(value):
    return f"{safe_number(value):.2f}x"


def format_signal_percent(value):
    if value is None:
        return "--"

    return f"{round_float(safe_number(value) * 100)}%"


def range_scoreboard_is_actionable(accuracy_summary):
    active_model = (
        accuracy_summary or {}
    ).get(
        "active_range_model"
    )

    if active_model:
        if active_model.get(
            "candidate_model"
        ) == "baseline":
            return False, None

        return True, active_model

    range_stats = (
        accuracy_summary or {}
    ).get(
        "range",
        {}
    ) or {}
    checked = int(
        safe_number(
            range_stats.get(
                "checked"
            )
        )
    )
    accuracy = range_stats.get(
        "accuracy"
    )

    if (
        checked >= MIN_ACTIONABLE_RANGE_CHECKED
        and accuracy is not None
        and safe_number(
            accuracy
        ) >= MIN_ACTIONABLE_RANGE_ACCURACY
    ):
        return True, {
            "candidate_model": "range_scoreboard",
            "checked": checked,
            "accuracy": accuracy,
        }

    return False, None


def defensive_low_call(predictions, accuracy_summary):
    target_stats = (
        accuracy_summary or {}
    ).get(
        "targets",
        {}
    ) or {}
    candidates = []

    for prediction in predictions or []:
        target = safe_number(
            prediction.get(
                "target"
            )
        )

        if target not in DEFENSIVE_LOW_TARGETS:
            continue

        if prediction.get(
            "predicted_high"
        ):
            continue

        low_probability = 1 - safe_number(
            prediction.get(
                "probability"
            )
        )
        minimum_probability = DEFENSIVE_LOW_MIN_PROBABILITY.get(
            target,
            0.65,
        )

        if low_probability < minimum_probability:
            continue

        stats = target_stats.get(
            f"{target:.2f}",
            {},
        ) or {}
        checked = int(
            safe_number(
                stats.get(
                    "checked"
                )
            )
        )
        accuracy = stats.get(
            "accuracy"
        )

        if (
            checked < DEFENSIVE_LOW_MIN_CHECKED
            or accuracy is None
            or safe_number(
                accuracy
            ) < DEFENSIVE_LOW_MIN_ACCURACY
        ):
            continue

        baseline_low = 1 - safe_number(
            prediction.get(
                "baseline_probability"
            )
        )
        candidates.append(
            {
                "status": "defensive",
                "label": "DEFENSIVE",
                "target": target,
                "main_call": f"Likely below {format_signal_multiplier(target)}",
                "cashout": (
                    "No high chase; if playing, protect before 2.00x."
                    if target >= 3
                    else "No high chase; cash out very early."
                ),
                "probability": low_probability,
                "baseline_probability": baseline_low,
                "accuracy": accuracy,
                "checked": checked,
                "detail": (
                    f"Model says below {format_signal_multiplier(target)} is "
                    f"{format_signal_percent(low_probability)}; history is "
                    f"{format_signal_percent(baseline_low)}; recent low-rate is "
                    f"{format_signal_percent(accuracy)} over {checked} rounds."
                ),
            }
        )

    return max(
        candidates,
        key=lambda item: (
            item["target"],
            item["probability"],
        ),
        default=None,
    )


def signal_quality(payload):
    summary = payload.get(
        "summary",
        {}
    )
    next_round = payload.get(
        "next_round",
        {}
    )
    timing = payload.get(
        "timing_insights",
        {}
    ) or {}
    range_estimate = next_round.get(
        "range_estimate",
        {}
    ) or {}
    predictions = next_round.get(
        "predictions",
        []
    )
    accuracy_summary = payload.get(
        "accuracy_summary",
        {},
    ) or {}
    ml_edge = best_ml_edge(
        payload.get(
            "ml_prediction"
        )
    )
    direct_edge = best_direct_edge(
        predictions
    )
    current_window = timing.get(
        "current_window"
    ) or {}
    current_timing_score = safe_number(
        current_window.get(
            "score"
        )
    )
    current_timing_rounds = int(
        safe_number(
            current_window.get(
                "rounds"
            )
        )
    )
    range_edge = safe_number(
        range_estimate.get(
            "edge"
        )
    )
    range_max = range_estimate.get(
        "maximum"
    )
    range_scoreboard_ok, range_scoreboard = range_scoreboard_is_actionable(
        accuracy_summary
    )
    range_is_actionable = (
        bool(
            range_estimate.get(
                "clear_signal"
            )
        )
        and range_edge >= 0.03
        and range_scoreboard_ok
        and (
            range_max is None
            or safe_number(
                range_max,
                default=99,
            ) <= 3
        )
    )
    rounds = int(
        safe_number(
            summary.get(
                "rounds"
            )
        )
    )
    points = 0
    reasons = []

    if rounds < 3000:
        return {
            "status": "wait",
            "label": "WAIT",
            "score": 0,
            "headline": "Wait - collecting enough history",
            "main_call": "No play signal",
            "cashout": "No reliable cashout target",
            "reasons": [
                compact_signal_reason(
                    "Data",
                    f"{rounds} rounds saved, needs 3000+ for signal quality.",
                    "wait",
                )
            ],
        }

    if ml_edge:
        edge = safe_number(
            ml_edge.get(
                "edge"
            )
        )
        points += min(
            55,
            35 + edge * 300,
        )
        reasons.append(
            compact_signal_reason(
                "ML edge",
                (
                    f"{ml_edge.get('target')}x target is "
                    f"{round_float(edge * 100)} pp above baseline."
                ),
                "good",
            )
        )
    else:
        reasons.append(
            compact_signal_reason(
                "ML edge",
                "No promoted model is beating history yet.",
                "wait",
            )
        )

    if direct_edge:
        edge = safe_number(
            direct_edge.get(
                "edge"
            )
        )
        points += min(
            35,
            20 + edge * 220,
        )
        reasons.append(
            compact_signal_reason(
                "Pattern edge",
                (
                    f"{direct_edge['target']:.2f}x+ is "
                    f"{round_float(edge * 100)} pp above history."
                ),
                "good",
            )
        )
    else:
        reasons.append(
            compact_signal_reason(
                "Pattern edge",
                "No clear next-round pattern edge.",
                "wait",
            )
        )

    if range_is_actionable:
        points += min(
            25,
            12 + range_edge * 220,
        )
        reasons.append(
            compact_signal_reason(
                "Range",
                (
                    f"{range_estimate.get('short') or range_estimate.get('label')} "
                    f"has a usable range edge and passed recent scoring."
                ),
                "good",
            )
        )
    else:
        range_reason = range_estimate.get(
            "clear_reason",
            "Range not strong enough.",
        )

        if (
            range_estimate.get(
                "clear_signal"
            )
            and not range_scoreboard_ok
        ):
            range_reason = "Range signal is blocked because its scoreboard is not good enough yet."

        reasons.append(
            compact_signal_reason(
                "Range",
                range_reason,
                "neutral",
            )
        )

    if current_window:
        if current_timing_score >= 0.06:
            points += 22
            tone = "good"
            detail = (
                f"{current_window.get('label')} is historically strong "
                f"over {current_timing_rounds} rounds."
            )
        elif current_timing_score >= 0.03:
            points += 12
            tone = "neutral"
            detail = (
                f"{current_window.get('label')} is slightly above average "
                f"over {current_timing_rounds} rounds."
            )
        else:
            tone = "wait"
            detail = (
                f"{current_window.get('label')} is not above average enough."
            )

        reasons.append(
            compact_signal_reason(
                "Timing",
                detail,
                tone,
            )
        )
    else:
        reasons.append(
            compact_signal_reason(
                "Timing",
                "Current weekday/hour does not have enough history yet.",
                "neutral",
            )
        )

    points = int(
        min(
            100,
            max(
                0,
                round(
                    points
                ),
            ),
        )
    )
    has_proven_edge = bool(
        ml_edge
        or direct_edge
        or range_is_actionable
    )
    defensive_call = defensive_low_call(
        predictions,
        accuracy_summary,
    )
    selective_call = {
        "status": "no_call",
        "label": "NO CALL",
        "headline": "No reliable call right now",
        "main_call": "No play signal",
        "cashout": "No reliable cashout target.",
        "reason": "No model or range is beating baseline enough.",
    }

    if points >= 65 and has_proven_edge:
        status = "active"
        label = "ACTIVE"
        headline = "Active signal - edge confirmed"
        main_call = (
            f"Target {direct_edge['target']:.2f}x+"
            if direct_edge
            else f"Target {ml_edge.get('target')}x+"
            if ml_edge
            else range_estimate.get(
                "short",
                "Use shown range",
            )
        )
        cashout = "Use only the shown target/range; avoid chasing big multipliers."
        selective_call = {
            "status": "active",
            "label": "ACTIVE",
            "headline": headline,
            "main_call": main_call,
            "cashout": cashout,
            "reason": "Validated edge signal.",
            "score": points,
        }
    elif defensive_call:
        points = max(
            points,
            42,
        )
        status = "watch"
        label = "DEFENSIVE"
        headline = "Defensive read - common outcome only"
        main_call = defensive_call["main_call"]
        cashout = defensive_call["cashout"]
        selective_call = {
            **defensive_call,
            "headline": headline,
            "reason": "This is a defensive common-outcome read, not a proven profit edge.",
            "score": points,
        }
        reasons.append(
            compact_signal_reason(
                "Defensive read",
                defensive_call["detail"],
                "neutral",
            )
        )
    elif points >= 30:
        status = "watch"
        label = "WATCH"
        headline = "Watch only - weak edge"
        main_call = "No strong play signal"
        cashout = "If playing, keep target small and do not chase 10x+."
    else:
        status = "wait"
        label = "WAIT"
        headline = "Wait - no proven edge"
        main_call = "No play signal"
        cashout = "No reliable cashout target."

    return {
        "status": status,
        "label": label,
        "score": points,
        "headline": headline,
        "main_call": main_call,
        "cashout": cashout,
        "reasons": reasons[:5],
        "current_window": current_window or None,
        "has_proven_edge": has_proven_edge,
        "selective_call": selective_call,
        "range_scoreboard": range_scoreboard,
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

        participants = payload["round_context"].get("participants")

        if (
            participants
            and participants.get("age_seconds") is not None
            and participants.get("age_seconds") > PARTICIPANT_CONTEXT_LIVE_SECONDS
        ):
            payload["round_context"]["participants"] = None

    payload["round_context"] = latest_round_context()
    payload["collector_status"] = collector_status_snapshot()
    cached_recent_rounds = payload.get(
        "recent_rounds",
        [],
    )
    chronological_cached_recent_rounds = list(
        reversed(
            cached_recent_rounds
        )
    )
    payload.setdefault(
        "edge_audit",
        edge_audit_snapshot(
            chronological_cached_recent_rounds
        ),
    )
    payload["sequence_watch"] = sequence_watch_snapshot(
        chronological_cached_recent_rounds
    )
    payload.setdefault(
        "data_quality",
        data_quality_snapshot(
            chronological_cached_recent_rounds
        ),
    )
    try:
        latest_rounds = load_rounds(CSV_PATH) if CSV_PATH.exists() else chronological_cached_recent_rounds
    except Exception:
        latest_rounds = chronological_cached_recent_rounds

    payload["strategy_audit"] = strategy_audit_snapshot(
        latest_rounds
    )
    payload["ml_retrain"] = ml_retrain_status_snapshot()
    payload["signal_quality"] = signal_quality(
        payload
    )

    with BACKTEST_LOCK:
        if (
            payload.get("_backtest_key") == BACKTEST_CACHE["key"]
            and BACKTEST_CACHE["items"]
        ):
            payload["backtests"] = list(BACKTEST_CACHE["items"])

    payload["generated_at"] = now_string()
    payload["cache_age_ms"] = int(
        (time.monotonic() - payload.get("_cached_at", time.monotonic()))
        * 1000
    )
    payload.pop("_cached_at", None)
    payload.pop("_backtest_key", None)
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


def load_json_file(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collector_status_snapshot():
    state = load_json_file(
        COLLECTOR_STATE_PATH
    ) or {}
    status = state.get(
        "game_status",
        {}
    )
    realtime_channels = state.get(
        "realtime_channels",
        {}
    )

    if not isinstance(
        realtime_channels,
        dict
    ):
        realtime_channels = {}

    channel_observed_at = realtime_channels.get(
        "observed_at",
        ""
    )
    parsed_channel_observed_at = parse_round_time(
        channel_observed_at
    )
    channel_age_seconds = (
        max(
            0,
            int(
                (datetime.now() - parsed_channel_observed_at).total_seconds()
            )
        )
        if parsed_channel_observed_at
        else None
    )
    safe_realtime_channels = {
        "available": bool(
            realtime_channels.get(
                "total"
            )
        ),
        "labels": realtime_channels.get(
            "labels",
            []
        ),
        "total": realtime_channels.get(
            "total",
            0
        ),
        "observed_at": channel_observed_at,
        "age_seconds": channel_age_seconds,
        "secrets_saved": False,
    }

    if not isinstance(
        status,
        dict
    ) or not status:
        return {
            "available": False,
            "phase": None,
            "label": "Waiting for game state",
            "live_multiplier": None,
            "age_seconds": None,
            "realtime_channels": safe_realtime_channels,
        }

    observed_at = status.get(
        "observed_at",
        ""
    )
    parsed = parse_round_time(
        observed_at
    )
    age_seconds = (
        max(
            0,
            int(
                (datetime.now() - parsed).total_seconds()
            )
        )
        if parsed
        else None
    )
    phase = status.get(
        "phase"
    )
    labels = {
        "preparing": "Preparing next round",
        "starting": "Round starting",
        "running": "Plane flying",
        "finished": "Round crashed",
    }

    return {
        "available": True,
        "phase": phase,
        "label": labels.get(
            phase,
            "Game state detected"
        ),
        "round_state": status.get(
            "round_state"
        ),
        "is_preparing": status.get(
            "is_preparing"
        ),
        "live_multiplier": round_float(
            status.get(
                "live_multiplier"
            )
        ),
        "observed_at": observed_at,
        "age_seconds": age_seconds,
        "source": status.get(
            "source"
        ),
        "game_source": status.get(
            "game_source"
        ),
        "last_run_at": status.get(
            "last_run_at"
        ),
        "last_finish_at": status.get(
            "last_finish_at"
        ),
        "realtime_channels": safe_realtime_channels,
    }


def edge_audit_rounds(rounds):
    try:
        from edge_audit import Round as EdgeRound
    except Exception:
        return []

    converted = []

    for item in rounds:
        timestamp = item.get(
            "timestamp",
            "",
        )

        try:
            multiplier = float(
                item.get(
                    "multiplier"
                )
            )
        except (TypeError, ValueError):
            continue

        converted.append(
            EdgeRound(
                timestamp=timestamp,
                timestamp_dt=parse_round_time(
                    timestamp
                ),
                multiplier=multiplier,
                round_id=item.get(
                    "round_id",
                    "",
                ),
                source=item.get(
                    "source",
                    "",
                ),
            )
        )

    return converted


def compact_edge_audit_item(item, is_active=False):
    train = item.get(
        "train",
        {}
    )
    holdout = item.get(
        "holdout",
        {}
    )
    walk_forward = item.get(
        "walk_forward",
        {}
    )

    return {
        "condition": item.get(
            "condition",
            "",
        ),
        "target": item.get(
            "target",
            "",
        ),
        "active": bool(
            is_active
        ),
        "status": item.get(
            "status",
            "",
        ),
        "fdr_confirmed": bool(
            item.get(
                "fdr_confirmed"
            )
        ),
        "walk_forward_stable": bool(
            item.get(
                "walk_forward_stable"
            )
        ),
        "strong_edge": bool(
            item.get(
                "strong_edge"
            )
        ),
        "watch_candidate": bool(
            item.get(
                "watch_candidate"
            )
        ),
        "holdout_rate": holdout.get(
            "rate"
        ),
        "holdout_baseline": holdout.get(
            "baseline"
        ),
        "holdout_lift": holdout.get(
            "lift"
        ),
        "holdout_lift_ci_low": holdout.get(
            "lift_ci_low"
        ),
        "holdout_lift_ci_high": holdout.get(
            "lift_ci_high"
        ),
        "holdout_lift_ci_excludes_zero": holdout.get(
            "lift_ci_excludes_zero"
        ),
        "holdout_z": holdout.get(
            "z"
        ),
        "holdout_q_value": holdout.get(
            "q_value"
        ),
        "holdout_fdr_significant": holdout.get(
            "fdr_significant"
        ),
        "holdout_checked": holdout.get(
            "checked"
        ),
        "train_rate": train.get(
            "rate"
        ),
        "train_checked": train.get(
            "checked"
        ),
        "walk_forward_verdict": walk_forward.get(
            "verdict"
        ),
        "walk_forward_valid_folds": walk_forward.get(
            "valid_folds"
        ),
        "walk_forward_positive_folds": walk_forward.get(
            "positive_folds"
        ),
        "walk_forward_significant_positive_folds": walk_forward.get(
            "significant_positive_folds"
        ),
        "walk_forward_positive_fold_share": walk_forward.get(
            "positive_fold_share"
        ),
        "walk_forward_average_lift": walk_forward.get(
            "average_lift"
        ),
    }


def edge_audit_snapshot(rounds):
    report = load_json_file(
        EDGE_AUDIT_PATH
    )

    if not isinstance(
        report,
        dict
    ) or not report:
        return {
            "available": False,
            "path": str(
                EDGE_AUDIT_PATH
            ),
            "message": "Run edge_audit.py to create AI watch candidates.",
            "refresh": edge_audit_status_snapshot(),
            "active": [],
            "watch": [],
        }

    active = []
    edge_rounds = edge_audit_rounds(
        rounds
    )

    if edge_rounds:
        try:
            from edge_audit import condition_matches
        except Exception:
            condition_matches = None

        if condition_matches:
            next_index = len(
                edge_rounds
            )

            for item in report.get(
                "validated_patterns",
                []
            ):
                spec = item.get(
                    "spec"
                )

                if not isinstance(
                    spec,
                    dict
                ):
                    continue

                try:
                    is_active = condition_matches(
                        edge_rounds,
                        next_index,
                        spec,
                    )
                except Exception:
                    is_active = False

                if is_active:
                    active.append(
                        compact_edge_audit_item(
                            item,
                            True,
                        )
                    )

    watch = [
        compact_edge_audit_item(
            item,
            False,
        )
        for item in report.get(
            "validated_patterns",
            []
        )[:8]
    ]

    return {
        "available": True,
        "path": str(
            EDGE_AUDIT_PATH
        ),
        "generated_at": report.get(
            "generated_at"
        ),
        "rounds": report.get(
            "rounds"
        ),
        "strong_edge_count": report.get(
            "strong_edge_count",
            0,
        ),
        "watch_candidate_count": report.get(
            "watch_candidate_count",
            0,
        ),
        "fdr_confirmed_count": report.get(
            "fdr_confirmed_count",
            0,
        ),
        "walk_forward_stable_count": report.get(
            "walk_forward_stable_count",
            0,
        ),
        "walk_forward_folds": report.get(
            "walk_forward_folds",
            0,
        ),
        "patterns_tested": report.get(
            "patterns_tested",
            0,
        ),
        "validated_test_count": report.get(
            "validated_test_count",
            0,
        ),
        "fdr_alpha": report.get(
            "fdr_alpha",
        ),
        "conclusion": report.get(
            "conclusion",
            "",
        ),
        "active": active[:5],
        "watch": watch,
        "big_multiplier_gaps": report.get(
            "big_multiplier_gaps",
            [],
        ),
        "refresh": edge_audit_status_snapshot(),
    }


def strategy_audit_snapshot(rounds):
    report = load_json_file(
        STRATEGY_AUDIT_PATH
    )

    if not isinstance(
        report,
        dict,
    ) or not report:
        return {
            "available": False,
            "status": "missing",
            "headline": "Strategy audit missing",
            "message": "Run strategy_audit.py to test cashout strategies on holdout data.",
            "path": str(
                STRATEGY_AUDIT_PATH
            ),
        }

    audited_rounds = int(
        safe_number(
            report.get(
                "rounds"
            )
        )
    )
    current_rounds = len(
        rounds or []
    )
    new_rounds = max(
        0,
        current_rounds - audited_rounds,
    )
    stale = new_rounds >= 500

    return {
        "available": bool(
            report.get(
                "available",
                True,
            )
        ),
        "status": (
            "stale"
            if stale
            else report.get(
                "status",
                "unknown",
            )
        ),
        "headline": report.get(
            "headline",
            "Strategy audit ready",
        ),
        "message": report.get(
            "message",
            "",
        ),
        "generated_at": report.get(
            "generated_at"
        ),
        "rounds": audited_rounds,
        "current_rounds": current_rounds,
        "new_rounds": new_rounds,
        "stale": stale,
        "train_rounds": report.get(
            "train_rounds"
        ),
        "holdout_rounds": report.get(
            "holdout_rounds"
        ),
        "best_train_strategy": report.get(
            "best_train_strategy"
        ),
        "best_forward_candidate": report.get(
            "best_forward_candidate"
        ),
        "positive_both_count": report.get(
            "positive_both_count",
            0,
        ),
        "family_winners": (
            report.get(
                "family_winners",
                []
            )[:5]
            if isinstance(
                report.get(
                    "family_winners",
                    []
                ),
                list,
            )
            else []
        ),
        "note": report.get(
            "note",
            "",
        ),
        "path": str(
            STRATEGY_AUDIT_PATH
        ),
    }


def sequence_watch_bucket(value):
    try:
        multiplier = float(
            value
        )
    except (TypeError, ValueError):
        return None

    if multiplier < 1.2:
        return "tiny"
    if multiplier < 2.0:
        return "small"
    if multiplier < 6.0:
        return "medium"
    if multiplier < 20.0:
        return "high"
    return "very_high"


def sequence_watch_label(keys):
    label_map = dict(
        SEQUENCE_WATCH_BUCKETS
    )

    return " -> ".join(
        label_map.get(
            key,
            key,
        )
        for key in keys
    )


def current_sequence_watch_keys(rounds):
    sequences = []

    for length in (
        2,
        3,
        4,
    ):
        if len(
            rounds
        ) < length:
            continue

        values = []

        for item in rounds[-length:]:
            bucket = sequence_watch_bucket(
                item.get(
                    "multiplier"
                )
            )

            if bucket is None:
                values = []
                break

            values.append(
                bucket
            )

        if not values:
            continue

        key = f"seq_{length}_{'_'.join(values)}"
        sequences.append(
            {
                "length": length,
                "key": key,
                "label": sequence_watch_label(
                    values
                ),
            }
        )

    return sequences


def compact_sequence_watch_item(item, is_active=False):
    train = item.get(
        "train",
        {}
    )
    holdout = item.get(
        "holdout",
        {}
    )
    walk_forward = item.get(
        "walk_forward",
        {}
    )

    return {
        "pattern": item.get(
            "pattern",
            "",
        ),
        "pattern_key": item.get(
            "pattern_key",
            "",
        ),
        "outcome": item.get(
            "outcome",
            "",
        ),
        "outcome_key": item.get(
            "outcome_key",
            "",
        ),
        "status": item.get(
            "status",
            "",
        ),
        "active": bool(
            is_active
        ),
        "holdout_rate": holdout.get(
            "rate"
        ),
        "holdout_baseline": holdout.get(
            "baseline"
        ),
        "holdout_lift": holdout.get(
            "lift"
        ),
        "holdout_lift_ci_low": holdout.get(
            "lift_ci_low"
        ),
        "holdout_lift_ci_high": holdout.get(
            "lift_ci_high"
        ),
        "holdout_lift_ci_excludes_zero": holdout.get(
            "lift_ci_excludes_zero"
        ),
        "holdout_z": holdout.get(
            "z"
        ),
        "holdout_q_value": holdout.get(
            "q_value"
        ),
        "holdout_fdr_significant": holdout.get(
            "fdr_significant"
        ),
        "holdout_checked": holdout.get(
            "checked"
        ),
        "train_rate": train.get(
            "rate"
        ),
        "train_checked": train.get(
            "checked"
        ),
        "walk_forward_valid_folds": walk_forward.get(
            "valid_folds"
        ),
        "walk_forward_positive_folds": walk_forward.get(
            "positive_folds"
        ),
        "walk_forward_significant_positive_folds": walk_forward.get(
            "significant_positive_folds"
        ),
        "walk_forward_average_lift": walk_forward.get(
            "average_lift"
        ),
    }


def sequence_watch_snapshot(rounds):
    report = load_json_file(
        PATTERN_DISCOVERY_PATH
    )
    current_sequences = current_sequence_watch_keys(
        rounds
    )

    if not isinstance(
        report,
        dict
    ) or not report:
        return {
            "available": False,
            "path": str(
                PATTERN_DISCOVERY_PATH
            ),
            "message": "Run pattern_discovery.py to create sequence watch data.",
            "active": [],
            "weak_active": [],
            "confirmed": [],
            "watch": [],
            "current_sequences": current_sequences,
        }

    groups = report.get(
        "groups",
        {}
    )
    sequence_items = []

    if isinstance(
        groups,
        dict
    ):
        sequence_items = [
            item
            for item in groups.get(
                "bucket_sequence",
                []
            )
            if isinstance(
                item,
                dict,
            )
        ]

    if not sequence_items:
        sequence_items = [
            item
            for item in report.get(
                "top",
                []
            )
            if isinstance(
                item,
                dict,
            )
            and item.get(
                "group"
            )
            == "bucket_sequence"
        ]

    active_keys = {
        item["key"]
        for item in current_sequences
    }
    active = [
        compact_sequence_watch_item(
            item,
            True,
        )
        for item in sequence_items
        if item.get(
            "pattern_key"
        )
        in active_keys
        and item.get(
            "status"
        )
        == "confirmed"
        and item.get(
            "holdout",
            {}
        ).get(
            "fdr_significant"
        )
        and item.get(
            "holdout",
            {}
        ).get(
            "lift_ci_excludes_zero"
        )
    ]
    weak_active = [
        compact_sequence_watch_item(
            item,
            True,
        )
        for item in sequence_items
        if item.get(
            "pattern_key"
        )
        in active_keys
        and item.get(
            "status"
        )
        in {
            "watch_strong",
            "watch",
        }
    ]
    confirmed = [
        compact_sequence_watch_item(
            item,
            False,
        )
        for item in sequence_items
        if item.get(
            "status"
        )
        == "confirmed"
    ]
    watch = [
        compact_sequence_watch_item(
            item,
            False,
        )
        for item in sequence_items
        if item.get(
            "status"
        )
        in {
            "watch_strong",
            "watch",
        }
    ]

    return {
        "available": True,
        "path": str(
            PATTERN_DISCOVERY_PATH
        ),
        "generated_at": report.get(
            "generated_at"
        ),
        "rounds": report.get(
            "rounds"
        ),
        "confirmed_count": len(
            confirmed
        ),
        "watch_count": len(
            watch
        ),
        "patterns_tested": report.get(
            "patterns_tested",
        ),
        "current_sequences": current_sequences,
        "active": active[:5],
        "weak_active": weak_active[:5],
        "confirmed": confirmed[:3],
        "watch": watch[:3],
    }


def compact_ml_prediction_payload(payload, round_count):
    if not payload:
        return {
            "available": False,
            "error": "ML prediction data is not available yet.",
            "predictions": {},
        }

    compact = {
        "available": not bool(payload.get("error")),
        "error": payload.get("error"),
        "generated_at": payload.get("generated_at"),
        "model_version": payload.get("model_version"),
        "metadata_source": payload.get("metadata_source"),
        "champion_version": payload.get("champion_version"),
        "feature_schema_version": payload.get("feature_schema_version"),
        "data_used_rounds": payload.get("data_used_rounds"),
        "rounds_in_csv": round_count,
        "is_current": (
            payload.get("data_used_rounds") >= round_count
            if payload.get("data_used_rounds") is not None
            else False
        ),
        "predictions": {},
    }

    for target, item in (payload.get("predictions") or {}).items():
        compact["predictions"][target] = {
            "probability": item.get("probability"),
            "historical_baseline": item.get("historical_baseline"),
            "edge": item.get("edge"),
            "model": item.get("model"),
            "model_version": item.get("model_version"),
            "model_path": item.get("model_path"),
            "validation_status": item.get("validation_status"),
            "holdout_status": item.get("holdout_status"),
            "holdout_brier_skill": item.get("holdout_brier_skill"),
            "note": item.get("note"),
        }

    return compact


def current_ml_prediction(round_count):
    signature = (
        csv_signature(),
        file_signature(ML_CHAMPION_METADATA_PATH),
        file_signature(ML_MANIFEST_PATH),
        file_signature(ML_REPORT_PATH),
    )

    with ML_PREDICTION_LOCK:
        if (
            ML_PREDICTION_CACHE["signature"] == signature
            and ML_PREDICTION_CACHE["payload"] is not None
        ):
            return compact_ml_prediction_payload(
                copy.deepcopy(ML_PREDICTION_CACHE["payload"]),
                round_count,
            )

    try:
        from ml_predict import make_predictions

        payload = make_predictions(
            csv_path=CSV_PATH,
            manifest_path=ML_MANIFEST_PATH,
            report_path=ML_REPORT_PATH,
            min_history=100,
            include_context=bool(
                ML_RETRAIN_SETTINGS.get(
                    "include_context",
                    False,
                )
            ),
        )
        payload.setdefault(
            "generated_at",
            now_string(),
        )
    except Exception as exc:
        payload = load_json_file(
            ML_PREDICTIONS_PATH
        ) or {
            "model_version": "ml-research-v1",
            "error": f"ML prediction unavailable: {type(exc).__name__}: {exc}",
            "predictions": {},
        }
        payload.setdefault(
            "generated_at",
            now_string(),
        )

    with ML_PREDICTION_LOCK:
        ML_PREDICTION_CACHE["signature"] = signature
        ML_PREDICTION_CACHE["payload"] = copy.deepcopy(payload)

    return compact_ml_prediction_payload(
        payload,
        round_count,
    )


def csv_data_row_count(path=CSV_PATH):
    if not path.exists():
        return 0

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def save_ml_retrain_state(state):
    DATA_DIR.mkdir(exist_ok=True)

    with ML_RETRAIN_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def load_ml_retrain_state():
    state = load_json_file(ML_RETRAIN_STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    return state


def infer_last_trained_rounds():
    state = load_ml_retrain_state()
    report_path = state.get(
        "report_path"
    )

    if report_path:
        try:
            from ml_auto_retrain import report_training_rounds

            trained_rounds = report_training_rounds(
                report_path
            )
        except Exception:
            trained_rounds = 0

        if trained_rounds > 0:
            return trained_rounds

    for key in ("last_trained_rounds", "last_checked_rounds"):
        try:
            value = int(state.get(key, 0))
        except (TypeError, ValueError):
            value = 0

        if value > 0:
            return value

    report = load_json_file(ML_REPORT_PATH) or {}
    data_quality = report.get("data_quality", {})
    dataset_statistics = report.get("dataset_statistics", {})

    for value in (
        data_quality.get("valid_rows"),
        dataset_statistics.get("feature_rows"),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0

        if parsed > 0:
            return parsed

    predictions = load_json_file(ML_PREDICTIONS_PATH) or {}

    try:
        return max(0, int(predictions.get("data_used_rounds", 0)))
    except (TypeError, ValueError):
        return 0


def output_tail(text, max_lines=35):
    lines = str(text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def ml_retrain_scheduler_config():
    return {
        "ml_retrain_every_rounds": ML_RETRAIN_SETTINGS["min_new_rounds"],
        "ml_minimum_training_rounds": ML_RETRAIN_SETTINGS[
            "minimum_training_rounds"
        ],
        "ml_include_context": ML_RETRAIN_SETTINGS.get(
            "include_context",
            False,
        ),
        "ml_promotion_min_skill_improvement": ML_RETRAIN_SETTINGS[
            "promotion_min_skill_improvement"
        ],
    }


def ml_retrain_scheduler_snapshot():
    from ml_auto_retrain import scheduler_status

    return scheduler_status(
        ml_retrain_scheduler_config(),
        CSV_PATH,
    )


def ml_retrain_status_snapshot():
    try:
        status = ml_retrain_scheduler_snapshot()
        status["enabled"] = bool(ML_RETRAIN_SETTINGS.get("enabled", True))
        status["check_seconds"] = int(ML_RETRAIN_SETTINGS.get("check_seconds", 30))

        thread = ML_RETRAIN_STATE.get("thread")
        if thread and thread.is_alive():
            status["status"] = "training"

        if not ML_RETRAIN_SETTINGS.get("enabled", True):
            status["status"] = "disabled"

        return status

    except Exception:
        pass

    settings = dict(ML_RETRAIN_SETTINGS)
    state = load_ml_retrain_state()
    current_rounds = csv_data_row_count()
    last_trained_rounds = infer_last_trained_rounds()
    new_rounds = max(0, current_rounds - last_trained_rounds)
    min_new_rounds = int(settings.get("min_new_rounds", 500))
    thread = ML_RETRAIN_STATE.get("thread")
    is_training = bool(thread and thread.is_alive())

    status = state.get("status") or "idle"

    if is_training:
        status = "training"
    elif not settings.get("enabled", True):
        status = "disabled"
    elif current_rounds <= 0:
        status = "waiting_for_data"
    elif new_rounds < min_new_rounds and status not in ("complete", "failed"):
        status = "waiting"

    return {
        "enabled": bool(settings.get("enabled", True)),
        "status": status,
        "current_rounds": current_rounds,
        "last_trained_rounds": last_trained_rounds,
        "new_rounds_since_train": new_rounds,
        "min_new_rounds": min_new_rounds,
        "rounds_until_next_train": max(0, min_new_rounds - new_rounds),
        "check_seconds": int(settings.get("check_seconds", 30)),
        "last_checked_at": state.get("last_checked_at"),
        "last_started_at": state.get("last_started_at"),
        "last_finished_at": state.get("last_finished_at"),
        "last_success_at": state.get("last_success_at"),
        "last_error": state.get("last_error"),
        "last_returncode": state.get("last_returncode"),
        "last_output_tail": state.get("last_output_tail"),
    }


def update_ml_retrain_status(**updates):
    state = load_ml_retrain_state()
    state.update(updates)
    state["last_checked_at"] = now_string()
    save_ml_retrain_state(state)
    return state


def clear_ml_dashboard_caches():
    with ML_PREDICTION_LOCK:
        ML_PREDICTION_CACHE["signature"] = None
        ML_PREDICTION_CACHE["payload"] = None

    DASHBOARD_CACHE.clear()


def run_ml_retrain(current_rounds):
    started_at = now_string()
    update_ml_retrain_status(
        status="training",
        last_started_at=started_at,
        last_error=None,
        last_returncode=None,
        started_rounds=current_rounds,
    )

    command = [
        sys.executable,
        str(ROOT / "ml_auto_retrain.py"),
        "--run-once",
        "--reason",
        "dashboard",
        "--csv",
        str(CSV_PATH),
        "--config",
        str(CONFIG_PATH),
        "--retrain-every-rounds",
        str(ML_RETRAIN_SETTINGS["min_new_rounds"]),
        "--minimum-training-rounds",
        str(ML_RETRAIN_SETTINGS["minimum_training_rounds"]),
        "--promotion-min-skill-improvement",
        str(ML_RETRAIN_SETTINGS["promotion_min_skill_improvement"]),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60 * 45,
        )
        combined_output = "\n".join(
            item
            for item in (result.stdout, result.stderr)
            if item
        )
        try:
            scheduler = ml_retrain_scheduler_snapshot()
        except Exception:
            scheduler = {}

        finished_rounds = scheduler.get(
            "current_rounds",
            csv_data_row_count(),
        )
        finished_trained_rounds = scheduler.get(
            "last_trained_rounds",
            finished_rounds,
        )

        if result.returncode == 0:
            update_ml_retrain_status(
                status=scheduler.get("status", "complete"),
                last_finished_at=now_string(),
                last_success_at=now_string(),
                last_trained_rounds=finished_trained_rounds,
                current_rounds=finished_rounds,
                new_rounds_since_train=scheduler.get(
                    "new_rounds_since_train",
                    0,
                ),
                min_new_rounds=scheduler.get(
                    "min_new_rounds",
                    ML_RETRAIN_SETTINGS["min_new_rounds"],
                ),
                rounds_until_next_train=scheduler.get(
                    "rounds_until_next_train",
                    ML_RETRAIN_SETTINGS["min_new_rounds"],
                ),
                promoted_targets=scheduler.get(
                    "promoted_targets",
                    [],
                ),
                kept_targets=scheduler.get(
                    "kept_targets",
                    [],
                ),
                last_returncode=result.returncode,
                last_output_tail=output_tail(combined_output),
                last_error=None,
            )
            clear_ml_dashboard_caches()
            return

        update_ml_retrain_status(
            status="failed",
            last_finished_at=now_string(),
            last_returncode=result.returncode,
            last_output_tail=output_tail(combined_output),
            last_error="ml_auto_retrain.py failed",
        )

    except Exception as exc:
        update_ml_retrain_status(
            status="failed",
            last_finished_at=now_string(),
            last_error=f"{type(exc).__name__}: {exc}",
        )


def maybe_start_ml_retrain():
    settings = ML_RETRAIN_SETTINGS

    if not settings.get("enabled", True):
        update_ml_retrain_status(
            status="disabled",
        )
        return

    try:
        from ml_auto_retrain import should_retrain

        retrain_due, reason, status = should_retrain(
            {
                "ml_retrain_every_rounds": settings.get("min_new_rounds", 500),
                "ml_minimum_training_rounds": settings.get(
                    "minimum_training_rounds",
                    3000,
                ),
                "ml_promotion_min_skill_improvement": settings.get(
                    "promotion_min_skill_improvement",
                    0.005,
                ),
                "ml_include_context": settings.get(
                    "include_context",
                    False,
                ),
            },
            CSV_PATH,
        )
        current_rounds = status.get("current_rounds", 0)
        last_trained_rounds = status.get("last_trained_rounds", 0)
    except Exception:
        retrain_due = False
        reason = "scheduler unavailable"
        current_rounds = csv_data_row_count()
        last_trained_rounds = infer_last_trained_rounds()
        fallback_new_rounds = max(
            0,
            current_rounds - last_trained_rounds,
        )
        status = {
            "new_rounds_since_train": fallback_new_rounds,
            "min_new_rounds": settings.get(
                "min_new_rounds",
                500,
            ),
            "rounds_until_next_train": max(
                0,
                int(
                    settings.get(
                        "min_new_rounds",
                        500,
                    )
                ) - fallback_new_rounds,
            ),
        }

    with ML_RETRAIN_LOCK:
        thread = ML_RETRAIN_STATE.get("thread")

        if thread and thread.is_alive():
            update_ml_retrain_status(
                status="training",
                current_rounds=current_rounds,
                last_trained_rounds=last_trained_rounds,
                new_rounds_since_train=status.get(
                    "new_rounds_since_train",
                    0,
                ),
                min_new_rounds=status.get(
                    "min_new_rounds",
                    settings.get("min_new_rounds", 500),
                ),
                rounds_until_next_train=status.get(
                    "rounds_until_next_train",
                    0,
                ),
            )
            return

        if not retrain_due:
            update_ml_retrain_status(
                status="waiting",
                current_rounds=current_rounds,
                last_trained_rounds=last_trained_rounds,
                new_rounds_since_train=status.get(
                    "new_rounds_since_train",
                    0,
                ),
                min_new_rounds=status.get(
                    "min_new_rounds",
                    settings.get("min_new_rounds", 500),
                ),
                rounds_until_next_train=status.get(
                    "rounds_until_next_train",
                    0,
                ),
                message=reason,
            )
            return

        thread = threading.Thread(
            target=run_ml_retrain,
            args=(current_rounds,),
            daemon=True,
        )
        ML_RETRAIN_STATE["thread"] = thread
        thread.start()


def start_ml_auto_retrainer(
    enabled=True,
    min_new_rounds=500,
    check_seconds=30,
    minimum_training_rounds=3000,
    include_context=False,
    promotion_min_skill_improvement=0.005,
):
    ML_RETRAIN_SETTINGS.update(
        {
            "enabled": bool(enabled),
            "min_new_rounds": max(1, int(min_new_rounds)),
            "check_seconds": max(5, int(check_seconds)),
            "minimum_training_rounds": max(100, int(minimum_training_rounds)),
            "include_context": bool(include_context),
            "promotion_min_skill_improvement": max(
                0.0,
                float(promotion_min_skill_improvement),
            ),
        }
    )

    def worker():
        while True:
            try:
                maybe_start_ml_retrain()
            except Exception as exc:
                update_ml_retrain_status(
                    status="failed",
                    last_error=f"{type(exc).__name__}: {exc}",
                )

            time.sleep(
                ML_RETRAIN_SETTINGS["check_seconds"]
            )

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )
    thread.start()
    return thread


def current_edge_audit_rounds():
    report = load_json_file(
        EDGE_AUDIT_PATH
    )

    if not isinstance(
        report,
        dict,
    ):
        return 0

    try:
        return max(
            0,
            int(
                report.get(
                    "rounds",
                    0,
                )
            ),
        )
    except (TypeError, ValueError):
        return 0


def edge_audit_status_snapshot():
    settings = dict(
        EDGE_AUDIT_SETTINGS
    )
    audited_rounds = current_edge_audit_rounds()
    current_rounds = csv_data_row_count()
    new_rounds = max(
        0,
        current_rounds - audited_rounds,
    )
    thread = EDGE_AUDIT_STATE.get(
        "thread"
    )
    is_running = bool(
        thread
        and thread.is_alive()
    )

    status = EDGE_AUDIT_STATE.get(
        "status",
        "idle",
    )

    if is_running:
        status = "running"
    elif not settings.get(
        "enabled",
        True,
    ):
        status = "disabled"
    elif not EDGE_AUDIT_PATH.exists():
        status = "due"
    elif new_rounds < settings.get(
        "every_rounds",
        250,
    ):
        status = "waiting"

    return {
        "enabled": bool(
            settings.get(
                "enabled",
                True,
            )
        ),
        "status": status,
        "current_rounds": current_rounds,
        "last_audited_rounds": audited_rounds,
        "new_rounds_since_audit": new_rounds,
        "every_rounds": int(
            settings.get(
                "every_rounds",
                250,
            )
        ),
        "rounds_until_next_audit": max(
            0,
            int(
                settings.get(
                    "every_rounds",
                    250,
                )
            ) - new_rounds,
        ),
        "check_seconds": int(
            settings.get(
                "check_seconds",
                60,
            )
        ),
        "last_checked_at": EDGE_AUDIT_STATE.get(
            "last_checked_at"
        ),
        "last_started_at": EDGE_AUDIT_STATE.get(
            "last_started_at"
        ),
        "last_finished_at": EDGE_AUDIT_STATE.get(
            "last_finished_at"
        ),
        "last_success_at": EDGE_AUDIT_STATE.get(
            "last_success_at"
        ),
        "last_error": EDGE_AUDIT_STATE.get(
            "last_error"
        ),
        "last_returncode": EDGE_AUDIT_STATE.get(
            "last_returncode"
        ),
        "last_output_tail": EDGE_AUDIT_STATE.get(
            "last_output_tail"
        ),
    }


def update_edge_audit_status(**updates):
    EDGE_AUDIT_STATE.update(
        updates
    )
    EDGE_AUDIT_STATE["last_checked_at"] = now_string()
    return EDGE_AUDIT_STATE


def run_edge_audit_refresh(current_rounds):
    update_edge_audit_status(
        status="running",
        last_started_at=now_string(),
        last_error=None,
        last_returncode=None,
        started_rounds=current_rounds,
    )

    command = [
        sys.executable,
        str(ROOT / "edge_audit.py"),
        "--min-sample",
        str(
            EDGE_AUDIT_SETTINGS["min_sample"]
        ),
        "--top",
        str(
            EDGE_AUDIT_SETTINGS["top"]
        ),
        "--walk-forward-folds",
        str(
            EDGE_AUDIT_SETTINGS["walk_forward_folds"]
        ),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60 * 10,
        )
        combined_output = "\n".join(
            item
            for item in (
                result.stdout,
                result.stderr,
            )
            if item
        )
        audited_rounds = current_edge_audit_rounds()

        if result.returncode == 0:
            update_edge_audit_status(
                status="complete",
                last_finished_at=now_string(),
                last_success_at=now_string(),
                last_returncode=result.returncode,
                last_output_tail=output_tail(
                    combined_output,
                    20,
                ),
                last_error=None,
                last_audited_rounds=audited_rounds,
            )
            DASHBOARD_CACHE.clear()
            return

        update_edge_audit_status(
            status="failed",
            last_finished_at=now_string(),
            last_returncode=result.returncode,
            last_output_tail=output_tail(
                combined_output,
                20,
            ),
            last_error="edge_audit.py failed",
        )

    except Exception as exc:
        update_edge_audit_status(
            status="failed",
            last_finished_at=now_string(),
            last_error=f"{type(exc).__name__}: {exc}",
        )


def maybe_start_edge_audit_refresh():
    settings = EDGE_AUDIT_SETTINGS
    current_rounds = csv_data_row_count()
    audited_rounds = current_edge_audit_rounds()
    new_rounds = max(
        0,
        current_rounds - audited_rounds,
    )
    audit_due = (
        current_rounds > 0
        and (
            not EDGE_AUDIT_PATH.exists()
            or new_rounds >= settings.get(
                "every_rounds",
                250,
            )
        )
    )

    with EDGE_AUDIT_LOCK:
        thread = EDGE_AUDIT_STATE.get(
            "thread"
        )

        if thread and thread.is_alive():
            update_edge_audit_status(
                status="running",
                current_rounds=current_rounds,
                last_audited_rounds=audited_rounds,
            )
            return

        if not settings.get(
            "enabled",
            True,
        ):
            update_edge_audit_status(
                status="disabled",
                current_rounds=current_rounds,
                last_audited_rounds=audited_rounds,
            )
            return

        if not audit_due:
            update_edge_audit_status(
                status="waiting",
                current_rounds=current_rounds,
                last_audited_rounds=audited_rounds,
                new_rounds_since_audit=new_rounds,
            )
            return

        thread = threading.Thread(
            target=run_edge_audit_refresh,
            args=(current_rounds,),
            daemon=True,
        )
        EDGE_AUDIT_STATE["thread"] = thread
        thread.start()


def start_edge_audit_refresher(
    enabled=True,
    every_rounds=250,
    check_seconds=60,
    min_sample=80,
    top=20,
    walk_forward_folds=6,
):
    EDGE_AUDIT_SETTINGS.update(
        {
            "enabled": bool(
                enabled
            ),
            "every_rounds": max(
                1,
                int(
                    every_rounds
                ),
            ),
            "check_seconds": max(
                10,
                int(
                    check_seconds
                ),
            ),
            "min_sample": max(
                10,
                int(
                    min_sample
                ),
            ),
            "top": max(
                1,
                int(
                    top
                ),
            ),
            "walk_forward_folds": max(
                2,
                int(
                    walk_forward_folds
                ),
            ),
        }
    )

    def worker():
        while True:
            try:
                maybe_start_edge_audit_refresh()
            except Exception as exc:
                update_edge_audit_status(
                    status="failed",
                    last_error=f"{type(exc).__name__}: {exc}",
                )

            time.sleep(
                EDGE_AUDIT_SETTINGS["check_seconds"]
            )

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )
    thread.start()
    return thread


def recent_csv_rows(path, max_rows, max_bytes):
    if not path.exists():
        return []

    try:
        with path.open("rb") as f:
            header = f.readline().decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if not header:
                return []

            f.seek(0, 2)
            size = f.tell()
            read_from = max(
                0,
                size - max_bytes,
            )
            f.seek(read_from)
            text = f.read().decode(
                "utf-8",
                errors="ignore",
            )
    except OSError:
        return []

    lines = text.splitlines()

    if read_from > 0 and lines:
        lines = lines[1:]

    if lines and lines[0].strip() == header:
        lines = lines[1:]

    lines = lines[-max_rows:]

    if not lines:
        return []

    return list(
        csv.DictReader(
            [header, *lines]
        )
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


def display_money_settings():
    now = time.monotonic()

    with DISPLAY_MONEY_LOCK:
        if (
            DISPLAY_MONEY_CACHE["settings"] is not None
            and now - DISPLAY_MONEY_CACHE["checked_at"]
            < DISPLAY_MONEY_REFRESH_SECONDS
        ):
            return DISPLAY_MONEY_CACHE["settings"]

    signature = file_signature(CONFIG_PATH)

    with DISPLAY_MONEY_LOCK:
        if (
            DISPLAY_MONEY_CACHE["signature"] == signature
            and DISPLAY_MONEY_CACHE["settings"] is not None
        ):
            DISPLAY_MONEY_CACHE["checked_at"] = now
            return DISPLAY_MONEY_CACHE["settings"]

    settings = {
        "currency": DEFAULT_DISPLAY_CURRENCY,
        "eur_to_inr_rate": DEFAULT_EUR_TO_INR_RATE,
        "dom_currency": DEFAULT_DOM_DISPLAY_CURRENCY,
    }

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}

        currency = str(
            config.get("display_currency", DEFAULT_DISPLAY_CURRENCY)
        ).strip().upper()
        dom_currency = str(
            config.get(
                "dom_display_currency",
                DEFAULT_DOM_DISPLAY_CURRENCY,
            )
        ).strip().upper()

        try:
            eur_to_inr_rate = float(
                config.get(
                    "display_eur_to_inr_rate",
                    DEFAULT_EUR_TO_INR_RATE,
                )
            )
        except (TypeError, ValueError):
            eur_to_inr_rate = DEFAULT_EUR_TO_INR_RATE

        if currency:
            settings["currency"] = currency

        if dom_currency:
            settings["dom_currency"] = dom_currency

        if eur_to_inr_rate > 0:
            settings["eur_to_inr_rate"] = eur_to_inr_rate

    with DISPLAY_MONEY_LOCK:
        DISPLAY_MONEY_CACHE["signature"] = signature
        DISPLAY_MONEY_CACHE["settings"] = settings
        DISPLAY_MONEY_CACHE["checked_at"] = now

    return settings


def context_money_source_unit(context):
    source = str(
        context.get("source", "")
    ).lower()

    if "worker_top" in source:
        return "EUR"

    if "participants_dom" in source:
        return display_money_settings()["dom_currency"]

    return ""


def decorate_context_display_money(context):
    if not context:
        return context

    settings = display_money_settings()
    display_currency = settings["currency"]
    source_unit = context_money_source_unit(context)
    display_rate = 1.0

    if display_currency == "INR" and source_unit == "EUR":
        display_rate = settings["eur_to_inr_rate"]

    context["display_currency"] = display_currency
    context["money_source_unit"] = source_unit
    context["display_money_rate"] = round_float(display_rate)

    for source_key, display_key in DISPLAY_MONEY_FIELDS.items():
        value = context.get(source_key)

        context[display_key] = (
            round_float(value * display_rate)
            if value is not None
            else None
        )

    return context


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
    context["net_result"] = (
        context["total_win"] - context["total_bet"]
        if (
            context.get("total_win") is not None
            and context.get("total_bet") is not None
        )
        else None
    )

    return decorate_context_display_money(
        update_round_context_age(
            context
        )
    )


def is_participant_context_source(source):
    lowered = str(
        source or ""
    ).lower()

    return (
        lowered == "participants_dom"
        or "participants" in lowered
        or "userbets" in lowered
    )


def best_participant_context(rows):
    candidates = []

    for row in rows:
        if not is_participant_context_source(
            row.get("source", "")
        ):
            continue

        context = context_from_row(
            row
        )

        if not context:
            continue

        candidates.append(
            context
        )

    if not candidates:
        return None

    fresh_candidates = [
        context
        for context in candidates
        if (
            context.get("age_seconds") is None
            or context.get("age_seconds") <= PARTICIPANT_CONTEXT_LIVE_SECONDS
        )
    ]

    active_candidates = [
        context
        for context in fresh_candidates
        if context.get("source") == "participants_worker_active"
    ]
    detail_candidates = [
        context
        for context in fresh_candidates
        if context.get("bet_count") is not None
    ]
    active_context = (
        max(
            active_candidates,
            key=lambda context: context.get("observed_at") or "",
        )
        if active_candidates
        else None
    )

    if detail_candidates:
        best_detail = max(
            detail_candidates,
            key=lambda context: (
                context.get("player_count") or 0,
                context.get("bet_count") or 0,
                context.get("payload_records") or 0,
                context.get("observed_at") or "",
            )
        )

        if (
            active_context
            and (active_context.get("player_count") or 0)
            > (best_detail.get("player_count") or 0)
        ):
            best_detail = dict(
                best_detail
            )
            best_detail["player_count"] = active_context.get(
                "player_count"
            )
            best_detail["active_source"] = active_context.get(
                "source"
            )
            best_detail["active_observed_at"] = active_context.get(
                "observed_at"
            )

        return best_detail

    if active_context:
        return active_context

    if fresh_candidates:
        return max(
            fresh_candidates,
            key=lambda context: (
                context.get("player_count") or 0,
                context.get("bet_count") or 0,
                context.get("payload_records") or 0,
                context.get("observed_at") or "",
            )
        )

    return max(
        candidates,
        key=lambda context: context.get("observed_at") or ""
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
    participant_rows = []

    rows = recent_csv_rows(
        ROUND_CONTEXT_PATH,
        RECENT_CONTEXT_MAX_ROWS,
        RECENT_CONTEXT_MAX_BYTES,
    )

    for row in rows:
        if (
            row.get("source") == "flight_radar_dom"
            and not parse_context_int(
                row.get("player_count")
            )
        ):
            continue

        latest = row
        by_source[row.get("source", "")] = row

        if is_participant_context_source(
            row.get("source", "")
        ):
            participant_rows.append(
                row
            )

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
    participants = best_participant_context(
        participant_rows
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
    include_backtests=True,
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

    if include_backtests:
        maybe_refresh_backtests(
            values,
            lookback,
            targets,
            min_matches,
            backtest_key,
        )

    with BACKTEST_LOCK:
        backtests = (
            list(BACKTEST_CACHE["items"])
            if include_backtests
            else []
        )

    return {
        "generated_at": now_string(),
        "csv_rounds": len(rounds),
        "_backtest_key": backtest_key,
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


def ensure_range_model_history():
    DATA_DIR.mkdir(exist_ok=True)
    headers = [
        "checked_at",
        "predicted_at",
        "predicted_after_round_count",
        "actual_round_count",
        "actual_timestamp",
        "model_version",
        "candidate_model",
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

    if RANGE_MODEL_HISTORY_PATH.exists():
        with RANGE_MODEL_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if all(header in fieldnames for header in headers):
                return

            rows = list(reader)

        with RANGE_MODEL_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        header: row.get(header, "")
                        for header in headers
                    }
                )

        return

    with RANGE_MODEL_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_range_model_result(row):
    ensure_range_model_history()

    with RANGE_MODEL_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
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


def range_model_history_rows():
    ensure_range_model_history()

    try:
        with RANGE_MODEL_HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def accuracy_for_rows(rows):
    if not rows:
        return {
            "checked": 0,
            "correct": 0,
            "accuracy": None,
            "baseline_accuracy": None,
            "skill": None,
        }

    correct = sum(
        1
        for row in rows
        if parse_bool_int(row.get("correct", "0"))
    )
    actual_high_values = [
        parse_bool_int(row.get("actual_high", "0"))
        for row in rows
        if row.get("actual_high", "") != ""
    ]
    baseline_accuracy = None
    skill = None
    accuracy = correct / len(rows)

    if len(actual_high_values) == len(rows):
        high_count = sum(1 for value in actual_high_values if value)
        low_count = len(actual_high_values) - high_count
        baseline_accuracy = max(high_count, low_count) / len(actual_high_values)
        skill = accuracy - baseline_accuracy

    return {
        "checked": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "skill": skill,
    }


def build_range_model_leaderboard():
    rows = [
        row
        for row in range_model_history_rows()
        if (
            row.get("model_version") == RANGE_MODEL_VERSION
            and parse_bool_int(row.get("scored", "0"))
        )
    ]
    by_model = {}

    for row in rows:
        model_name = row.get("candidate_model", "") or "unknown"
        by_model.setdefault(model_name, []).append(row)

    leaderboard = []

    for model_name, model_rows in by_model.items():
        recent_rows = model_rows[-100:]
        stats = accuracy_for_rows(recent_rows)
        all_stats = accuracy_for_rows(model_rows[-300:])
        leaderboard.append(
            {
                "candidate_model": model_name,
                "checked": stats["checked"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy"],
                "long_checked": all_stats["checked"],
                "long_accuracy": all_stats["accuracy"],
            }
        )

    leaderboard.sort(
        key=lambda item: (
            item["accuracy"] if item["accuracy"] is not None else -1,
            item["checked"],
        ),
        reverse=True,
    )
    active = None

    for item in leaderboard:
        if item.get("candidate_model") == "baseline":
            continue

        if (
            item.get("checked", 0) >= MIN_LEADERBOARD_CHECKED
            and item.get("accuracy") is not None
            and item["accuracy"] >= MIN_LEADERBOARD_ACCURACY
            and item.get("long_checked", 0) >= MIN_LEADERBOARD_LONG_CHECKED
            and item.get("long_accuracy") is not None
            and item["long_accuracy"] >= MIN_LEADERBOARD_LONG_ACCURACY
        ):
            active = item
            break

    return {
        "items": leaderboard,
        "active": active,
    }


def build_self_learning_status(range_leaderboard):
    items = range_leaderboard.get("items", [])
    active = range_leaderboard.get("active")
    best = items[0] if items else None
    checked = max(
        (
            int(item.get("checked", 0))
            for item in items
        ),
        default=0,
    )
    remaining = max(
        0,
        MIN_LEADERBOARD_CHECKED - checked,
    )

    return {
        "enabled": True,
        "updates_after_each_round": True,
        "status": "active" if active else "learning",
        "active_model": active,
        "best_model": best,
        "models_tracked": len(items),
        "scored_rounds": checked,
        "minimum_scored_rounds": MIN_LEADERBOARD_CHECKED,
        "rounds_until_auto_select": remaining,
        "minimum_accuracy": MIN_LEADERBOARD_ACCURACY,
    }


def build_accuracy_summary():
    rows = prediction_history_rows()
    range_rows = range_prediction_history_rows()
    range_leaderboard = build_range_model_leaderboard()
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
    useful_targets = [
        {
            "target": target,
            **item,
        }
        for target, item in target_accuracy.items()
        if item.get("checked", 0) >= 50 and item.get("skill") is not None
    ]
    best_target = (
        max(
            useful_targets,
            key=lambda item: item["skill"],
        )
        if useful_targets
        else None
    )

    return {
        "windows": windows,
        "range": accuracy_for_rows(scored_range_rows[-300:]),
        "range_skipped": len(current_range_rows) - len(scored_range_rows),
        "clear": accuracy_for_rows(clear_rows[-300:]),
        "weak": accuracy_for_rows(weak_rows[-300:]),
        "targets": target_accuracy,
        "best_target": best_target,
        "range_model_leaderboard": range_leaderboard["items"],
        "active_range_model": range_leaderboard["active"],
        "self_learning": build_self_learning_status(
            range_leaderboard
        ),
    }


def cached_accuracy_summary():
    signature = (
        file_signature(PREDICTION_HISTORY_PATH),
        file_signature(RANGE_PREDICTION_HISTORY_PATH),
        file_signature(RANGE_MODEL_HISTORY_PATH),
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


def candidate_from_active_model(range_estimate, active_model):
    if not range_estimate or not active_model:
        return None

    model_name = active_model.get("candidate_model")

    if not model_name:
        return None

    for candidate in range_estimate.get("model_candidates", []):
        if candidate.get("candidate_model") == model_name:
            return candidate

    return None


def apply_active_range_model(report, accuracy_summary):
    active_model = (
        accuracy_summary or {}
    ).get("active_range_model")

    if not active_model:
        return report

    next_round = report.get("next_round", {})
    range_estimate = next_round.get("range_estimate")
    candidate = candidate_from_active_model(
        range_estimate,
        active_model,
    )

    if not candidate or not is_actionable_candidate_range(candidate):
        return report

    selected = {
        **range_estimate,
        **candidate,
        "coverage_range": (
            range_estimate or {}
        ).get("coverage_range"),
        "model_candidates": (
            range_estimate or {}
        ).get("model_candidates", []),
        "selected_by": "range_model_leaderboard",
        "active_model": active_model,
    }
    report["next_round"]["range_estimate"] = selected
    return report


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
    majority_accuracy = majority_accuracy_for_calls(component_calls)

    if len(component_calls) < 12:
        return {
            "profile": "balanced",
            "decision_margin": learn_decision_margin(calls),
            "profile_accuracy": None,
            "majority_accuracy": majority_accuracy,
            "profile_skill": None,
        }

    best = {
        "profile": "balanced",
        "decision_margin": 0,
        "profile_accuracy": -1,
        "majority_accuracy": majority_accuracy,
        "profile_skill": None,
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
                    "majority_accuracy": majority_accuracy,
                    "profile_skill": (
                        accuracy - majority_accuracy
                        if majority_accuracy is not None
                        else None
                    ),
                    "balanced_accuracy": balanced_accuracy,
                    "brier": brier,
                }

    return best


def majority_accuracy_for_calls(calls):
    if not calls:
        return None

    positives = sum(
        1
        for call in calls
        if bool(call.get("actual_high"))
    )
    negatives = len(calls) - positives
    return max(positives, negatives) / len(calls)


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
            "profile_majority_accuracy": strategy.get("majority_accuracy"),
            "profile_skill": strategy.get("profile_skill"),
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
        "target_confidence": range_estimate.get("target_confidence"),
        "cashout_target": range_estimate.get("cashout_target"),
        "coverage_gap": range_estimate.get("coverage_gap"),
        "bucket_count": range_estimate.get("bucket_count"),
        "coverage_range": range_estimate.get("coverage_range"),
        "model_candidates": range_estimate.get("model_candidates", []),
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


def is_actionable_candidate_range(range_prediction):
    if not range_prediction:
        return False

    minimum = range_prediction.get("minimum")
    maximum = range_prediction.get("maximum")

    if minimum is None or maximum is None:
        return False

    try:
        return float(maximum) <= ACTIONABLE_LEADERBOARD_MAXIMUM
    except (TypeError, ValueError):
        return False


def score_range_model_candidate(range_prediction, actual_multiplier):
    if not range_prediction:
        return None

    minimum = range_prediction.get("minimum")
    maximum = range_prediction.get("maximum")

    if minimum is None:
        return None

    try:
        minimum_value = float(minimum)
        maximum_value = None if maximum is None else float(maximum)
        actual = float(actual_multiplier)
    except (TypeError, ValueError):
        return None

    correct = actual >= minimum_value and (
        maximum_value is None
        or actual < maximum_value
    )
    scored = is_actionable_candidate_range(range_prediction)

    return {
        "candidate_model": range_prediction.get("candidate_model", "unknown"),
        "label": range_prediction.get("label", ""),
        "minimum": minimum_value,
        "maximum": maximum_value,
        "probability": range_prediction.get("probability"),
        "confidence": range_prediction.get("confidence", ""),
        "source": range_prediction.get("source", ""),
        "range_type": range_prediction.get("range_type", ""),
        "clear_signal": bool(range_prediction.get("clear_signal", False)),
        "clear_reason": range_prediction.get("clear_reason", ""),
        "scored": scored,
        "correct": correct if scored else None,
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
    range_candidate_results = []
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

    for candidate in pending.get("range_prediction", {}).get("model_candidates", []):
        candidate_result = score_range_model_candidate(
            candidate,
            actual_multiplier,
        )

        if not candidate_result:
            continue

        append_range_model_result(
            [
                now_string(),
                pending.get("predicted_at", ""),
                after_round_count,
                actual_round_count,
                actual_round.get("timestamp", ""),
                RANGE_MODEL_VERSION,
                candidate_result.get("candidate_model", ""),
                candidate_result.get("label", ""),
                f"{float(candidate_result['minimum']):.2f}",
                (
                    ""
                    if candidate_result.get("maximum") is None
                    else f"{float(candidate_result['maximum']):.2f}"
                ),
                (
                    ""
                    if candidate_result.get("probability") is None
                    else f"{float(candidate_result['probability']):.6f}"
                ),
                candidate_result.get("confidence", ""),
                candidate_result.get("source", ""),
                candidate_result.get("range_type", ""),
                int(candidate_result.get("clear_signal", False)),
                candidate_result.get("clear_reason", ""),
                int(candidate_result.get("scored", False)),
                f"{actual_multiplier:.2f}",
                (
                    ""
                    if candidate_result.get("correct") is None
                    else int(candidate_result["correct"])
                ),
            ]
        )
        range_candidate_results.append(
            candidate_result
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
        "range_candidate_results": range_candidate_results,
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
    next_round = copy.deepcopy(report["next_round"])
    range_estimate = next_round.get("range_estimate")

    if range_estimate:
        range_estimate.setdefault("model_version", RANGE_MODEL_VERSION)

    return {
        "generated_at": report["generated_at"],
        "_backtest_key": report.get("_backtest_key"),
        "warning": report["warning"],
        "data_selection": report.get("data_selection", {}),
        "ingest": ingest_status(rounds),
        "data_quality": data_quality_snapshot(rounds),
        "collector_status": collector_status_snapshot(),
        "round_context": latest_round_context(),
        "big_rounds": big_round_watch(rounds),
        "timing_insights": timing_insights(rounds),
        "edge_audit": edge_audit_snapshot(rounds),
        "strategy_audit": strategy_audit_snapshot(rounds),
        "sequence_watch": sequence_watch_snapshot(rounds),
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
        "next_round": next_round,
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


def compact_live_prediction(prediction):
    return {
        "target": prediction.get("target"),
        "probability": prediction.get("probability"),
        "baseline_probability": prediction.get("baseline_probability"),
        "edge": prediction.get("edge"),
        "predicted_high": prediction.get("predicted_high"),
        "confidence": prediction.get("confidence"),
        "signal": prediction.get("signal"),
        "clear_signal": prediction.get("clear_signal"),
        "clear_reason": prediction.get("clear_reason"),
    }


def compact_live_tracking(tracking):
    if not tracking:
        return None

    pending = tracking.get("pending")
    compact_pending = None

    if pending:
        compact_pending = {
            "predicted_at": pending.get("predicted_at"),
            "after_round_count": pending.get("after_round_count"),
            "latest_multiplier": pending.get("latest_multiplier"),
            "range_prediction": pending.get("range_prediction"),
        }

    last_result = tracking.get("last_result")
    compact_last_result = None

    if last_result:
        compact_last_result = {
            "score_id": last_result.get("score_id"),
            "actual_multiplier": last_result.get("actual_multiplier"),
            "range_result": last_result.get("range_result"),
            "results": [
                {
                    "target": item.get("target"),
                    "predicted_high": item.get("predicted_high"),
                    "actual_high": item.get("actual_high"),
                    "correct": item.get("correct"),
                    "probability": item.get("probability"),
                    "baseline_probability": item.get("baseline_probability"),
                }
                for item in last_result.get("results", [])
            ],
        }

    return {
        "pending": compact_pending,
        "metrics": {},
        "range_metrics": tracking.get("range_metrics", {}),
        "last_result": compact_last_result,
    }


def compact_live_payload(payload):
    summary = payload.get("summary", {})
    next_round = payload.get("next_round", {})

    return {
        "generated_at": payload.get("generated_at"),
        "warning": payload.get("warning"),
        "data_selection": payload.get("data_selection", {}),
        "ingest": payload.get("ingest", {}),
        "data_quality": payload.get("data_quality"),
        "collector_status": payload.get("collector_status"),
        "round_context": payload.get("round_context"),
        "big_rounds": payload.get("big_rounds"),
        "timing_insights": payload.get("timing_insights"),
        "edge_audit": payload.get("edge_audit"),
        "strategy_audit": payload.get("strategy_audit"),
        "sequence_watch": payload.get("sequence_watch"),
        "signal_quality": payload.get("signal_quality"),
        "summary": {
            "rounds": summary.get("rounds", 0),
            "latest_multiplier": summary.get("latest_multiplier"),
            "average": summary.get("average"),
            "median": summary.get("median"),
            "p90": summary.get("p90"),
            "maximum": summary.get("maximum"),
            "buckets": summary.get("buckets", {}),
        },
        "next_round": {
            "lookback": next_round.get("lookback"),
            "latest_pattern": next_round.get("latest_pattern", []),
            "pattern_match_count": next_round.get("pattern_match_count", 0),
            "range_estimate": next_round.get("range_estimate"),
            "predictions": [
                compact_live_prediction(prediction)
                for prediction in next_round.get("predictions", [])
            ],
        },
        "tracking": compact_live_tracking(
            payload.get("tracking")
        ),
        "accuracy_summary": payload.get("accuracy_summary"),
        "ml_prediction": payload.get("ml_prediction"),
        "ml_retrain": payload.get("ml_retrain"),
        "cache_age_ms": payload.get("cache_age_ms"),
    }


def build_dashboard_payload(query, include_backtests=True):
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
        current_config_signature = file_signature(
            CONFIG_PATH
        )
        current_ml_signature = (
            file_signature(ML_CHAMPION_METADATA_PATH),
            file_signature(ML_MANIFEST_PATH),
            file_signature(ML_REPORT_PATH),
        )
        current_pattern_signature = file_signature(
            PATTERN_DISCOVERY_PATH
        )
        current_strategy_signature = file_signature(
            STRATEGY_AUDIT_PATH
        )

        cache_key = (
            lookback,
            min_matches,
            include_backtests,
            current_csv_signature,
            current_config_signature,
            current_ml_signature,
            current_pattern_signature,
            current_strategy_signature,
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
                "data_quality": data_quality_snapshot([]),
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
                "collector_status": collector_status_snapshot(),
                "round_context": latest_round_context(),
                "big_rounds": big_round_watch([]),
                "timing_insights": timing_insights([]),
                "sequence_watch": sequence_watch_snapshot([]),
                "strategy_audit": strategy_audit_snapshot([]),
                "signal_quality": signal_quality(
                    {
                        "summary": {
                            "rounds": 0,
                        },
                        "next_round": {},
                        "timing_insights": timing_insights([]),
                    }
                ),
                "ml_prediction": current_ml_prediction(0),
                "ml_retrain": ml_retrain_status_snapshot(),
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
            include_backtests=include_backtests,
        )
        accuracy_summary = cached_accuracy_summary()
        report = apply_active_range_model(
            report,
            accuracy_summary,
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
        payload["ml_prediction"] = current_ml_prediction(
            len(all_rounds)
        )
        try:
            from ml_auto_retrain import update_live_prediction_tracking

            update_live_prediction_tracking(
                rounds,
                payload["ml_prediction"],
                source="dashboard",
            )
        except Exception:
            pass
        payload["ml_retrain"] = ml_retrain_status_snapshot()
        payload["signal_quality"] = signal_quality(
            payload
        )
        payload["_cached_at"] = time.monotonic()

        if len(DASHBOARD_CACHE) >= CACHE_MAX_ITEMS:
            DASHBOARD_CACHE.clear()

        DASHBOARD_CACHE[cache_key] = copy.deepcopy(payload)

        return refresh_cached_ingest(payload)


def warm_dashboard_cache_once():
    for query in WARM_CACHE_QUERIES:
        build_dashboard_payload(
            query,
            include_backtests=False,
        )


def start_cache_warmer():
    def worker():
        last_signature = object()

        while True:
            try:
                current_signature = (
                    csv_signature(),
                    file_signature(CONFIG_PATH),
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

        if parsed.path == "/api/live":
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

        if parsed.path == "/api/live":
            self.send_json(
                compact_live_payload(
                    build_dashboard_payload(
                        parse_qs(parsed.query),
                        include_backtests=False,
                    )
                )
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
    parser.add_argument(
        "--disable-ml-auto-retrain",
        action="store_true",
        help="Do not refresh ML models automatically while the dashboard runs.",
    )
    parser.add_argument(
        "--ml-retrain-min-new-rounds",
        type=int,
        default=None,
        help="Retrain after this many new CSV rounds since the last successful ML train.",
    )
    parser.add_argument(
        "--ml-minimum-training-rounds",
        type=int,
        default=None,
        help="Minimum valid rounds required before ML retraining can run.",
    )
    parser.add_argument(
        "--ml-promotion-min-skill-improvement",
        type=float,
        default=None,
        help="Minimum Brier skill improvement required to promote a challenger.",
    )
    parser.add_argument(
        "--ml-retrain-check-seconds",
        type=int,
        default=None,
        help="How often the dashboard checks whether ML retraining is due.",
    )
    args = parser.parse_args()
    config = load_json_file(CONFIG_PATH) or {}
    ml_auto_retrain_enabled = bool(
        config.get(
            "ml_auto_retrain",
            True,
        )
    ) and not args.disable_ml_auto_retrain
    ml_retrain_min_new_rounds = (
        args.ml_retrain_min_new_rounds
        if args.ml_retrain_min_new_rounds is not None
        else int(
            config.get(
                "ml_retrain_every_rounds",
                config.get(
                    "ml_retrain_min_new_rounds",
                    500,
                ),
            )
        )
    )
    ml_minimum_training_rounds = (
        args.ml_minimum_training_rounds
        if args.ml_minimum_training_rounds is not None
        else int(
            config.get(
                "ml_minimum_training_rounds",
                3000,
            )
        )
    )
    ml_promotion_min_skill_improvement = (
        args.ml_promotion_min_skill_improvement
        if args.ml_promotion_min_skill_improvement is not None
        else float(
            config.get(
                "ml_promotion_min_skill_improvement",
                0.005,
            )
        )
    )
    ml_include_context = bool(
        config.get(
            "ml_include_context",
            False,
        )
    )
    ml_retrain_check_seconds = (
        args.ml_retrain_check_seconds
        if args.ml_retrain_check_seconds is not None
        else int(
            config.get(
                "ml_retrain_check_seconds",
                30,
            )
        )
    )
    edge_audit_auto_refresh_enabled = bool(
        config.get(
            "edge_audit_auto_refresh",
            True,
        )
    )
    edge_audit_every_rounds = int(
        config.get(
            "edge_audit_every_rounds",
            250,
        )
    )
    edge_audit_check_seconds = int(
        config.get(
            "edge_audit_check_seconds",
            60,
        )
    )
    edge_audit_min_sample = int(
        config.get(
            "edge_audit_min_sample",
            80,
        )
    )
    edge_audit_top = int(
        config.get(
            "edge_audit_top",
            20,
        )
    )
    edge_audit_walk_forward_folds = int(
        config.get(
            "edge_audit_walk_forward_folds",
            6,
        )
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        DashboardHandler,
    )
    start_cache_warmer()
    start_ml_auto_retrainer(
        enabled=ml_auto_retrain_enabled,
        min_new_rounds=ml_retrain_min_new_rounds,
        check_seconds=ml_retrain_check_seconds,
        minimum_training_rounds=ml_minimum_training_rounds,
        include_context=ml_include_context,
        promotion_min_skill_improvement=ml_promotion_min_skill_improvement,
    )
    start_edge_audit_refresher(
        enabled=edge_audit_auto_refresh_enabled,
        every_rounds=edge_audit_every_rounds,
        check_seconds=edge_audit_check_seconds,
        min_sample=edge_audit_min_sample,
        top=edge_audit_top,
        walk_forward_folds=edge_audit_walk_forward_folds,
    )

    print(f"Dashboard running at http://{args.host}:{args.port}")
    if ml_auto_retrain_enabled:
        print(
            "ML auto-retrain enabled "
            f"(every {ml_retrain_min_new_rounds} new rounds)."
        )
    else:
        print("ML auto-retrain disabled.")
    if edge_audit_auto_refresh_enabled:
        print(
            "Edge audit auto-refresh enabled "
            f"(every {edge_audit_every_rounds} new rounds)."
        )
    else:
        print("Edge audit auto-refresh disabled.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
