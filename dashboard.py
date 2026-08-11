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
CONFIG_PATH = ROOT / "config.json"
PREDICTION_STATE_PATH = DATA_DIR / "prediction_state.json"
PREDICTION_HISTORY_PATH = DATA_DIR / "prediction_history.csv"
RANGE_PREDICTION_HISTORY_PATH = DATA_DIR / "range_prediction_history.csv"
RANGE_MODEL_HISTORY_PATH = DATA_DIR / "range_model_history.csv"
ROUND_CONTEXT_PATH = DATA_DIR / "round_context.csv"
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
TRACKING_LOCK = threading.Lock()
BACKTEST_LOCK = threading.Lock()
DISPLAY_MONEY_LOCK = threading.Lock()
ML_PREDICTION_LOCK = threading.Lock()
ML_RETRAIN_LOCK = threading.Lock()
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
ML_RETRAIN_STATE = {
    "thread": None,
}
ML_RETRAIN_SETTINGS = {
    "enabled": True,
    "min_new_rounds": 500,
    "check_seconds": 30,
    "minimum_training_rounds": 3000,
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
    payload["ml_retrain"] = ml_retrain_status_snapshot()

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
            include_context=False,
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


def ml_retrain_status_snapshot():
    try:
        from ml_auto_retrain import scheduler_status

        status = scheduler_status(
            {
                "ml_retrain_every_rounds": ML_RETRAIN_SETTINGS["min_new_rounds"],
                "ml_minimum_training_rounds": ML_RETRAIN_SETTINGS[
                    "minimum_training_rounds"
                ],
                "ml_promotion_min_skill_improvement": ML_RETRAIN_SETTINGS[
                    "promotion_min_skill_improvement"
                ],
            },
            CSV_PATH,
        )
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
        finished_rounds = csv_data_row_count()

        if result.returncode == 0:
            update_ml_retrain_status(
                status="complete",
                last_finished_at=now_string(),
                last_success_at=now_string(),
                last_trained_rounds=finished_rounds,
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

    with ML_RETRAIN_LOCK:
        thread = ML_RETRAIN_STATE.get("thread")

        if thread and thread.is_alive():
            update_ml_retrain_status(
                status="training",
                current_rounds=current_rounds,
                last_trained_rounds=last_trained_rounds,
            )
            return

        if not retrain_due:
            update_ml_retrain_status(
                status="waiting",
                current_rounds=current_rounds,
                last_trained_rounds=last_trained_rounds,
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
    promotion_min_skill_improvement=0.005,
):
    ML_RETRAIN_SETTINGS.update(
        {
            "enabled": bool(enabled),
            "min_new_rounds": max(1, int(min_new_rounds)),
            "check_seconds": max(5, int(check_seconds)),
            "minimum_training_rounds": max(100, int(minimum_training_rounds)),
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
        if (
            item.get("checked", 0) >= MIN_LEADERBOARD_CHECKED
            and item.get("accuracy") is not None
            and item["accuracy"] >= MIN_LEADERBOARD_ACCURACY
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
        "round_context": latest_round_context(),
        "big_rounds": big_round_watch(rounds),
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
        "round_context": payload.get("round_context"),
        "big_rounds": payload.get("big_rounds"),
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

        cache_key = (
            lookback,
            min_matches,
            include_backtests,
            current_csv_signature,
            current_config_signature,
            current_ml_signature,
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
                "big_rounds": big_round_watch([]),
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
        promotion_min_skill_improvement=ml_promotion_min_skill_improvement,
    )

    print(f"Dashboard running at http://{args.host}:{args.port}")
    if ml_auto_retrain_enabled:
        print(
            "ML auto-retrain enabled "
            f"(every {ml_retrain_min_new_rounds} new rounds)."
        )
    else:
        print("ML auto-retrain disabled.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
