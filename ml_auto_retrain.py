"""Controlled ML retraining, champion promotion, and live tracking.

This module deliberately separates two ideas:

1. Training challengers on a schedule.
2. Promoting a challenger only when it beats the current champion on honest
   chronological validation.

It never tunes settings after seeing an untouched holdout result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ml_backtest import (
    WalkForwardConfig,
    calculate_metrics,
    fixed_model_registry,
    run_walk_forward_backtest,
)
from ml_features import (
    DEFAULT_ROUNDS_PATH,
    DEFAULT_TARGETS,
    FEATURE_SCHEMA_VERSION,
    clean_rounds,
    load_feature_dataset,
    target_name,
    write_json,
)
from ml_train import (
    MANIFEST_PATH as LEGACY_MANIFEST_PATH,
    REPORT_PATH as LEGACY_REPORT_PATH,
    evaluate_holdout,
    extract_feature_importance,
    fit_final_model,
    load_model,
    model_filename,
    save_model,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
CHALLENGERS_DIR = MODELS_DIR / "challengers"
ARCHIVE_DIR = MODELS_DIR / "archive"
CHAMPION_METADATA_PATH = MODELS_DIR / "champion.json"
TRAINING_LOCK_PATH = DATA_DIR / "ml_training.lock"
RETRAIN_STATE_PATH = DATA_DIR / "ml_retrain_state.json"
LIVE_STATE_PATH = DATA_DIR / "ml_live_state.json"
LIVE_PREDICTIONS_PATH = DATA_DIR / "ml_live_predictions.csv"

MODEL_VERSION = "ml-champion-v1"
DEFAULT_CONFIG = {
    "ml_retrain_every_rounds": 500,
    "ml_minimum_training_rounds": 3000,
    "ml_promotion_min_skill_improvement": 0.005,
    "ml_promotion_min_evaluated_rounds": 500,
    "ml_max_calibration_deterioration": 0.03,
    "ml_max_fold_brier_skill_std": 0.08,
    "ml_live_rollback_min_predictions": 200,
    "ml_live_rollback_min_brier_skill": -0.03,
}
LIVE_HISTORY_MODELS = {
    "historical_frequency",
    "recent_frequency_100",
    "majority",
}

LIVE_PREDICTION_FIELDS = [
    "prediction_time",
    "resolved_time",
    "target",
    "round_id_predicted_for",
    "round_number_predicted_for",
    "model_version",
    "model_type",
    "predicted_probability",
    "historical_baseline",
    "predicted_class",
    "actual_multiplier",
    "actual_class",
    "correct",
    "brier_loss",
    "source",
]


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def version_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict:
    if not Path(path).exists():
        return {}

    try:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    os.replace(tmp_path, path)


def root_relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


@contextmanager
def training_lock(path: Path = TRAINING_LOCK_PATH):
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "started_at": now_string(),
    }

    try:
        fd = os.open(
            str(path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise RuntimeError(f"Training already running: {path}") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.write("\n")
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_config_file(path: Optional[Path] = None) -> dict:
    path = path or ROOT / "config.json"
    raw_config = load_json(path)
    config = DEFAULT_CONFIG.copy()
    config.update(raw_config)

    if (
        "ml_retrain_min_new_rounds" in raw_config
        and "ml_retrain_every_rounds" not in raw_config
    ):
        config["ml_retrain_every_rounds"] = raw_config["ml_retrain_min_new_rounds"]

    return config


def normalized_config(config: Optional[dict] = None) -> dict:
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)

    return {
        "ml_retrain_every_rounds": max(1, int(merged["ml_retrain_every_rounds"])),
        "ml_minimum_training_rounds": max(100, int(merged["ml_minimum_training_rounds"])),
        "ml_promotion_min_skill_improvement": max(
            0.0,
            float(merged["ml_promotion_min_skill_improvement"]),
        ),
        "ml_promotion_min_evaluated_rounds": max(
            50,
            int(merged["ml_promotion_min_evaluated_rounds"]),
        ),
        "ml_max_calibration_deterioration": max(
            0.0,
            float(merged["ml_max_calibration_deterioration"]),
        ),
        "ml_max_fold_brier_skill_std": max(
            0.0,
            float(merged["ml_max_fold_brier_skill_std"]),
        ),
        "ml_live_rollback_min_predictions": max(
            50,
            int(merged["ml_live_rollback_min_predictions"]),
        ),
        "ml_live_rollback_min_brier_skill": float(
            merged["ml_live_rollback_min_brier_skill"]
        ),
    }


def csv_valid_round_count(csv_path: Path = DEFAULT_ROUNDS_PATH) -> int:
    _, quality = clean_rounds(csv_path)
    return int(quality.get("valid_rows", 0))


def save_retrain_state(**updates) -> dict:
    state = load_json(RETRAIN_STATE_PATH)
    state.update(updates)
    state["last_checked_at"] = now_string()
    atomic_write_json(RETRAIN_STATE_PATH, state)
    return state


def champion_target_items(champion: dict) -> dict:
    return champion.get("targets", {}) if isinstance(champion, dict) else {}


def load_champion_metadata() -> dict:
    return load_json(CHAMPION_METADATA_PATH)


def write_champion_manifest(champion: dict) -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    targets = champion_target_items(champion)
    feature_names = champion.get("feature_names", [])
    manifest = {
        "model_version": champion.get("model_version", MODEL_VERSION),
        "feature_schema_version": champion.get(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        ),
        "generated_at": now_string(),
        "feature_names": feature_names,
        "targets": {},
    }

    for label, item in targets.items():
        manifest["targets"][label] = {
            "target": item.get("target"),
            "target_column": item.get("target_column"),
            "model_name": item.get("model_name"),
            "model_path": item.get("model_path"),
            "champion_model_path": item.get("champion_model_path"),
            "historical_baseline_probability": item.get(
                "historical_baseline_probability"
            ),
            "version": item.get("version"),
        }

    atomic_write_json(CHAMPION_DIR / "manifest.json", manifest)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def bootstrap_champion_from_legacy() -> dict:
    champion = load_champion_metadata()
    if champion.get("targets"):
        return champion

    legacy_manifest = load_json(LEGACY_MANIFEST_PATH)
    if not legacy_manifest.get("targets"):
        return {}

    legacy_report = load_json(LEGACY_REPORT_PATH)
    version = f"legacy_{version_string()}"
    archive_version_dir = ARCHIVE_DIR / version
    archive_version_dir.mkdir(parents=True, exist_ok=True)
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    targets = {}

    for label, item in legacy_manifest.get("targets", {}).items():
        source_path = MODELS_DIR / item.get("model_path", "")
        archive_path = archive_version_dir / model_filename(label)
        champion_path = CHAMPION_DIR / model_filename(label)

        if source_path.exists():
            copy_if_exists(source_path, archive_path)
            copy_if_exists(source_path, champion_path)
        else:
            archive_path = source_path
            champion_path = source_path

        validation_metrics = (
            legacy_report.get("selected_models", {})
            .get(label, {})
            .get("validation_metrics", {})
        )
        holdout_metrics = (
            legacy_report.get("final_holdout_results", {})
            .get(label, {})
            .get("holdout_metrics", {})
        )
        data_quality = legacy_report.get("data_quality", {})
        dataset_statistics = legacy_report.get("dataset_statistics", {})
        trained_rounds = int(
            data_quality.get(
                "valid_rows",
                dataset_statistics.get("feature_rows", 0),
            )
            or 0
        )
        model_name = item.get("model_name", "historical_frequency")

        targets[label] = {
            "target": item.get("target"),
            "target_column": item.get("target_column"),
            "model_name": model_name,
            "model_type": model_name,
            "model_path": root_relative(archive_path),
            "champion_model_path": root_relative(champion_path),
            "save_format": item.get("save_format"),
            "trained_rounds": trained_rounds,
            "trained_at": legacy_manifest.get("generated_at"),
            "promoted_at": now_string(),
            "version": version,
            "source": "legacy_bootstrap",
            "feature_names": legacy_manifest.get("feature_names", []),
            "historical_baseline_probability": item.get(
                "historical_baseline_probability"
            ),
            "validation_metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "brier_skill": validation_metrics.get("brier_skill_score"),
            "balanced_accuracy": validation_metrics.get("balanced_accuracy"),
            "roc_auc": validation_metrics.get("roc_auc"),
            "calibration_error": validation_metrics.get("calibration_error"),
            "previous_champion": None,
        }

    champion = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": legacy_manifest.get(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        ),
        "generated_at": now_string(),
        "feature_names": legacy_manifest.get("feature_names", []),
        "targets": targets,
    }
    atomic_write_json(CHAMPION_METADATA_PATH, champion)
    write_champion_manifest(champion)
    return champion


def ensure_champion() -> dict:
    champion = load_champion_metadata()
    if champion.get("targets"):
        return champion

    return bootstrap_champion_from_legacy()


def metric_number(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_folds(candidate_metrics: dict, champion_metrics: dict, config: dict) -> Tuple[bool, str]:
    folds = candidate_metrics.get("folds") or []
    if len(folds) < 2:
        return False, "not enough folds"

    candidate_fold_values = [
        metric_number(item.get("metrics", {}), "brier_skill_score", -1.0)
        for item in folds
    ]
    champion_folds_by_id = {
        item.get("fold"): item.get("metrics", {})
        for item in champion_metrics.get("folds", [])
    }
    better_folds = 0

    for item, candidate_value in zip(folds, candidate_fold_values):
        champion_value = metric_number(
            champion_folds_by_id.get(item.get("fold"), {}),
            "brier_skill_score",
            0.0,
        )
        if candidate_value >= champion_value:
            better_folds += 1

    std_value = metric_number(
        candidate_metrics,
        "fold_brier_skill_std",
        float(np.std(candidate_fold_values)),
    )
    required_better = max(1, (len(folds) + 1) // 2)

    if better_folds < required_better:
        return False, "performance came from too few folds"

    if std_value > config["ml_max_fold_brier_skill_std"]:
        return False, "fold performance too unstable"

    return True, "stable folds"


def should_promote(
    candidate_name: str,
    candidate_metrics: dict,
    champion_name: Optional[str],
    champion_metrics: dict,
    holdout_metrics: dict,
    config: dict,
) -> Tuple[bool, str, dict]:
    if not candidate_metrics:
        return False, "no candidate metrics", {}

    if champion_name and candidate_name == champion_name:
        return False, "candidate is already champion model type", {}

    evaluated = int(candidate_metrics.get("predictions", 0) or 0)
    if evaluated < config["ml_promotion_min_evaluated_rounds"]:
        return False, "not enough evaluated validation rounds", {}

    candidate_brier = metric_number(candidate_metrics, "brier_skill_score", -1.0)
    champion_brier = metric_number(champion_metrics, "brier_skill_score", 0.0)
    candidate_balanced = metric_number(candidate_metrics, "balanced_accuracy", 0.0)
    champion_balanced = metric_number(champion_metrics, "balanced_accuracy", 0.5)
    candidate_calibration = metric_number(candidate_metrics, "calibration_error", 1.0)
    champion_calibration = metric_number(champion_metrics, "calibration_error", 0.08)
    holdout_brier = metric_number(holdout_metrics, "brier_skill_score", -1.0)
    holdout_balanced_skill = metric_number(
        holdout_metrics,
        "balanced_accuracy_skill",
        -1.0,
    )
    min_improvement = config["ml_promotion_min_skill_improvement"]
    reasons = {
        "candidate_brier_skill": candidate_brier,
        "champion_brier_skill": champion_brier,
        "brier_skill_improvement": candidate_brier - champion_brier,
        "candidate_balanced_accuracy": candidate_balanced,
        "champion_balanced_accuracy": champion_balanced,
        "candidate_calibration_error": candidate_calibration,
        "champion_calibration_error": champion_calibration,
        "holdout_brier_skill": holdout_brier,
        "holdout_balanced_accuracy_skill": holdout_balanced_skill,
    }

    if candidate_brier < champion_brier + min_improvement:
        return False, "brier skill improvement too small", reasons

    if candidate_balanced < champion_balanced:
        return False, "balanced accuracy did not improve", reasons

    if candidate_brier <= 0:
        return False, "candidate did not beat historical baseline", reasons

    if candidate_calibration > champion_calibration + config["ml_max_calibration_deterioration"]:
        return False, "calibration deteriorated too much", reasons

    if candidate_calibration > 0.12:
        return False, "calibration error too high", reasons

    folds_ok, fold_reason = stable_folds(candidate_metrics, champion_metrics, config)
    if not folds_ok:
        return False, fold_reason, reasons

    if holdout_brier < -0.002:
        return False, "untouched holdout brier skill deteriorated", reasons

    if holdout_balanced_skill < -0.02:
        return False, "untouched holdout balanced accuracy deteriorated", reasons

    return True, "challenger clearly beat champion", reasons


def evaluate_heuristic_benchmark(
    frame,
    targets: Iterable[float],
    min_history: int = 100,
    max_rows: int = 1000,
) -> Dict[str, dict]:
    """Evaluate the existing heuristic ensemble as a benchmark only."""

    try:
        from aviator_analyzer import ensemble_prediction_for_target
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
        }

    values = frame["multiplier"].astype(float).tolist()
    benchmark = {}

    for target in targets:
        records = []
        label = f"{float(target):.2f}"
        start_index = max(max(min_history, 10), len(values) - max_rows)
        for index in range(start_index, len(values)):
            previous_values = values[:index]
            actual = int(values[index] >= float(target))
            try:
                probability = float(
                    ensemble_prediction_for_target(
                        previous_values,
                        lookback=2,
                        target=float(target),
                    )["probability"]
                )
            except Exception:
                probability = float(np.mean([value >= float(target) for value in previous_values]))
            baseline = float(np.mean([value >= float(target) for value in previous_values]))
            records.append(
                {
                    "actual": actual,
                    "probability": probability,
                    "predicted": int(probability >= 0.5),
                    "baseline_probability": baseline,
                    "majority_prediction": int(baseline >= 0.5),
                }
            )
        metrics = calculate_metrics(records)
        benchmark[label] = metrics

    return benchmark


def train_challengers(
    csv_path: Path = DEFAULT_ROUNDS_PATH,
    targets: Sequence[float] = tuple(DEFAULT_TARGETS),
    config: Optional[dict] = None,
    force: bool = False,
    reason: str = "scheduler",
) -> dict:
    config = normalized_config(config)
    version = version_string()

    with training_lock():
        champion = ensure_champion()
        champion_targets = champion_target_items(champion)
        valid_rounds = csv_valid_round_count(csv_path)
        last_trained_rounds = max(
            [
                int(item.get("trained_rounds", 0) or 0)
                for item in champion_targets.values()
            ]
            or [0]
        )
        new_rounds = valid_rounds - last_trained_rounds

        if valid_rounds < config["ml_minimum_training_rounds"]:
            state = save_retrain_state(
                status="waiting_for_minimum_rounds",
                current_rounds=valid_rounds,
                last_trained_rounds=last_trained_rounds,
                new_rounds_since_train=max(0, new_rounds),
                message="minimum training rounds not reached",
            )
            return {
                "status": "skipped",
                "reason": state["message"],
                "state": state,
            }

        if not force and new_rounds < config["ml_retrain_every_rounds"]:
            state = save_retrain_state(
                status="waiting",
                current_rounds=valid_rounds,
                last_trained_rounds=last_trained_rounds,
                new_rounds_since_train=max(0, new_rounds),
                rounds_until_next_train=config["ml_retrain_every_rounds"] - max(0, new_rounds),
                message="retrain threshold not reached",
            )
            return {
                "status": "skipped",
                "reason": state["message"],
                "state": state,
            }

        save_retrain_state(
            status="training",
            last_started_at=now_string(),
            current_rounds=valid_rounds,
            last_trained_rounds=last_trained_rounds,
            new_rounds_since_train=max(0, new_rounds),
            reason=reason,
        )

        dataset = load_feature_dataset(
            csv_path,
            targets=targets,
            min_history=100,
            include_context=False,
        )
        frame = dataset.frame
        if frame.empty:
            state = save_retrain_state(
                status="failed",
                last_error="No feature rows available.",
            )
            return {
                "status": "failed",
                "state": state,
            }

        holdout_size = max(1, int(len(frame) * 0.18))
        trainval = frame.iloc[: len(frame) - holdout_size].reset_index(drop=True)
        holdout = frame.iloc[len(frame) - holdout_size :].reset_index(drop=True)
        wf_config = WalkForwardConfig(
            min_train=5000,
            test_size=1000,
            holdout_fraction=0.18,
            max_folds=None,
            min_history=100,
            include_context=False,
        )
        models = fixed_model_registry(calibration_methods=("uncalibrated",))
        records, validation_summary, selected = run_walk_forward_backtest(
            trainval,
            dataset.feature_names,
            targets=targets,
            config=wf_config,
            models=models,
            calibration_methods=("uncalibrated",),
        )
        challenger_dir = CHALLENGERS_DIR / version
        challenger_dir.mkdir(parents=True, exist_ok=True)
        X_trainval = trainval[dataset.feature_names].astype(float)
        X_holdout = holdout[dataset.feature_names].astype(float)
        X_all = frame[dataset.feature_names].astype(float)
        champion_changed = False
        promoted_targets = []
        kept_targets = []
        challenger_results = {}
        updated_champion = {
            **champion,
            "model_version": MODEL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": dataset.feature_names,
            "generated_at": now_string(),
            "targets": dict(champion_targets),
        }

        for target in targets:
            label = f"{float(target):.2f}"
            selected_item = selected.get(label)
            if not selected_item:
                kept_targets.append(
                    {
                        "target": label,
                        "reason": "no challenger selected",
                    }
                )
                continue

            candidate_name = selected_item["model"]
            candidate_model = models[candidate_name]
            target_column = target_name(float(target))
            y_trainval = trainval[target_column].astype(int).to_numpy()
            y_holdout = holdout[target_column].astype(int).to_numpy()
            y_all = frame[target_column].astype(int).to_numpy()
            baseline_probability = float(y_trainval.mean())
            eval_estimator = fit_final_model(
                candidate_model,
                X_trainval,
                y_trainval,
            )
            holdout_metrics = evaluate_holdout(
                eval_estimator,
                X_holdout,
                y_holdout,
                baseline_probability,
            )
            final_estimator = fit_final_model(
                candidate_model,
                X_all,
                y_all,
            )
            challenger_path = challenger_dir / model_filename(label)
            save_format = save_model(challenger_path, final_estimator)
            load_model(challenger_path)
            candidate_metrics = selected_item.get("metrics", {})
            champion_entry = champion_targets.get(label, {})
            champion_name = champion_entry.get("model_name")
            champion_metrics = (
                validation_summary.get(label, {}).get(champion_name, {})
                if champion_name
                else {}
            )
            should_upgrade, promotion_reason, promotion_details = should_promote(
                candidate_name,
                candidate_metrics,
                champion_name,
                champion_metrics,
                holdout_metrics,
                config,
            )
            importance = extract_feature_importance(
                final_estimator,
                dataset.feature_names,
            )
            challenger_results[label] = {
                "target": float(target),
                "candidate_model": candidate_name,
                "candidate_model_path": root_relative(challenger_path),
                "save_format": save_format,
                "validation_metrics": candidate_metrics,
                "champion_model": champion_name,
                "champion_validation_metrics": champion_metrics,
                "holdout_metrics": holdout_metrics,
                "promotion_recommended": should_upgrade,
                "promotion_reason": promotion_reason,
                "promotion_details": promotion_details,
                "feature_importance": importance,
            }

            if should_upgrade:
                archive_path = ARCHIVE_DIR / version / model_filename(label)
                champion_path = CHAMPION_DIR / model_filename(label)
                copy_if_exists(challenger_path, archive_path)
                copy_if_exists(challenger_path, champion_path)
                new_entry = {
                    "target": float(target),
                    "target_column": target_column,
                    "model_name": candidate_name,
                    "model_type": candidate_name,
                    "model_path": root_relative(archive_path),
                    "champion_model_path": root_relative(champion_path),
                    "save_format": save_format,
                    "trained_rounds": valid_rounds,
                    "trained_at": now_string(),
                    "promoted_at": now_string(),
                    "version": version,
                    "source": "challenger_promotion",
                    "feature_names": dataset.feature_names,
                    "historical_baseline_probability": float(np.mean(y_all)),
                    "validation_metrics": candidate_metrics,
                    "holdout_metrics": holdout_metrics,
                    "brier_skill": candidate_metrics.get("brier_skill_score"),
                    "balanced_accuracy": candidate_metrics.get("balanced_accuracy"),
                    "roc_auc": candidate_metrics.get("roc_auc"),
                    "calibration_error": candidate_metrics.get("calibration_error"),
                    "previous_champion": champion_entry or None,
                    "promotion_details": promotion_details,
                }
                updated_champion["targets"][label] = new_entry
                promoted_targets.append(label)
                champion_changed = True
            else:
                kept_targets.append(
                    {
                        "target": label,
                        "candidate_model": candidate_name,
                        "champion_model": champion_name,
                        "reason": promotion_reason,
                        "details": promotion_details,
                    }
                )

        heuristic_benchmark = evaluate_heuristic_benchmark(
            trainval,
            targets=targets,
            min_history=100,
        )
        report = {
            "version": version,
            "generated_at": now_string(),
            "reason": reason,
            "config": config,
            "data_quality": dataset.quality_report,
            "source_report": dataset.source_report,
            "dataset_statistics": {
                "feature_rows": int(len(frame)),
                "training_rounds": int(valid_rounds),
                "selection_rows": int(len(trainval)),
                "holdout_rows": int(len(holdout)),
                "feature_count": len(dataset.feature_names),
                "targets": list(targets),
            },
            "validation_summary": validation_summary,
            "selected_models": selected,
            "challenger_results": challenger_results,
            "heuristic_ensemble_benchmark": heuristic_benchmark,
            "promoted_targets": promoted_targets,
            "kept_targets": kept_targets,
            "champion_changed": champion_changed,
            "guardrails": {
                "promotion": "Only promotes when challenger beats champion on the same walk-forward periods and passes holdout/calibration checks.",
                "holdout": "The recent holdout is used once for promotion evidence and not for tuning thresholds after results are seen.",
            },
        }
        atomic_write_json(challenger_dir / "report.json", report)

        if champion_changed:
            atomic_write_json(CHAMPION_METADATA_PATH, updated_champion)
            write_champion_manifest(updated_champion)

        state = save_retrain_state(
            status="complete",
            last_finished_at=now_string(),
            last_success_at=now_string(),
            last_trained_rounds=valid_rounds,
            current_rounds=valid_rounds,
            new_rounds_since_train=0,
            challenger_version=version,
            promoted_targets=promoted_targets,
            kept_targets=kept_targets,
            champion_changed=champion_changed,
            report_path=root_relative(challenger_dir / "report.json"),
            message=(
                "promoted challenger"
                if promoted_targets
                else "kept current champion"
            ),
        )
        return {
            "status": "complete",
            "state": state,
            "report": report,
        }


def ensure_live_prediction_csv() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if LIVE_PREDICTIONS_PATH.exists():
        return

    with LIVE_PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_PREDICTION_FIELDS)
        writer.writeheader()


def append_live_rows(rows: Sequence[dict]) -> None:
    ensure_live_prediction_csv()
    existing_keys = set()
    try:
        with LIVE_PREDICTIONS_PATH.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add(
                    (
                        row.get("prediction_time", ""),
                        row.get("target", ""),
                        row.get("round_number_predicted_for", ""),
                    )
                )
    except OSError:
        existing_keys = set()

    with LIVE_PREDICTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_PREDICTION_FIELDS)
        for row in rows:
            row_key = (
                str(row.get("prediction_time", "")),
                str(row.get("target", "")),
                str(row.get("round_number_predicted_for", "")),
            )
            if row_key in existing_keys:
                continue
            writer.writerow({field: row.get(field, "") for field in LIVE_PREDICTION_FIELDS})
            existing_keys.add(row_key)


def normalize_rounds(rounds: Sequence[dict]) -> List[dict]:
    return [
        {
            "timestamp": item.get("timestamp", ""),
            "multiplier": float(item.get("multiplier")),
            "round_id": item.get("round_id", ""),
            "source": item.get("source", ""),
        }
        for item in rounds
        if item.get("multiplier") is not None
    ]


def update_live_prediction_tracking(
    rounds: Sequence[dict],
    ml_prediction: dict,
    source: str = "dashboard",
) -> dict:
    rounds = normalize_rounds(rounds)
    if not rounds:
        return load_json(LIVE_STATE_PATH)

    state = load_json(LIVE_STATE_PATH)
    pending = state.get("pending")

    if pending and len(rounds) > int(pending.get("after_round_count", 0)):
        after_round_count = int(pending.get("after_round_count", 0))
        actual_round = rounds[after_round_count]
        actual_multiplier = float(actual_round["multiplier"])
        resolved_rows = []

        for prediction in pending.get("predictions", []):
            target = float(prediction["target"])
            probability = float(prediction["predicted_probability"])
            baseline = float(prediction["historical_baseline"])
            actual_class = int(actual_multiplier >= target)
            predicted_class = int(prediction["predicted_class"])
            resolved_rows.append(
                {
                    "prediction_time": pending.get("prediction_time", ""),
                    "resolved_time": now_string(),
                    "target": f"{target:.2f}",
                    "round_id_predicted_for": actual_round.get("round_id", ""),
                    "round_number_predicted_for": after_round_count + 1,
                    "model_version": prediction.get("model_version", ""),
                    "model_type": prediction.get("model_type", ""),
                    "predicted_probability": f"{probability:.8f}",
                    "historical_baseline": f"{baseline:.8f}",
                    "predicted_class": predicted_class,
                    "actual_multiplier": f"{actual_multiplier:.2f}",
                    "actual_class": actual_class,
                    "correct": int(predicted_class == actual_class),
                    "brier_loss": f"{(probability - actual_class) ** 2:.8f}",
                    "source": actual_round.get("source", source),
                }
            )

        if resolved_rows:
            append_live_rows(resolved_rows)

        state["last_resolved_at"] = now_string()
        state["last_resolved_round_count"] = after_round_count + 1
        state["pending"] = None

    if not ml_prediction or not ml_prediction.get("available"):
        atomic_write_json(LIVE_STATE_PATH, state)
        return state

    current_pending = state.get("pending") or {}
    if current_pending.get("after_round_count") == len(rounds):
        return state

    pending_predictions = []
    for target, item in (ml_prediction.get("predictions") or {}).items():
        probability = item.get("probability")
        baseline = item.get("historical_baseline")

        if probability is None or baseline is None:
            continue

        target_value = float(target)
        pending_predictions.append(
            {
                "target": f"{target_value:.2f}",
                "model_version": item.get("model_version")
                or ml_prediction.get("champion_version")
                or ml_prediction.get("model_version", ""),
                "model_type": item.get("model", ""),
                "predicted_probability": float(probability),
                "historical_baseline": float(baseline),
                "predicted_class": int(float(probability) >= 0.5),
            }
        )

    if pending_predictions:
        state["pending"] = {
            "prediction_time": now_string(),
            "after_round_count": len(rounds),
            "round_number_predicted_for": len(rounds) + 1,
            "predictions": pending_predictions,
            "source": source,
        }

    atomic_write_json(LIVE_STATE_PATH, state)
    return state


def live_prediction_rows(target: Optional[str] = None) -> List[dict]:
    if not LIVE_PREDICTIONS_PATH.exists():
        return []

    with LIVE_PREDICTIONS_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if target is not None:
        rows = [row for row in rows if row.get("target") == target]

    return rows


def calibration_error(actual, probabilities, bins: int = 10) -> Optional[float]:
    if len(actual) == 0:
        return None

    actual = np.asarray(actual, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    error = 0.0
    for left in np.linspace(0, 1, bins, endpoint=False):
        right = left + 1 / bins
        mask = (probabilities >= left) & (
            probabilities <= right if right >= 1 else probabilities < right
        )
        if not mask.any():
            continue
        error += (mask.sum() / len(actual)) * abs(
            float(probabilities[mask].mean()) - float(actual[mask].mean())
        )
    return float(error)


def metrics_for_live_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "predictions": 0,
        }

    actual = np.asarray([int(row["actual_class"]) for row in rows], dtype=int)
    predicted = np.asarray([int(row["predicted_class"]) for row in rows], dtype=int)
    probabilities = np.asarray(
        [float(row["predicted_probability"]) for row in rows],
        dtype=float,
    )
    baselines = np.asarray(
        [float(row["historical_baseline"]) for row in rows],
        dtype=float,
    )
    true_positive = int(((predicted == 1) & (actual == 1)).sum())
    true_negative = int(((predicted == 0) & (actual == 0)).sum())
    false_positive = int(((predicted == 1) & (actual == 0)).sum())
    false_negative = int(((predicted == 0) & (actual == 1)).sum())
    positives = int((actual == 1).sum())
    negatives = int((actual == 0).sum())
    accuracy = float((predicted == actual).mean())
    sensitivity = None if positives == 0 else true_positive / positives
    specificity = None if negatives == 0 else true_negative / negatives
    balanced = (
        accuracy
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2
    )
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = 0.0 if positives == 0 else true_positive / positives
    brier = float(np.mean((probabilities - actual) ** 2))
    baseline_brier = float(np.mean((baselines - actual) ** 2))
    return {
        "predictions": int(len(rows)),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": (
            None
            if baseline_brier <= 0
            else 1.0 - (brier / baseline_brier)
        ),
        "calibration_error": calibration_error(actual, probabilities),
        "average_predicted_probability": float(probabilities.mean()),
        "actual_target_frequency": float(actual.mean()),
        "average_model_edge": float((probabilities - baselines).mean()),
        "confusion_matrix": {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        },
    }


def live_metrics() -> dict:
    rows = live_prediction_rows()
    targets = sorted({row.get("target") for row in rows if row.get("target")})
    output = {}

    for target in targets:
        target_rows = [row for row in rows if row.get("target") == target]
        output[target] = {
            "100": metrics_for_live_rows(target_rows[-100:]),
            "250": metrics_for_live_rows(target_rows[-250:]),
            "500": metrics_for_live_rows(target_rows[-500:]),
            "all": metrics_for_live_rows(target_rows),
        }

    return output


def rollback_if_needed(config: Optional[dict] = None) -> dict:
    config = normalized_config(config)
    champion = ensure_champion()
    metrics = live_metrics()
    changed = False
    rolled_back = []

    for label, item in champion_target_items(champion).items():
        previous = item.get("previous_champion")
        if not previous:
            continue

        window = metrics.get(label, {}).get("250", {})
        if int(window.get("predictions", 0) or 0) < config["ml_live_rollback_min_predictions"]:
            continue

        brier_skill = window.get("brier_skill")
        if brier_skill is None or float(brier_skill) >= config["ml_live_rollback_min_brier_skill"]:
            continue

        champion["targets"][label] = {
            **previous,
            "rolled_back_at": now_string(),
            "rollback_from": item,
            "rollback_reason": "live brier skill materially negative",
        }
        previous_path = resolve_project_path(previous.get("model_path", ""))
        champion_path = CHAMPION_DIR / model_filename(label)
        copy_if_exists(previous_path, champion_path)
        champion["targets"][label]["champion_model_path"] = root_relative(champion_path)
        changed = True
        rolled_back.append(label)

    if changed:
        atomic_write_json(CHAMPION_METADATA_PATH, champion)
        write_champion_manifest(champion)
        save_retrain_state(
            status="rollback",
            rolled_back_targets=rolled_back,
            last_rollback_at=now_string(),
        )

    return {
        "changed": changed,
        "rolled_back_targets": rolled_back,
    }


def scheduler_status(config: Optional[dict] = None, csv_path: Path = DEFAULT_ROUNDS_PATH) -> dict:
    config = normalized_config(config)
    champion = ensure_champion()
    targets = champion_target_items(champion)
    valid_rounds = csv_valid_round_count(csv_path)
    state = load_json(RETRAIN_STATE_PATH)
    champion_trained_rounds = max(
        [int(item.get("trained_rounds", 0) or 0) for item in targets.values()] or [0]
    )
    state_trained_rounds = int(state.get("last_trained_rounds", 0) or 0)
    last_trained_rounds = max(champion_trained_rounds, state_trained_rounds)
    new_rounds = max(0, valid_rounds - last_trained_rounds)
    lock_exists = TRAINING_LOCK_PATH.exists()
    return {
        "enabled": True,
        "status": "training" if lock_exists else state.get("status", "waiting"),
        "current_rounds": valid_rounds,
        "last_trained_rounds": last_trained_rounds,
        "new_rounds_since_train": new_rounds,
        "min_new_rounds": config["ml_retrain_every_rounds"],
        "rounds_until_next_train": max(0, config["ml_retrain_every_rounds"] - new_rounds),
        "minimum_training_rounds": config["ml_minimum_training_rounds"],
        "promotion_min_skill_improvement": config["ml_promotion_min_skill_improvement"],
        "training_lock": str(TRAINING_LOCK_PATH),
        "lock_exists": lock_exists,
        "champion_targets": {
            label: {
                "model_name": item.get("model_name"),
                "version": item.get("version"),
                "trained_rounds": item.get("trained_rounds"),
                "brier_skill": item.get("brier_skill"),
                "balanced_accuracy": item.get("balanced_accuracy"),
            }
            for label, item in targets.items()
        },
        "last_success_at": state.get("last_success_at"),
        "last_started_at": state.get("last_started_at"),
        "last_finished_at": state.get("last_finished_at"),
        "last_error": state.get("last_error"),
        "last_message": state.get("message"),
        "last_report_path": state.get("report_path"),
        "promoted_targets": state.get("promoted_targets", []),
        "kept_targets": state.get("kept_targets", []),
        "live_metrics": live_metrics(),
    }


def should_retrain(config: Optional[dict] = None, csv_path: Path = DEFAULT_ROUNDS_PATH) -> Tuple[bool, str, dict]:
    status = scheduler_status(config, csv_path)
    config = normalized_config(config)

    if status["lock_exists"]:
        return False, "training already running", status

    if status["current_rounds"] < config["ml_minimum_training_rounds"]:
        return False, "minimum training rounds not reached", status

    if status["new_rounds_since_train"] < config["ml_retrain_every_rounds"]:
        return False, "retrain threshold not reached", status

    return True, "retrain due", status


def run_once(args) -> int:
    config = read_config_file(Path(args.config)) if args.config else read_config_file()
    config = normalized_config(
        {
            **config,
            **{
                key: value
                for key, value in {
                    "ml_retrain_every_rounds": args.retrain_every_rounds,
                    "ml_minimum_training_rounds": args.minimum_training_rounds,
                    "ml_promotion_min_skill_improvement": args.promotion_min_skill_improvement,
                }.items()
                if value is not None
            },
        }
    )
    needed, reason, status = should_retrain(config, Path(args.csv))

    if not args.force and not needed:
        print(json.dumps({"status": "skipped", "reason": reason, "scheduler": status}, indent=2))
        return 0

    try:
        result = train_challengers(
            csv_path=Path(args.csv),
            targets=parse_targets(args.targets),
            config=config,
            force=args.force,
            reason=args.reason,
        )
        rollback_if_needed(config)
    except Exception as exc:
        save_retrain_state(
            status="failed",
            last_error=f"{type(exc).__name__}: {exc}",
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.get("state", result), indent=2, default=str))
    return 0


def parse_targets(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_ROUNDS_PATH))
    parser.add_argument("--config", default=None)
    parser.add_argument("--targets", default=",".join(str(item) for item in DEFAULT_TARGETS))
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--init-champion", action="store_true")
    parser.add_argument("--live-metrics", action="store_true")
    parser.add_argument("--retrain-every-rounds", type=int, default=None)
    parser.add_argument("--minimum-training-rounds", type=int, default=None)
    parser.add_argument("--promotion-min-skill-improvement", type=float, default=None)
    args = parser.parse_args(argv)

    if args.init_champion:
        champion = ensure_champion()
        print(json.dumps(champion, indent=2, default=str))
        return 0 if champion.get("targets") else 1

    if args.live_metrics:
        print(json.dumps(live_metrics(), indent=2, default=str))
        return 0

    if args.status:
        config = read_config_file(Path(args.config)) if args.config else read_config_file()
        print(json.dumps(scheduler_status(config, Path(args.csv)), indent=2, default=str))
        return 0

    if args.run_once or args.force:
        return run_once(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
