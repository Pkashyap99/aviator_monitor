"""Train final per-target ML models after honest walk-forward selection."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ml_backtest import (
    RANDOM_SEED,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    WalkForwardConfig,
    calculate_metrics,
    clone_model,
    concise_summary,
    fixed_model_registry,
    run_walk_forward_backtest,
    safe_positive_probability,
    validation_status,
    write_backtest_csv,
)
from ml_features import (
    DEFAULT_ROUNDS_PATH,
    DEFAULT_TARGETS,
    FEATURE_SCHEMA_VERSION,
    load_feature_dataset,
    next_round_feature_frame,
    target_name,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORT_PATH = DATA_DIR / "ml_report.json"
BACKTEST_PATH = DATA_DIR / "ml_backtest.csv"
PREDICTIONS_PATH = DATA_DIR / "ml_predictions.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
MODEL_VERSION = "ml-research-v1"


try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


def parse_targets(value: str) -> list[float]:
    """Parse comma-separated target thresholds."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def model_filename(target: str) -> str:
    """Return a stable model filename for a target label."""

    return f"target_{target.replace('.', '_')}.joblib"


def save_model(path: Path, estimator) -> str:
    """Save a model with joblib when available, otherwise pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if JOBLIB_AVAILABLE:
        joblib.dump(estimator, path)
        return "joblib"

    fallback = path.with_suffix(".pkl")
    with fallback.open("wb") as f:
        pickle.dump(estimator, f)
    return "pickle"


def load_model(path: Path):
    """Load a saved model using joblib/pickle fallback."""

    if path.exists() and JOBLIB_AVAILABLE:
        return joblib.load(path)
    fallback = path.with_suffix(".pkl")
    with fallback.open("rb") as f:
        return pickle.load(f)


def baseline_records(y_true, probability, majority_probability) -> list[Dict]:
    """Create metric records for final holdout evaluation."""

    probability = np.asarray(probability, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    majority_prediction = np.ones(len(y_true), dtype=int) if majority_probability >= 0.5 else np.zeros(len(y_true), dtype=int)
    baseline_probability = np.full(len(y_true), majority_probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    return [
        {
            "actual": int(actual),
            "probability": float(prob),
            "predicted": int(pred),
            "baseline_probability": float(base_prob),
            "majority_prediction": int(majority),
        }
        for actual, prob, pred, base_prob, majority in zip(
            y_true,
            probability,
            predicted,
            baseline_probability,
            majority_prediction,
        )
    ]


def fit_final_model(model, X_train, y_train):
    """Fit the selected model on the full pre-holdout block."""

    estimator = clone_model(model)
    estimator.fit(X_train, y_train)
    return estimator


def extract_feature_importance(estimator, feature_names: Sequence[str], top_n: int = 20) -> Dict:
    """Extract simple feature importance/coefficient summaries."""

    target = estimator
    if hasattr(target, "estimator_"):
        target = target.estimator_
    if hasattr(target, "named_steps"):
        model = target.named_steps.get("model")
    else:
        model = target

    if model is None:
        return {}

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        ranked = sorted(
            zip(feature_names, values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]
        return {
            "type": "feature_importances",
            "top": [
                {
                    "feature": name,
                    "importance": float(value),
                }
                for name, value in ranked
            ],
        }

    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_[0], dtype=float)
        positive = sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)[:top_n]
        negative = sorted(zip(feature_names, values), key=lambda item: item[1])[:top_n]
        return {
            "type": "logistic_coefficients",
            "largest_positive": [
                {
                    "feature": name,
                    "coefficient": float(value),
                }
                for name, value in positive
            ],
            "largest_negative": [
                {
                    "feature": name,
                    "coefficient": float(value),
                }
                for name, value in negative
            ],
        }

    return {}


def evaluate_holdout(estimator, X_holdout, y_holdout, baseline_probability: float) -> Dict:
    """Evaluate final holdout once."""

    probabilities = safe_positive_probability(estimator, X_holdout)
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    records = baseline_records(y_holdout, probabilities, baseline_probability)
    metrics = calculate_metrics(records)
    metrics["validation_status"] = validation_status(metrics)
    return metrics


def run_permutation_diagnostic(
    model_name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    baseline_probability: float,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Shuffle training labels only as a null diagnostic."""

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(y_train).copy()
    rng.shuffle(shuffled)
    try:
        estimator = fit_final_model(model, X_train, shuffled)
        probabilities = safe_positive_probability(estimator, X_test)
        records = baseline_records(y_test, probabilities, baseline_probability)
        return {
            "model": model_name,
            "label_shuffle": "training_labels_only",
            "metrics": calculate_metrics(records),
        }
    except Exception as exc:
        return {
            "model": model_name,
            "label_shuffle": "training_labels_only",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_next_predictions(
    csv_path: Path,
    manifest: Dict,
    report: Dict,
    include_context: bool,
    min_history: int,
) -> Dict:
    """Generate current next-round probability estimates from saved models."""

    feature_names = manifest.get("feature_names", [])
    next_features, quality = next_round_feature_frame(
        csv_path,
        feature_names=feature_names,
        min_history=min_history,
        include_context=include_context,
    )
    predictions = {}
    if next_features.empty:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "No feature row available.",
            "predictions": predictions,
        }

    for target, item in manifest.get("targets", {}).items():
        path = MODELS_DIR / item["model_path"]
        estimator = load_model(path)
        probability = float(safe_positive_probability(estimator, next_features)[0])
        baseline = item.get("historical_baseline_probability")
        validation = report.get("selected_models", {}).get(target, {})
        metrics = validation.get("validation_metrics", {})
        predictions[target] = {
            "probability": probability,
            "historical_baseline": baseline,
            "edge": None if baseline is None else probability - float(baseline),
            "model": item.get("model_name"),
            "validation_status": metrics.get(
                "validation_status",
                "UNKNOWN",
            ),
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "data_used_rounds": quality.get("valid_rows"),
        "predictions": predictions,
    }


def overall_conclusion(holdout_results: Dict[str, Dict]) -> str:
    """Summarize whether the final untouched holdout supports a real edge."""

    useful = []
    weak = []
    for target, payload in holdout_results.items():
        status = payload.get("holdout_metrics", {}).get("validation_status", "")
        if status == "CONSISTENT OUT-OF-SAMPLE EDGE":
            useful.append(target)
        elif status in {"WEAK / UNSTABLE EDGE", "POTENTIAL EDGE — NEEDS MORE DATA"}:
            weak.append(target)

    if useful:
        return (
            "Some targets showed a positive untouched-holdout edge. Treat this as "
            "research evidence only and continue collecting future data."
        )
    if weak:
        return (
            "Only weak or unstable holdout evidence was observed. Do not treat the "
            "model as reliably predictive yet."
        )
    return (
        "NO PREDICTIVE EDGE DETECTED on the untouched holdout. The current data "
        "does not prove repeatable next-round predictability."
    )


def train_pipeline(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_ROUNDS_PATH))
    parser.add_argument(
        "--targets",
        default=",".join(str(item) for item in DEFAULT_TARGETS),
    )
    parser.add_argument("--min-train", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--holdout-fraction", type=float, default=0.18)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument(
        "--calibration",
        default="uncalibrated",
        help="Comma-separated calibration methods.",
    )
    args = parser.parse_args(argv)

    targets = parse_targets(args.targets)
    calibration_methods = [item.strip() for item in args.calibration.split(",") if item.strip()]
    dataset = load_feature_dataset(
        Path(args.csv),
        targets=targets,
        min_history=args.min_history,
        include_context=args.include_context,
    )
    frame = dataset.frame
    if frame.empty:
        raise SystemExit("No valid feature rows available for ML training.")

    holdout_size = max(1, int(len(frame) * args.holdout_fraction))
    trainval = frame.iloc[: len(frame) - holdout_size].reset_index(drop=True)
    holdout = frame.iloc[len(frame) - holdout_size :].reset_index(drop=True)

    config = WalkForwardConfig(
        min_train=args.min_train,
        test_size=args.test_size,
        holdout_fraction=args.holdout_fraction,
        max_folds=args.max_folds,
        min_history=args.min_history,
        include_context=args.include_context,
    )
    models = fixed_model_registry(calibration_methods=calibration_methods)
    records, validation_summary, selected = run_walk_forward_backtest(
        trainval,
        dataset.feature_names,
        targets=targets,
        config=config,
        models=models,
        calibration_methods=calibration_methods,
    )
    write_backtest_csv(BACKTEST_PATH, records)

    X_train = trainval[dataset.feature_names].astype(float)
    X_holdout = holdout[dataset.feature_names].astype(float)
    manifest = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": dataset.feature_names,
        "targets": {},
    }
    selected_models = {}
    holdout_results = {}
    permutation_diagnostics = {}

    for target in targets:
        label = f"{target:.2f}"
        column = target_name(target)
        selected_item = selected.get(label)
        if not selected_item:
            continue

        model_name = selected_item["model"]
        model = models[model_name]
        y_train = trainval[column].astype(int).to_numpy()
        y_holdout = holdout[column].astype(int).to_numpy()
        baseline_probability = float(y_train.mean())
        estimator = fit_final_model(model, X_train, y_train)
        save_format = save_model(MODELS_DIR / model_filename(label), estimator)
        holdout_metrics = evaluate_holdout(
            estimator,
            X_holdout,
            y_holdout,
            baseline_probability,
        )
        importance = extract_feature_importance(estimator, dataset.feature_names)
        selected_models[label] = {
            "model_name": model_name,
            "validation_metrics": selected_item["metrics"],
            "selection_rank": selected_item.get("rank_tuple"),
            "feature_importance": importance,
            "selected_before_holdout": True,
        }
        holdout_results[label] = {
            "model_name": model_name,
            "holdout_metrics": holdout_metrics,
            "holdout_round_range": {
                "start_round_number": int(holdout["round_number"].min()),
                "end_round_number": int(holdout["round_number"].max()),
                "start_timestamp": str(holdout.iloc[0]["timestamp"]),
                "end_timestamp": str(holdout.iloc[-1]["timestamp"]),
            },
            "evaluated_once_after_model_selection": True,
        }
        permutation_diagnostics[label] = run_permutation_diagnostic(
            model_name,
            model,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            baseline_probability,
        )
        manifest["targets"][label] = {
            "target": target,
            "target_column": column,
            "model_name": model_name,
            "model_path": model_filename(label),
            "save_format": save_format,
            "historical_baseline_probability": baseline_probability,
        }

    report = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "guardrails": {
            "holdout_rule": (
                "Final holdout is reserved before model selection and evaluated once. "
                "Do not tune thresholds/models after seeing holdout results."
            ),
            "objective": "honest out-of-sample edge detection, not impressive training accuracy",
        },
        "config": vars(args),
        "library_status": {
            "sklearn_available": SKLEARN_AVAILABLE,
            "xgboost_available": XGBOOST_AVAILABLE,
            "joblib_available": JOBLIB_AVAILABLE,
        },
        "dataset_statistics": {
            "feature_rows": int(len(frame)),
            "selection_rows": int(len(trainval)),
            "holdout_rows": int(len(holdout)),
            "holdout_fraction_actual": len(holdout) / len(frame),
            "feature_count": len(dataset.feature_names),
            "targets": targets,
        },
        "data_quality": dataset.quality_report,
        "source_report": dataset.source_report,
        "validation_summary": validation_summary,
        "selected_models": selected_models,
        "final_holdout_results": holdout_results,
        "permutation_diagnostics": permutation_diagnostics,
        "overall_conclusion": overall_conclusion(holdout_results),
        "artifacts": {
            "backtest_csv": str(BACKTEST_PATH),
            "report_json": str(REPORT_PATH),
            "predictions_json": str(PREDICTIONS_PATH),
            "models_manifest": str(MANIFEST_PATH),
        },
    }
    write_json(REPORT_PATH, report)
    write_json(MANIFEST_PATH, manifest)
    predictions = build_next_predictions(
        Path(args.csv),
        manifest,
        report,
        include_context=args.include_context,
        min_history=args.min_history,
    )
    write_json(PREDICTIONS_PATH, predictions)
    print(concise_summary(validation_summary, selected))
    print("\nFINAL UNTOUCHED HOLDOUT")
    for target, result in holdout_results.items():
        metrics = result["holdout_metrics"]
        print(
            f">={target}x {result['model_name']}: "
            f"brier_skill={metrics.get('brier_skill_score')} "
            f"balanced_skill={metrics.get('balanced_accuracy_skill')} "
            f"roc_auc={metrics.get('roc_auc')} "
            f"status={metrics.get('validation_status')}"
        )
    print(f"\n{report['overall_conclusion']}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    train_pipeline(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
