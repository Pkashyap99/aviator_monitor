"""Walk-forward ML backtesting for next-round Aviatrix threshold targets."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

_ORIGINAL_SHOWWARNING = warnings.showwarning


def _showwarning_without_loky_core_noise(message, category, filename, lineno, file=None, line=None):
    if "joblib/externals/loky/backend/context.py" in filename:
        return
    _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _showwarning_without_loky_core_noise
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

from ml_features import (
    DEFAULT_ROUNDS_PATH,
    DEFAULT_TARGETS,
    load_feature_dataset,
    target_name,
    write_json,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BACKTEST_PATH = DATA_DIR / "ml_backtest.csv"
RANDOM_SEED = 20260811


try:
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.calibration import calibration_curve
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised when deps are absent.
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for chronological validation."""

    min_train: int = 5000
    test_size: int = 1000
    holdout_fraction: float = 0.18
    max_folds: Optional[int] = None
    random_seed: int = RANDOM_SEED
    min_history: int = 100
    include_context: bool = False


class ConstantProbabilityClassifier:
    """A deterministic probability predictor used for baselines/fallbacks."""

    def __init__(self, probability: Optional[float] = None, recent_window: Optional[int] = None):
        self.probability = probability
        self.recent_window = recent_window
        self.fitted_probability_ = 0.5

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        if len(y_array) == 0:
            self.fitted_probability_ = 0.5
            return self

        if self.probability is not None:
            self.fitted_probability_ = float(self.probability)
        elif self.recent_window:
            self.fitted_probability_ = float(y_array[-self.recent_window :].mean())
        else:
            self.fitted_probability_ = float(y_array.mean())
        self.fitted_probability_ = min(max(self.fitted_probability_, 1e-6), 1 - 1e-6)
        return self

    def predict_proba(self, X):
        probability = float(self.fitted_probability_)
        return np.column_stack(
            [
                np.full(len(X), 1.0 - probability),
                np.full(len(X), probability),
            ]
        )


class MajorityClassifier(ConstantProbabilityClassifier):
    """Majority-class baseline with probability equal to 0 or 1 after fitting."""

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        prevalence = 0.0 if len(y_array) == 0 else float(y_array.mean())
        self.fitted_probability_ = 1.0 - 1e-6 if prevalence >= 0.5 else 1e-6
        return self


if SKLEARN_AVAILABLE:

    class ChronologicalCalibratedClassifier(BaseEstimator, ClassifierMixin):
        """Calibrate probabilities on the last slice of the training fold only."""

        def __init__(
            self,
            base_estimator,
            method: str = "sigmoid",
            calibration_fraction: float = 0.2,
            min_calibration_size: int = 500,
            random_state: int = RANDOM_SEED,
        ):
            self.base_estimator = base_estimator
            self.method = method
            self.calibration_fraction = calibration_fraction
            self.min_calibration_size = min_calibration_size
            self.random_state = random_state

        def fit(self, X, y):
            X_frame = pd.DataFrame(X).reset_index(drop=True)
            y_series = pd.Series(y).reset_index(drop=True)
            n_rows = len(y_series)
            calibration_size = max(
                self.min_calibration_size,
                int(n_rows * self.calibration_fraction),
            )

            if (
                n_rows < calibration_size + 200
                or y_series.nunique() < 2
            ):
                self.estimator_ = clone(self.base_estimator)
                self.estimator_.fit(X_frame, y_series)
                self.calibrator_ = None
                self.classes_ = np.array([0, 1])
                return self

            split = n_rows - calibration_size
            X_train = X_frame.iloc[:split]
            y_train = y_series.iloc[:split]
            X_cal = X_frame.iloc[split:]
            y_cal = y_series.iloc[split:]

            if y_train.nunique() < 2 or y_cal.nunique() < 2:
                self.estimator_ = clone(self.base_estimator)
                self.estimator_.fit(X_frame, y_series)
                self.calibrator_ = None
                self.classes_ = np.array([0, 1])
                return self

            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(X_train, y_train)
            probabilities = safe_positive_probability(self.estimator_, X_cal)

            if self.method == "isotonic":
                self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
                self.calibrator_.fit(probabilities, y_cal)
            else:
                self.calibrator_ = LogisticRegression(random_state=self.random_state)
                self.calibrator_.fit(probabilities.reshape(-1, 1), y_cal)

            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            probabilities = safe_positive_probability(self.estimator_, X)
            if self.calibrator_ is None:
                calibrated = probabilities
            elif self.method == "isotonic":
                calibrated = self.calibrator_.predict(probabilities)
            else:
                calibrated = self.calibrator_.predict_proba(
                    probabilities.reshape(-1, 1)
                )[:, 1]
            calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
            return np.column_stack([1.0 - calibrated, calibrated])


def safe_positive_probability(model, X) -> np.ndarray:
    """Return positive-class probabilities, falling back safely when needed."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities, dtype=float).reshape(-1)

    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def fixed_model_registry(
    calibration_methods: Sequence[str] = ("uncalibrated",),
    include_xgboost: bool = True,
) -> Dict[str, object]:
    """Create fixed model configurations without using holdout feedback."""

    models: Dict[str, object] = {
        "majority": MajorityClassifier(),
        "historical_frequency": ConstantProbabilityClassifier(),
        "recent_frequency_100": ConstantProbabilityClassifier(recent_window=100),
    }

    if not SKLEARN_AVAILABLE:
        return models

    base_models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=1200,
                        tol=1e-3,
                        alpha=0.0005,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=90,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=90,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        ),
    }

    if include_xgboost and XGBOOST_AVAILABLE:
        base_models["xgboost"] = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )

    for name, estimator in base_models.items():
        models[f"{name}__uncalibrated"] = estimator
        if "sigmoid" in calibration_methods:
            models[f"{name}__sigmoid"] = ChronologicalCalibratedClassifier(
                estimator,
                method="sigmoid",
            )
        if "isotonic" in calibration_methods:
            models[f"{name}__isotonic"] = ChronologicalCalibratedClassifier(
                estimator,
                method="isotonic",
                min_calibration_size=1000,
            )
    return models


def clone_model(model):
    """Clone sklearn estimators while handling custom baselines."""

    if SKLEARN_AVAILABLE and not isinstance(model, ConstantProbabilityClassifier):
        return clone(model)
    if isinstance(model, MajorityClassifier):
        return MajorityClassifier()
    if isinstance(model, ConstantProbabilityClassifier):
        return ConstantProbabilityClassifier(
            probability=model.probability,
            recent_window=model.recent_window,
        )
    return model


def chronological_splits(
    n_rows: int,
    min_train: int,
    test_size: int,
    max_folds: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build walk-forward train/test index splits."""

    if n_rows < 80:
        return []

    min_train = min(int(min_train), max(40, int(n_rows * 0.65)))
    test_size = min(int(test_size), max(20, n_rows - min_train))
    if min_train >= n_rows:
        min_train = max(20, int(n_rows * 0.6))
    if test_size <= 0 or min_train >= n_rows:
        return []

    splits = []
    train_end = min_train
    while train_end < n_rows:
        test_end = min(train_end + test_size, n_rows)
        if test_end <= train_end:
            break
        train_index = np.arange(0, train_end)
        test_index = np.arange(train_end, test_end)
        splits.append((train_index, test_index))
        if test_end == n_rows:
            break
        train_end = test_end

    if max_folds is not None and len(splits) > max_folds:
        splits = splits[-int(max_folds) :]
    return splits


def ece_score(y_true, y_probability, bins: int = 10) -> Optional[float]:
    """Expected calibration error with equal-width probability bins."""

    if len(y_true) == 0:
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    total = len(y_true)
    error = 0.0
    for left in np.linspace(0, 1, bins, endpoint=False):
        right = left + 1 / bins
        if right >= 1:
            mask = (y_probability >= left) & (y_probability <= right)
        else:
            mask = (y_probability >= left) & (y_probability < right)
        if not mask.any():
            continue
        confidence = float(y_probability[mask].mean())
        observed = float(y_true[mask].mean())
        error += (mask.sum() / total) * abs(confidence - observed)
    return float(error)


def bootstrap_ci(values, metric_fn, samples: int = 300, seed: int = RANDOM_SEED):
    """Bootstrap a simple metric confidence interval."""

    values = list(values)
    if len(values) < 20:
        return None

    rng = random.Random(seed)
    metrics = []
    for _ in range(samples):
        sampled = [values[rng.randrange(len(values))] for _ in range(len(values))]
        metrics.append(metric_fn(sampled))
    metrics = sorted(item for item in metrics if item is not None and not math.isnan(item))
    if not metrics:
        return None
    lower = metrics[int(0.025 * (len(metrics) - 1))]
    upper = metrics[int(0.975 * (len(metrics) - 1))]
    return [float(lower), float(upper)]


def calculate_metrics(records: Sequence[Dict]) -> Dict:
    """Calculate classification, probability, skill, and calibration metrics."""

    if not records:
        return {
            "predictions": 0,
        }

    y_true = np.asarray([row["actual"] for row in records], dtype=int)
    y_probability = np.asarray([row["probability"] for row in records], dtype=float)
    y_pred = np.asarray([row["predicted"] for row in records], dtype=int)
    majority_pred = np.asarray([row["majority_prediction"] for row in records], dtype=int)
    baseline_probability = np.asarray(
        [row["baseline_probability"] for row in records],
        dtype=float,
    )
    labels = [0, 1]
    accuracy = float(accuracy_score(y_true, y_pred))
    majority_accuracy = float(accuracy_score(y_true, majority_pred))
    balanced_accuracy = (
        None
        if len(set(y_true)) < 2
        else float(balanced_accuracy_score(y_true, y_pred))
    )
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_probability))
    baseline_brier = float(brier_score_loss(y_true, baseline_probability))
    probability = np.clip(y_probability, 1e-6, 1 - 1e-6)
    try:
        logloss = float(log_loss(y_true, probability, labels=labels))
    except Exception:
        logloss = None
    if len(set(y_true)) >= 2:
        roc_auc = float(roc_auc_score(y_true, y_probability))
        pr_auc = float(average_precision_score(y_true, y_probability))
    else:
        roc_auc = None
        pr_auc = None
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    rows_for_ci = [
        {
            "actual": int(actual),
            "predicted": int(predicted),
            "majority": int(majority),
        }
        for actual, predicted, majority in zip(y_true, y_pred, majority_pred)
    ]
    accuracy_skill_ci = bootstrap_ci(
        rows_for_ci,
        lambda sample: (
            sum(row["actual"] == row["predicted"] for row in sample) / len(sample)
            - sum(row["actual"] == row["majority"] for row in sample) / len(sample)
        ),
    )

    return {
        "predictions": int(len(records)),
        "target_prevalence": float(y_true.mean()),
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_accuracy,
        "accuracy_skill": accuracy - majority_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_skill": (
            None if balanced_accuracy is None else balanced_accuracy - 0.5
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill_score": (
            None
            if baseline_brier <= 0
            else 1.0 - (brier / baseline_brier)
        ),
        "log_loss": logloss,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
        "calibration_error": ece_score(y_true, y_probability),
        "accuracy_skill_ci_95": accuracy_skill_ci,
    }


def validation_status(metrics: Dict, fold_stability: Optional[float] = None) -> str:
    """Convert metrics into an honest edge label."""

    predictions = metrics.get("predictions", 0)
    brier_skill = metrics.get("brier_skill_score")
    balanced_skill = metrics.get("balanced_accuracy_skill")
    roc_auc = metrics.get("roc_auc")
    calibration_error = metrics.get("calibration_error")

    if predictions < 200:
        return "INSUFFICIENT DATA"

    brier_skill = brier_skill if brier_skill is not None else -1.0
    balanced_skill = balanced_skill if balanced_skill is not None else -1.0
    roc_auc = roc_auc if roc_auc is not None else 0.5
    calibration_error = calibration_error if calibration_error is not None else 1.0
    fold_stability = fold_stability if fold_stability is not None else 0.0

    if (
        brier_skill > 0.03
        and balanced_skill > 0.03
        and roc_auc > 0.54
        and calibration_error < 0.06
        and fold_stability < 0.05
    ):
        return "CONSISTENT OUT-OF-SAMPLE EDGE"

    if (
        brier_skill > 0.015
        and balanced_skill > 0.015
        and roc_auc > 0.525
    ):
        return "POTENTIAL EDGE — NEEDS MORE DATA"

    if (
        brier_skill > 0.004
        or balanced_skill > 0.01
        or roc_auc > 0.515
    ):
        return "WEAK / UNSTABLE EDGE"

    return "NO PREDICTIVE EDGE DETECTED"


def aggregate_backtest_records(records: Sequence[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Aggregate raw prediction records by target and model."""

    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    fold_brier_skill: Dict[Tuple[str, str], List[float]] = {}
    for record in records:
        grouped.setdefault(record["target"], {}).setdefault(record["model"], []).append(record)
        key = (record["target"], record["model"])
        fold_brier_skill.setdefault(key, [])

    summary: Dict[str, Dict[str, Dict]] = {}
    for target, model_records in grouped.items():
        summary[target] = {}
        for model_name, rows in model_records.items():
            metrics = calculate_metrics(rows)
            per_fold = []
            for fold in sorted(set(row["fold"] for row in rows)):
                fold_rows = [row for row in rows if row["fold"] == fold]
                fold_metrics = calculate_metrics(fold_rows)
                per_fold.append(
                    {
                        "fold": int(fold),
                        "metrics": fold_metrics,
                    }
                )
                if fold_metrics.get("brier_skill_score") is not None:
                    fold_brier_skill[(target, model_name)].append(
                        fold_metrics["brier_skill_score"]
                    )
            stability_values = fold_brier_skill[(target, model_name)]
            fold_stability = (
                None
                if len(stability_values) < 2
                else float(np.std(stability_values))
            )
            metrics["fold_brier_skill_std"] = fold_stability
            metrics["validation_status"] = validation_status(metrics, fold_stability)
            metrics["folds"] = per_fold
            summary[target][model_name] = metrics
    return summary


def rank_model(metrics: Dict) -> Tuple[float, float, float, float, float]:
    """Rank models without looking at final holdout data."""

    brier_skill = metrics.get("brier_skill_score")
    balanced_skill = metrics.get("balanced_accuracy_skill")
    roc_auc = metrics.get("roc_auc")
    calibration_error = metrics.get("calibration_error")
    fold_std = metrics.get("fold_brier_skill_std")
    brier_skill = -1.0 if brier_skill is None else float(brier_skill)
    balanced_skill = -1.0 if balanced_skill is None else float(balanced_skill)
    roc_auc = 0.5 if roc_auc is None else float(roc_auc)
    calibration_error = 1.0 if calibration_error is None else float(calibration_error)
    fold_std = 0.2 if fold_std is None else float(fold_std)
    return (
        brier_skill - fold_std * 0.30,
        balanced_skill,
        roc_auc,
        -calibration_error,
        -fold_std,
    )


def choose_best_models(summary: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Choose one best validation model per target using only validation metrics."""

    selected = {}
    for target, model_metrics in summary.items():
        if not model_metrics:
            continue
        best_name, best_metrics = sorted(
            model_metrics.items(),
            key=lambda item: rank_model(item[1]),
            reverse=True,
        )[0]
        selected[target] = {
            "model": best_name,
            "metrics": best_metrics,
            "rank_tuple": rank_model(best_metrics),
        }
    return selected


def fit_predict_fold(model, X_train, y_train, X_test):
    """Fit one fold and return probabilities with safe failure handling."""

    if len(np.unique(y_train)) < 2:
        fallback = ConstantProbabilityClassifier()
        fallback.fit(X_train, y_train)
        return fallback.predict_proba(X_test)[:, 1], "single_class_train_fallback"

    estimator = clone_model(model)
    try:
        estimator.fit(X_train, y_train)
        probabilities = safe_positive_probability(estimator, X_test)
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return probabilities, ""
    except Exception as exc:
        fallback = ConstantProbabilityClassifier()
        fallback.fit(X_train, y_train)
        return (
            fallback.predict_proba(X_test)[:, 1],
            f"model_failed_fallback:{type(exc).__name__}:{exc}",
        )


def run_walk_forward_backtest(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    targets: Iterable[float] = DEFAULT_TARGETS,
    config: WalkForwardConfig = WalkForwardConfig(),
    models: Optional[Dict[str, object]] = None,
    calibration_methods: Sequence[str] = ("uncalibrated",),
) -> Tuple[List[Dict], Dict[str, Dict[str, Dict]], Dict[str, Dict]]:
    """Run walk-forward validation over an already-prepared feature frame."""

    if models is None:
        models = fixed_model_registry(calibration_methods=calibration_methods)

    if frame.empty:
        return [], {}, {}

    X_all = frame[list(feature_names)].astype(float)
    splits = chronological_splits(
        len(frame),
        min_train=config.min_train,
        test_size=config.test_size,
        max_folds=config.max_folds,
    )
    records: List[Dict] = []

    for target in targets:
        target_column = target_name(float(target))
        if target_column not in frame.columns:
            continue
        y_all = frame[target_column].astype(int).to_numpy()
        target_label = f"{float(target):.2f}"

        for fold_number, (train_index, test_index) in enumerate(splits, start=1):
            X_train = X_all.iloc[train_index]
            X_test = X_all.iloc[test_index]
            y_train = y_all[train_index]
            y_test = y_all[test_index]
            historical_baseline = ConstantProbabilityClassifier()
            historical_baseline.fit(X_train, y_train)
            baseline_probability = historical_baseline.predict_proba(X_test)[:, 1]
            majority_prediction = (
                np.ones(len(test_index), dtype=int)
                if float(np.mean(y_train)) >= 0.5
                else np.zeros(len(test_index), dtype=int)
            )

            for model_name, model in models.items():
                probability, warning = fit_predict_fold(model, X_train, y_train, X_test)
                predicted = (probability >= 0.5).astype(int)
                for offset, row_index in enumerate(test_index):
                    records.append(
                        {
                            "target": target_label,
                            "model": model_name,
                            "fold": fold_number,
                            "row_index": int(row_index),
                            "round_number": int(frame.iloc[row_index]["round_number"]),
                            "timestamp": str(frame.iloc[row_index]["timestamp"]),
                            "actual_multiplier": float(frame.iloc[row_index]["multiplier"]),
                            "actual": int(y_test[offset]),
                            "probability": float(probability[offset]),
                            "predicted": int(predicted[offset]),
                            "baseline_probability": float(baseline_probability[offset]),
                            "majority_prediction": int(majority_prediction[offset]),
                            "warning": warning,
                        }
                    )

    summary = aggregate_backtest_records(records)
    selected = choose_best_models(summary)
    return records, summary, selected


def write_backtest_csv(path: Path, records: Sequence[Dict]) -> None:
    """Persist raw fold predictions."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "model",
        "fold",
        "row_index",
        "round_number",
        "timestamp",
        "actual_multiplier",
        "actual",
        "probability",
        "predicted",
        "baseline_probability",
        "majority_prediction",
        "warning",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_targets(value: str) -> List[float]:
    """Parse comma-separated target thresholds."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def concise_summary(summary: Dict[str, Dict[str, Dict]], selected: Dict[str, Dict]) -> str:
    """Render a compact terminal summary."""

    lines = ["ML WALK-FORWARD BACKTEST", ""]
    if not summary:
        return "ML WALK-FORWARD BACKTEST\nNo backtest rows were produced."

    for target in sorted(summary, key=lambda item: float(item)):
        selected_item = selected.get(target, {})
        selected_model = selected_item.get("model", "none")
        metrics = selected_item.get("metrics", {})
        lines.append(f">={target}x best validation model: {selected_model}")
        lines.append(
            "  "
            f"predictions={metrics.get('predictions', 0)} "
            f"brier_skill={metrics.get('brier_skill_score')} "
            f"balanced_skill={metrics.get('balanced_accuracy_skill')} "
            f"roc_auc={metrics.get('roc_auc')} "
            f"status={metrics.get('validation_status', 'UNKNOWN')}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_ROUNDS_PATH))
    parser.add_argument("--target", type=float, default=None)
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
        help="Comma-separated calibration methods: uncalibrated,sigmoid,isotonic",
    )
    parser.add_argument("--output", default=str(DEFAULT_BACKTEST_PATH))
    parser.add_argument("--summary-json", default=str(DATA_DIR / "ml_backtest_summary.json"))
    args = parser.parse_args(argv)

    targets = [args.target] if args.target is not None else parse_targets(args.targets)
    calibration_methods = [item.strip() for item in args.calibration.split(",") if item.strip()]
    dataset = load_feature_dataset(
        Path(args.csv),
        targets=targets,
        min_history=args.min_history,
        include_context=args.include_context,
    )

    frame = dataset.frame
    if frame.empty:
        print("No valid feature rows available.")
        return 1

    holdout_size = max(1, int(len(frame) * args.holdout_fraction))
    selection_frame = frame.iloc[: len(frame) - holdout_size].reset_index(drop=True)
    config = WalkForwardConfig(
        min_train=args.min_train,
        test_size=args.test_size,
        holdout_fraction=args.holdout_fraction,
        max_folds=args.max_folds,
        min_history=args.min_history,
        include_context=args.include_context,
    )
    records, summary, selected = run_walk_forward_backtest(
        selection_frame,
        dataset.feature_names,
        targets=targets,
        config=config,
        calibration_methods=calibration_methods,
    )
    write_backtest_csv(Path(args.output), records)
    write_json(
        Path(args.summary_json),
        {
            "config": vars(args),
            "sklearn_available": SKLEARN_AVAILABLE,
            "xgboost_available": XGBOOST_AVAILABLE,
            "data_quality": dataset.quality_report,
            "source_report": dataset.source_report,
            "holdout_rows_reserved_and_not_used": holdout_size,
            "validation_summary": summary,
            "selected_models": selected,
        },
    )
    print(concise_summary(summary, selected))
    if not SKLEARN_AVAILABLE:
        print(f"\nscikit-learn unavailable: {SKLEARN_IMPORT_ERROR}", file=sys.stderr)
        print("Install requirements.txt to compare ML models.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
