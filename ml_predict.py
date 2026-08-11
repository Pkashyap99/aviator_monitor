"""Generate saved-model next-round ML probability estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ml_features import (
    DEFAULT_ROUNDS_PATH,
    clean_rounds,
    next_round_feature_frame,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
REPORT_PATH = DATA_DIR / "ml_report.json"
PREDICTIONS_PATH = DATA_DIR / "ml_predictions.json"
MODEL_VERSION = "ml-research-v1"
LIVE_HISTORY_MODELS = {
    "historical_frequency",
    "recent_frequency_100",
    "majority",
}


def load_json(path: Path) -> dict:
    """Load JSON if present."""

    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_percent(value) -> str:
    """Format a probability-like number."""

    if value is None:
        return "--"
    return f"{float(value) * 100:.1f}%"


def clipped_probability(value: float) -> float:
    """Keep probabilities away from exact 0/1 for stable scoring."""

    return min(max(float(value), 1e-6), 1 - 1e-6)


def history_probability(rounds, target: float, model_name: str) -> Optional[float]:
    """Return a live probability for baseline-style models."""

    if rounds.empty:
        return None

    values = rounds["multiplier"].astype(float)
    hits = values >= float(target)

    if model_name == "historical_frequency":
        return clipped_probability(hits.mean())

    if model_name == "recent_frequency_100":
        return clipped_probability(hits.tail(100).mean())

    if model_name == "majority":
        return clipped_probability(1.0 if hits.mean() >= 0.5 else 0.0)

    return None


def make_predictions(
    csv_path: Path = DEFAULT_ROUNDS_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    min_history: int = 100,
    include_context: bool = False,
) -> dict:
    """Return current next-round estimates from trained ML artifacts."""

    manifest = load_json(manifest_path)
    report = load_json(report_path)
    if not manifest:
        return {
            "model_version": MODEL_VERSION,
            "error": "No trained ML manifest found. Run python3 ml_train.py first.",
            "predictions": {},
        }

    rounds, quality = clean_rounds(csv_path)
    if rounds.empty:
        return {
            "model_version": manifest.get("model_version", MODEL_VERSION),
            "error": "No completed rounds available for ML prediction.",
            "predictions": {},
        }

    target_items = manifest.get("targets", {})
    needs_feature_frame = any(
        item.get("model_name") not in LIVE_HISTORY_MODELS
        for item in target_items.values()
    )
    frame = None

    if needs_feature_frame:
        feature_names = manifest.get("feature_names", [])
        frame, quality = next_round_feature_frame(
            csv_path,
            feature_names=feature_names,
            min_history=min_history,
            include_context=include_context,
        )
        if frame.empty:
            return {
                "model_version": manifest.get("model_version", MODEL_VERSION),
                "error": "No feature row available for ML prediction.",
                "predictions": {},
            }

    predictions = {}
    selected_models = report.get("selected_models", {})
    holdout_results = report.get("final_holdout_results", {})
    for target, item in target_items.items():
        model_name = item.get("model_name")
        target_value = float(item.get("target", target))
        baseline = history_probability(
            rounds,
            target_value,
            "historical_frequency",
        )
        probability = history_probability(
            rounds,
            target_value,
            model_name,
        )

        if probability is None:
            from ml_backtest import safe_positive_probability
            from ml_train import load_model

            model_path = Path(manifest_path).parent / item["model_path"]
            estimator = load_model(model_path)
            probability = float(safe_positive_probability(estimator, frame)[0])

        validation_metrics = selected_models.get(target, {}).get("validation_metrics", {})
        holdout_metrics = holdout_results.get(target, {}).get("holdout_metrics", {})
        predictions[target] = {
            "probability": probability,
            "historical_baseline": baseline,
            "edge": None if baseline is None else probability - float(baseline),
            "model": item.get("model_name"),
            "validation_status": validation_metrics.get("validation_status", "UNKNOWN"),
            "holdout_status": holdout_metrics.get("validation_status", "UNKNOWN"),
            "holdout_brier_skill": holdout_metrics.get("brier_skill_score"),
            "note": (
                "Probability is useful only if validation/holdout skill beats baseline."
            ),
        }

    return {
        "model_version": manifest.get("model_version", MODEL_VERSION),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "data_used_rounds": quality.get("valid_rows"),
        "predictions": predictions,
    }


def print_predictions(payload: dict) -> None:
    """Print a concise user-facing prediction report."""

    if payload.get("error"):
        print(payload["error"])
        return

    print("ML NEXT-ROUND ESTIMATE")
    print("")
    print(f"Data used: {payload.get('data_used_rounds', '--')} rounds")
    print("")
    for target, item in sorted(payload.get("predictions", {}).items(), key=lambda row: float(row[0])):
        edge = item.get("edge")
        edge_text = "--" if edge is None else f"{edge * 100:+.2f} pp"
        print(f">={float(target):.2f}x")
        print(f"Probability: {format_percent(item.get('probability'))}")
        print(f"Historical baseline: {format_percent(item.get('historical_baseline'))}")
        print(f"Edge: {edge_text}")
        print(f"Model: {item.get('model', '--')}")
        print(f"Validation status: {item.get('validation_status', 'UNKNOWN')}")
        print(f"Holdout status: {item.get('holdout_status', 'UNKNOWN')}")
        print("")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_ROUNDS_PATH))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--output", default=str(PREDICTIONS_PATH))
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--include-context", action="store_true")
    args = parser.parse_args(argv)

    payload = make_predictions(
        csv_path=Path(args.csv),
        manifest_path=Path(args.manifest),
        report_path=Path(args.report),
        min_history=args.min_history,
        include_context=args.include_context,
    )
    write_json(Path(args.output), payload)
    print_predictions(payload)
    return 0 if not payload.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
