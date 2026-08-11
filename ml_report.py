"""Print the saved ML research report in a concise form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ml_train import REPORT_PATH


def load_report(path: Path = REPORT_PATH) -> dict:
    """Load the ML report JSON."""

    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value, digits: int = 4) -> str:
    """Format metric values safely."""

    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def print_report(report: dict) -> None:
    """Print a compact terminal report."""

    if not report:
        print("No ML report found. Run python3 ml_train.py first.")
        return

    stats = report.get("dataset_statistics", {})
    quality = report.get("data_quality", {})
    print("AVIATRIX ML RESEARCH REPORT")
    print("")
    print(f"Generated: {report.get('generated_at', '--')}")
    print(f"Feature rows: {stats.get('feature_rows', '--')}")
    print(f"Selection rows: {stats.get('selection_rows', '--')}")
    print(f"Untouched holdout rows: {stats.get('holdout_rows', '--')}")
    print(f"Feature count: {stats.get('feature_count', '--')}")
    print("")
    print("DATA QUALITY")
    print(f"Total CSV rows: {quality.get('total_rows', '--')}")
    print(f"Valid rows: {quality.get('valid_rows', '--')}")
    print(f"Invalid multipliers: {quality.get('invalid_multipliers', '--')}")
    print(f"Duplicate round-id rows: {quality.get('duplicate_round_id_rows', '--')}")
    print(f"Rows without round id: {quality.get('rows_without_round_id', '--')}")
    print(f"Suspicious gaps: {quality.get('suspicious_gap_count', '--')}")
    print("")
    print("SELECTED MODELS AND FINAL HOLDOUT")
    selected = report.get("selected_models", {})
    holdout = report.get("final_holdout_results", {})
    if not selected:
        print("No selected models.")
    for target in sorted(selected, key=lambda item: float(item)):
        validation = selected[target].get("validation_metrics", {})
        holdout_metrics = holdout.get(target, {}).get("holdout_metrics", {})
        print(f">={float(target):.2f}x")
        print(f"  model: {selected[target].get('model_name', '--')}")
        print(
            "  validation: "
            f"brier_skill={format_metric(validation.get('brier_skill_score'))}, "
            f"balanced_skill={format_metric(validation.get('balanced_accuracy_skill'))}, "
            f"roc_auc={format_metric(validation.get('roc_auc'))}, "
            f"status={validation.get('validation_status', '--')}"
        )
        print(
            "  holdout: "
            f"brier_skill={format_metric(holdout_metrics.get('brier_skill_score'))}, "
            f"balanced_skill={format_metric(holdout_metrics.get('balanced_accuracy_skill'))}, "
            f"roc_auc={format_metric(holdout_metrics.get('roc_auc'))}, "
            f"status={holdout_metrics.get('validation_status', '--')}"
        )
    print("")
    print("CONCLUSION")
    print(report.get("overall_conclusion", "--"))
    print("")
    print("Guardrail:")
    print(report.get("guardrails", {}).get("holdout_rule", "--"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args(argv)
    report = load_report(Path(args.report))
    print_report(report)
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
