"""Feature generation and data-quality utilities for Aviatrix ML research.

The key invariant in this module is chronological safety: features for round
``t`` are built only from completed rounds before ``t``.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_ROUNDS_PATH = DATA_DIR / "rounds.csv"
DEFAULT_CONTEXT_PATH = DATA_DIR / "round_context.csv"

FEATURE_SCHEMA_VERSION = "multiplier-history-v1"
DEFAULT_TARGETS = [1.5, 2.0, 3.0, 5.0, 10.0]
ROLLING_WINDOWS = [3, 5, 10, 20, 50, 100]
MAX_LAG = 50
MAX_MULTIPLIER_CLIP = 100.0
SUSPICIOUS_GAP_SECONDS = 180
CONTEXT_FIELDS = [
    "player_count",
    "bet_count",
    "total_bet",
    "avg_bet",
    "max_bet",
    "cashed_out_count",
    "avg_cashout",
    "total_win",
]


@dataclass(frozen=True)
class FeatureDataset:
    """Container returned by feature generation."""

    frame: pd.DataFrame
    feature_names: List[str]
    target_names: List[str]
    quality_report: Dict
    source_report: Dict


def target_name(target: float) -> str:
    """Return a stable column name for a multiplier threshold."""

    return f"target_ge_{target:.2f}".replace(".", "_")


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse timestamps in the repository CSV format."""

    return pd.to_datetime(
        values,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def read_rounds_csv(path: Path = DEFAULT_ROUNDS_PATH) -> pd.DataFrame:
    """Read the rounds CSV without deduplicating legitimate identical rounds."""

    if not Path(path).exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "multiplier",
                "round_id",
                "source",
                "_original_index",
            ]
        )

    df = pd.read_csv(
        path,
        dtype={
            "timestamp": "string",
            "multiplier": "string",
            "round_id": "string",
            "source": "string",
        },
        keep_default_na=False,
    )
    for column in ("timestamp", "multiplier", "round_id", "source"):
        if column not in df.columns:
            df[column] = ""

    df = df[["timestamp", "multiplier", "round_id", "source"]].copy()
    df["_original_index"] = np.arange(len(df))
    return df


def clean_rounds(
    path: Path = DEFAULT_ROUNDS_PATH,
    suspicious_gap_seconds: int = SUSPICIOUS_GAP_SECONDS,
) -> Tuple[pd.DataFrame, Dict]:
    """Load, validate, sort, and deduplicate only by non-empty round_id."""

    raw = read_rounds_csv(Path(path))
    total_rows = int(len(raw))
    raw["multiplier_value"] = pd.to_numeric(raw["multiplier"], errors="coerce")
    invalid_mask = raw["multiplier_value"].isna() | (raw["multiplier_value"] < 1)
    invalid_rows = raw.loc[
        invalid_mask,
        ["_original_index", "timestamp", "multiplier", "round_id", "source"],
    ]

    valid = raw.loc[~invalid_mask].copy()
    valid["multiplier"] = valid["multiplier_value"].astype(float)
    valid.drop(columns=["multiplier_value"], inplace=True)
    valid["timestamp_dt"] = parse_timestamp_series(valid["timestamp"])
    valid["source"] = valid["source"].fillna("").astype(str)
    valid["round_id"] = valid["round_id"].fillna("").astype(str)

    timestamp_invalid = int(valid["timestamp_dt"].isna().sum())
    original_time = valid.sort_values("_original_index")["timestamp_dt"]
    non_monotonic_original = int(
        (original_time.dropna().diff().dt.total_seconds() < 0).sum()
    )

    valid["_sort_ts"] = valid["timestamp_dt"].fillna(pd.Timestamp.max)
    valid = valid.sort_values(["_sort_ts", "_original_index"]).reset_index(drop=True)
    valid.drop(columns=["_sort_ts"], inplace=True)

    has_round_id = valid["round_id"].str.len() > 0
    duplicate_round_id_mask = has_round_id & valid.duplicated("round_id", keep="first")
    duplicate_round_ids = (
        valid.loc[duplicate_round_id_mask, "round_id"].value_counts().to_dict()
    )
    valid = valid.loc[~duplicate_round_id_mask].copy().reset_index(drop=True)
    valid["round_number"] = np.arange(1, len(valid) + 1)

    sorted_times = valid["timestamp_dt"].dropna()
    gaps = sorted_times.diff().dt.total_seconds()
    suspicious_gaps = []
    if not gaps.empty:
        for index, gap in gaps[gaps > suspicious_gap_seconds].items():
            previous_index = index - 1
            suspicious_gaps.append(
                {
                    "previous_timestamp": (
                        valid.loc[previous_index, "timestamp"]
                        if previous_index in valid.index
                        else ""
                    ),
                    "timestamp": valid.loc[index, "timestamp"],
                    "gap_seconds": float(gap),
                }
            )

    source_counts = Counter(valid["source"].replace("", "unlabeled"))
    report = {
        "path": str(Path(path)),
        "total_rows": total_rows,
        "valid_rows": int(len(valid)),
        "invalid_multipliers": int(invalid_mask.sum()),
        "invalid_multiplier_examples": invalid_rows.head(20).to_dict("records"),
        "duplicate_round_id_rows": int(duplicate_round_id_mask.sum()),
        "duplicate_round_ids": duplicate_round_ids,
        "rows_without_round_id": int((valid["round_id"].str.len() == 0).sum()),
        "source_counts": dict(source_counts),
        "minimum_timestamp": (
            None if sorted_times.empty else sorted_times.min().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "maximum_timestamp": (
            None if sorted_times.empty else sorted_times.max().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "invalid_timestamps": timestamp_invalid,
        "timestamp_anomalies": {
            "non_monotonic_in_original_order": non_monotonic_original,
        },
        "suspicious_gap_seconds": suspicious_gap_seconds,
        "suspicious_gaps": suspicious_gaps[:50],
        "suspicious_gap_count": len(suspicious_gaps),
    }
    return valid, report


def multiplier_bucket(value: float) -> int:
    """Encode a multiplier into a coarse ordered bucket."""

    if value < 1.2:
        return 0
    if value < 1.5:
        return 1
    if value < 2.0:
        return 2
    if value < 3.0:
        return 3
    if value < 5.0:
        return 4
    if value < 10.0:
        return 5
    if value < 20.0:
        return 6
    if value < 50.0:
        return 7
    if value < 100.0:
        return 8
    return 9


def bucket_entropy(values: Sequence[float]) -> float:
    """Calculate bucket entropy for a rolling window."""

    clean = [float(value) for value in values if not pd.isna(value)]
    if not clean:
        return 0.0

    counts = Counter(multiplier_bucket(value) for value in clean)
    total = len(clean)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return float(entropy)


def pattern_code(bucket_values: Sequence[int], length: int) -> int:
    """Encode the latest bucket pattern as a bounded deterministic integer."""

    code = 0
    recent = list(bucket_values)[-length:]
    padded = [0] * max(0, length - len(recent)) + recent
    for item in padded:
        code = (code * 11 + int(item) + 1) % 1_000_003
    return int(code)


def consecutive_count(values: Sequence[float], predicate) -> int:
    """Count consecutive previous rounds satisfying predicate."""

    count = 0
    for value in reversed(values):
        if predicate(float(value)):
            count += 1
        else:
            break
    return count


def rounds_since_last(values: Sequence[float], threshold: float) -> int:
    """Count rounds since the most recent previous value at or above threshold."""

    for offset, value in enumerate(reversed(values), start=1):
        if float(value) >= threshold:
            return offset - 1
    return len(values) + 1


def add_streak_features(features: pd.DataFrame, values: np.ndarray) -> None:
    """Add leakage-safe streak and rounds-since features."""

    below_thresholds = [1.2, 1.5, 2.0, 3.0]
    since_thresholds = [2.0, 3.0, 5.0, 10.0, 20.0]
    below_streaks = {threshold: [] for threshold in below_thresholds}
    above_2_streak = []
    since_values = {threshold: [] for threshold in since_thresholds}
    previous_values: List[float] = []

    for value in values:
        for threshold in below_thresholds:
            below_streaks[threshold].append(
                consecutive_count(previous_values, lambda item, t=threshold: item < t)
            )
        above_2_streak.append(
            consecutive_count(previous_values, lambda item: item >= 2.0)
        )
        for threshold in since_thresholds:
            since_values[threshold].append(rounds_since_last(previous_values, threshold))
        previous_values.append(float(value))

    for threshold in below_thresholds:
        features[f"streak_below_{threshold:.1f}".replace(".", "_")] = below_streaks[
            threshold
        ]
    features["streak_ge_2_0"] = above_2_streak
    for threshold in since_thresholds:
        features[f"rounds_since_ge_{threshold:.1f}".replace(".", "_")] = since_values[
            threshold
        ]


def add_pattern_features(features: pd.DataFrame, values: np.ndarray) -> None:
    """Add bucket and recent pattern features based only on previous rounds."""

    buckets = [multiplier_bucket(float(value)) for value in values]
    current_buckets = []
    pattern_3 = []
    pattern_5 = []
    pattern_10 = []
    previous_buckets: List[int] = []

    for bucket in buckets:
        current_buckets.append(previous_buckets[-1] if previous_buckets else 0)
        pattern_3.append(pattern_code(previous_buckets, 3))
        pattern_5.append(pattern_code(previous_buckets, 5))
        pattern_10.append(pattern_code(previous_buckets, 10))
        previous_buckets.append(bucket)

    features["current_multiplier_bucket"] = current_buckets
    features["bucket_pattern_3"] = pattern_3
    features["bucket_pattern_5"] = pattern_5
    features["bucket_pattern_10"] = pattern_10


def build_context_features(rounds: pd.DataFrame, context_path: Path) -> pd.DataFrame:
    """Attach optional context rows observed before the target round timestamp."""

    output = pd.DataFrame(index=rounds.index)
    if not Path(context_path).exists() or rounds.empty:
        return output

    try:
        context = pd.read_csv(context_path, keep_default_na=False)
    except Exception:
        return output

    if "observed_at" not in context.columns:
        return output

    context["observed_dt"] = parse_timestamp_series(context["observed_at"].astype(str))
    context = context.dropna(subset=["observed_dt"]).sort_values("observed_dt")
    if context.empty:
        return output

    for field in CONTEXT_FIELDS:
        if field not in context.columns:
            context[field] = np.nan
        context[field] = pd.to_numeric(context[field], errors="coerce")

    left = rounds[["timestamp_dt"]].copy()
    left["_row_index"] = rounds.index
    left = left.dropna(subset=["timestamp_dt"]).sort_values("timestamp_dt")
    merged = pd.merge_asof(
        left,
        context[["observed_dt", *CONTEXT_FIELDS]],
        left_on="timestamp_dt",
        right_on="observed_dt",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.set_index("_row_index")
    for field in CONTEXT_FIELDS:
        output[f"context_{field}"] = merged[field].reindex(rounds.index).fillna(0.0)
        output[f"context_{field}_known"] = (
            merged[field].reindex(rounds.index).notna().astype(int)
        )
    return output


def generate_features(
    rounds: pd.DataFrame,
    targets: Iterable[float] = DEFAULT_TARGETS,
    min_history: int = 100,
    include_context: bool = False,
    context_path: Path = DEFAULT_CONTEXT_PATH,
) -> FeatureDataset:
    """Generate leakage-safe features and binary threshold targets."""

    if rounds.empty:
        quality = {
            "total_rows": 0,
            "valid_rows": 0,
        }
        return FeatureDataset(pd.DataFrame(), [], [], quality, {})

    values = rounds["multiplier"].astype(float).to_numpy()
    features = pd.DataFrame(index=rounds.index)
    previous = rounds["multiplier"].astype(float).shift(1)
    clipped_previous = previous.clip(upper=MAX_MULTIPLIER_CLIP)

    for lag in range(1, MAX_LAG + 1):
        lagged = rounds["multiplier"].astype(float).shift(lag).clip(
            upper=MAX_MULTIPLIER_CLIP
        )
        features[f"lag_{lag}"] = lagged.fillna(0.0)

    features["lag1_log"] = np.log1p(clipped_previous.fillna(0.0))
    features["lag2_log"] = np.log1p(
        rounds["multiplier"].astype(float).shift(2).clip(upper=MAX_MULTIPLIER_CLIP).fillna(0.0)
    )
    features["change_lag1_lag2"] = (
        previous - rounds["multiplier"].astype(float).shift(2)
    ).clip(lower=-MAX_MULTIPLIER_CLIP, upper=MAX_MULTIPLIER_CLIP).fillna(0.0)
    features["log_change_lag1_lag2"] = (
        features["lag1_log"] - features["lag2_log"]
    ).fillna(0.0)

    for window in ROLLING_WINDOWS:
        rolling = clipped_previous.rolling(window=window, min_periods=1)
        features[f"roll_{window}_count"] = rolling.count().fillna(0.0)
        features[f"roll_{window}_mean"] = rolling.mean().fillna(0.0)
        features[f"roll_{window}_median"] = rolling.median().fillna(0.0)
        std = rolling.std(ddof=0).fillna(0.0)
        features[f"roll_{window}_std"] = std
        features[f"roll_{window}_min"] = rolling.min().fillna(0.0)
        features[f"roll_{window}_max"] = rolling.max().fillna(0.0)
        features[f"roll_{window}_q25"] = rolling.quantile(0.25).fillna(0.0)
        features[f"roll_{window}_q75"] = rolling.quantile(0.75).fillna(0.0)
        mean = features[f"roll_{window}_mean"].replace(0, np.nan)
        features[f"roll_{window}_cv"] = (std / mean).replace([np.inf, -np.inf], 0).fillna(0.0)
        features[f"roll_{window}_entropy"] = (
            clipped_previous.rolling(window=window, min_periods=1)
            .apply(bucket_entropy, raw=True)
            .fillna(0.0)
        )

        for threshold in [1.2, 1.5, 2.0]:
            name = str(threshold).replace(".", "_")
            features[f"roll_{window}_prop_lt_{name}"] = (
                (previous < threshold).astype(float)
                .rolling(window=window, min_periods=1)
                .mean()
                .fillna(0.0)
            )
        for threshold in [2.0, 3.0, 5.0, 10.0]:
            name = str(threshold).replace(".", "_")
            above = (previous >= threshold).astype(float)
            features[f"roll_{window}_prop_ge_{name}"] = (
                above.rolling(window=window, min_periods=1).mean().fillna(0.0)
            )
            features[f"roll_{window}_count_ge_{name}"] = (
                above.rolling(window=window, min_periods=1).sum().fillna(0.0)
            )

    add_streak_features(features, values)
    add_pattern_features(features, values)

    if include_context:
        context_features = build_context_features(rounds, Path(context_path))
        if not context_features.empty:
            features = pd.concat([features, context_features], axis=1)

    metadata = rounds[
        ["timestamp", "timestamp_dt", "round_id", "source", "round_number", "multiplier"]
    ].copy()
    frame = pd.concat([metadata, features], axis=1)
    target_names = []
    for target in targets:
        name = target_name(float(target))
        target_names.append(name)
        frame[name] = (frame["multiplier"] >= float(target)).astype(int)

    min_history = max(1, int(min_history))
    if len(frame) > min_history:
        frame = frame.loc[frame["round_number"] > min_history].copy()
    elif len(frame) > 1:
        frame = frame.iloc[1:].copy()

    feature_names = [
        column
        for column in frame.columns
        if column
        not in {
            "timestamp",
            "timestamp_dt",
            "round_id",
            "source",
            "round_number",
            "multiplier",
            *target_names,
        }
    ]
    frame[feature_names] = frame[feature_names].replace([np.inf, -np.inf], 0).fillna(0.0)
    source_report = analyze_sources(rounds)

    return FeatureDataset(
        frame=frame.reset_index(drop=True),
        feature_names=feature_names,
        target_names=target_names,
        quality_report={},
        source_report=source_report,
    )


def analyze_sources(rounds: pd.DataFrame) -> Dict:
    """Compare source labels without assuming demo/real are equivalent."""

    if rounds.empty:
        return {
            "source_counts": {},
            "source_stats": {},
        }

    source_counts = rounds["source"].replace("", "unlabeled").value_counts().to_dict()
    source_stats = {}
    for source, group in rounds.groupby(rounds["source"].replace("", "unlabeled")):
        values = group["multiplier"].astype(float)
        source_stats[source] = {
            "count": int(len(group)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "rate_ge_2": float((values >= 2.0).mean()),
            "rate_ge_10": float((values >= 10.0).mean()),
            "maximum": float(values.max()),
        }

    overlap_matches = 0
    overlap_total = 0
    if {"demo", "real"}.issubset(set(rounds["source"].unique())):
        demo = rounds.loc[rounds["source"] == "demo", ["timestamp", "multiplier"]]
        real = rounds.loc[rounds["source"] == "real", ["timestamp", "multiplier"]]
        overlap = demo.merge(real, on="timestamp", suffixes=("_demo", "_real"))
        overlap_total = int(len(overlap))
        if overlap_total:
            overlap_matches = int(
                np.isclose(
                    overlap["multiplier_demo"].astype(float),
                    overlap["multiplier_real"].astype(float),
                    rtol=0,
                    atol=0.005,
                ).sum()
            )

    return {
        "source_counts": source_counts,
        "source_stats": source_stats,
        "demo_real_overlap": {
            "timestamp_overlap_rows": overlap_total,
            "exact_multiplier_matches": overlap_matches,
            "match_rate": (
                None if overlap_total == 0 else overlap_matches / overlap_total
            ),
        },
    }


def load_feature_dataset(
    csv_path: Path = DEFAULT_ROUNDS_PATH,
    targets: Iterable[float] = DEFAULT_TARGETS,
    min_history: int = 100,
    include_context: bool = False,
    context_path: Path = DEFAULT_CONTEXT_PATH,
) -> FeatureDataset:
    """Load rounds and return quality plus leakage-safe features."""

    rounds, quality_report = clean_rounds(Path(csv_path))
    dataset = generate_features(
        rounds,
        targets=targets,
        min_history=min_history,
        include_context=include_context,
        context_path=context_path,
    )
    return FeatureDataset(
        frame=dataset.frame,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        quality_report=quality_report,
        source_report=dataset.source_report,
    )


def next_round_feature_frame(
    csv_path: Path = DEFAULT_ROUNDS_PATH,
    feature_names: Optional[Sequence[str]] = None,
    min_history: int = 100,
    include_context: bool = False,
    context_path: Path = DEFAULT_CONTEXT_PATH,
) -> Tuple[pd.DataFrame, Dict]:
    """Build one feature row for the next unknown round from completed history."""

    rounds, quality_report = clean_rounds(Path(csv_path))
    if rounds.empty:
        return pd.DataFrame(), quality_report

    pseudo = rounds.copy()
    latest = rounds.tail(1).copy()
    latest.loc[:, "timestamp"] = ""
    last_timestamp = rounds["timestamp_dt"].dropna().max()
    latest.loc[:, "timestamp_dt"] = (
        last_timestamp + pd.Timedelta(seconds=1)
        if pd.notna(last_timestamp)
        else pd.Timestamp.now()
    )
    latest.loc[:, "round_id"] = ""
    latest.loc[:, "round_number"] = int(rounds["round_number"].max()) + 1
    latest.loc[:, "multiplier"] = 1.0
    pseudo = pd.concat([pseudo, latest], ignore_index=True, sort=False)
    dataset = generate_features(
        pseudo,
        targets=DEFAULT_TARGETS,
        min_history=min_history,
        include_context=include_context,
        context_path=context_path,
    )
    frame = dataset.frame.tail(1).copy()
    if feature_names is not None:
        for name in feature_names:
            if name not in frame.columns:
                frame[name] = 0.0
        frame = frame[list(feature_names)]
    return frame, quality_report


def write_json(path: Path, payload: Dict) -> None:
    """Write deterministic JSON output."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
