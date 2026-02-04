"""Compare each time slice to baseline quality; log metrics and alert flags. Writes data_quality_log.csv."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

ALERT_MISSING_DELTA = 0.10   # Alert if missing rate spike vs baseline exceeds this
ALERT_OUTLIER_DELTA = 0.05   # Alert if avg outlier rate exceeds this
ALERT_QUALITY_THRESHOLD = 0.85  # Alert if quality_score falls below this


def load_baseline_stat(baseline_stat_path: Path) -> Dict:
    """Load baseline quality stats JSON. Raises FileNotFoundError if missing."""
    if not baseline_stat_path.exists():
        raise FileNotFoundError(f"Baseline stats file not found at {baseline_stat_path}")

    with open(baseline_stat_path, "r") as f:
       return json.load(f)


def run_quality_check(slice_df: pd.DataFrame, baseline_missing: pd.Series, outlier_bounds: Dict, numeric_features: List) -> dict:
    """Compute avg_missing_rate, avg_missing_delta, avg_outlier_rate, quality_score for one slice vs baseline."""
    current_missing = slice_df.isna().mean()
    common_cols = [col for col in current_missing.index if col in baseline_missing.index]
    missing_delta = current_missing[common_cols] - baseline_missing[common_cols]

    avg_missing_delta = missing_delta.abs().mean()
    avg_missing_rate = current_missing[common_cols].mean()

    outlier_rates = []
    for col in numeric_features:
        if col not in slice_df.columns:
            continue
        bounds = outlier_bounds[col]
        lower, upper = bounds["lower"], bounds["upper"]

        valid = slice_df[col].dropna()
        if len(valid) == 0:
            continue

        outside = ((valid < lower) | (valid > upper)).sum()
        outlier_rates.append(outside / len(valid))

    avg_outlier_rate = np.mean(outlier_rates) if outlier_rates else 0.0

    quality_score = 1.0 - (avg_missing_rate + avg_outlier_rate)

    return {
        "avg_missing_rate": float(avg_missing_rate),
        "avg_missing_delta": float(avg_missing_delta),
        "avg_outlier_rate": float(avg_outlier_rate),
        "quality_score": float(quality_score)
    }

def trigger_quality_alert(slice_id: str, metrics: Dict, baseline_missing: pd.Series) -> Dict:
    """Add slice_id, timestamp, and alert_flag to metrics. alert_flag True if missing spike, outlier rate, or low quality_score."""
    baseline_avg_missing = baseline_missing.mean()

    missing_spike = metrics["avg_missing_rate"] - baseline_avg_missing

    alert_flag = (
        missing_spike > ALERT_MISSING_DELTA
        or metrics["avg_outlier_rate"] > ALERT_OUTLIER_DELTA
        or metrics["quality_score"] < ALERT_QUALITY_THRESHOLD
    )

    return {
        "slice_id": slice_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metrics,
        "alert_flag": alert_flag,
    }


def main():
    """Run quality check on each slice_*.csv, write data_quality_log.csv."""
    project_root = Path(__file__).resolve().parents[2]
    baseline_stats_path = (project_root / "artifacts" / "baseline" / "quality_stats.json")
    slice_dir = project_root / "data" / "time_slices"
    output_log = (project_root / "artifacts" / "monitoring" / "data_quality_log.csv")
    output_log.parent.mkdir(parents=True, exist_ok=True)

    baseline_stats = load_baseline_stat(baseline_stats_path)
    baseline_missing = pd.Series(baseline_stats["missing_rate"])
    outlier_bounds = baseline_stats["outlier_bounds"]
    numeric_features = list(outlier_bounds.keys())

    results = []
    for slice_path in sorted(slice_dir.glob("slice_*.csv")):
        slice_id = slice_path.stem 
        slice_df = pd.read_csv(slice_path)

        metrics = run_quality_check(
            slice_df,
            baseline_missing,
            outlier_bounds,
            numeric_features,
        )

        record = trigger_quality_alert(
            slice_id=slice_id,
            metrics=metrics,
            baseline_missing=baseline_missing,
        )

        results.append(record)
    if results:
        df_log = pd.DataFrame(results)
        df_log.to_csv(output_log, index=False)
        print(f"Quality check completed for {len(results)} slices.")
    else:
        print("No slice_*.csv files found. Nothing to evaluate.")
    
    print("\n\n2.2 - quality check completion")


def run():
    """Entry point: run main()."""
    main()
