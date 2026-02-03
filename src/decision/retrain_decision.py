"""Combine performance, quality, drift, and feature-impact signals. Writes retraining_decision.json."""
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


RECALL_DROP_THRESHOLD = 0.40      # Flag bad performance if recall drop exceeds this
F1_DROP_THRESHOLD = 0.25         # Flag bad performance if F1 drop exceeds this
QUALITY_ALERT_RATE = 0.5        # Flag quality issues if alert rate across slices exceeds this
SEVERE_PSI_THRESHOLD = 0.25  # PSI above this is treated as severe drift
MIN_SEVERE_DRIFT_FEATURES = 3   # Need this many severe-PSI features to flag drift
MIN_IMPACT_COLLAPSE_FEATURES = 3  # Need this many negative importance_delta to flag impact collapse


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV from path. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)



def check_performance(perf_df: pd.DataFrame) -> tuple[bool, list]:
    """True if recall_drop or f1_drop exceeds thresholds. Returns (is_bad, list of reason strings)."""
    reasons = []
    if perf_df["recall_drop"].max() > RECALL_DROP_THRESHOLD:
        reasons.append(f"Recall dropped by {perf_df['recall_drop'].max():.0%}")

    if perf_df["f1_drop"].max() > F1_DROP_THRESHOLD:
        reasons.append(f"F1-score dropped by {perf_df['f1_drop'].max():.0%}")

    return len(reasons) > 0, reasons


def check_quality(quality_df: pd.DataFrame) -> tuple[bool, list]:
    """True if fraction of slices with alert_flag exceeds QUALITY_ALERT_RATE. Returns (is_bad, reasons)."""
    alert_rate = quality_df["alert_flag"].mean()

    if alert_rate > QUALITY_ALERT_RATE:
        return True, [f"Quality alerts triggered in {alert_rate:.0%} of slices"]
    return False, []


def check_drift(drift_df: pd.DataFrame) -> tuple[bool, list]:
    """True if number of features with psi > SEVERE_PSI_THRESHOLD meets MIN_SEVERE_DRIFT_FEATURES. Returns (is_bad, reasons)."""
    severe = drift_df[drift_df["psi"] > SEVERE_PSI_THRESHOLD]
    count = severe["feature"].nunique()

    if count >= MIN_SEVERE_DRIFT_FEATURES:
        return True, [f"{count} features show severe PSI drift (> {SEVERE_PSI_THRESHOLD})"]

    return False, []


def check_feature_impact(impact_df: pd.DataFrame) -> tuple[bool, list]:
    """True if count of features with negative importance_delta meets MIN_IMPACT_COLLAPSE_FEATURES. Returns (is_bad, reasons)."""
    collapsed = impact_df[impact_df["importance_delta"] < 0]
    count = collapsed.shape[0]

    if count >= MIN_IMPACT_COLLAPSE_FEATURES:
        return True, [f"{count} high-importance features lost predictive power"]

    return False, []


def decide(performance_bad: bool, quality_bad: bool, drift_bad: bool, impact_bad: bool, reasons: list) -> dict:
    """Choose RETRAIN, INVESTIGATE_DATA_PIPELINE, MONITOR, or NO_ACTION from flags and reasons. Returns decision dict with timestamp."""
    if performance_bad and drift_bad and impact_bad:
        decision = "RETRAIN"
        confidence = "HIGH"

    elif quality_bad and performance_bad:
        decision = "INVESTIGATE_DATA_PIPELINE"
        confidence = "HIGH"

    elif drift_bad and not performance_bad:
        decision = "MONITOR"
        confidence = "MEDIUM"

    else:
        decision = "NO_ACTION"
        confidence = "LOW"

    return {
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
        "timestamp": datetime.utcnow().isoformat()
    }


def main():
    """Load monitoring CSVs and impact CSV, run checks, decide action, write retraining_decision.json."""
    project_root = Path(__file__).resolve().parents[2]

    quality_path = project_root / "artifacts" / "monitoring" / "data_quality_log.csv"
    perf_path = project_root / "artifacts" / "monitoring" / "performance_log.csv"
    drift_path = project_root / "artifacts" / "monitoring" / "drift_log.csv"
    impact_path = project_root / "artifacts" / "explainability" / "feature_impact_delta.csv"

    output_dir = project_root / "artifacts" / "decision"
    output_dir.mkdir(parents=True, exist_ok=True)


    quality_df = load_csv(quality_path)
    perf_df = load_csv(perf_path)
    drift_df = load_csv(drift_path)
    impact_df = load_csv(impact_path)


    reasons = []

    performance_bad, r = check_performance(perf_df)
    reasons.extend(r)

    quality_bad, r = check_quality(quality_df)
    reasons.extend(r)

    drift_bad, r = check_drift(drift_df)
    reasons.extend(r)

    impact_bad, r = check_feature_impact(impact_df)
    reasons.extend(r)


    decision = decide(
        performance_bad,
        quality_bad,
        drift_bad,
        impact_bad,
        reasons
    )


    output_path = output_dir / "retraining_decision.json"
    with open(output_path, "w") as f:
        json.dump(decision, f, indent=2)

    print("Retraining Decision Engine completed")
    print(json.dumps(decision, indent=2))


def run():
    """Entry point: run main()."""
    main()