"""Evaluate baseline model on each time slice; compare to baseline stats. Writes performance_log.csv."""
from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def load_model(model_path: Path):
    """Load pickled model from path. Raises FileNotFoundError if missing."""
    if not model_path.exists():
        raise FileNotFoundError(f"model is not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluate_slice(slice_df: pd.DataFrame, model):
    """Predict on slice, return precision, recall, f1 (zero_division=0)."""
    df = slice_df.copy()
    df.drop(["TransactionID", "time_slice"], axis=1, inplace=True, errors="ignore")
    TARGET = "isFraud"
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    y_pred = model.predict(X)

    return {
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

def compare_stats(baseline_stats_path: Path, slice_metrics: dict) -> dict:
    """Load baseline stats JSON and return precision_drop, recall_drop, f1_drop (baseline minus slice)."""
    with open(baseline_stats_path) as f:
        baseline = json.load(f)
    return {
        "precision_drop": baseline["precision"] - slice_metrics["precision"],
        "recall_drop": baseline["recall"] - slice_metrics["recall"],
        "f1_drop": baseline["f1"] - slice_metrics["f1"],
    }



def main():
    """Evaluate baseline model on each slice_*.csv, compare to baseline stats, write performance_log.csv."""
    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "models" / "baseline_model.pkl"
    slice_dir = project_root / "data" / "time_slices"
    baseline_stats_path = project_root / "artifacts" / "monitoring" / "baseline_model_stats.json"
    output_path = (project_root / "artifacts" / "monitoring" / "performance_log.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    records = []
    for slice_path in sorted(slice_dir.glob("slice_*.csv")):
        slice_id = slice_path.stem
        slice_df = pd.read_csv(slice_path)
        metrics = evaluate_slice(slice_df, model)
        deltas = compare_stats(baseline_stats_path, metrics)
        records.append({
            "slice_id": slice_id,
            **metrics,
            **deltas
        })

    if records:
        pd.DataFrame(records).to_csv(output_path, index=False)
        print(f"Performance evaluation completed for {len(records)} slices.")
    else:
        print("No slices found.")


def run():
    """Entry point: run main()."""
    main()