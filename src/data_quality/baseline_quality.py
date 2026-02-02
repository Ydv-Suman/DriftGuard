from pathlib import Path
import pandas as pd
import numpy as np
import json


ignore_cols = ["TransactionID", "isFraud", "time_slice"]

def load_baseline(baseline_path: Path) -> pd.DataFrame:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {baseline_path}")
    return pd.read_csv(baseline_path)


def get_feature_groups(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    numeric_cols = [col for col in numeric_cols if col not in ignore_cols]
    categorical_cols = [col for col in categorical_cols if col not in ignore_cols]

    return numeric_cols, categorical_cols

def compute_missing_rate(df: pd.DataFrame):
    return df.isna().mean().to_dict()

def compute_outlier_bounds(df: pd.DataFrame, numeric_cols):
    bounds = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "lower": float(lower),
            "upper": float(upper)
        }

    return bounds

def compute_feature_stats(df: pd.DataFrame, numeric_cols):
    stats = {}

    for col in numeric_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    return stats


def save_quality_artifact(artifact: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=4)

    print(f"Baseline quality artifact saved to {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]

    baseline_path = project_root / "data" / "time_slices" / "baseline.csv"
    output_path = project_root / "artifacts" / "baseline" / "quality_stats.json"

    df = load_baseline(baseline_path)
    numeric_cols, categorical_cols = get_feature_groups(df)

    baseline_quality = {
        "missing_rate": compute_missing_rate(df),
        "outlier_bounds": compute_outlier_bounds(df, numeric_cols),
        "feature_stats": compute_feature_stats(df, numeric_cols),
        "metadata": {
            "num_rows": len(df),
            "num_numeric_features": len(numeric_cols),
            "num_categorical_features": len(categorical_cols)
        }
    }

    save_quality_artifact(baseline_quality, output_path)


if __name__ == "__main__":
    main()