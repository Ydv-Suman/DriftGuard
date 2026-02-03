from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Dict
from sklearn.metrics import f1_score, make_scorer
from sklearn.inspection import permutation_importance as sk_permutation_importance


BASELINE_SLICE = "baseline.csv"
DEGRADED_SLICE = "slice_6.csv"
N_PERMUTATION_REPEATS = 3
RANDOM_STATE = 42
MAX_SAMPLES_FOR_IMPORTANCE = 5000


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"model is not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_slice(slice_path: Path):
    if not slice_path.exists():
        raise FileNotFoundError(f"slice df not found at {slice_path}")
    return pd.read_csv(slice_path, low_memory=False)

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=["TransactionID", "time_slice"], errors="ignore")
    TARGET = "isFraud"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


F1_SCORER = make_scorer(f1_score, zero_division=0)


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = N_PERMUTATION_REPEATS,
    max_samples: int | None = MAX_SAMPLES_FOR_IMPORTANCE,
    n_jobs: int = -1,
) -> Dict[str, float]:
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X.iloc[idx].copy(), y.iloc[idx]
    result = sk_permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring=F1_SCORER,
        n_jobs=n_jobs,
    )
    return dict(zip(X.columns, result.importances_mean))

def main():
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "baseline_model.pkl"
    slice_dir = project_root / "data" / "time_slices"
    output_dir = project_root / "artifacts" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    print("model loaded")
    baseline_df = load_slice(slice_dir / BASELINE_SLICE)
    degraded_df = load_slice(slice_dir / DEGRADED_SLICE)
    print("slice loaded")
    
    X_base, y_base = prepare_xy(baseline_df)
    X_deg, y_deg = prepare_xy(degraded_df)
    print("data prepared")

    base_importance = permutation_importance(model, X_base, y_base)
    deg_importance = permutation_importance(model, X_deg, y_deg)
    print("permutation")

    base_df = pd.DataFrame(
        base_importance.items(),
        columns=["feature", "importance_baseline"]
    )

    deg_df = pd.DataFrame(
        deg_importance.items(),
        columns=["feature", "importance_degraded"]
    )

    delta_df = base_df.merge(deg_df, on="feature", how="inner")
    delta_df["importance_delta"] = (
        delta_df["importance_degraded"] -
        delta_df["importance_baseline"]
    )
    delta_df["abs_importance_delta"] = delta_df["importance_delta"].abs()

    delta_df = delta_df.sort_values(
        by="abs_importance_delta",
        ascending=False
    )

    base_df.to_csv(output_dir / "feature_impact_baseline.csv", index=False)
    deg_df.to_csv(output_dir / "feature_impact_degraded.csv", index=False)
    delta_df.to_csv(output_dir / "feature_impact_delta.csv", index=False)
    print("saved")

if __name__ == "__main__":
    main()