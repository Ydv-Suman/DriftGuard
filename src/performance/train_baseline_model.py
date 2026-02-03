from pathlib import Path
import pandas as pd
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def load_baseline(baseline_path: Path) -> pd.DataFrame:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {baseline_path}")
    return pd.read_csv(baseline_path)


def train_baseline_model(df: pd.DataFrame):
    df = df.copy()

    df = df.drop(columns=["TransactionID", "time_slice"], errors="ignore")

    TARGET = "isFraud"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "string"]).columns

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model, X, y


def main():
    project_root = Path(__file__).resolve().parents[2]

    baseline_path = project_root / "data" / "time_slices" / "baseline.csv"
    model_path = project_root / "models" / "baseline_model.pkl"
    stats_path = project_root / "artifacts" / "monitoring" / "baseline_model_stats.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_baseline(baseline_path)
    model, X, y = train_baseline_model(df)

    y_pred = model.predict(X)

    stats = {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "support": int(y.sum())
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
