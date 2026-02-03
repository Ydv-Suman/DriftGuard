from pathlib import Path
import json
import pickle
from typing import dict

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"model is not found at {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluate_slice(slice_df: pd.DataFrame, model):
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


def evaluate_performance(baseline_metrics: dict, slice_metrics:dict) -> dict:
    output = {}
    