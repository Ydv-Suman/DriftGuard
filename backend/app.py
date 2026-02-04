from fastapi import FastAPI
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import main as run_pipeline

app = FastAPI(title="DriftGuard API")


@app.post("/run")
def run_driftguard():
    """
    Trigger full DriftGuard pipeline
    """
    run_pipeline()
    return {"status": "completed"}


@app.get("/decision")
def get_decision():
    """
    Fetch latest retraining decision
    """
    decision_path = Path("artifacts/decision/retraining_decision.json")
    if not decision_path.exists():
        return {"error": "Decision not found"}

    with open(decision_path) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}
