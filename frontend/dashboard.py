import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="DriftGuard Dashboard", layout="wide")

st.title("DriftGuard – Model Monitoring Dashboard")

# Resolve paths from repo root so artifacts are found from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.header(" Retraining Decision")

## Decision
DECISION_PATH = PROJECT_ROOT / "artifacts" / "decision" / "retraining_decisions.json"
if DECISION_PATH.exists():
    decisions = json.loads(DECISION_PATH.read_text())

    if not decisions:
        st.warning("No per-slice decisions yet. Run the pipeline.")
        st.stop()

    slice_id = st.selectbox(
        "Select time slice",
        options=sorted(decisions.keys())
    )

    decision = decisions[slice_id]

    col1, col2 = st.columns(2)
    col1.metric("Decision", decision["decision"])
    col2.metric("Confidence", decision["confidence"])

    st.subheader("Reasons")
    for r in decision["reasons"]:
        st.write(f"- {r}")

else:
    st.warning("No decision file found. Run the pipeline.")
    st.stop()

st.header("Explainability (Feature Impact)")


## Feature inpact - Analysis
EXPLAIN_DIR = PROJECT_ROOT / "artifacts" / "explainability"

impact_path = EXPLAIN_DIR / f"{slice_id}_feature_impact_delta.csv"

if impact_path.exists():
    impact_df = pd.read_csv(impact_path)

    st.caption("Top features whose importance changed the most vs baseline")
    st.dataframe(
        impact_df.head(15),
        use_container_width=True
    )
else:
    st.info("No explainability generated for this slice (slice considered healthy).")


# Model performance
PERF_PATH = PROJECT_ROOT / "artifacts" / "monitoring" / "performance_log.csv"

st.header("Model Performance Over Time")

if PERF_PATH.exists():
    perf_df = pd.read_csv(PERF_PATH)

    metric_choice = st.multiselect(
        "Select metrics to view",
        options=["recall", "f1", "precision"],
        default=["recall", "f1"]
    )

    if metric_choice:
        st.line_chart(
            perf_df.set_index("slice_id")[metric_choice]
        )
else:
    st.warning("Performance log not found.")


## Drift Detail
DRIFT_PATH = PROJECT_ROOT / "artifacts" / "monitoring" / "drift_log.csv"

st.header("Feature Drift Overview")

if DRIFT_PATH.exists():
    drift_df = pd.read_csv(DRIFT_PATH)

    top_drift = (
        drift_df.groupby("feature")["psi"]
        .max()
        .sort_values(ascending=False)
        .head(10)
    )

    st.caption("Top features with highest PSI drift across slices")
    st.bar_chart(top_drift)
else:
    st.warning("Drift log not found.")
