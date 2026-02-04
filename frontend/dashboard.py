import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="DriftGuard Dashboard", layout="wide")

st.title("🛡 DriftGuard – Model Monitoring Dashboard")

st.header("Retraining Decision")

decisions_path = Path("artifacts/decision/retraining_decisions.json")

if decisions_path.exists():
    with open(decisions_path) as f:
        decisions = json.load(f)

    if not decisions:
        st.warning("No per-slice decisions yet. Run the pipeline.")
    else:
        slice_id = st.selectbox(
            "Select time slice",
            options=list(decisions.keys()),
        )

        decision = decisions[slice_id]
        st.metric("Decision", decision["decision"])
        st.metric("Confidence", decision["confidence"])

        st.subheader("Reasons")
        for r in decision["reasons"]:
            st.write(f"- {r}")
else:
    st.warning("No decision available. Run the pipeline.")


st.header("Model Performance Over Time")

perf_path = Path("artifacts/monitoring/performance_log.csv")
if perf_path.exists():
    perf_df = pd.read_csv(perf_path)
    st.line_chart(perf_df.set_index("slice_id")[["recall", "f1"]])


st.header("Top Drifted Features")

drift_path = Path("artifacts/monitoring/drift_log.csv")
if drift_path.exists():
    drift_df = pd.read_csv(drift_path)
    top_drift = (
        drift_df.groupby("feature")["psi"]
        .max()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_drift)
