# DriftGuard — ML Drift Detection & Retraining Decision System

DriftGuard is an end-to-end machine learning monitoring system that detects data quality issues, distribution drift, and model performance degradation over time, explains why failures occur, and determines when retraining is necessary.

This project deliberately focuses not on maximizing predictive accuracy, but on understanding, diagnosing, and responding to model failure in production-like environments.

---

## Project Motivation

Machine learning models deployed in production rarely fail abruptly. Instead, they degrade silently due to changing data distributions, data pipeline issues, or shifts in feature importance. Without monitoring, these failures often go unnoticed until business impact becomes severe.

DriftGuard addresses the core production question:

**If this model were deployed, how would we detect failure early and understand its root cause?**

---

## System Overview

DriftGuard implements a time-aware monitoring pipeline that simulates real production behavior using sequential data slices. The system continuously evaluates incoming data and model behavior, producing actionable diagnostics and retraining recommendations.

---

## Folder Structure

```
DriftGuard/
├── src/
│   ├── main.py                    # Pipeline orchestrator
│   ├── data_ingestion/
│   │   └── data.py                # Time-based data loading and slicing
│   ├── data_quality/
│   │   ├── baseline_quality.py    # Baseline profiling
│   │   └── quality_check.py       # Per-slice quality monitoring
│   ├── drift/
│   │   └── feature_drift.py       # PSI & KS drift detection
│   ├── performance/
│   │   ├── train_baseline_model.py # Baseline model training
│   │   └── evaluation_over_time.py # Per-slice performance evaluation
│   ├── explanability/
│   │   └── analysis.py            # Permutation importance analysis
│   └── decision/
│       └── retrain_decision.py    # Retraining decision engine
├── backend/
│   └── app.py                     # API backend
├── frontend/
│   └── dashboard.py               # Streamlit monitoring dashboard
├── artifacts/                     # Pipeline outputs (generated)
│   ├── baseline/
│   │   └── quality_stats.json
│   ├── monitoring/
│   │   ├── baseline_model_stats.json
│   │   ├── data_quality_log.csv
│   │   ├── drift_log.csv
│   │   └── performance_log.csv
│   ├── explainability/
│   │   ├── feature_impact_*.csv   # Global & per-slice importance
│   │   └── slice_*_feature_impact_*.csv
│   └── decision/
│       └── retraining_decisions.json
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

### 1. Time-Based Data Ingestion

* Merges transactional and identity datasets
* Orders data chronologically using timestamps
* Splits data into sequential time slices representing production windows

---

### 2. Baseline Data Quality Profiling

Computed on the initial baseline slice:

* Missing value rates per feature
* Outlier thresholds using IQR
* Feature distribution statistics

Baseline quality statistics are stored and used as reference points for future comparisons.

---

### 3. Continuous Data Quality Monitoring

For each subsequent time slice:

* Missing rate deltas are computed relative to baseline
* Outlier rates are measured using baseline thresholds
* A composite quality score is produced
* Quality alerts are triggered when thresholds are exceeded

This step detects data pipeline failures and schema instability.

---

### 4. Baseline Model Training

* A classification model is trained once using the baseline slice
* Feature preprocessing is performed via a unified pipeline
* The model is persisted and treated as a deployed production model

The model is not retrained during monitoring, allowing natural degradation to occur.

---

### 5. Performance Evaluation Over Time

The baseline model is evaluated on each time slice:

* Precision, recall, and F1-score are tracked
* Metric deltas relative to baseline are computed
* Performance decay trends are logged

This step reveals silent performance degradation.

---

### 6. Distribution Drift Detection

Feature-level covariate drift is detected using:

* Population Stability Index (PSI)
* Kolmogorov–Smirnov (KS) tests

Drift is evaluated per feature and aggregated across slices to identify severe distribution shifts.

---

### 7. Conditional Explainability

Explainability is triggered only for slices exhibiting:

* Significant performance degradation
* Severe feature drift

Permutation importance (F1-based) is computed to compare:

* Baseline feature importance
* Degraded slice feature importance

This reveals which features lost or gained predictive influence over time.

---

### 8. Retraining Decision Engine

Signals from quality monitoring, performance decay, drift detection, and explainability are combined into a rule-based decision system that outputs:

* RETRAIN
* INVESTIGATE_DATA_PIPELINE
* MONITOR
* NO_ACTION

Decisions are generated per time slice and include supporting reasons and confidence levels.

---

### 9. Interactive Monitoring Dashboard

A Streamlit-based dashboard allows users to:

* Select time slices from a dropdown
* View retraining decisions and confidence
* Inspect performance trends
* Examine feature drift summaries
* Review explainability outputs when available

The dashboard consumes stored monitoring artifacts without recomputation.
