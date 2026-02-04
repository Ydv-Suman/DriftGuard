"""
DriftGuard pipeline: data, then baseline quality, then quality checks, then train baseline model,
then performance over time, then feature drift, then explainability, then retrain decision.
"""
import sys
from pathlib import Path
import pandas as pd

from data_ingestion import data
from data_quality import baseline_quality, quality_check
from performance import train_baseline_model, evaluation_over_time
from drift import feature_drift
from explanability import analysis
from decision import retrain_decision


# Ensure project root is on sys.path (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

perf_path = PROJECT_ROOT / "artifacts" / "monitoring" / "performance_log.csv"
drift_path = PROJECT_ROOT / "artifacts" / "monitoring" / "drift_log.csv"
slice_dir = PROJECT_ROOT / "data" / "time_slices"
perf_df = pd.read_csv(perf_path)
drift_df = pd.read_csv(drift_path)
slice_ids = sorted(p.stem for p in slice_dir.glob("slice_*.csv"))


# identify degraded slices
def get_degraded_slices(perf_df: pd.DataFrame, drift_df: pd.DataFrame) -> list[str]:
    """
        A slice is considered degraded if:
        - recall_drop > 0.40 OR f1_drop > 0.25
        - OR any feature shows severe PSI drift (> 0.25)
    """

    bad_perf = perf_df[(perf_df["recall_drop"] > 0.40) |(perf_df["f1_drop"] > 0.25)]["slice_id"].unique()
    severe_drift = drift_df[drift_df["psi"] > 0.25]["slice_id"].unique()

    return sorted(set(bad_perf) | set(severe_drift))


def main():
    """Run the full DriftGuard pipeline in order."""
    data.run()                    # Load raw data, merge, create time slices
    baseline_quality.run()       # Compute baseline missing/outlier/stats, writes quality_stats.json
    quality_check.run()          # Compare each slice to baseline, writes data_quality_log.csv
    train_baseline_model.run()   # Train on baseline slice, save model + baseline_model_stats.json
    evaluation_over_time.run()  # Evaluate model on each slice, writes performance_log.csv
    feature_drift.run()         # PSI/KS drift per feature vs baseline, writes drift_log.csv
    degraded_slices = get_degraded_slices(perf_df, drift_df)
    for slice_id in degraded_slices:
        analysis.run(slice_id)              # Permutation importance baseline vs degraded, writes feature_impact_*.csv
    print("\n\nanalysis completion")
    retrain_decision.run(slice_ids)  # Decide RETRAIN or INVESTIGATE or MONITOR or NO_ACTION, writes retraining_decisions.json

    print("piprline completed")


if __name__ == "__main__":
    main()
