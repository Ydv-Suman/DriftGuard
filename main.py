"""
DriftGuard pipeline: data, then baseline quality, then quality checks, then train baseline model,
then performance over time, then feature drift, then explainability, then retrain decision.
"""
from src.data_ingestion import data
from src.data_quality import baseline_quality, quality_check
from src.performance import train_baseline_model, evaluation_over_time
from src.drift import feature_drift
from src.explanability import analysis
from src.decision import retrain_decision


def main():
    """Run the full DriftGuard pipeline in order."""
    data.run()                    # Load raw data, merge, create time slices
    baseline_quality.run()       # Compute baseline missing/outlier/stats, writes quality_stats.json
    quality_check.run()          # Compare each slice to baseline, writes data_quality_log.csv
    train_baseline_model.run()   # Train on baseline slice, save model + baseline_model_stats.json
    evaluation_over_time.run()  # Evaluate model on each slice, writes performance_log.csv
    feature_drift.run()         # PSI/KS drift per feature vs baseline, writes drift_log.csv
    analysis.run()              # Permutation importance baseline vs degraded, writes feature_impact_*.csv
    retrain_decision.run()      # Decide RETRAIN or INVESTIGATE or MONITOR or NO_ACTION, writes retraining_decision.json


if __name__ == "__main__":
    main()