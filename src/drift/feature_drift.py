"""Detect feature drift per slice vs baseline using PSI and KS; write drift_log.csv."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

IGNORE_COLS = ["TransactionID", "isFraud", "time_slice"]
N_PSI_BINS = 10  # number of bins for Population Stability Index
EPS = 1e-10  # avoid log(0) in PSI
MAX_KS_SAMPLE = 5_000  # subsample for KS when n > this (speed + scipy stability)


def load_baseline(baseline_path: Path) -> pd.DataFrame:
    """Load baseline CSV. Raises FileNotFoundError if missing."""
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {baseline_path}")
    return pd.read_csv(baseline_path)


def _get_numeric_features(df: pd.DataFrame) -> list:
    """Return list of numeric column names excluding IGNORE_COLS."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in IGNORE_COLS]


def compute_psi(baseline_series: pd.Series, slice_series: pd.Series, n_bins: int = N_PSI_BINS) -> float:
    """Population Stability Index between baseline and slice distributions. Returns float or nan if empty."""
    base = baseline_series.dropna()
    slc = slice_series.dropna()
    if len(base) == 0 or len(slc) == 0:
        return np.nan

    try:
        edges = np.percentile(base, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
    except Exception:
        return np.nan

    base_counts = np.histogram(base, bins=edges)[0]
    slice_counts = np.histogram(slc, bins=edges)[0]

    base_pct = base_counts / (base_counts.sum() + EPS) + EPS
    slice_pct = slice_counts / (slice_counts.sum() + EPS) + EPS

    psi = np.sum((slice_pct - base_pct) * np.log(slice_pct / base_pct))
    return float(psi)


def compute_ks(baseline_series: pd.Series, slice_series: pd.Series) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value. Subsamples if len > MAX_KS_SAMPLE. Returns (stat, pval)."""
    base = baseline_series.dropna()
    slc = slice_series.dropna()
    if len(base) == 0 or len(slc) == 0:
        return np.nan, np.nan
    if len(base) > MAX_KS_SAMPLE:
        base = base.sample(n=MAX_KS_SAMPLE, random_state=42)
    if len(slc) > MAX_KS_SAMPLE:
        slc = slc.sample(n=MAX_KS_SAMPLE, random_state=42)
    stat, pval = ks_2samp(base, slc)
    return float(stat), float(pval)


def psi_to_drift_flag(psi: float) -> str:
    """Map PSI to label: unknown, no_drift, moderate_drift, or severe_drift."""
    if np.isnan(psi):
        return "unknown"
    if psi < 0.1:
        return "no_drift"
    if psi <= 0.25:
        return "moderate_drift"
    return "severe_drift"


def run_drift_detection(baseline_df: pd.DataFrame, slice_df: pd.DataFrame, slice_id: str, numeric_features: list[str] | None = None) -> list[dict]:
    """Compute PSI, KS stat, KS p-value, drift_flag per numeric feature. Returns list of record dicts."""
    if numeric_features is None:
        base_num = set(_get_numeric_features(baseline_df))
        slice_num = set(_get_numeric_features(slice_df))
        numeric_features = sorted(base_num & slice_num)

    records = []
    for feature in numeric_features:
        if feature not in baseline_df.columns or feature not in slice_df.columns:
            continue
        base_col = baseline_df[feature]
        slice_col = slice_df[feature]

        psi = compute_psi(base_col, slice_col)
        ks_stat, ks_pvalue = compute_ks(base_col, slice_col)
        drift_flag = psi_to_drift_flag(psi)

        records.append({
            "slice_id": slice_id,
            "feature": feature,
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "drift_flag": drift_flag,
        })
    return records


def main():
    """Run drift detection for each slice vs baseline, write drift_log.csv."""
    project_root = Path(__file__).resolve().parents[2]
    slice_dir = project_root / "data" / "time_slices"
    baseline_path = slice_dir / "baseline.csv"
    output_path = project_root / "artifacts" / "monitoring" / "drift_log.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline(baseline_path)
    numeric_features = _get_numeric_features(baseline_df)

    all_records = []
    for slice_path in sorted(slice_dir.glob("slice_*.csv")):
        slice_id = slice_path.stem
        slice_df = pd.read_csv(slice_path, low_memory=False)
        records = run_drift_detection(
            baseline_df,
            slice_df,
            slice_id=slice_id,
            numeric_features=numeric_features,
        )
        all_records.extend(records)

    if all_records:
        log_df = pd.DataFrame(all_records)
        log_df = log_df[["slice_id", "feature", "psi", "ks_stat", "ks_pvalue", "drift_flag"]]
        log_df.to_csv(output_path, index=False)
        print(f"Drift log written to {output_path} ({len(all_records)} rows).")
    else:
        print("No slice_*.csv files or no common numeric features. drift_log.csv not written.")
    
    print("\n\n4 -  feature drift completion")


def run():
    """Entry point: run main()."""
    main()
