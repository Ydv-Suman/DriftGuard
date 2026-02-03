"""Load raw transaction/identity CSVs, merge, split into time slices (baseline + slice_1..N)."""
import pandas as pd
from pathlib import Path


def load_raw_data(raw_dir: Path):
    """Load train_transaction.csv and train_identity.csv from raw_dir. Returns (transaction_df, identity_df)."""
    train_transaction_df = pd.read_csv(raw_dir / "train_transaction.csv")
    train_identity_df = pd.read_csv(raw_dir / "train_identity.csv")
    return train_transaction_df, train_identity_df


def merge_data(transaction_df: pd.DataFrame, identity_df: pd.DataFrame):
    """Left-merge identity onto transaction on TransactionID. Returns merged DataFrame."""
    df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left"
    )
    return df


def create_time_slices(df: pd.DataFrame, n_slices: int = 8):
    """Sort by TransactionDT and assign time_slice via quantile cut. Returns DataFrame with time_slice column."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["time_slice"] = pd.qcut(
        df["TransactionDT"],
        q=n_slices,
        labels=False
    )
    return df


def save_time_slices(df: pd.DataFrame, output_dir: Path):
    """Write each time_slice to a CSV: slice 0 as baseline.csv, others as slice_N.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for slice_idx in sorted(df["time_slice"].unique()):
        slice_df = df[df["time_slice"] == slice_idx]

        if slice_idx == 0:
            filename = "baseline.csv"
        else:
            filename = f"slice_{slice_idx}.csv"

        slice_df.to_csv(output_dir / filename, index=False)


def main():
    """Load raw data, merge, create time slices, save merged and per-slice CSVs."""
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "time_slices"
    merged_path = project_root / "data" / "merged"

    transaction_df, identity_df = load_raw_data(raw_dir)
    merged_df = merge_data(transaction_df, identity_df)
    merged_df.to_csv(merged_path / "merged.csv", index=False)
    sliced_df = create_time_slices(merged_df, n_slices=8)
    save_time_slices(sliced_df, output_dir)

    print("Time slices created successfully.")


def run():
    """Entry point: run main()."""
    main()