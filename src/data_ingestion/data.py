import pandas as pd
from pathlib import Path


def load_raw_data(raw_dir: Path):
    train_transaction_df = pd.read_csv(raw_dir / "train_transaction.csv")
    train_identity_df = pd.read_csv(raw_dir / "train_identity.csv")
    return train_transaction_df, train_identity_df


def merge_data(transaction_df: pd.DataFrame, identity_df: pd.DataFrame):
    df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left"
    )
    return df


def create_time_slices(df: pd.DataFrame, n_slices: int = 8):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["time_slice"] = pd.qcut(
        df["TransactionDT"],
        q=n_slices,
        labels=False
    )
    return df


def save_time_slices(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for slice_idx in sorted(df["time_slice"].unique()):
        slice_df = df[df["time_slice"] == slice_idx]

        if slice_idx == 0:
            filename = "baseline.csv"
        else:
            filename = f"slice_{slice_idx}.csv"

        slice_df.to_csv(output_dir / filename, index=False)


def main():
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "time_slices"

    transaction_df, identity_df = load_raw_data(raw_dir)
    merged_df = merge_data(transaction_df, identity_df)
    sliced_df = create_time_slices(merged_df, n_slices=8)
    save_time_slices(sliced_df, output_dir)

    print("Time slices created successfully.")


if __name__ == "__main__":
    main()
