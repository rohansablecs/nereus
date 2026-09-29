from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "02_DATA/processed/paradip"
    / "paradip_turnaround_labels_v1.csv"
)


def mae(y_true, y_pred):
    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def main():

    df = pd.read_csv(INPUT)

    df["berth_start_timestamp"] = pd.to_datetime(
        df["berth_start_timestamp"],
        errors="coerce",
    )

    df["berth_duration_days"] = pd.to_numeric(
        df["berth_duration_days"],
        errors="coerce",
    )

    df["quantity_mt"] = pd.to_numeric(
        df["quantity_mt"],
        errors="coerce",
    )

    df["cargo_clean"] = (
        df["cargo"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["berth_start_timestamp"].notna()
        &
        df["berth_duration_days"].notna()
        &
        (df["berth_duration_days"] >= 0)
    ].copy()

    df = df.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Chronological split.
    #
    # First 70% = train
    # Last 30% = test
    # ---------------------------------------------------------------

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    y_train = train[
        "berth_duration_days"
    ]

    y_test = test[
        "berth_duration_days"
    ]

    predictions = {}

    # ---------------------------------------------------------------
    # Baseline 1: global median
    # ---------------------------------------------------------------

    global_median = y_train.median()

    predictions[
        "global_median"
    ] = np.full(
        len(test),
        global_median,
    )

    # ---------------------------------------------------------------
    # Baseline 2: cargo median
    #
    # Unknown/unseen cargo falls back to global median.
    # ---------------------------------------------------------------

    cargo_medians = (
        train.groupby(
            "cargo_clean"
        )[
            "berth_duration_days"
        ]
        .median()
    )

    predictions[
        "cargo_median"
    ] = test[
        "cargo_clean"
    ].map(
        cargo_medians
    ).fillna(
        global_median
    ).to_numpy()

    # ---------------------------------------------------------------
    # Baseline 3:
    # cargo + quantity bucket
    #
    # Buckets:
    #   <=10k
    #   10-25k
    #   25-50k
    #   50-75k
    #   >75k
    # ---------------------------------------------------------------

    bins = [
        -np.inf,
        10000,
        25000,
        50000,
        75000,
        np.inf,
    ]

    labels = [
        "<=10k",
        "10-25k",
        "25-50k",
        "50-75k",
        ">75k",
    ]

    train["quantity_bucket"] = pd.cut(
        train["quantity_mt"],
        bins=bins,
        labels=labels,
    )

    test["quantity_bucket"] = pd.cut(
        test["quantity_mt"],
        bins=bins,
        labels=labels,
    )

    train["cargo_quantity_key"] = (
        train["cargo_clean"]
        + "|"
        + train["quantity_bucket"]
        .astype(str)
    )

    test["cargo_quantity_key"] = (
        test["cargo_clean"]
        + "|"
        + test["quantity_bucket"]
        .astype(str)
    )

    cq_medians = (
        train.groupby(
            "cargo_quantity_key",
            observed=True,
        )[
            "berth_duration_days"
        ]
        .median()
    )

    predictions[
        "cargo_quantity_median"
    ] = test[
        "cargo_quantity_key"
    ].map(
        cq_medians
    ).fillna(
        test["cargo_clean"].map(
            cargo_medians
        )
    ).fillna(
        global_median
    ).to_numpy()

    # ---------------------------------------------------------------
    # Baseline 4: cargo + berth median
    # ---------------------------------------------------------------

    train["cargo_berth_key"] = (
        train["cargo_clean"]
        + "|"
        + train["berth"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    test["cargo_berth_key"] = (
        test["cargo_clean"]
        + "|"
        + test["berth"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    cb_medians = (
        train.groupby(
            "cargo_berth_key",
            observed=True,
        )[
            "berth_duration_days"
        ]
        .median()
    )

    predictions[
        "cargo_berth_median"
    ] = test[
        "cargo_berth_key"
    ].map(
        cb_medians
    ).fillna(
        test["cargo_clean"].map(
            cargo_medians
        )
    ).fillna(
        global_median
    ).to_numpy()

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    results = []

    for name, pred in predictions.items():

        results.append(
            {
                "model": name,
                "mae_days": mae(
                    y_test.to_numpy(),
                    pred,
                ),
                "rmse_days": rmse(
                    y_test.to_numpy(),
                    pred,
                ),
            }
        )

    results = pd.DataFrame(
        results
    ).sort_values(
        "mae_days"
    )

    print("=" * 80)
    print("PARADIP BERTH-CYCLE BASELINES")
    print("=" * 80)

    print(
        "Total rows:",
        len(df),
    )

    print(
        "Train:",
        len(train),
    )

    print(
        "Test:",
        len(test),
    )

    print(
        "Train period:",
        train["berth_start_timestamp"].min(),
        "→",
        train["berth_start_timestamp"].max(),
    )

    print(
        "Test period:",
        test["berth_start_timestamp"].min(),
        "→",
        test["berth_start_timestamp"].max(),
    )

    print()

    print(
        results
        .round(3)
        .to_string(index=False)
    )

    print()

    print(
        "Training global median:",
        round(
            global_median,
            3,
        ),
        "days",
    )

    print(
        "Best baseline:",
        results.iloc[0]["model"],
    )

    print(
        "Best MAE:",
        round(
            results.iloc[0]["mae_days"],
            3,
        ),
        "days",
    )

    print()

    print("Test target distribution:")

    print(
        y_test
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
