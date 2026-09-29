from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "02_DATA/processed/paradip/"
    / "paradip_cycle_operational_join_v1.csv"
)

TARGET = "berth_duration_days"

BASE = [
    "cargo",
    "berth",
    "quantity_mt",
]

CURRENT = [
    "dry_bulk_vessels_waiting",
]

TREND_1D = [
    "waiting_vessels_change_1d",
]

TREND_3D = [
    "waiting_vessels_change_3d",
]

ROLLING_3D = [
    "waiting_vessels_rolling_3d",
]


def build_operational_history_features(df):
    """
    Build historical trend features from the operational snapshots.

    IMPORTANT:
    These are calculated on the operational snapshot timeline BEFORE
    joining to berth cycles.

    Therefore a cycle can only receive information available from its
    snapshot date or earlier.
    """

    ops = (
        df[
            [
                "snapshot_date",
                "dry_bulk_vessels_waiting",
            ]
        ]
        .drop_duplicates(
            subset=["snapshot_date"]
        )
        .sort_values("snapshot_date")
        .copy()
    )

    ops["waiting_vessels_change_1d"] = (
        ops["dry_bulk_vessels_waiting"]
        .diff(1)
    )

    ops["waiting_vessels_change_3d"] = (
        ops["dry_bulk_vessels_waiting"]
        .diff(3)
    )

    ops["waiting_vessels_rolling_3d"] = (
        ops["dry_bulk_vessels_waiting"]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    return ops


def attach_features(df, ops):
    """
    Attach the latest operational state on or before cycle date.
    """

    left = df.copy()

    left["cycle_date"] = (
        left["berth_start_timestamp"]
        .dt.normalize()
    )

    right = ops.copy()

    right["snapshot_date"] = (
        right["snapshot_date"]
        .dt.normalize()
    )

    right = right.sort_values(
        "snapshot_date"
    )

    merged = pd.merge_asof(
        left.sort_values("cycle_date"),
        right.sort_values("snapshot_date"),
        left_on="cycle_date",
        right_on="snapshot_date",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_trend"),
    )

    return merged


def make_model(features):

    categorical = [
        f
        for f in features
        if f in ["cargo", "berth"]
    ]

    numeric = [
        f
        for f in features
        if f not in categorical
    ]

    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipe,
                numeric,
            ),
            (
                "categorical",
                categorical_pipe,
                categorical,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        [
            (
                "preprocess",
                preprocess,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate(train, test, features):

    model = make_model(
        features
    )

    model.fit(
        train[features],
        train[TARGET].astype(float),
    )

    pred = model.predict(
        test[features]
    )

    y = test[TARGET].astype(float)

    mae = mean_absolute_error(
        y,
        pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred,
        )
    )

    return mae, rmse


def main():

    df = pd.read_csv(
        DATA,
        parse_dates=[
            "berth_start_timestamp",
            "sail_timestamp",
            "cycle_date",
            "snapshot_date",
        ],
    )

    df = (
        df
        .sort_values(
            "berth_start_timestamp"
        )
        .reset_index(drop=True)
    )

    print("=" * 100)
    print("PARADIP WAITING-VESSEL TREND EXPERIMENT")
    print("=" * 100)

    print(
        f"\nBerth cycles: {len(df)}"
    )

    # Build trend from operational snapshot history.
    ops = build_operational_history_features(
        df
    )

    print(
        f"Operational snapshots: {len(ops)}"
    )

    # Attach point-in-time trend state.
    merged = attach_features(
        df,
        ops,
    )

    print()
    print("TREND FEATURE COVERAGE")
    print("-" * 100)

    for col in [
        "dry_bulk_vessels_waiting",
        "waiting_vessels_change_1d",
        "waiting_vessels_change_3d",
        "waiting_vessels_rolling_3d",
    ]:
        count = merged[col].notna().sum()

        print(
            f"{col:<35} "
            f"{count}/{len(merged)} "
            f"({count/len(merged):.1%})"
        )

    print()
    print("TREND STATISTICS")
    print("-" * 100)

    stats = merged[
        [
            "dry_bulk_vessels_waiting",
            "waiting_vessels_change_1d",
            "waiting_vessels_change_3d",
            "waiting_vessels_rolling_3d",
        ]
    ].describe()

    print(
        stats.to_string()
    )

    # Same chronological split as previous experiments.
    split = int(
        len(merged) * 0.70
    )

    train = merged.iloc[:split].copy()
    test = merged.iloc[split:].copy()

    print()
    print("CHRONOLOGICAL SPLIT")
    print("-" * 100)

    print(
        f"Train: {len(train)}"
    )

    print(
        f"Test:  {len(test)}"
    )

    print(
        f"Test period: "
        f"{test['berth_start_timestamp'].min().date()} "
        f"→ "
        f"{test['berth_start_timestamp'].max().date()}"
    )

    experiments = {
        "A_BASELINE": BASE,

        "B_CURRENT_WAITING": (
            BASE
            + CURRENT
        ),

        "C_1D_CHANGE": (
            BASE
            + TREND_1D
        ),

        "D_3D_CHANGE": (
            BASE
            + TREND_3D
        ),

        "E_3D_ROLLING_MEAN": (
            BASE
            + ROLLING_3D
        ),

        "F_CURRENT_PLUS_1D_TREND": (
            BASE
            + CURRENT
            + TREND_1D
        ),

        "G_CURRENT_PLUS_3D_TREND": (
            BASE
            + CURRENT
            + TREND_3D
        ),

        "H_CURRENT_PLUS_TREND_AND_ROLLING": (
            BASE
            + CURRENT
            + TREND_1D
            + TREND_3D
            + ROLLING_3D
        ),
    }

    results = []

    print()
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    for name, features in experiments.items():

        # Trend features naturally have missing values at the beginning
        # of the operational history. The model's median imputer handles
        # these without fabricating historical observations.

        mae, rmse = evaluate(
            train,
            test,
            features,
        )

        results.append(
            {
                "experiment": name,
                "mae_days": mae,
                "rmse_days": rmse,
            }
        )

        print()
        print(name)

        print(
            f"  MAE:  {mae:.4f} days"
        )

        print(
            f"  RMSE: {rmse:.4f} days"
        )

    result = pd.DataFrame(
        results
    )

    baseline_mae = result.loc[
        result["experiment"]
        == "A_BASELINE",
        "mae_days",
    ].iloc[0]

    result["mae_improvement_pct"] = (
        (
            baseline_mae
            - result["mae_days"]
        )
        / baseline_mae
        * 100
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        result.to_string(
            index=False
        )
    )

    output = (
        ROOT
        / "02_DATA/processed/paradip/"
        / "paradip_waiting_trend_results.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    # Save the enriched point-in-time dataset too.
    joined_output = (
        ROOT
        / "02_DATA/processed/paradip/"
        / "paradip_cycle_waiting_trend_join_v1.csv"
    )

    merged.to_csv(
        joined_output,
        index=False,
    )

    print()
    print(f"Wrote results: {output}")
    print(f"Wrote dataset: {joined_output}")


if __name__ == "__main__":
    main()
