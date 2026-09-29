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
    / "02_DATA/processed/paradip/paradip_cycle_operational_join_v1.csv"
)

TARGET = "berth_duration_days"

BASE = [
    "cargo",
    "berth",
    "quantity_mt",
]

WAITING = [
    "vessels_waiting_anchorage",
    "dry_bulk_vessels_waiting",
    "dry_bulk_cargo_mt_waiting",
]

ACTIVITY = [
    "berth_events",
    "sail_events",
    "shift_events",
    "dry_bulk_berth_events",
    "dry_bulk_sail_events",
    "unique_berth_events",
]

AGE = [
    "snapshot_age_days",
]


def make_model(numeric, categorical):

    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
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
        ],
        remainder="drop",
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


def score(y, pred):

    return {
        "mae": mean_absolute_error(
            y,
            pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y,
                pred,
            )
        ),
    }


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

    df = df.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    experiments = {
        "A_BASELINE": BASE,
        "B_WAITING_PRESSURE": (
            BASE
            + WAITING
        ),
        "C_ACTIVITY": (
            BASE
            + ACTIVITY
        ),
        "D_WAITING_PLUS_ACTIVITY": (
            BASE
            + WAITING
            + ACTIVITY
        ),
        "E_ALL_PLUS_AGE": (
            BASE
            + WAITING
            + ACTIVITY
            + AGE
        ),
    }

    results = []

    print("=" * 100)
    print("PARADIP OPERATIONAL FEATURE ABLATION")
    print("=" * 100)

    print()
    print(f"Total rows:  {len(df)}")
    print(f"Train rows:  {len(train)}")
    print(f"Test rows:   {len(test)}")
    print(
        f"Test period: "
        f"{test['berth_start_timestamp'].min().date()} "
        f"→ "
        f"{test['berth_start_timestamp'].max().date()}"
    )

    for name, features in experiments.items():

        numeric = [
            f
            for f in features
            if f not in [
                "cargo",
                "berth",
            ]
        ]

        categorical = [
            f
            for f in features
            if f in [
                "cargo",
                "berth",
            ]
        ]

        model = make_model(
            numeric,
            categorical,
        )

        model.fit(
            train[features],
            y_train,
        )

        pred = model.predict(
            test[features]
        )

        metrics = score(
            y_test,
            pred,
        )

        results.append(
            {
                "experiment": name,
                "features": len(features),
                "mae_days": metrics["mae"],
                "rmse_days": metrics["rmse"],
            }
        )

        print()
        print(name)
        print("-" * 100)
        print(
            "Features:"
        )

        for feature in features:
            print(
                f"  {feature}"
            )

        print(
            f"\nMAE:  "
            f"{metrics['mae']:.4f} days"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.4f} days"
        )

    result_df = pd.DataFrame(
        results
    )

    baseline_mae = result_df.loc[
        result_df["experiment"]
        == "A_BASELINE",
        "mae_days",
    ].iloc[0]

    result_df[
        "mae_improvement_vs_baseline_pct"
    ] = (
        (
            baseline_mae
            - result_df["mae_days"]
        )
        / baseline_mae
        * 100
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        result_df.to_string(
            index=False
        )
    )

    output = (
        ROOT
        / "02_DATA/processed/paradip/"
        "paradip_operational_ablation_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    print()
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
