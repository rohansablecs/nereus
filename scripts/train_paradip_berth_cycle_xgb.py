from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "02_DATA/processed/paradip"
    / "paradip_turnaround_labels_v1.csv"
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

    for column in [
        "loa_m",
        "beam_m",
        "draft_m",
        "shift_count",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["cargo_clean"] = (
        df["cargo"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["berth_clean"] = (
        df["berth"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Calendar features available at berth-start time.
    # ---------------------------------------------------------------

    df["month"] = (
        df["berth_start_timestamp"]
        .dt.month
    )

    df["day_of_year"] = (
        df["berth_start_timestamp"]
        .dt.dayofyear
    )

    df["day_of_week"] = (
        df["berth_start_timestamp"]
        .dt.dayofweek
    )

    df = df[
        df["berth_start_timestamp"].notna()
        &
        df["berth_duration_days"].notna()
    ].copy()

    df = df.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Same chronological 70/30 split as baseline.
    # ---------------------------------------------------------------

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    target = "berth_duration_days"

    feature_columns = [
        "cargo_clean",
        "berth_clean",
        "quantity_mt",
        "loa_m",
        "beam_m",
        "draft_m",
        "shift_count",
        "month",
        "day_of_year",
        "day_of_week",
    ]

    categorical_features = [
        "cargo_clean",
        "berth_clean",
    ]

    numeric_features = [
        "quantity_mt",
        "loa_m",
        "beam_m",
        "draft_m",
        "shift_count",
        "month",
        "day_of_year",
        "day_of_week",
    ]

    X_train = train[
        feature_columns
    ]

    X_test = test[
        feature_columns
    ]

    y_train = train[
        target
    ]

    y_test = test[
        target
    ]

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
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

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    # ---------------------------------------------------------------
    # Baseline: cargo + berth median.
    # ---------------------------------------------------------------

    train["cargo_berth_key"] = (
        train["cargo_clean"]
        + "|"
        + train["berth_clean"]
    )

    test["cargo_berth_key"] = (
        test["cargo_clean"]
        + "|"
        + test["berth_clean"]
    )

    medians = (
        train.groupby(
            "cargo_berth_key"
        )[target]
        .median()
    )

    global_median = y_train.median()

    baseline = (
        test["cargo_berth_key"]
        .map(medians)
        .fillna(
            test["cargo_clean"].map(
                train.groupby(
                    "cargo_clean"
                )[target]
                .median()
            )
        )
        .fillna(
            global_median
        )
        .to_numpy()
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    xgb_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline,
        )
    )

    improvement = (
        1
        -
        xgb_mae / baseline_mae
    ) * 100

    print("=" * 80)
    print("PARADIP BERTH-CYCLE XGBOOST")
    print("=" * 80)

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
        train[
            "berth_start_timestamp"
        ].min(),
        "→",
        train[
            "berth_start_timestamp"
        ].max(),
    )

    print(
        "Test period:",
        test[
            "berth_start_timestamp"
        ].min(),
        "→",
        test[
            "berth_start_timestamp"
        ].max(),
    )

    print()

    print(
        "Cargo + berth baseline:"
    )

    print(
        "  MAE :",
        round(
            baseline_mae,
            3,
        ),
        "days",
    )

    print(
        "  RMSE:",
        round(
            baseline_rmse,
            3,
        ),
        "days",
    )

    print()

    print(
        "XGBoost:"
    )

    print(
        "  MAE :",
        round(
            xgb_mae,
            3,
        ),
        "days",
    )

    print(
        "  RMSE:",
        round(
            xgb_rmse,
            3,
        ),
        "days",
    )

    print()

    print(
        "Improvement vs cargo+berth baseline:",
        round(
            improvement,
            2,
        ),
        "%",
    )

    if xgb_mae < baseline_mae:

        print()
        print(
            "RESULT: XGBoost BEATS the simple baseline."
        )

    else:

        print()
        print(
            "RESULT: XGBoost DOES NOT BEAT the simple baseline."
        )


if __name__ == "__main__":
    main()
