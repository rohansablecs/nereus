from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


DATA = Path(
    "02_DATA/processed/freight_port_features.parquet"
)

TARGETS = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]

HORIZONS = [7, 14, 30, 60]

FREIGHT_COLUMNS = [
    "kdci",
    "cape",
    "panamax",
    "supramax",
    "handy",
]


def build_features(df):

    df = df.copy()

    for column in FREIGHT_COLUMNS:

        # Historical lags.
        for lag in [1, 2, 3, 5, 7, 14, 21, 30, 60]:

            df[f"{column}_lag_{lag}"] = (
                df[column].shift(lag)
            )

        # Rolling statistics.
        for window in [3, 7, 14, 30, 60]:

            shifted = df[column].shift(1)

            df[f"{column}_mean_{window}"] = (
                shifted.rolling(window).mean()
            )

            df[f"{column}_std_{window}"] = (
                shifted.rolling(window).std()
            )

            df[f"{column}_min_{window}"] = (
                shifted.rolling(window).min()
            )

            df[f"{column}_max_{window}"] = (
                shifted.rolling(window).max()
            )

        # Momentum.
        df[f"{column}_momentum_3"] = (
            df[column].shift(1)
            - df[column].shift(4)
        )

        df[f"{column}_momentum_7"] = (
            df[column].shift(1)
            - df[column].shift(8)
        )

        df[f"{column}_momentum_14"] = (
            df[column].shift(1)
            - df[column].shift(15)
        )

        # Percentage momentum.
        previous = df[column].shift(1)

        df[f"{column}_return_3"] = (
            previous
            / df[column].shift(4)
            - 1
        )

        df[f"{column}_return_7"] = (
            previous
            / df[column].shift(8)
            - 1
        )

        df[f"{column}_return_14"] = (
            previous
            / df[column].shift(15)
            - 1
        )

    # Calendar features.
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Cyclic seasonality.
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return mae, rmse


def persistence_prediction(df, target, horizon):

    return (
        df[target]
        .shift(0)
    )


def main():

    df = pd.read_parquet(DATA)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    print("=" * 90)
    print("NEREUS — FREIGHT TEMPORAL MODEL EXPERIMENT")
    print("=" * 90)

    print(
        f"\nObservations : {len(df):,}"
    )

    print(
        f"Date range   : "
        f"{df.date.min().date()} → "
        f"{df.date.max().date()}"
    )

    feature_df = build_features(df)

    all_results = []

    for target in TARGETS:

        print("\n" + "=" * 90)
        print(f"### {target.upper()}")
        print("=" * 90)

        for horizon in HORIZONS:

            work = feature_df.copy()

            # Future target.
            work["target"] = (
                work[target]
                .shift(-horizon)
            )

            # Remove rows without complete target.
            work = work.dropna(
                subset=["target"]
            )

            # Never use date itself as ML feature.
            excluded = {
                "date",
                "record_number",
                "target",
            }

            feature_columns = [
                c for c in work.columns
                if c not in excluded
            ]

            X = work[feature_columns]
            y = work["target"]

            # Remove rows with incomplete features.
            valid = X.notna().all(axis=1)

            X = X.loc[valid]
            y = y.loc[valid]
            dates = work.loc[valid, "date"]

            # --------------------------------------------------
            # Chronological holdout.
            # --------------------------------------------------

            split = int(
                len(X) * 0.80
            )

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]

            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            # --------------------------------------------------
            # Persistence baseline.
            #
            # Forecast future value using today's value.
            # --------------------------------------------------

            persistence = (
                df[target]
                .shift(0)
                .shift(-horizon)
            )

            # Align persistence with work rows.
            persistence_work = (
                df[target]
                .iloc[
                    work.index
                ]
                .values
            )

            persistence_pred = (
                persistence_work[valid.values]
            )

            persistence_test = (
                persistence_pred[split:]
            )

            p_mae, p_rmse = metrics(
                y_test,
                persistence_test,
            )

            # --------------------------------------------------
            # RIDGE
            # --------------------------------------------------

            ridge = Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        Ridge(
                            alpha=10.0
                        )
                    ),
                ]
            )

            ridge.fit(
                X_train,
                y_train,
            )

            ridge_pred = ridge.predict(
                X_test
            )

            r_mae, r_rmse = metrics(
                y_test,
                ridge_pred,
            )

            # --------------------------------------------------
            # XGBOOST
            # --------------------------------------------------

            xgb = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            )

            xgb.fit(
                X_train,
                y_train,
            )

            xgb_pred = xgb.predict(
                X_test
            )

            x_mae, x_rmse = metrics(
                y_test,
                xgb_pred,
            )

            ridge_improvement = (
                1 - r_mae / p_mae
            ) * 100

            xgb_improvement = (
                1 - x_mae / p_mae
            ) * 100

            print(
                f"\n{horizon} observations"
            )

            print(
                f"  Persistence "
                f"MAE: {p_mae:,.1f}"
            )

            print(
                f"  Ridge       "
                f"MAE: {r_mae:,.1f} "
                f"({ridge_improvement:+.2f}%)"
            )

            print(
                f"  XGBoost     "
                f"MAE: {x_mae:,.1f} "
                f"({xgb_improvement:+.2f}%)"
            )

            all_results.extend(
                [
                    {
                        "target": target,
                        "horizon": horizon,
                        "model": "persistence",
                        "mae": p_mae,
                        "rmse": p_rmse,
                    },
                    {
                        "target": target,
                        "horizon": horizon,
                        "model": "ridge",
                        "mae": r_mae,
                        "rmse": r_rmse,
                    },
                    {
                        "target": target,
                        "horizon": horizon,
                        "model": "xgboost",
                        "mae": x_mae,
                        "rmse": x_rmse,
                    },
                ]
            )

    results = pd.DataFrame(
        all_results
    )

    output = Path(
        "02_DATA/processed/"
        "freight_temporal_model_results.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    print(
        results.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
