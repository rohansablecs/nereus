from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA = Path("02_DATA/raw/freight/kobc/kobc_drybulk_daily.csv")

TARGETS = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]

HORIZONS = [7, 14, 30, 60]

LAGS = [1, 2, 3, 5, 10, 20]

ROLLING_WINDOWS = [7, 14, 30]


def build_features(df, target):
    x = pd.DataFrame(index=df.index)

    # Historical lag features
    for lag in LAGS:
        x[f"lag_{lag}"] = df[target].shift(lag)

    # Rolling statistics.
    # Shift first so today's value cannot leak into the features.
    shifted = df[target].shift(1)

    for window in ROLLING_WINDOWS:
        x[f"mean_{window}"] = shifted.rolling(window).mean()
        x[f"std_{window}"] = shifted.rolling(window).std()

    return x


def evaluate(df, target, horizon):
    X = build_features(df, target)
    y = df[target].shift(-horizon)

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    dates = df.loc[valid, "date"]

    # Chronological split.
    # Final 20% is untouched test data.
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    test_dates = dates.iloc[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))

    # Persistence on exactly the same test rows.
    persistence = df.loc[
        y_test.index, target
    ].to_numpy()

    persistence_mae = mean_absolute_error(
        y_test,
        persistence,
    )

    persistence_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            persistence,
        )
    )

    return {
        "target": target,
        "horizon": horizon,
        "train": len(X_train),
        "test": len(X_test),
        "ridge_mae": mae,
        "ridge_rmse": rmse,
        "persistence_mae": persistence_mae,
        "persistence_rmse": persistence_rmse,
        "improvement_pct": (
            (persistence_mae - mae)
            / persistence_mae
            * 100
        ),
        "test_start": test_dates.iloc[0],
        "test_end": test_dates.iloc[-1],
    }


def main():
    df = pd.read_csv(DATA)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    print("=" * 80)
    print("NEREUS — FREIGHT RIDGE EXPERIMENT")
    print("=" * 80)

    results = []

    for target in TARGETS:
        print(f"\n### {target.upper()}")

        for horizon in HORIZONS:
            result = evaluate(
                df,
                target,
                horizon,
            )

            results.append(result)

            print(
                f"{horizon:>2} obs | "
                f"Ridge MAE {result['ridge_mae']:>9,.1f} | "
                f"Persistence {result['persistence_mae']:>9,.1f} | "
                f"Improvement {result['improvement_pct']:>7.2f}%"
            )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        results_df[
            [
                "target",
                "horizon",
                "ridge_mae",
                "persistence_mae",
                "improvement_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
