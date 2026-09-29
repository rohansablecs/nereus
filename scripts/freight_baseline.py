from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA = Path("02_DATA/raw/freight/kobc/kobc_drybulk_daily.csv")

TARGETS = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]

HORIZONS = [7, 14, 30, 60]


def make_target(df, target, horizon):
    """
    Create a future target using observation-based horizon.

    horizon=7 means the value 7 observations ahead,
    not necessarily 7 calendar days.
    """
    out = df.copy()
    out["target"] = out[target].shift(-horizon)
    return out.dropna(subset=["target"])


def persistence_forecast(df, target, horizon):
    """
    Naive forecast:
    future rate = latest currently known rate.
    """
    data = make_target(df, target, horizon)

    y_true = data["target"].to_numpy()
    y_pred = data[target].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse, len(data)


def main():
    df = pd.read_csv(DATA)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("NEREUS — FREIGHT PERSISTENCE BASELINE")
    print("=" * 70)

    print(f"Observations : {len(df):,}")
    print(f"Date range   : {df.date.min().date()} → {df.date.max().date()}")

    results = []

    for target in TARGETS:
        print(f"\n--- {target.upper()} ---")

        for horizon in HORIZONS:
            mae, rmse, n = persistence_forecast(
                df,
                target,
                horizon,
            )

            results.append(
                {
                    "target": target,
                    "horizon_observations": horizon,
                    "mae": mae,
                    "rmse": rmse,
                    "n": n,
                }
            )

            print(
                f"{horizon:>2} obs | "
                f"MAE: {mae:,.1f} | "
                f"RMSE: {rmse:,.1f} | "
                f"N: {n:,}"
            )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
