from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "02_DATA/processed/forecast_calendar"
)

OUTPUT_DIR = (
    ROOT
    / "03_ML/freight/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VESSELS = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):
    return float(
        np.mean(
            np.abs(y_true - y_pred)
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


def direction_accuracy(
    y_true,
    y_pred,
):
    return float(
        np.mean(
            np.sign(y_true)
            == np.sign(y_pred)
        )
        * 100
    )


# ============================================================
# MOMENTUM BASELINE
# ============================================================

def build_momentum_return(df, vessel):

    current = df["current_freight"]

    r1 = (
        current
        / current.shift(1)
        - 1
    )

    r7 = (
        current
        / current.shift(7)
        - 1
    )

    r30 = (
        current
        / current.shift(30)
        - 1
    )

    momentum = (
        0.50 * r1
        + 0.30 * r7
        + 0.20 * r30
    )

    return momentum


# ============================================================
# TIME SPLIT
# ============================================================

def split_time(df):

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    return (
        df.iloc[:train_end].copy(),
        df.iloc[
            train_end:validation_end
        ].copy(),
        df.iloc[
            validation_end:
        ].copy(),
    )


# ============================================================
# MAIN
# ============================================================

results = []

print("=== NEREUS FREIGHT RETURN BACKTEST ===")
print()

for vessel in VESSELS:

    path = (
        DATA_DIR
        / f"{vessel}_30d.parquet"
    )

    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove invalid target-return rows.
    # --------------------------------------------------------

    df = df[
        df["target_return"].notna()
    ].copy()

    # --------------------------------------------------------
    # Momentum feature.
    # --------------------------------------------------------

    df["momentum_return"] = (
        build_momentum_return(
            df,
            vessel,
        )
    )

    df = df.dropna(
        subset=[
            "momentum_return"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    excluded = {
        "date",
        "requested_target_date",
        "target_date",
        "target",
        "target_change",
        "target_return",
        "vessel_class",
        "forecast_horizon_days",
        "target_delay_days",
        "momentum_return",
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    train, validation, test = (
        split_time(df)
    )

    X_train = train[
        feature_cols
    ]

    y_train = train[
        "target_return"
    ]

    X_val = validation[
        feature_cols
    ]

    y_val = validation[
        "target_return"
    ]

    X_test = test[
        feature_cols
    ]

    y_test = test[
        "target_return"
    ]

    # ========================================================
    # BASELINE 1 — ZERO RETURN
    # ========================================================

    zero_pred = np.zeros(
        len(test)
    )

    results.append(
        {
            "vessel": vessel,
            "model": "zero_return",
            "mae": mae(
                y_test,
                zero_pred,
            ),
            "rmse": rmse(
                y_test,
                zero_pred,
            ),
            "direction_accuracy_pct":
                direction_accuracy(
                    y_test,
                    zero_pred,
                ),
        }
    )

    # ========================================================
    # BASELINE 2 — MOMENTUM
    # ========================================================

    momentum_pred = (
        test["momentum_return"]
        .to_numpy()
    )

    results.append(
        {
            "vessel": vessel,
            "model": "momentum",
            "mae": mae(
                y_test,
                momentum_pred,
            ),
            "rmse": rmse(
                y_test,
                momentum_pred,
            ),
            "direction_accuracy_pct":
                direction_accuracy(
                    y_test,
                    momentum_pred,
                ),
        }
    )

    # ========================================================
    # RIDGE
    # ========================================================

    ridge = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )

    ridge.fit(
        X_train,
        y_train,
    )

    # Validation prediction generated
    # chronologically but not used for
    # test evaluation.
    ridge.predict(
        X_val
    )

    ridge_pred = ridge.predict(
        X_test
    )

    results.append(
        {
            "vessel": vessel,
            "model": "ridge",
            "mae": mae(
                y_test,
                ridge_pred,
            ),
            "rmse": rmse(
                y_test,
                ridge_pred,
            ),
            "direction_accuracy_pct":
                direction_accuracy(
                    y_test,
                    ridge_pred,
                ),
        }
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=4,
    )

    xgb.fit(
        X_train,
        y_train,
    )

    xgb.predict(
        X_val
    )

    xgb_pred = xgb.predict(
        X_test
    )

    results.append(
        {
            "vessel": vessel,
            "model": "xgboost",
            "mae": mae(
                y_test,
                xgb_pred,
            ),
            "rmse": rmse(
                y_test,
                xgb_pred,
            ),
            "direction_accuracy_pct":
                direction_accuracy(
                    y_test,
                    xgb_pred,
                ),
        }
    )

    print(
        f"{vessel.upper():9s} | "
        f"test={len(test)}"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

output = (
    OUTPUT_DIR
    / "freight_return_backtest.csv"
)

results_df.to_csv(
    output,
    index=False,
)


print()
print("=== RESULTS ===")
print()

print(
    results_df
    .sort_values(
        [
            "vessel",
            "mae",
        ]
    )
    .to_string(
        index=False
    )
)

print()
print("=== BEST VS ZERO RETURN ===")
print()

for vessel in VESSELS:

    subset = results_df[
        results_df["vessel"]
        == vessel
    ]

    baseline = subset[
        subset["model"]
        == "zero_return"
    ].iloc[0]

    best = subset[
        subset["model"]
        != "zero_return"
    ].sort_values(
        "mae"
    ).iloc[0]

    improvement = (
        1
        -
        best["mae"]
        /
        baseline["mae"]
    ) * 100

    print(
        f"{vessel:9s} | "
        f"best={best['model']:8s} | "
        f"MAE={best['mae']:.4f} | "
        f"zero={baseline['mae']:.4f} | "
        f"improvement={improvement:+.2f}%"
    )


print()
print(f"Saved: {output}")
