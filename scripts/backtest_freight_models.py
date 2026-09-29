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

HORIZONS = [
    7,
    14,
    30,
    60,
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {

    "ridge": Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
    ),

    "xgboost": XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=4,
    ),
}


# ============================================================
# METRICS
# ============================================================

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


def mape(y_true, y_pred):
    denominator = np.maximum(
        np.abs(y_true),
        1e-9,
    )

    return float(
        np.mean(
            np.abs(
                (y_true - y_pred)
                / denominator
            )
        )
        * 100
    )


def directional_accuracy(
    current,
    actual,
    predicted,
):
    actual_direction = np.sign(
        actual - current
    )

    predicted_direction = np.sign(
        predicted - current
    )

    return float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100
    )


# ============================================================
# TIME SPLIT
#
# 70% train
# 15% validation
# 15% untouched test
#
# Chronological only.
# ============================================================

def split_time(df):

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    return train, validation, test


# ============================================================
# MAIN
# ============================================================

results = []

print("=== NEREUS FREIGHT MODEL BACKTEST ===")
print()

for vessel in VESSELS:

    for horizon in HORIZONS:

        path = (
            DATA_DIR
            / f"{vessel}_{horizon}d.parquet"
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

        # ----------------------------------------------------
        # Identify features.
        #
        # Explicitly exclude target-related fields.
        # ----------------------------------------------------

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
        }

        feature_cols = [
            c for c in df.columns
            if c not in excluded
        ]

        X = df[
            feature_cols
        ]

        y = df["target"]

        train, validation, test = split_time(
            df
        )

        X_train = train[
            feature_cols
        ]

        y_train = train[
            "target"
        ]

        X_val = validation[
            feature_cols
        ]

        y_val = validation[
            "target"
        ]

        X_test = test[
            feature_cols
        ]

        y_test = test[
            "target"
        ]

        current_test = test[
            "current_freight"
        ]

        print(
            f"{vessel.upper():9s} "
            f"{horizon:2d}d | "
            f"train={len(train):4d} "
            f"val={len(validation):4d} "
            f"test={len(test):4d}"
        )

        # ====================================================
        # PERSISTENCE BASELINE
        #
        # Forecast = current freight rate.
        # ====================================================

        persistence_pred = (
            current_test.to_numpy()
        )

        persistence_mae = mae(
            y_test,
            persistence_pred,
        )

        persistence_rmse = rmse(
            y_test,
            persistence_pred,
        )

        persistence_mape = mape(
            y_test,
            persistence_pred,
        )

        persistence_direction = (
            directional_accuracy(
                current_test.to_numpy(),
                y_test.to_numpy(),
                persistence_pred,
            )
        )

        results.append(
            {
                "vessel": vessel,
                "horizon_days": horizon,
                "model": "persistence",
                "test_rows": len(test),
                "mae": persistence_mae,
                "rmse": persistence_rmse,
                "mape_pct": persistence_mape,
                "directional_accuracy_pct":
                    persistence_direction,
            }
        )

        # ====================================================
        # ML MODELS
        # ====================================================

        for model_name, model in MODELS.items():

            model.fit(
                X_train,
                y_train,
            )

            # Validation prediction is generated so the
            # workflow remains explicitly chronological.
            _ = model.predict(
                X_val
            )

            predictions = model.predict(
                X_test
            )

            model_mae = mae(
                y_test,
                predictions,
            )

            model_rmse = rmse(
                y_test,
                predictions,
            )

            model_mape = mape(
                y_test,
                predictions,
            )

            model_direction = (
                directional_accuracy(
                    current_test.to_numpy(),
                    y_test.to_numpy(),
                    predictions,
                )
            )

            results.append(
                {
                    "vessel": vessel,
                    "horizon_days": horizon,
                    "model": model_name,
                    "test_rows": len(test),
                    "mae": model_mae,
                    "rmse": model_rmse,
                    "mape_pct": model_mape,
                    "directional_accuracy_pct":
                        model_direction,
                }
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "vessel",
        "horizon_days",
        "mae",
    ]
)

output = (
    OUTPUT_DIR
    / "freight_model_backtest.csv"
)

results_df.to_csv(
    output,
    index=False,
)


# ============================================================
# MODEL VS PERSISTENCE
# ============================================================

print()
print("=== RESULTS ===")
print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print("=== BEST MODEL VS PERSISTENCE ===")
print()

for vessel in VESSELS:

    for horizon in HORIZONS:

        subset = results_df[
            (results_df["vessel"] == vessel)
            &
            (
                results_df["horizon_days"]
                == horizon
            )
        ]

        persistence = subset[
            subset["model"]
            == "persistence"
        ].iloc[0]

        ml = subset[
            subset["model"]
            != "persistence"
        ].sort_values(
            "mae"
        ).iloc[0]

        improvement = (
            1
            -
            ml["mae"]
            /
            persistence["mae"]
        ) * 100

        print(
            f"{vessel:9s} "
            f"{horizon:2d}d | "
            f"best={ml['model']:8s} | "
            f"MAE={ml['mae']:,.0f} | "
            f"baseline={persistence['mae']:,.0f} | "
            f"improvement={improvement:+.2f}%"
        )


print()
print(f"Saved: {output}")
