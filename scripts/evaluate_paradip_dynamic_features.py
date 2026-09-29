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

LABELS = (
    ROOT
    / "02_DATA/processed/paradip/paradip_turnaround_labels_v1.csv"
)

OPS = (
    ROOT
    / "02_DATA/processed/paradip/paradip_operational_history_daily.csv"
)

OUT = (
    ROOT
    / "02_DATA/processed/paradip/paradip_cycle_operational_join_v1.csv"
)


TARGET = "berth_duration_days"

BASE_FEATURES = [
    "cargo",
    "berth",
    "quantity_mt",
]

DYNAMIC_FEATURES = [
    "vessels_waiting_anchorage",
    "dry_bulk_vessels_waiting",
    "dry_bulk_cargo_mt_waiting",
    "berth_events",
    "sail_events",
    "shift_events",
    "dry_bulk_berth_events",
    "dry_bulk_sail_events",
    "unique_berth_events",
]

NUMERIC_BASE = [
    "quantity_mt",
]

CATEGORICAL_BASE = [
    "cargo",
    "berth",
]

NUMERIC_DYNAMIC = DYNAMIC_FEATURES + [
    "snapshot_age_days",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data():

    labels = pd.read_csv(
        LABELS,
        parse_dates=[
            "berth_start_timestamp",
            "sail_timestamp",
        ],
    )

    ops = pd.read_csv(
        OPS,
        parse_dates=[
            "date",
        ],
    )

    labels = labels.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    ops = ops.sort_values(
        "date"
    ).reset_index(drop=True)

    return labels, ops


# ---------------------------------------------------------------------------
# Leakage-safe operational snapshot attachment
# ---------------------------------------------------------------------------

def attach_latest_snapshot(
    labels: pd.DataFrame,
    ops: pd.DataFrame,
) -> pd.DataFrame:

    left = labels.copy()

    # We deliberately use DATE, not time-of-day.
    #
    # The source does not establish an exact publication timestamp for
    # each daily operational report. Therefore we conservatively associate
    # a cycle with the latest report whose report date is <= the cycle's
    # calendar date.
    left["cycle_date"] = (
        left["berth_start_timestamp"]
        .dt.normalize()
    )

    right = ops.copy()

    right["snapshot_date"] = (
        right["date"]
        .dt.normalize()
    )

    right = right.sort_values(
        "snapshot_date"
    )

    merged = pd.merge_asof(
        left.sort_values(
            "cycle_date"
        ),
        right,
        left_on="cycle_date",
        right_on="snapshot_date",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_ops"),
    )

    merged["snapshot_age_days"] = (
        merged["cycle_date"]
        - merged["snapshot_date"]
    ).dt.total_seconds() / 86400.0

    return merged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y = np.asarray(y_true)[mask]
    p = np.asarray(y_pred)[mask]

    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
        }

    return {
        "n": len(y),
        "mae": mean_absolute_error(y, p),
        "rmse": np.sqrt(
            mean_squared_error(y, p)
        ),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model(
    numeric_features,
    categorical_features,
):

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
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
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(df):

    df = df.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    return train, test


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def signal_correlation(df):

    print()
    print("=" * 100)
    print("OPERATIONAL SIGNAL CORRELATION")
    print("=" * 100)

    rows = []

    for feature in DYNAMIC_FEATURES:

        subset = df[
            [feature, TARGET]
        ].dropna()

        if len(subset) < 5:
            rho = np.nan
        else:
            rho = subset[
                feature
            ].corr(
                subset[TARGET],
                method="spearman",
            )

        rows.append(
            {
                "feature": feature,
                "n": len(subset),
                "spearman_rho": rho,
                "abs_rho": (
                    abs(rho)
                    if pd.notna(rho)
                    else np.nan
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "abs_rho",
            ascending=False,
        )
    )

    print(
        result.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------------
# Freshness experiment
# ---------------------------------------------------------------------------

def run_experiment(
    merged,
    max_age,
):

    subset = merged[
        merged["snapshot_age_days"].notna()
        & (
            merged["snapshot_age_days"]
            <= max_age
        )
    ].copy()

    subset = subset.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    if len(subset) < 20:
        return None

    train, test = chronological_split(
        subset
    )

    if len(train) < 10 or len(test) < 5:
        return None

    y_train = train[
        TARGET
    ].astype(float)

    y_test = test[
        TARGET
    ].astype(float)

    # ---------------------------------------------------------------
    # BASELINE
    # cargo + berth + quantity
    # ---------------------------------------------------------------

    baseline = make_model(
        NUMERIC_BASE,
        CATEGORICAL_BASE,
    )

    baseline.fit(
        train[
            BASE_FEATURES
        ],
        y_train,
    )

    baseline_pred = baseline.predict(
        test[
            BASE_FEATURES
        ]
    )

    baseline_metrics = metrics(
        y_test,
        baseline_pred,
    )

    # ---------------------------------------------------------------
    # DYNAMIC
    # baseline + operational state + snapshot age
    # ---------------------------------------------------------------

    dynamic_features = (
        BASE_FEATURES
        + NUMERIC_DYNAMIC
    )

    dynamic = make_model(
        NUMERIC_BASE + NUMERIC_DYNAMIC,
        CATEGORICAL_BASE,
    )

    dynamic.fit(
        train[
            dynamic_features
        ],
        y_train,
    )

    dynamic_pred = dynamic.predict(
        test[
            dynamic_features
        ]
    )

    dynamic_metrics = metrics(
        y_test,
        dynamic_pred,
    )

    improvement = (
        (
            baseline_metrics["mae"]
            - dynamic_metrics["mae"]
        )
        / baseline_metrics["mae"]
        * 100
    )

    return {
        "max_snapshot_age_days": max_age,
        "rows": len(subset),
        "train": len(train),
        "test": len(test),
        "test_start": (
            test[
                "berth_start_timestamp"
            ].min()
            .strftime("%Y-%m-%d")
        ),
        "test_end": (
            test[
                "berth_start_timestamp"
            ].max()
            .strftime("%Y-%m-%d")
        ),
        "baseline_mae_days": (
            baseline_metrics["mae"]
        ),
        "dynamic_mae_days": (
            dynamic_metrics["mae"]
        ),
        "mae_improvement_pct": improvement,
        "baseline_rmse_days": (
            baseline_metrics["rmse"]
        ),
        "dynamic_rmse_days": (
            dynamic_metrics["rmse"]
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    labels, ops = load_data()

    print("=" * 100)
    print("PARADIP LEAKAGE-SAFE DYNAMIC OPERATIONAL FEATURE EXPERIMENT")
    print("=" * 100)

    print()
    print(f"Berth-cycle labels:       {len(labels)}")
    print(f"Operational snapshots:    {len(ops)}")

    merged = attach_latest_snapshot(
        labels,
        ops,
    )

    print()
    print("SNAPSHOT ATTACHMENT")
    print("-" * 100)

    attached = (
        merged[
            "snapshot_date"
        ].notna()
    )

    print(
        f"Cycles with snapshot:     "
        f"{attached.sum()}/{len(merged)} "
        f"({attached.mean():.1%})"
    )

    print(
        f"Cycles without snapshot: "
        f"{(~attached).sum()}/{len(merged)} "
        f"({(~attached).mean():.1%})"
    )

    if attached.any():

        ages = merged.loc[
            attached,
            "snapshot_age_days",
        ]

        print()
        print("SNAPSHOT AGE")

        print(
            ages.describe(
                percentiles=[
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

        for threshold in [1, 3, 7]:

            count = (
                ages <= threshold
            ).sum()

            print(
                f"  <= {threshold} day: "
                f"{count}/{len(merged)} "
                f"({count/len(merged):.1%})"
            )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    signal_correlation(
        merged
    )

    # ------------------------------------------------------------------
    # Freshness experiments
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHRONOLOGICAL MODEL COMPARISON")
    print("=" * 100)

    results = []

    for max_age in [1, 3, 7]:

        result = run_experiment(
            merged,
            max_age,
        )

        if result is None:
            print(
                f"\n<= {max_age} day: "
                f"insufficient observations"
            )
            continue

        results.append(
            result
        )

        print()
        print(
            f"MAX SNAPSHOT AGE: <= {max_age} DAY(S)"
        )

        print(
            f"Rows:            {result['rows']}"
        )

        print(
            f"Train:           {result['train']}"
        )

        print(
            f"Test:            {result['test']}"
        )

        print(
            f"Test period:     "
            f"{result['test_start']} → "
            f"{result['test_end']}"
        )

        print(
            f"Baseline MAE:    "
            f"{result['baseline_mae_days']:.3f} days"
        )

        print(
            f"Dynamic MAE:     "
            f"{result['dynamic_mae_days']:.3f} days"
        )

        print(
            f"MAE improvement: "
            f"{result['mae_improvement_pct']:+.2f}%"
        )

        print(
            f"Baseline RMSE:   "
            f"{result['baseline_rmse_days']:.3f} days"
        )

        print(
            f"Dynamic RMSE:    "
            f"{result['dynamic_rmse_days']:.3f} days"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    if results:

        result_df = pd.DataFrame(
            results
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

    # ------------------------------------------------------------------
    # Save joined data
    # ------------------------------------------------------------------

    merged.to_csv(
        OUT,
        index=False,
    )

    print()
    print(f"Wrote joined dataset:")
    print(OUT)


if __name__ == "__main__":
    main()
