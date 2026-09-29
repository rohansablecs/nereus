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

LABELS = (
    ROOT
    / "02_DATA/processed/paradip"
    / "paradip_turnaround_labels_v1.csv"
)

PORT_STATE = (
    ROOT
    / "02_DATA/processed/port_state"
    / "port_operational_state.csv"
)

PRESSURE = (
    ROOT
    / "02_DATA/processed/port_state"
    / "port_pressure_v1.csv"
)


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def main():

    labels = pd.read_csv(LABELS)

    labels["berth_start_timestamp"] = pd.to_datetime(
        labels["berth_start_timestamp"],
        errors="coerce",
    )

    labels["berth_duration_days"] = pd.to_numeric(
        labels["berth_duration_days"],
        errors="coerce",
    )

    numeric_label_cols = [
        "quantity_mt",
        "loa_m",
        "beam_m",
        "draft_m",
        "shift_count",
    ]

    for col in numeric_label_cols:
        labels[col] = pd.to_numeric(
            labels[col],
            errors="coerce",
        )

    labels["cargo_clean"] = (
        labels["cargo"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    labels["berth_clean"] = (
        labels["berth"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Calendar features.
    # ---------------------------------------------------------------

    labels["month"] = (
        labels["berth_start_timestamp"].dt.month
    )

    labels["day_of_week"] = (
        labels["berth_start_timestamp"].dt.dayofweek
    )

    labels = labels[
        labels["berth_start_timestamp"].notna()
        &
        labels["berth_duration_days"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Load dynamic Paradip state.
    # ---------------------------------------------------------------

    state = pd.read_csv(PORT_STATE)

    state = state[
        state["port"].astype(str).str.upper().eq(
            "PARADIP"
        )
    ].copy()

    if state.empty:
        raise RuntimeError(
            "No Paradip operational state found."
        )

    state_time = pd.to_datetime(
        state["snapshot_time"],
        errors="coerce",
        utc=True,
    )

    state["state_timestamp"] = state_time

    pressure = pd.read_csv(PRESSURE)

    pressure = pressure[
        pressure["port"].astype(str).str.upper().eq(
            "PARADIP"
        )
    ].copy()

    if pressure.empty:
        raise RuntimeError(
            "No Paradip pressure state found."
        )

    pressure_cols = [
        "pressure_score",
        "pressure_confidence",
    ]

    pressure_values = {}

    for col in pressure_cols:
        if col in pressure.columns:
            pressure_values[col] = pd.to_numeric(
                pressure.iloc[0][col],
                errors="coerce",
            )

    # ---------------------------------------------------------------
    # Important:
    #
    # The current operational state file contains ONE current
    # snapshot. It cannot legitimately be backfilled across the
    # entire historical label set.
    #
    # Therefore this experiment uses the current state only for
    # observations whose berth start occurs sufficiently close to
    # that snapshot.
    #
    # Historical rows retain NaN rather than receiving fake state.
    # ---------------------------------------------------------------

    state_features = [
        "occupancy_ratio",
        "vessels_expected",
        "dry_bulk_vessels_expected",
        "dry_bulk_cargo_mt_current",
        "recent_departures",
    ]

    for col in state_features:
        if col in state.columns:
            state[col] = pd.to_numeric(
                state[col],
                errors="coerce",
            )

    latest_state = state.iloc[0]

    latest_timestamp = latest_state[
        "state_timestamp"
    ]

    if pd.isna(latest_timestamp):
        raise RuntimeError(
            "Paradip operational state has no valid timestamp."
        )

    # Use state only for labels within 24 hours of the snapshot.
    labels["start_timestamp_utc"] = (
        labels["berth_start_timestamp"]
        .dt.tz_localize("UTC")
    )

    age_hours = (
        labels["start_timestamp_utc"]
        - latest_timestamp
    ).dt.total_seconds().abs() / 3600.0

    near_state = age_hours <= 24

    for col in state_features:
        if col in latest_state.index:
            value = latest_state[col]

            labels.loc[
                near_state,
                col,
            ] = value

    labels.loc[
        near_state,
        "pressure_score",
    ] = pressure_values.get(
        "pressure_score"
    )

    labels.loc[
        near_state,
        "pressure_confidence",
    ] = pressure_values.get(
        "pressure_confidence"
    )

    # ---------------------------------------------------------------
    # Report coverage.
    # ---------------------------------------------------------------

    dynamic_columns = state_features + [
        "pressure_score",
        "pressure_confidence",
    ]

    print("=" * 80)
    print("PARADIP DYNAMIC-STATE EXPERIMENT")
    print("=" * 80)

    print(
        "Total label rows:",
        len(labels),
    )

    print(
        "Rows within 24h of available operational snapshot:",
        int(near_state.sum()),
    )

    print(
        "Rows without historical dynamic state:",
        int((~near_state).sum()),
    )

    print()
    print(
        "IMPORTANT: historical rows are NOT filled with today's"
        " state. Missing dynamic state remains missing."
    )

    # ---------------------------------------------------------------
    # Because we only have a current state snapshot, a full
    # historical ML comparison would be invalid.
    #
    # We therefore restrict this experiment to rows for which the
    # dynamic state genuinely exists.
    # ---------------------------------------------------------------

    experiment = labels[
        near_state
    ].copy()

    if len(experiment) < 30:
        print()
        print(
            "NOT ENOUGH HISTORICAL DYNAMIC-STATE OBSERVATIONS"
        )
        print(
            "Dynamic ML experiment is not statistically meaningful."
        )
        print(
            "Rows available:",
            len(experiment),
        )
        print()
        print(
            "RESULT: Dynamic-state model deferred until historical"
            " operational snapshots are ingested."
        )
        return

    # ---------------------------------------------------------------
    # Chronological split inside the valid dynamic-state subset.
    # ---------------------------------------------------------------

    experiment = experiment.sort_values(
        "berth_start_timestamp"
    ).reset_index(drop=True)

    split = int(
        len(experiment) * 0.70
    )

    train = experiment.iloc[:split].copy()
    test = experiment.iloc[split:].copy()

    target = "berth_duration_days"

    categorical = [
        "cargo_clean",
        "berth_clean",
    ]

    numeric = [
        "quantity_mt",
        "loa_m",
        "beam_m",
        "draft_m",
        "shift_count",
        "month",
        "day_of_week",
    ] + dynamic_columns

    features = categorical + numeric

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
                categorical,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
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

    pipeline.fit(
        train[features],
        train[target],
    )

    predictions = pipeline.predict(
        test[features]
    )

    # ---------------------------------------------------------------
    # Simple cargo + berth baseline.
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

    cargo_medians = (
        train.groupby(
            "cargo_clean"
        )[target]
        .median()
    )

    global_median = train[target].median()

    baseline = (
        test["cargo_berth_key"]
        .map(medians)
        .fillna(
            test["cargo_clean"].map(
                cargo_medians
            )
        )
        .fillna(
            global_median
        )
        .to_numpy()
    )

    baseline_mae = mean_absolute_error(
        test[target],
        baseline,
    )

    baseline_rmse = rmse(
        test[target],
        baseline,
    )

    model_mae = mean_absolute_error(
        test[target],
        predictions,
    )

    model_rmse = rmse(
        test[target],
        predictions,
    )

    improvement = (
        1
        -
        model_mae / baseline_mae
    ) * 100

    print()
    print("=" * 80)
    print("VALID DYNAMIC-STATE COMPARISON")
    print("=" * 80)

    print(
        "Train:",
        len(train),
    )

    print(
        "Test:",
        len(test),
    )

    print()

    print(
        "Cargo + berth baseline"
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
        "Dynamic XGBoost"
    )

    print(
        "  MAE :",
        round(
            model_mae,
            3,
        ),
        "days",
    )

    print(
        "  RMSE:",
        round(
            model_rmse,
            3,
        ),
        "days",
    )

    print()

    print(
        "MAE improvement:",
        round(
            improvement,
            2,
        ),
        "%",
    )

    if improvement >= 5:
        print()
        print(
            "RESULT: Dynamic state shows meaningful predictive lift."
        )
    else:
        print()
        print(
            "RESULT: No meaningful dynamic-state lift demonstrated."
        )


if __name__ == "__main__":
    main()
