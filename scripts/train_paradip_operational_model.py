from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]

LABELS = (
    ROOT
    / "02_DATA/processed/paradip/"
    / "paradip_turnaround_labels_v1.csv"
)

OPS = (
    ROOT
    / "02_DATA/processed/paradip/"
    / "paradip_operational_history_daily.csv"
)

MODEL_DIR = (
    ROOT
    / "03_ML/risk/paradip"
)

MODEL_PATH = (
    MODEL_DIR
    / "paradip_berth_duration_model.joblib"
)

META_PATH = (
    MODEL_DIR
    / "paradip_berth_duration_model_metadata.json"
)


FEATURES = [
    "cargo",
    "berth",
    "quantity_mt",
    "dry_bulk_vessels_waiting",
]

TARGET = "berth_duration_days"


def load_training_data():

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

    labels = (
        labels
        .sort_values("berth_start_timestamp")
        .reset_index(drop=True)
    )

    ops = (
        ops
        .sort_values("date")
        .reset_index(drop=True)
    )

    labels["cycle_date"] = (
        labels[
            "berth_start_timestamp"
        ].dt.normalize()
    )

    ops["snapshot_date"] = (
        ops["date"].dt.normalize()
    )

    # Point-in-time join.
    merged = pd.merge_asof(
        labels.sort_values("cycle_date"),
        ops.sort_values("snapshot_date"),
        left_on="cycle_date",
        right_on="snapshot_date",
        direction="backward",
        allow_exact_matches=True,
    )

    return merged


def build_model():

    categorical = [
        "cargo",
        "berth",
    ]

    numeric = [
        "quantity_mt",
        "dry_bulk_vessels_waiting",
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


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_training_data()

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing features: {missing}"
        )

    df = df.dropna(
        subset=[
            TARGET,
            "cargo",
            "berth",
        ]
    ).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    model = build_model()

    model.fit(
        X,
        y,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    metadata = {
        "model": "HistGradientBoostingRegressor",
        "purpose": "Paradip berth-cycle duration prediction",
        "target": TARGET,
        "features": FEATURES,
        "training_rows": len(df),
        "training_start": (
            df[
                "berth_start_timestamp"
            ].min()
            .isoformat()
        ),
        "training_end": (
            df[
                "berth_start_timestamp"
            ].max()
            .isoformat()
        ),
        "random_state": 42,
        "validation_status": (
            "Research-validated feature set. "
            "Chronological holdout showed lower MAE/RMSE "
            "when dry_bulk_vessels_waiting was added to "
            "cargo + berth + quantity."
        ),
        "important_caveat": (
            "Operational snapshot date is used as the "
            "point-in-time state. Exact publication time "
            "of the daily source report is not established."
        ),
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PARADIP BERTH-DURATION MODEL")
    print("=" * 100)

    print(
        f"Training rows: {len(df)}"
    )

    print(
        f"Training period: "
        f"{df['berth_start_timestamp'].min()} "
        f"→ "
        f"{df['berth_start_timestamp'].max()}"
    )

    print()
    print("Features:")

    for feature in FEATURES:
        print(
            f"  {feature}"
        )

    print()
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {META_PATH}")


if __name__ == "__main__":
    main()
