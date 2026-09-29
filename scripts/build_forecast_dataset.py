from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "02_DATA/processed/point_in_time_features.parquet"
OUTPUT_DIR = ROOT / "02_DATA/processed/forecast"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# MODEL TARGETS
# ============================================================

TARGETS = {
    "cape": "cape",
    "panamax": "panamax",
    "supramax": "supramax",
    "handy": "handy",
}

HORIZONS = {
    7: 7,
    14: 14,
    30: 30,
    60: 60,
}


# ============================================================
# FEATURE SELECTION
#
# We intentionally DO NOT feed the entire 211-column feature
# warehouse into the first model.
#
# Start with variables that have a defensible relationship
# with bulk freight:
#
# 1. Historical freight state
# 2. Commodity state
# 3. Port activity
# 4. Seasonality
#
# Economic variables remain available in the warehouse and
# can be tested later through controlled experiments.
# ============================================================

feature_cols = []


# ------------------------------------------------------------
# Freight history
# ------------------------------------------------------------

for vessel in TARGETS:

    feature_cols.extend(
        [
            f"{vessel}_lag_1",
            f"{vessel}_lag_7obs",
            f"{vessel}_lag_30obs",
            f"{vessel}_mean_7obs",
            f"{vessel}_mean_30obs",
            f"{vessel}_std_30obs",
        ]
    )


# ------------------------------------------------------------
# Commodity history
# ------------------------------------------------------------

for col in [
    "commodity_crude_oil_brent",
    "commodity_iron_ore_global_cfr_china",
    "commodity_thermal_coal_australia",
    "commodity_thermal_coal_south_africa",
]:

    feature_cols.extend(
        [
            col,
            f"{col}_lag_1",
            f"{col}_change",
            f"{col}_mean_3",
        ]
    )


# ------------------------------------------------------------
# PortWatch operational activity
# ------------------------------------------------------------

for col in [
    "port_dhamra_export_dry_bulk",
    "port_dhamra_import_dry_bulk",
    "port_dhamra_portcalls_dry_bulk",

    "port_haldia_export_dry_bulk",
    "port_haldia_import_dry_bulk",
    "port_haldia_portcalls_dry_bulk",

    "port_paradip_export_dry_bulk",
    "port_paradip_import_dry_bulk",
    "port_paradip_portcalls_dry_bulk",

    "port_visakhapatnam_export_dry_bulk",
    "port_visakhapatnam_import_dry_bulk",
    "port_visakhapatnam_portcalls_dry_bulk",
]:

    feature_cols.extend(
        [
            f"{col}_lag_1",
            f"{col}_mean_7",
            f"{col}_std_30",
        ]
    )


# ------------------------------------------------------------
# Calendar
# ------------------------------------------------------------

feature_cols.extend(
    [
        "day_of_week",
        "month",
        "quarter",
        "month_sin",
        "month_cos",
    ]
)


# Remove accidental duplicates while preserving order.
feature_cols = list(dict.fromkeys(feature_cols))


# ============================================================
# BUILD EACH FORECAST DATASET
# ============================================================

print("=== FORECAST DATASET BUILD ===")
print(f"Source rows: {len(df)}")
print(f"Features: {len(feature_cols)}")

for vessel, target_col in TARGETS.items():

    for horizon, observation_shift in HORIZONS.items():

        data = df[
            [
                "date",
                target_col,
            ]
            + feature_cols
        ].copy()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Freight observations are NOT calendar-daily.
        #
        # Therefore this means:
        #
        # "7 observed market records ahead"
        #
        # NOT:
        #
        # "exactly seven calendar days ahead".
        #
        # We keep this explicit rather than fabricating
        # missing market observations.
        # ----------------------------------------------------

        data["target_date"] = data["date"].shift(-observation_shift)

        data["target"] = (
            data[target_col]
            .shift(-observation_shift)
        )

        data["horizon_observations"] = observation_shift

        # Percentage change between current and future rate.
        data["target_return"] = (
            data["target"] / data[target_col] - 1
        )

        # Absolute change in USD/day.
        data["target_change"] = (
            data["target"] - data[target_col]
        )

        data["vessel_class"] = vessel
        data["forecast_horizon"] = horizon

        # ----------------------------------------------------
        # Remove rows where future target does not exist.
        # ----------------------------------------------------

        data = data.dropna(
            subset=["target"]
        ).reset_index(drop=True)

        output = (
            OUTPUT_DIR
            / f"{vessel}_{horizon}obs.parquet"
        )

        data.to_parquet(
            output,
            index=False,
        )

        print(
            f"{vessel:9s} "
            f"{horizon:2d} obs | "
            f"rows={len(data):4d} | "
            f"features={len(feature_cols):2d} | "
            f"{output.name}"
        )


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "source": str(INPUT),
    "targets": list(TARGETS.keys()),
    "horizons_observations": list(HORIZONS.keys()),
    "feature_count": len(feature_cols),
    "feature_columns": feature_cols,
    "interpretation": (
        "Forecast horizons are measured in future freight "
        "market observations, not calendar days, because "
        "the historical KOBC series contains non-trading/"
        "non-publication gaps."
    ),
}

import json

with open(
    OUTPUT_DIR / "manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


print("\n=== COMPLETE ===")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")
