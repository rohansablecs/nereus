from pathlib import Path

import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "02_DATA/processed/point_in_time_features.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "02_DATA/processed/forecast_calendar"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
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
# FEATURES
#
# Same controlled feature set as before.
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
# Commodity
# ------------------------------------------------------------

commodity_cols = [
    "commodity_crude_oil_brent",
    "commodity_iron_ore_global_cfr_china",
    "commodity_thermal_coal_australia",
    "commodity_thermal_coal_south_africa",
]

for col in commodity_cols:

    feature_cols.extend(
        [
            col,
            f"{col}_lag_1",
            f"{col}_change",
            f"{col}_mean_3",
        ]
    )


# ------------------------------------------------------------
# PortWatch
# ------------------------------------------------------------

port_cols = [
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
]

for col in port_cols:

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

feature_cols = list(
    dict.fromkeys(feature_cols)
)


# ============================================================
# BUILD CALENDAR TARGETS
#
# For each decision date T:
#
# target date = T + horizon
#
# Actual target = FIRST available freight observation
# ON OR AFTER target date.
#
# We do not interpolate.
# ============================================================

dates = df["date"].values

print("=== CALENDAR FORECAST DATASET ===")
print(f"Source observations: {len(df)}")
print(f"Features: {len(feature_cols)}")
print()


for vessel in TARGETS:

    target_series = (
        df[
            [
                "date",
                vessel,
            ]
        ]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    target_dates = target_series["date"].values
    target_values = target_series[vessel].values

    for horizon in HORIZONS:

        decision_dates = df["date"]

        requested_dates = (
            decision_dates
            + pd.Timedelta(days=horizon)
        )

        # ----------------------------------------------------
        # Find first actual market observation >= requested
        # calendar target.
        # ----------------------------------------------------

        positions = np.searchsorted(
            target_dates,
            requested_dates.values,
            side="left",
        )

        valid = (
            positions < len(target_series)
        )

        data = df[
            [
                "date",
                vessel,
            ]
            + feature_cols
        ].copy()

        data = data.rename(
            columns={
                vessel: "current_freight"
            }
        )

        data["requested_target_date"] = requested_dates

        data["target_date"] = pd.NaT

        data["target"] = np.nan

        data.loc[
            valid,
            "target_date"
        ] = target_series.loc[
            positions[valid],
            "date"
        ].values

        data.loc[
            valid,
            "target"
        ] = target_values[
            positions[valid]
        ]

        # ----------------------------------------------------
        # How far after the requested date did the market
        # observation actually occur?
        # ----------------------------------------------------

        data["target_delay_days"] = (
            data["target_date"]
            - data["requested_target_date"]
        ).dt.total_seconds() / 86400

        # ----------------------------------------------------
        # Target movement
        # ----------------------------------------------------

        current = df[vessel]

        data["target_change"] = (
            data["target"] - current
        )

        data["target_return"] = (
            data["target"] / current - 1
        )

        data["vessel_class"] = vessel

        data["forecast_horizon_days"] = horizon

        # ----------------------------------------------------
        # Remove rows without future observation.
        # ----------------------------------------------------

        data = data.dropna(
            subset=["target"]
        ).reset_index(drop=True)

        output = (
            OUTPUT_DIR
            / f"{vessel}_{horizon}d.parquet"
        )

        data.to_parquet(
            output,
            index=False,
        )

        print(
            f"{vessel:9s} "
            f"{horizon:2d}d | "
            f"rows={len(data):4d} | "
            f"median target delay="
            f"{data['target_delay_days'].median():.1f}d | "
            f"p90 delay="
            f"{data['target_delay_days'].quantile(.90):.1f}d"
        )


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "source": str(INPUT),
    "target_type": "calendar_horizon",
    "targets": TARGETS,
    "horizons_days": HORIZONS,
    "target_definition": (
        "For decision date T and horizon H, the target is "
        "the first available freight observation on or "
        "after T + H. No interpolation is performed."
    ),
    "feature_count": len(feature_cols),
    "feature_columns": feature_cols,
}


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


print()
print("=== COMPLETE ===")
print(f"Output: {OUTPUT_DIR}")
