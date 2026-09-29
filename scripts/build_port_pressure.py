from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "02_DATA/processed/port_state"
    / "port_operational_state.csv"
)

OUTPUT_DIR = (
    ROOT
    / "02_DATA/processed/port_state"
)

OUTPUT = (
    OUTPUT_DIR
    / "port_pressure_v1.csv"
)


def clip01(x):
    if pd.isna(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))


def percentile_like(
    value,
    low,
    high,
):
    """
    Deterministic normalization.

    Values at/below low -> 0
    Values at/above high -> 1
    """
    if pd.isna(value):
        return np.nan

    if high <= low:
        return np.nan

    return clip01(
        (value - low)
        / (high - low)
    )


def classify_pressure(score):
    if pd.isna(score):
        return "UNKNOWN"

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


def calculate_port_pressure(row):

    components = {}

    # ---------------------------------------------------------------
    # 1. Berth pressure
    # ---------------------------------------------------------------

    occupancy = row[
        "occupancy_ratio"
    ]

    if pd.notna(occupancy):

        components["berth"] = clip01(
            occupancy
        )

    # ---------------------------------------------------------------
    # 2. Expected-vessel pressure
    # ---------------------------------------------------------------

    expected = row[
        "vessels_expected"
    ]

    if pd.notna(expected):

        # This is intentionally a broad operational
        # normalization, not a claim that N vessels
        # equals N% congestion.
        components["expected"] = (
            percentile_like(
                expected,
                0,
                50,
            )
        )

    # ---------------------------------------------------------------
    # 3. Dry-bulk workload pressure
    # ---------------------------------------------------------------

    dry_expected = row[
        "dry_bulk_vessels_expected"
    ]

    if pd.notna(dry_expected):

        components["dry_bulk_expected"] = (
            percentile_like(
                dry_expected,
                0,
                25,
            )
        )

    # ---------------------------------------------------------------
    # 4. Cargo workload pressure
    # ---------------------------------------------------------------

    cargo = row[
        "dry_bulk_cargo_mt_current"
    ]

    if pd.notna(cargo):

        components["cargo"] = (
            percentile_like(
                cargo,
                0,
                1_000_000,
            )
        )

    # ---------------------------------------------------------------
    # 5. Recent turnover / activity
    # ---------------------------------------------------------------

    departures = row[
        "recent_departures"
    ]

    if pd.notna(departures):

        components["departures"] = (
            percentile_like(
                departures,
                0,
                10,
            )
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Weights are applied only to available signals.
    # Missing != zero.
    # ---------------------------------------------------------------

    weights = {
        "berth": 0.35,
        "expected": 0.20,
        "dry_bulk_expected": 0.20,
        "cargo": 0.15,
        "departures": 0.10,
    }

    available = []

    for name, value in components.items():

        if pd.notna(value):

            available.append(
                (
                    name,
                    value,
                    weights[name],
                )
            )

    if not available:

        return {
            "pressure_score": np.nan,
            "pressure_band": "UNKNOWN",
            "pressure_confidence": 0.0,
            "available_components": 0,
            "component_json": json.dumps(
                components
            ),
        }

    total_weight = sum(
        weight
        for _, _, weight
        in available
    )

    weighted_score = sum(
        value * weight
        for _, value, weight
        in available
    )

    normalized = (
        weighted_score
        / total_weight
    )

    # Confidence reflects signal coverage,
    # not model accuracy.
    confidence = (
        total_weight
        / sum(weights.values())
    )

    return {
        "pressure_score": round(
            normalized * 100,
            2,
        ),
        "pressure_band": classify_pressure(
            normalized * 100
        ),
        "pressure_confidence": round(
            confidence,
            3,
        ),
        "available_components": len(
            available
        ),
        "component_json": json.dumps(
            components
        ),
    }


def main():

    df = pd.read_csv(
        INPUT
    )

    pressure_rows = []

    for _, row in df.iterrows():

        pressure_rows.append(
            calculate_port_pressure(
                row
            )
        )

    pressure = pd.DataFrame(
        pressure_rows
    )

    output = pd.concat(
        [
            df.reset_index(drop=True),
            pressure,
        ],
        axis=1,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print("PORT PRESSURE v1")
    print("=" * 80)

    print(
        output[
            [
                "port",
                "pressure_score",
                "pressure_band",
                "pressure_confidence",
                "available_components",
                "occupancy_ratio",
                "vessels_expected",
                "dry_bulk_vessels_expected",
                "dry_bulk_cargo_mt_current",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Output: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
