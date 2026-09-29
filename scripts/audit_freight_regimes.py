from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "02_DATA/processed/forecast_calendar"
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


print("=== FREIGHT REGIME AUDIT ===\n")


for vessel in VESSELS:

    path = (
        DATA_DIR
        / f"{vessel}_30d.parquet"
    )

    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    )

    print(
        f"--- {vessel.upper()} ---"
    )

    print(
        f"Date range: "
        f"{df.date.min().date()} → "
        f"{df.date.max().date()}"
    )

    print(
        f"Current freight:"
        f" min={df.current_freight.min():,.0f}"
        f" median={df.current_freight.median():,.0f}"
        f" max={df.current_freight.max():,.0f}"
    )

    # Monthly return volatility
    monthly = (
        df.set_index("date")
        ["current_freight"]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
    )

    print(
        f"Monthly volatility: "
        f"{monthly.std():.2%}"
    )

    print(
        f"Monthly p05/p95: "
        f"{monthly.quantile(.05):+.2%} / "
        f"{monthly.quantile(.95):+.2%}"
    )

    # Test period
    n = len(df)

    test_start = int(
        n * 0.85
    )

    test = df.iloc[
        test_start:
    ]

    print(
        f"Test period: "
        f"{test.date.min().date()} → "
        f"{test.date.max().date()}"
    )

    print(
        f"Test current median: "
        f"{test.current_freight.median():,.0f}"
    )

    print(
        f"Test target median: "
        f"{test.target.median():,.0f}"
    )

    print(
        f"Test target return: "
        f"median={test.target_return.median():+.2%} "
        f"mean={test.target_return.mean():+.2%}"
    )

    print(
        f"Test target return std: "
        f"{test.target_return.std():.2%}"
    )

    print()


print("=== RECENT FREIGHT LEVELS ===")

foundation = pd.read_parquet(
    ROOT
    / "02_DATA/processed/point_in_time_features.parquet"
)

foundation["date"] = pd.to_datetime(
    foundation["date"]
)

print(
    foundation[
        [
            "date",
            "cape",
            "panamax",
            "supramax",
            "handy",
        ]
    ]
    .tail(15)
    .to_string(index=False)
)
