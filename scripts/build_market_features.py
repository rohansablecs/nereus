from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FREIGHT_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/freight/kobc/kobc_drybulk_daily.csv"
)

PORTWATCH_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/portwatch_daily_ports.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "02_DATA/processed/market_features.parquet"
)


PORTS = {
    "dhamra": "Dhamra Port",
    "gopalpur": "Gopalpur",
    "haldia": "Haldia",
    "paradip": "Paradip",
    "visakhapatnam": "Visakhapatnam",
}

FREIGHT_CLASSES = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    """
    Percentage change expressed in percent.

    Division by zero is treated as missing rather than infinity.
    """
    previous = series.shift(periods)

    result = pd.Series(np.nan, index=series.index, dtype="float64")

    valid = previous.notna() & previous.ne(0)
    result.loc[valid] = (
        (series.loc[valid] - previous.loc[valid])
        / previous.loc[valid]
        * 100.0
    )

    return result


def add_freight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add level, change, volatility and relative-spread features
    to the KOBC daily freight dataset.
    """

    df = df.sort_values("date").copy()

    # Make sure all freight columns are numeric.
    for col in FREIGHT_CLASSES + ["kdci"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------------------------
    # Freight momentum / changes
    # ---------------------------------------------------------

    for vessel_class in FREIGHT_CLASSES:
        series = df[vessel_class]

        for window in [7, 30]:
            df[f"{vessel_class}_{window}d_change_pct"] = pct_change(
                series,
                window,
            )

        # Absolute change is useful alongside percentage change.
        for window in [7, 30]:
            df[f"{vessel_class}_{window}d_change"] = (
                series - series.shift(window)
            )

    # ---------------------------------------------------------
    # Rolling volatility
    # ---------------------------------------------------------

    for vessel_class in FREIGHT_CLASSES:
        series = df[vessel_class]

        daily_return = series.pct_change()

        for window in [7, 30]:
            df[f"{vessel_class}_{window}d_volatility_pct"] = (
                daily_return
                .rolling(window, min_periods=window)
                .std()
                * 100.0
            )

    # ---------------------------------------------------------
    # Vessel-class relative spreads
    # ---------------------------------------------------------

    spread_pairs = [
        ("cape", "panamax"),
        ("panamax", "supramax"),
        ("supramax", "handy"),
        ("cape", "supramax"),
        ("cape", "handy"),
    ]

    for left, right in spread_pairs:
        df[f"{left}_vs_{right}_spread"] = (
            df[left] - df[right]
        )

        df[f"{left}_vs_{right}_ratio"] = (
            df[left] / df[right].replace(0, np.nan)
        )

    # ---------------------------------------------------------
    # KDCI changes
    # ---------------------------------------------------------

    df["kdci_7d_change_pct"] = pct_change(df["kdci"], 7)
    df["kdci_30d_change_pct"] = pct_change(df["kdci"], 30)

    df["kdci_7d_change"] = df["kdci"] - df["kdci"].shift(7)
    df["kdci_30d_change"] = df["kdci"] - df["kdci"].shift(30)

    kdci_returns = df["kdci"].pct_change()

    df["kdci_7d_volatility_pct"] = (
        kdci_returns
        .rolling(7, min_periods=7)
        .std()
        * 100.0
    )

    df["kdci_30d_volatility_pct"] = (
        kdci_returns
        .rolling(30, min_periods=30)
        .std()
        * 100.0
    )

    return df


def build_port_features(
    freight_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate PortWatch daily data into features aligned to
    the freight observation dates.

    Missing source dates remain missing.
    """

    portwatch = pd.read_csv(PORTWATCH_FILE)

    portwatch["date"] = pd.to_datetime(
        portwatch["date"],
        errors="coerce",
    )

    portwatch = portwatch.dropna(subset=["date"]).copy()

    portwatch["portname"] = (
        portwatch["portname"]
        .astype(str)
        .str.strip()
    )

    numeric_columns = [
        "portcalls_dry_bulk",
        "portcalls",
        "import_dry_bulk",
        "export_dry_bulk",
    ]

    for col in numeric_columns:
        portwatch[col] = pd.to_numeric(
            portwatch[col],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Keep only the five destination ports.
    # ---------------------------------------------------------

    portwatch = portwatch[
        portwatch["portname"].isin(PORTS.values())
    ].copy()

    portwatch["port_key"] = (
        portwatch["portname"]
        .map({value: key for key, value in PORTS.items()})
    )

    # One row per port/date is expected.
    portwatch = (
        portwatch
        .sort_values(["port_key", "date"])
        .drop_duplicates(
            subset=["date", "port_key"],
            keep="last",
        )
    )

    # ---------------------------------------------------------
    # Port-level features
    # ---------------------------------------------------------

    feature_frames = []

    for port_key, group in portwatch.groupby("port_key"):
        group = group.sort_values("date").copy()

        prefix = port_key

        for col in numeric_columns:
            series = group[col]

            group[f"{prefix}_{col}"] = series

            group[f"{prefix}_{col}_7d_change_pct"] = pct_change(
                series,
                7,
            )

            group[f"{prefix}_{col}_30d_change_pct"] = pct_change(
                series,
                30,
            )

            group[f"{prefix}_{col}_7d_mean"] = (
                series
                .rolling(7, min_periods=7)
                .mean()
            )

            group[f"{prefix}_{col}_30d_mean"] = (
                series
                .rolling(30, min_periods=30)
                .mean()
            )

        # Share of this port's dry-bulk calls among the five
        # tracked destination ports.
        feature_frames.append(group)

    # ---------------------------------------------------------
    # Pivot port features into one row per date.
    # ---------------------------------------------------------

    port_feature_frames = []

    for group in feature_frames:
        port_key = group["port_key"].iloc[0]

        group = group.set_index("date")

        feature_columns = [
            col
            for col in group.columns
            if col.startswith(f"{port_key}_")
        ]

        port_feature_frames.append(
            group[feature_columns]
        )

    if port_feature_frames:
        port_features = pd.concat(
            port_feature_frames,
            axis=1,
        ).sort_index()
    else:
        port_features = pd.DataFrame()

    # ---------------------------------------------------------
    # Cross-port activity.
    # ---------------------------------------------------------

    dry_bulk_cols = [
        f"{port}_portcalls_dry_bulk"
        for port in PORTS
    ]

    if not port_features.empty:
        existing = [
            col
            for col in dry_bulk_cols
            if col in port_features.columns
        ]

        if existing:
            port_features["all_ports_dry_bulk_calls"] = (
                port_features[existing].sum(
                    axis=1,
                    min_count=1,
                )
            )

            for port in PORTS:
                col = f"{port}_portcalls_dry_bulk"

                if col in port_features.columns:
                    denominator = (
                        port_features["all_ports_dry_bulk_calls"]
                    )

                    port_features[
                        f"{port}_dry_bulk_call_share"
                    ] = (
                        port_features[col]
                        / denominator.replace(0, np.nan)
                    )

    # ---------------------------------------------------------
    # Explicitly reindex to freight dates.
    #
    # This is important:
    # if PortWatch has no observation for a date, we retain NaN.
    # We do NOT forward-fill.
    # ---------------------------------------------------------

    freight_dates = (
        freight_dates[["date"]]
        .drop_duplicates()
        .set_index("date")
        .sort_index()
    )

    return freight_dates.join(
        port_features,
        how="left",
    )


def main() -> None:
    print("Loading KOBC freight data...")

    freight = pd.read_csv(FREIGHT_FILE)

    freight["date"] = pd.to_datetime(
        freight["date"],
        errors="coerce",
    )

    freight = (
        freight
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    print(f"KOBC rows: {len(freight):,}")
    print(
        f"Date range: "
        f"{freight['date'].min().date()} → "
        f"{freight['date'].max().date()}"
    )

    print("\nBuilding freight features...")
    freight = add_freight_features(freight)

    print("Building PortWatch features...")
    port_features = build_port_features(freight[["date"]])

    print("Joining feature layers...")

    result = (
        freight
        .set_index("date")
        .join(
            port_features,
            how="left",
            rsuffix="_port",
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Quality checks
    # ---------------------------------------------------------

    if result["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    if result["date"].isna().any():
        raise ValueError("Missing dates detected.")

    # Make sure infinity never enters downstream ML.
    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    inf_count = np.isinf(
        result[numeric_columns].to_numpy()
    ).sum()

    if inf_count:
        print(
            f"Replacing {inf_count} infinite numeric values "
            "with NaN."
        )

        result[numeric_columns] = (
            result[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n========================================")
    print("MARKET FEATURES V1 COMPLETE")
    print("========================================")
    print(f"Rows:    {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(
        f"Dates:   "
        f"{result['date'].min().date()} → "
        f"{result['date'].max().date()}"
    )
    print(f"Output:  {OUTPUT_FILE}")

    print("\nSample columns:")

    for column in result.columns[:40]:
        print(f"  {column}")

    print("\nLatest feature row:")
    print(
        result.tail(1).T.to_string(
            header=False
        )
    )


if __name__ == "__main__":
    main()