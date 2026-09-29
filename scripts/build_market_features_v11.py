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
    / "02_DATA/processed/market_features_v11.parquet"
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

PORT_METRICS = [
    "portcalls_dry_bulk",
    "portcalls",
    "import_dry_bulk",
    "export_dry_bulk",
]


def calendar_reference_value(
    dates: pd.Series,
    values: pd.Series,
    days: int,
) -> pd.Series:
    """
    For every observation date, find the latest available
    observation on or before date - N calendar days.

    This is important because KOBC observations are not
    guaranteed to exist on every calendar day.
    """

    current = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values.to_numpy(),
            "row_id": np.arange(len(dates)),
        }
    ).sort_values("date")

    targets = pd.DataFrame(
        {
            "target_date": (
                pd.to_datetime(dates)
                - pd.Timedelta(days=days)
            ),
            "row_id": np.arange(len(dates)),
        }
    ).sort_values("target_date")

    history = (
        current[
            ["date", "value"]
        ]
        .dropna(subset=["date"])
        .sort_values("date")
    )

    matched = pd.merge_asof(
        targets,
        history,
        left_on="target_date",
        right_on="date",
        direction="backward",
    )

    matched = matched.sort_values("row_id")

    result = pd.Series(
        np.nan,
        index=np.arange(len(dates)),
        dtype="float64",
    )

    valid = matched["value"].notna()

    result.loc[
        matched.loc[valid, "row_id"].astype(int)
    ] = matched.loc[
        valid,
        "value",
    ].to_numpy()

    return result


def calendar_change_pct(
    dates: pd.Series,
    values: pd.Series,
    days: int,
) -> pd.Series:
    """
    Percentage change against the latest available
    observation on or before N calendar days ago.
    """

    reference = calendar_reference_value(
        dates,
        values,
        days,
    )

    reference = reference.replace(
        0,
        np.nan,
    )

    return (
        (values - reference)
        / reference
        * 100.0
    )


def calendar_change(
    dates: pd.Series,
    values: pd.Series,
    days: int,
) -> pd.Series:
    """
    Absolute change against the latest available
    observation on or before N calendar days ago.
    """

    reference = calendar_reference_value(
        dates,
        values,
        days,
    )

    return values - reference


def calendar_volatility(
    dates: pd.Series,
    values: pd.Series,
    days: int,
) -> pd.Series:
    """
    Rolling volatility using an actual calendar-time window.

    Returns are calculated from consecutive available
    observations, while the rolling window itself is
    measured in calendar time.
    """

    temp = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values.to_numpy(),
        }
    ).sort_values("date")

    temp["return"] = (
        temp["value"]
        .pct_change()
    )

    temp = temp.set_index("date")

    volatility = (
        temp["return"]
        .rolling(
            f"{days}D",
            min_periods=2,
        )
        .std()
        * 100.0
    )

    return volatility.to_numpy()


def add_freight_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build compact freight-market state features.

    Temporal features use calendar-time windows rather
    than row-count windows.
    """

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
        .copy()
    )

    for column in FREIGHT_CLASSES + ["kdci"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Freight class features
    # ---------------------------------------------------------

    for vessel_class in FREIGHT_CLASSES:

        series = df[vessel_class]

        df[
            f"{vessel_class}_7d_change"
        ] = calendar_change(
            df["date"],
            series,
            7,
        )

        df[
            f"{vessel_class}_30d_change"
        ] = calendar_change(
            df["date"],
            series,
            30,
        )

        df[
            f"{vessel_class}_7d_change_pct"
        ] = calendar_change_pct(
            df["date"],
            series,
            7,
        )

        df[
            f"{vessel_class}_30d_change_pct"
        ] = calendar_change_pct(
            df["date"],
            series,
            30,
        )

        df[
            f"{vessel_class}_7d_volatility_pct"
        ] = calendar_volatility(
            df["date"],
            series,
            7,
        )

        df[
            f"{vessel_class}_30d_volatility_pct"
        ] = calendar_volatility(
            df["date"],
            series,
            30,
        )

    # ---------------------------------------------------------
    # Relative vessel-class economics
    # ---------------------------------------------------------

    spread_pairs = [
        ("cape", "panamax"),
        ("panamax", "supramax"),
        ("supramax", "handy"),
    ]

    for left, right in spread_pairs:

        right_safe = df[right].replace(
            0,
            np.nan,
        )

        df[
            f"{left}_vs_{right}_spread"
        ] = (
            df[left] - df[right]
        )

        df[
            f"{left}_vs_{right}_ratio"
        ] = (
            df[left] / right_safe
        )

    # ---------------------------------------------------------
    # KDCI
    # ---------------------------------------------------------

    df["kdci_7d_change"] = calendar_change(
        df["date"],
        df["kdci"],
        7,
    )

    df["kdci_30d_change"] = calendar_change(
        df["date"],
        df["kdci"],
        30,
    )

    df["kdci_7d_change_pct"] = calendar_change_pct(
        df["date"],
        df["kdci"],
        7,
    )

    df["kdci_30d_change_pct"] = calendar_change_pct(
        df["date"],
        df["kdci"],
        30,
    )

    df["kdci_7d_volatility_pct"] = calendar_volatility(
        df["date"],
        df["kdci"],
        7,
    )

    df["kdci_30d_volatility_pct"] = calendar_volatility(
        df["date"],
        df["kdci"],
        30,
    )

    return df


def build_port_activity_features(
    freight_dates: pd.Series,
) -> pd.DataFrame:
    """
    Build compact PortWatch activity features.

    PortWatch activity is operational/market context,
    NOT a direct congestion measurement.

    Missing observations are preserved as NaN.
    No forward filling is performed.
    """

    portwatch = pd.read_csv(
        PORTWATCH_FILE
    )

    portwatch["date"] = pd.to_datetime(
        portwatch["date"],
        errors="coerce",
    )

    portwatch = portwatch.dropna(
        subset=["date"]
    ).copy()

    portwatch["portname"] = (
        portwatch["portname"]
        .astype(str)
        .str.strip()
    )

    for column in PORT_METRICS:
        portwatch[column] = pd.to_numeric(
            portwatch[column],
            errors="coerce",
        )

    portwatch = portwatch[
        portwatch["portname"].isin(
            PORTS.values()
        )
    ].copy()

    portwatch["port_key"] = (
        portwatch["portname"].map(
            {
                name: key
                for key, name in PORTS.items()
            }
        )
    )

    portwatch = (
        portwatch
        .sort_values(
            ["port_key", "date"]
        )
        .drop_duplicates(
            subset=[
                "date",
                "port_key",
            ],
            keep="last",
        )
    )

    port_frames = []

    for port_key, group in portwatch.groupby(
        "port_key"
    ):

        group = (
            group
            .sort_values("date")
            .copy()
        )

        output = pd.DataFrame(
            index=group["date"]
        )

        # -----------------------------------------------------
        # Raw activity + rolling levels
        # -----------------------------------------------------

        for metric in PORT_METRICS:

            series = group[metric]

            output[
                f"{port_key}_{metric}"
            ] = series.to_numpy()

            output[
                f"{port_key}_{metric}_7d_mean"
            ] = (
                series
                .rolling(
                    7,
                    min_periods=7,
                )
                .mean()
                .to_numpy()
            )

            output[
                f"{port_key}_{metric}_30d_mean"
            ] = (
                series
                .rolling(
                    30,
                    min_periods=30,
                )
                .mean()
                .to_numpy()
            )

        # -----------------------------------------------------
        # Dry-bulk activity anomaly
        # -----------------------------------------------------

        dry_bulk = group[
            "portcalls_dry_bulk"
        ]

        rolling_mean = (
            dry_bulk
            .rolling(
                30,
                min_periods=30,
            )
            .mean()
        )

        rolling_std = (
            dry_bulk
            .rolling(
                30,
                min_periods=30,
            )
            .std()
        )

        output[
            f"{port_key}_dry_bulk_activity_zscore"
        ] = (
            (
                dry_bulk
                - rolling_mean
            )
            / rolling_std.replace(
                0,
                np.nan,
            )
        ).to_numpy()

        output.index.name = "date"

        port_frames.append(output)

    if not port_frames:
        return pd.DataFrame(
            index=pd.DatetimeIndex(
                freight_dates.unique(),
                name="date",
            )
        )

    port_features = (
        pd.concat(
            port_frames,
            axis=1,
        )
        .sort_index()
    )

    # ---------------------------------------------------------
    # Cross-port dry-bulk activity
    # ---------------------------------------------------------

    dry_bulk_columns = [
        f"{port}_portcalls_dry_bulk"
        for port in PORTS
    ]

    available_columns = [
        column
        for column in dry_bulk_columns
        if column in port_features.columns
    ]

    if available_columns:

        port_features[
            "all_ports_dry_bulk_calls"
        ] = (
            port_features[
                available_columns
            ]
            .sum(
                axis=1,
                min_count=1,
            )
        )

        total = port_features[
            "all_ports_dry_bulk_calls"
        ]

        share_columns = {}

        for port in PORTS:

            column = (
                f"{port}_portcalls_dry_bulk"
            )

            if column in port_features.columns:

                share_columns[
                    f"{port}_dry_bulk_call_share"
                ] = (
                    port_features[column]
                    / total.replace(
                        0,
                        np.nan,
                    )
                )

        if share_columns:
            port_features = pd.concat(
                [
                    port_features,
                    pd.DataFrame(
                        share_columns,
                        index=port_features.index,
                    ),
                ],
                axis=1,
            )

    # ---------------------------------------------------------
    # Align to freight dates.
    #
    # If PortWatch has no observation for a date,
    # the resulting value stays NaN.
    # ---------------------------------------------------------

    target_index = pd.DatetimeIndex(
        freight_dates.unique(),
        name="date",
    ).sort_values()

    return port_features.reindex(
        target_index
    )


def main() -> None:

    print("Loading KOBC freight data...")

    freight = pd.read_csv(
        FREIGHT_FILE
    )

    freight["date"] = pd.to_datetime(
        freight["date"],
        errors="coerce",
    )

    freight = (
        freight
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"KOBC rows: {len(freight):,}"
    )

    print(
        f"Date range: "
        f"{freight['date'].min().date()} → "
        f"{freight['date'].max().date()}"
    )

    print(
        "\nBuilding calendar-time freight features..."
    )

    freight = add_freight_features(
        freight
    )

    print(
        "Building compact PortWatch features..."
    )

    port_features = (
        build_port_activity_features(
            freight["date"]
        )
    )

    print(
        "Joining feature layers..."
    )

    result = (
        freight
        .set_index("date")
        .join(
            port_features,
            how="left",
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Replace infinities.
    # ---------------------------------------------------------

    numeric_columns = (
        result
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    result[numeric_columns] = (
        result[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    if result["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected."
        )

    if result["date"].isna().any():
        raise ValueError(
            "Missing dates detected."
        )

    if np.isinf(
        result[numeric_columns]
        .to_numpy()
    ).any():
        raise ValueError(
            "Infinite values remain."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n========================================"
    )
    print(
        "MARKET FEATURES V1.1 COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"Rows:    {len(result):,}"
    )

    print(
        f"Columns: {len(result.columns):,}"
    )

    print(
        f"Dates:   "
        f"{result['date'].min().date()} → "
        f"{result['date'].max().date()}"
    )

    print(
        f"Output:  {OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # Latest freight state
    # ---------------------------------------------------------

    print(
        "\nLatest freight state:"
    )

    latest_columns = [
        "date",
        "cape",
        "panamax",
        "supramax",
        "handy",
        "cape_7d_change_pct",
        "panamax_7d_change_pct",
        "supramax_7d_change_pct",
        "handy_7d_change_pct",
        "cape_7d_volatility_pct",
        "panamax_7d_volatility_pct",
        "supramax_7d_volatility_pct",
        "handy_7d_volatility_pct",
        "cape_vs_panamax_spread",
        "panamax_vs_supramax_spread",
        "supramax_vs_handy_spread",
    ]

    available = [
        column
        for column in latest_columns
        if column in result.columns
    ]

    print(
        result[available]
        .tail(1)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Latest PortWatch state
    # ---------------------------------------------------------

    print(
        "\nPortWatch latest observations:"
    )

    port_columns = [
        f"{port}_portcalls_dry_bulk"
        for port in PORTS
    ]

    port_columns = [
        column
        for column in port_columns
        if column in result.columns
    ]

    print(
        result[
            ["date"] + port_columns
        ]
        .tail(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Missingness summary
    # ---------------------------------------------------------

    print(
        "\nTop missingness:"
    )

    missingness = (
        result
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print(
        missingness.to_string()
    )


if __name__ == "__main__":
    main()