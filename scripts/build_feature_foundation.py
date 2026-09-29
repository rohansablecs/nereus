from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "02_DATA" / "processed"

FREIGHT = (
    ROOT
    / "02_DATA/raw/freight/kobc/kobc_drybulk_daily.csv"
)

COMMODITY = (
    ROOT
    / "02_DATA/raw/commodity/world_bank/"
    "world_bank_commodity_monthly.csv"
)

ECONOMIC = (
    ROOT
    / "02_DATA/raw/economic/imf/"
    "imf_weo_economic_annual.csv"
)

WEATHER = (
    ROOT
    / "02_DATA/raw/weather/open_meteo/"
    "open_meteo_weather_current.csv"
)

MARINE = (
    ROOT
    / "02_DATA/raw/weather/open_meteo/"
    "open_meteo_marine_current.csv"
)

PORTWATCH = (
    ROOT
    / "02_DATA/raw/port/"
    "portwatch_daily_ports.csv"
)

PORTWATCH_DISRUPTIONS = (
    ROOT
    / "02_DATA/raw/port/"
    "portwatch_disruptions.csv"
)


def load_freight():

    df = pd.read_csv(FREIGHT)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    value_columns = [
        "kdci",
        "cape",
        "panamax",
        "supramax",
        "handy",
    ]

    for column in value_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df[
        ["date"] + value_columns
    ].sort_values("date")


def build_commodity_features():

    df = pd.read_csv(COMMODITY)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    pivot = (
        df.pivot_table(
            index="date",
            columns=[
                "commodity",
                "market",
            ],
            values="price",
            aggfunc="last",
        )
    )

    pivot.columns = [
        f"commodity_{commodity}_{market}"
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for commodity, market in pivot.columns
    ]

    return (
        pivot
        .reset_index()
        .sort_values("date")
    )


def build_economic_features():

    df = pd.read_csv(ECONOMIC)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    pivot = (
        df.pivot_table(
            index=[
                "year",
                "country_code",
            ],
            columns="indicator",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )

    indicator_columns = [
        column
        for column in pivot.columns
        if column not in [
            "year",
            "country_code",
        ]
    ]

    rename = {}

    for column in indicator_columns:
        rename[column] = (
            f"econ_{column}"
            .lower()
        )

    pivot = pivot.rename(
        columns=rename
    )

    # Keep the country code in the column
    # so we never accidentally mix countries.
    pivot = pivot.rename(
        columns={
            column: (
                f"{column}_"
                f"{pivot.loc[pivot.index[0], 'country_code']}"
            )
            if False
            else column
            for column in []
        }
    )

    frames = []

    for country_code, group in pivot.groupby(
        "country_code"
    ):

        group = group.drop(
            columns="country_code"
        ).copy()

        group = group.rename(
            columns={
                column: (
                    f"{column}_"
                    f"{country_code.lower()}"
                )
                for column in group.columns
                if column != "year"
            }
        )

        frames.append(group)

    result = frames[0]

    for frame in frames[1:]:
        result = result.merge(
            frame,
            on="year",
            how="outer",
        )

    result["date"] = pd.to_datetime(
        result["year"].astype("Int64").astype(str)
        + "-01-01",
        errors="coerce",
    )

    return result.drop(
        columns="year"
    )


def build_port_features():

    df = pd.read_csv(
        PORTWATCH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # Only retain dry-bulk-relevant activity.
    keep = [
        "date",
        "portname",
        "portcalls_dry_bulk",
        "import_dry_bulk",
        "export_dry_bulk",
    ]

    df = df[keep].copy()

    mapping = {
        "Dhamra Port": "dhamra",
        "Gopalpur Port": "gopalpur",
        "Haldia": "haldia",
        "Paradip": "paradip",
        "Visakhapatnam": "visakhapatnam",
    }

    df["port_id"] = (
        df["portname"]
        .map(mapping)
    )

    df = df[
        df["port_id"].notna()
    ]

    for column in [
        "portcalls_dry_bulk",
        "import_dry_bulk",
        "export_dry_bulk",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    pivot = df.pivot_table(
        index="date",
        columns="port_id",
        values=[
            "portcalls_dry_bulk",
            "import_dry_bulk",
            "export_dry_bulk",
        ],
        aggfunc="last",
    )

    pivot.columns = [
        f"port_{port}_{metric}"
        for metric, port in pivot.columns
    ]

    return (
        pivot
        .reset_index()
        .sort_values("date")
    )


def build_weather_snapshot():

    weather = pd.read_csv(
        WEATHER
    )

    marine = pd.read_csv(
        MARINE
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
        utc=True,
    )

    marine["timestamp"] = pd.to_datetime(
        marine["timestamp"],
        errors="coerce",
        utc=True,
    )

    # Current/live weather only.
    # Keep the timestamp explicit.
    weather = weather.rename(
        columns={
            column: (
                f"weather_{column}"
                if column not in [
                    "timestamp",
                    "location_id",
                ]
                else column
            )
            for column in weather.columns
        }
    )

    marine = marine.rename(
        columns={
            column: (
                f"marine_{column}"
                if column not in [
                    "timestamp",
                    "location_id",
                ]
                else column
            )
            for column in marine.columns
        }
    )

    return weather, marine


def main():

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Building NEREUS feature foundation...")

    freight = load_freight()

    commodity = build_commodity_features()

    economic = build_economic_features()

    ports = build_port_features()

    weather, marine = build_weather_snapshot()

    # Freight is the primary historical daily timeline.
    features = freight.copy()

    # Monthly commodity values are joined by month.
    features["month"] = (
        features["date"]
        .dt.to_period("M")
    )

    commodity["month"] = (
        commodity["date"]
        .dt.to_period("M")
    )

    features = features.merge(
        commodity.drop(
            columns="date"
        ),
        on="month",
        how="left",
    )

    features = features.drop(
        columns="month"
    )

    # Annual economic values.
    features["year"] = (
        features["date"].dt.year
    )

    economic["year"] = (
        economic["date"].dt.year
    )

    features = features.merge(
        economic.drop(
            columns="date"
        ),
        on="year",
        how="left",
    )

    features = features.drop(
        columns="year"
    )

    # Daily PortWatch signals.
    features = features.merge(
        ports,
        on="date",
        how="left",
    )

    # Do NOT merge live weather into the historical
    # freight table. The current weather snapshot
    # is a separate operational state.
    #
    # This prevents accidental temporal leakage.

    output = (
        PROCESSED
        / "feature_foundation_daily.parquet"
    )

    features.to_parquet(
        output,
        index=False,
    )

    weather_output = (
        PROCESSED
        / "weather_live_state.parquet"
    )

    marine_output = (
        PROCESSED
        / "marine_live_state.parquet"
    )

    weather.to_parquet(
        weather_output,
        index=False,
    )

    marine.to_parquet(
        marine_output,
        index=False,
    )

    print()
    print("FEATURE FOUNDATION COMPLETE")
    print(
        f"Historical rows: {len(features):,}"
    )
    print(
        f"Historical columns: {len(features.columns)}"
    )
    print(
        f"Date range: "
        f"{features['date'].min().date()} → "
        f"{features['date'].max().date()}"
    )

    print()
    print("HISTORICAL FEATURE GROUPS:")

    print(
        "Freight:",
        len(
            [
                c for c in features.columns
                if c in [
                    "kdci",
                    "cape",
                    "panamax",
                    "supramax",
                    "handy",
                ]
            ]
        )
    )

    print(
        "Commodity:",
        len(
            [
                c for c in features.columns
                if c.startswith("commodity_")
            ]
        )
    )

    print(
        "Economic:",
        len(
            [
                c for c in features.columns
                if c.startswith("econ_")
            ]
        )
    )

    print(
        "Port:",
        len(
            [
                c for c in features.columns
                if c.startswith("port_")
            ]
        )
    )

    print()
    print("LIVE WEATHER:")
    print(
        f"Rows: {len(weather):,}"
    )

    print("LIVE MARINE:")
    print(
        f"Rows: {len(marine):,}"
    )

    print()
    print("OUTPUT:")
    print(output)
    print(weather_output)
    print(marine_output)


if __name__ == "__main__":
    main()
