from pathlib import Path

import numpy as np
import pandas as pd


RAW = Path("02_DATA/raw")
OUT = Path("02_DATA/processed/freight_port_features.parquet")

FREIGHT_FILE = RAW / "freight/kobc/kobc_drybulk_daily.csv"
PORTWATCH_FILE = RAW / "port/portwatch_daily_ports.csv"
DISRUPTION_FILE = RAW / "port/portwatch_disruptions.csv"


PORTS = {
    "dhamra": "port290",
    "gopalpur": "port2299",
    "haldia": "port442",
    "paradip": "port883",
    "vizag": "port1367",
}


PORT_METRICS = [
    "portcalls_dry_bulk",
    "portcalls",
    "import_dry_bulk",
    "export_dry_bulk",
]


def load_freight():
    df = pd.read_csv(FREIGHT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    columns = [
        "date",
        "kdci",
        "cape",
        "panamax",
        "supramax",
        "handy",
    ]

    return df[columns]


def load_portwatch():
    df = pd.read_csv(PORTWATCH_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df[df["portid"].isin(PORTS.values())].copy()

    # Ensure numerical fields are numeric.
    for column in PORT_METRICS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    # One row per port/date is expected.
    duplicate_count = df.duplicated(
        subset=["date", "portid"]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"PortWatch has {duplicate_count} duplicate date/port rows."
        )

    return df


def make_port_features(portwatch):
    frames = []

    for port_name, port_id in PORTS.items():

        port = portwatch[
            portwatch["portid"] == port_id
        ].copy()

        port = port[
            [
                "date",
                *PORT_METRICS,
            ]
        ].copy()

        rename = {
            metric: f"{port_name}_{metric}"
            for metric in PORT_METRICS
        }

        port = port.rename(columns=rename)

        frames.append(port)

    result = frames[0]

    for frame in frames[1:]:
        result = result.merge(
            frame,
            on="date",
            how="outer",
        )

    result = result.sort_values("date")

    return result


def make_aggregate_features(port_features):
    result = port_features[["date"]].copy()

    for metric in PORT_METRICS:
        columns = [
            f"{port}_{metric}"
            for port in PORTS
        ]

        result[f"all_ports_{metric}"] = (
            port_features[columns]
            .sum(axis=1, min_count=1)
        )

    # Share of dry-bulk activity by port.
    total = result["all_ports_portcalls_dry_bulk"]

    for port in PORTS:
        result[f"{port}_dry_bulk_share"] = (
            port_features[f"{port}_portcalls_dry_bulk"]
            / total.replace(0, np.nan)
        )

    return result


def parse_epoch_ms(series):
    """
    PortWatch timestamps are Unix epoch milliseconds.
    """
    return pd.to_datetime(
        pd.to_numeric(series, errors="coerce"),
        unit="ms",
        errors="coerce",
    )


def make_disruption_features(disruptions):
    """
    Convert PortWatch event intervals into daily indicators.

    Only disruptions explicitly affecting one of our five
    PortWatch-covered Indian ports are included.
    """

    columns = [
        "date",
        "disruption_any",
        "disruption_count",
        "disruption_red_count",
    ]

    # Determine the complete date range from freight history.
    freight = pd.read_csv(FREIGHT_FILE)
    freight["date"] = pd.to_datetime(freight["date"])

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                freight["date"].min(),
                freight["date"].max(),
                freq="D",
            )
        }
    )

    events = disruptions.copy()

    events["start"] = parse_epoch_ms(events["fromdate"])
    events["end"] = parse_epoch_ms(events["todate"])

    # If an event has no end date, treat it as active
    # through the end of the modelling period.
    model_end = calendar["date"].max()

    events["end"] = events["end"].fillna(model_end)

    # Normalize to dates.
    events["start"] = events["start"].dt.normalize()
    events["end"] = events["end"].dt.normalize()

    # Keep events that mention at least one recognized port.
    recognized = set(PORTS.values())

    def affected_recognized(value):
        if pd.isna(value):
            return False

        affected = {
            item.strip()
            for item in str(value).split(";")
            if item.strip()
        }

        return bool(affected & recognized)

    events = events[
        events["affectedports"].apply(affected_recognized)
    ].copy()

    daily = []

    for _, event in events.iterrows():

        if pd.isna(event["start"]):
            continue

        start = max(event["start"], calendar["date"].min())
        end = min(event["end"], calendar["date"].max())

        if end < start:
            continue

        dates = pd.date_range(
            start,
            end,
            freq="D",
        )

        is_red = (
            str(event.get("alertlevel", ""))
            .strip()
            .upper()
            == "RED"
        )

        for date in dates:
            daily.append(
                {
                    "date": date,
                    "disruption_count": 1,
                    "disruption_red_count": int(is_red),
                }
            )

    if not daily:
        result = calendar.copy()
        result["disruption_any"] = 0
        result["disruption_count"] = 0
        result["disruption_red_count"] = 0
        return result[columns]

    event_daily = pd.DataFrame(daily)

    event_daily = (
        event_daily
        .groupby("date", as_index=False)
        .agg(
            disruption_count=(
                "disruption_count",
                "sum",
            ),
            disruption_red_count=(
                "disruption_red_count",
                "sum",
            ),
        )
    )

    result = calendar.merge(
        event_daily,
        on="date",
        how="left",
    )

    result["disruption_count"] = (
        result["disruption_count"]
        .fillna(0)
        .astype(int)
    )

    result["disruption_red_count"] = (
        result["disruption_red_count"]
        .fillna(0)
        .astype(int)
    )

    result["disruption_any"] = (
        result["disruption_count"] > 0
    ).astype(int)

    return result[columns]


def main():

    print("=" * 80)
    print("NEREUS — BUILD FREIGHT + PORT FEATURE TABLE V1")
    print("=" * 80)

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # FREIGHT
    # ------------------------------------------------------------

    freight = load_freight()

    print("\nFreight:")
    print(
        f"  {freight.date.min().date()} → "
        f"{freight.date.max().date()}"
    )
    print(f"  rows: {len(freight):,}")

    # ------------------------------------------------------------
    # PORTWATCH
    # ------------------------------------------------------------

    portwatch = load_portwatch()

    print("\nPortWatch:")
    print(
        f"  {portwatch.date.min().date()} → "
        f"{portwatch.date.max().date()}"
    )
    print(f"  rows: {len(portwatch):,}")

    port_features = make_port_features(
        portwatch
    )

    aggregate = make_aggregate_features(
        port_features
    )

    # ------------------------------------------------------------
    # DISRUPTIONS
    # ------------------------------------------------------------

    disruptions = pd.read_csv(
        DISRUPTION_FILE
    )

    disruption_features = make_disruption_features(
        disruptions
    )

    # ------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------

    df = freight.merge(
        port_features,
        on="date",
        how="left",
    )

    df = df.merge(
        aggregate,
        on="date",
        how="left",
    )

    df = df.merge(
        disruption_features,
        on="date",
        how="left",
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # QUALITY CHECKS
    # ------------------------------------------------------------

    if df["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates found after merge."
        )

    # PortWatch has no observations after 2026-08-28.
    # Therefore these features are legitimately missing
    # for the final KOBC observations.
    port_columns = [
        c for c in df.columns
        if c not in {
            "date",
            "record_number",
            "kdci",
            "cape",
            "panamax",
            "supramax",
            "handy",
            "disruption_any",
            "disruption_count",
            "disruption_red_count",
        }
    ]

    print("\nFinal feature table:")
    print(f"  rows    : {len(df):,}")
    print(f"  columns : {len(df.columns)}")
    print(
        f"  dates   : "
        f"{df.date.min().date()} → "
        f"{df.date.max().date()}"
    )

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nMissingness:")
    missing = df.isna().sum()

    for column, count in missing[missing > 0].items():
        print(
            f"  {column}: "
            f"{count:,} "
            f"({count / len(df) * 100:.2f}%)"
        )

    print("\nLatest rows:")
    print(
        df.tail(5).to_string(index=False)
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    df.to_parquet(
        OUT,
        index=False,
    )

    print("\nSaved:")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
