from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FREIGHT_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/freight/kobc/kobc_drybulk_daily.csv"
)

PORTWATCH_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/portwatch_daily_ports.csv"
)

DISRUPTION_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/portwatch_disruptions.csv"
)

VIZAG_OCCUPANCY_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/vizag_berth_occupancy_current.csv"
)

VIZAG_WAITING_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/vizag_waiting_expected_current.csv"
)

HALDIA_OCCUPANCY_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/haldia_berth_position_current.csv"
)

GANGAVARAM_SCHEDULE_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/gangavaram_vessel_schedule_current.csv"
)


PORTWATCH_PORTS = {
    "dhamra": "Dhamra Port",
    "gopalpur": "Gopalpur",
    "haldia": "Haldia",
    "paradip": "Paradip",
    "visakhapatnam": "Visakhapatnam",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def _latest_freight() -> dict[str, Any]:
    df = _load_csv(FREIGHT_FILE)

    if df.empty:
        return {
            "available": False,
            "latest_date": None,
            "values": {},
        }

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    latest = df.iloc[-1]

    values = {}

    for column in ["kdci", "cape", "panamax", "supramax", "handy"]:
        value = latest.get(column)

        if pd.notna(value):
            values[column] = float(value)

    return {
        "available": True,
        "latest_date": latest["date"].date().isoformat(),
        "values": values,
        "unit": "USD/day",
        "source": "KOBC",
    }


def _latest_portwatch() -> dict[str, Any]:
    df = _load_csv(PORTWATCH_FILE)

    if df.empty:
        return {
            "available": False,
            "latest_date": None,
            "ports": {},
        }

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    latest_date = df["date"].max()

    latest = df[df["date"] == latest_date]

    result = {}

    for key, port_name in PORTWATCH_PORTS.items():
        row = latest[
            latest["portname"].astype(str).str.strip().str.lower()
            == port_name.lower()
        ]

        if row.empty:
            result[key] = {
                "available": False,
                "source": "PortWatch",
            }
            continue

        r = row.iloc[0]

        fields = [
            "portcalls_dry_bulk",
            "portcalls",
            "import_dry_bulk",
            "export_dry_bulk",
        ]

        values = {}

        for field in fields:
            if field in r.index and pd.notna(r[field]):
                values[field] = float(r[field])

        result[key] = {
            "available": True,
            "values": values,
            "source": "PortWatch",
        }

    return {
        "available": True,
        "latest_date": latest_date.date().isoformat(),
        "ports": result,
    }


def _latest_disruptions() -> dict[str, Any]:
    df = _load_csv(DISRUPTION_FILE)

    if df.empty:
        return {
            "available": False,
            "events": 0,
        }

    date_columns = [
        c for c in ["fromdate", "todate", "editdate"]
        if c in df.columns
    ]

    parsed_dates = []

    for column in date_columns:
        parsed = pd.to_numeric(df[column], errors="coerce")

        parsed_dates.append(
            pd.to_datetime(
                parsed,
                unit="ms",
                errors="coerce",
                utc=True,
            )
        )

    if not parsed_dates:
        return {
            "available": True,
            "events": int(len(df)),
            "latest_event_date": None,
        }

    combined = pd.concat(parsed_dates, axis=1)

    latest_event_date = combined.max(axis=1).max()

    return {
        "available": True,
        "events": int(len(df)),
        "latest_event_date": (
            latest_event_date.isoformat()
            if pd.notna(latest_event_date)
            else None
        ),
        "source": "PortWatch disruptions database",
    }


def _operational_snapshot(
    path: Path,
    count_name: str,
) -> dict[str, Any]:

    df = _load_csv(path)

    if df.empty:
        return {
            "available": False,
            count_name: None,
        }

    return {
        "available": True,
        count_name: int(len(df)),
    }


def build_market_state() -> dict[str, Any]:

    freight = _latest_freight()
    portwatch = _latest_portwatch()
    disruptions = _latest_disruptions()

    vizag = {
        "berth_occupancy": _operational_snapshot(
            VIZAG_OCCUPANCY_FILE,
            "rows",
        ),
        "waiting_expected": _operational_snapshot(
            VIZAG_WAITING_FILE,
            "vessels",
        ),
    }

    haldia = {
        "berth_position": _operational_snapshot(
            HALDIA_OCCUPANCY_FILE,
            "rows",
        ),
    }

    gangavaram = {
        "vessel_schedule": _operational_snapshot(
            GANGAVARAM_SCHEDULE_FILE,
            "vessels",
        ),
    }

    return {
        "state_version": "v1",
        "freight": freight,
        "port_activity": portwatch,
        "disruptions": disruptions,
        "operational": {
            "visakhapatnam": vizag,
            "haldia": haldia,
            "gangavaram": gangavaram,
        },
    }


def save_market_state(output_path: Path) -> None:

    state = build_market_state()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":

    output = (
        PROJECT_ROOT
        / "02_DATA/processed/current_market_state.json"
    )

    save_market_state(output)

    print(f"Market state written to: {output}")

    state = build_market_state()

    print(
        json.dumps(
            state,
            indent=2,
        )
    )
