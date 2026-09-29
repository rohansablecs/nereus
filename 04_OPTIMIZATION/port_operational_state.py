from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PARADIP_HISTORY = (
    ROOT
    / "02_DATA/processed/paradip/"
    / "paradip_operational_history_daily.csv"
)


def get_paradip_operational_state(
    as_of: pd.Timestamp | None = None,
) -> dict:

    df = pd.read_csv(
        PARADIP_HISTORY,
        parse_dates=["date"],
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        return {
            "status": "unavailable",
            "reason": "No Paradip operational snapshots.",
        }

    if as_of is None:
        as_of = pd.Timestamp.now(
            tz="Asia/Kolkata"
        ).tz_localize(None)

    as_of_date = (
        pd.Timestamp(as_of)
        .normalize()
    )

    available = df[
        df["date"] <= as_of_date
    ]

    if available.empty:
        return {
            "status": "unavailable",
            "reason": (
                "No operational snapshot exists "
                "on or before requested date."
            ),
        }

    row = available.iloc[-1]

    snapshot_date = pd.Timestamp(
        row["date"]
    )

    age_days = (
        as_of_date - snapshot_date
    ).total_seconds() / 86400.0

    return {
        "status": "available",
        "snapshot_date": (
            snapshot_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "snapshot_age_days": age_days,
        "dry_bulk_vessels_waiting": (
            float(
                row[
                    "dry_bulk_vessels_waiting"
                ]
            )
        ),
        "source_file": str(
            row["source_file"]
        ),
    }


if __name__ == "__main__":

    state = (
        get_paradip_operational_state()
    )

    print(state)
