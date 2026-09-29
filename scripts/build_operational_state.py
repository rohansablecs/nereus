from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIZAG_OCCUPANCY = (
    PROJECT_ROOT
    / "02_DATA/raw/port/vizag_berth_occupancy_current.csv"
)

VIZAG_WAITING = (
    PROJECT_ROOT
    / "02_DATA/raw/port/vizag_waiting_expected_current.csv"
)

GANGAVARAM_SCHEDULE = (
    PROJECT_ROOT
    / "02_DATA/raw/port/gangavaram_vessel_schedule_current.csv"
)

GANGAVARAM_EXPECTED = (
    PROJECT_ROOT
    / "02_DATA/raw/port/gangavaram_expected_vessels_current.csv"
)

GANGAVARAM_SAILED = (
    PROJECT_ROOT
    / "02_DATA/raw/port/gangavaram_sailed_24h.csv"
)

HALDIA_BERTH = (
    PROJECT_ROOT
    / "02_DATA/raw/port/haldia_berth_position_current.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "02_DATA/processed/current_operational_state.json"
)


def clean_text(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def read_csv(path: Path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def valid_haldia_data(df):
    """
    The current Haldia file contains a header row reproduced
    as data. Do not treat that as a real vessel/berth record.
    """

    if df is None or df.empty:
        return False

    if "vessel_name" not in df.columns:
        return False

    real_vessels = (
        df["vessel_name"]
        .astype(str)
        .str.strip()
        .ne("")
        & ~df["vessel_name"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "vessel name",
                "nan",
                "none",
            ]
        )
    )

    return bool(real_vessels.any())


def build_vizag_state():
    occupancy = read_csv(
        VIZAG_OCCUPANCY
    )

    waiting = read_csv(
        VIZAG_WAITING
    )

    state = {
        "available": False,
        "source": "Visakhapatnam Port",
    }

    if occupancy is not None:

        status = (
            occupancy["status"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        occupied = int(
            (status == "OCCUPIED").sum()
        )

        vacant = int(
            (status == "VACANT").sum()
        )

        unavailable = int(
            (status == "UNAVAILABLE").sum()
        )

        known_status = (
            occupied
            + vacant
            + unavailable
        )

        state.update(
            {
                "available": True,
                "programme_date": clean_text(
                    occupancy[
                        "programme_date"
                    ].iloc[0]
                ),
                "snapshot_time_utc": clean_text(
                    occupancy[
                        "snapshot_time_utc"
                    ].iloc[0]
                ),
                "berths": {
                    "total_rows": int(
                        len(occupancy)
                    ),
                    "occupied": occupied,
                    "vacant": vacant,
                    "unavailable": unavailable,
                    "unknown_status": int(
                        len(occupancy)
                        - known_status
                    ),
                },
            }
        )

        if known_status > 0:
            state["berths"][
                "occupancy_ratio"
            ] = round(
                occupied / known_status,
                4,
            )
        else:
            state["berths"][
                "occupancy_ratio"
            ] = None

    # ---------------------------------------------------------
    # Waiting / expected vessels
    # ---------------------------------------------------------

    if waiting is not None:

        waiting = waiting.copy()

        waiting["tonnes_mt"] = pd.to_numeric(
            waiting["tonnes_mt"],
            errors="coerce",
        )

        expected_dates = pd.to_datetime(
            waiting["expected_date"],
            errors="coerce",
        )

        state["waiting_expected"] = {
            "available": True,
            "vessels": int(
                len(waiting)
            ),
            "cargo_mt": float(
                waiting["tonnes_mt"]
                .sum(
                    min_count=1
                )
            )
            if waiting["tonnes_mt"].notna().any()
            else None,
            "priority_vessels": int(
                waiting["priority_marker"]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "earliest_expected_date": (
                expected_dates.min()
                .strftime("%Y-%m-%d")
                if expected_dates.notna().any()
                else None
            ),
            "latest_expected_date": (
                expected_dates.max()
                .strftime("%Y-%m-%d")
                if expected_dates.notna().any()
                else None
            ),
        }

    else:

        state["waiting_expected"] = {
            "available": False,
            "vessels": None,
            "cargo_mt": None,
            "priority_vessels": None,
            "earliest_expected_date": None,
            "latest_expected_date": None,
        }

    return state


def build_gangavaram_state():
    schedule = read_csv(
        GANGAVARAM_SCHEDULE
    )

    expected = read_csv(
        GANGAVARAM_EXPECTED
    )

    sailed = read_csv(
        GANGAVARAM_SAILED
    )

    state = {
        "source": "Gangavaram Port",
    }

    # ---------------------------------------------------------
    # Current schedule
    # ---------------------------------------------------------

    if schedule is not None:

        state["schedule"] = {
            "available": True,
            "vessels": int(
                len(schedule)
            ),
            "imports": int(
                (
                    schedule["imp_exp"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == "I"
                ).sum()
            ),
            "exports": int(
                (
                    schedule["imp_exp"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == "E"
                ).sum()
            ),
            "berths_in_use": int(
                schedule[
                    "berth_no"
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .nunique()
            ),
        }

    else:

        state["schedule"] = {
            "available": False,
            "vessels": None,
            "imports": None,
            "exports": None,
            "berths_in_use": None,
        }

    # ---------------------------------------------------------
    # Expected vessels
    # ---------------------------------------------------------

    if expected is not None:

        expected = expected.copy()

        eta = pd.to_datetime(
            expected["eta"],
            errors="coerce",
            dayfirst=True,
        )

        state["expected"] = {
            "available": True,
            "vessels": int(
                len(expected)
            ),
            "imports": int(
                (
                    expected["imp_exp"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == "I"
                ).sum()
            ),
            "exports": int(
                (
                    expected["imp_exp"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == "E"
                ).sum()
            ),
            "earliest_eta": (
                eta.min()
                .isoformat()
                if eta.notna().any()
                else None
            ),
        }

    else:

        state["expected"] = {
            "available": False,
            "vessels": None,
            "imports": None,
            "exports": None,
            "earliest_eta": None,
        }

    # ---------------------------------------------------------
    # Sailed in last 24h
    # ---------------------------------------------------------

    if sailed is not None:

        state["recent_sailed"] = {
            "available": True,
            "vessels": int(
                len(sailed)
            ),
        }

    else:

        state["recent_sailed"] = {
            "available": False,
            "vessels": None,
        }

    return state


def build_haldia_state():
    berth = read_csv(
        HALDIA_BERTH
    )

    if not valid_haldia_data(
        berth
    ):
        return {
            "available": False,
            "reason": (
                "Current berth-position "
                "file contains no valid "
                "vessel records."
            ),
            "source": "Haldia Dock Complex",
        }

    return {
        "available": True,
        "source": "Haldia Dock Complex",
        "vessels": int(
            len(berth)
        ),
    }


def main():

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    result = {
        "state_version": "v1",
        "generated_at_utc": generated_at,
        "ports": {
            "visakhapatnam": (
                build_vizag_state()
            ),
            "gangavaram": (
                build_gangavaram_state()
            ),
            "haldia": (
                build_haldia_state()
            ),
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "Operational state written to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()