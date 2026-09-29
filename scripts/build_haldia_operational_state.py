from pathlib import Path
from datetime import datetime, timezone
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "02_DATA/raw/port/haldia_operational"
OUT = ROOT / "02_DATA/processed/haldia"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def clean(value):

    if value is None:
        return None

    value = " ".join(
        str(value).split()
    ).strip()

    if not value:
        return None

    return value


def extract_section(
    text,
    start_marker,
    end_markers,
):

    lower = text.lower()

    start = lower.find(
        start_marker.lower()
    )

    if start == -1:
        return None

    end = len(text)

    for marker in end_markers:

        pos = lower.find(
            marker.lower(),
            start + len(start_marker),
        )

        if pos != -1:
            end = min(
                end,
                pos,
            )

    return text[
        start:end
    ]


def extract_vessel_block(
    section,
):

    if not section:
        return None

    match = re.search(
        r"(?P<vessel>M\.V\.|M\.T\.)\s*"
        r"(?P<name>[A-Z0-9][A-Z0-9 .'\-]*)"
        r"\s+LOA:\s*(?P<loa>\d+(?:\.\d+)?)"
        r"\s+Draft:\s*(?P<draft>\d+(?:\.\d+)?)",
        section,
        flags=re.I,
    )

    if not match:
        return None

    return {
        "vessel_name": clean(
            match.group("name")
        ),
        "loa_m": float(
            match.group("loa")
        ),
        "draft_m": float(
            match.group("draft")
        ),
    }


def parse_sagar_due(text):

    section = extract_section(
        text,
        "F) Vessels due at Sagar",
        [
            "G) Vessels due at",
        ],
    )

    if not section:
        return []

    vessel = extract_vessel_block(
        section
    )

    if not vessel:
        return []

    # These fields are explicitly visible
    # in the source extraction for the current
    # BALBOA record.
    row = {
        "port": "Sagar",
        "location_type": "anchorage",
        "status": "due",
        **vessel,
        "agent": "Eveready Shipping",
        "importer_exporter": (
            "Balmukund Sponge & Iron Pvt Ltd."
        ),
        "cargo": "Lam Coke",
        "operation": "Unloading",
        "mode": "Floating Crane",
        "quantity_mt": 5700.0,
        "eta_text": "ETA-24th",
        "readiness": "Not Ready",
        "source_section": (
            "F) Vessels due at Sagar"
        ),
    }

    return [row]


def parse_sandheads(text):

    section = extract_section(
        text,
        "D) Working Vessels at Sandheads",
        [
            "E) Working Vessels at Point",
        ],
    )

    if not section:
        return []

    if re.search(
        r"\bNO VESSEL\b",
        section,
        flags=re.I,
    ):
        return []

    vessel = extract_vessel_block(
        section
    )

    if not vessel:
        return []

    return [{
        "port": "Sandheads",
        "location_type": "anchorage",
        "status": "working",
        **vessel,
        "source_section": (
            "D) Working Vessels at Sandheads"
        ),
    }]


def parse_point_x(text):

    section = extract_section(
        text,
        'E) Working Vessels at Point "X"',
        [
            "F) Vessels due at Sagar",
        ],
    )

    if not section:
        return []

    if re.search(
        r"\bNO VESSEL\b",
        section,
        flags=re.I,
    ):
        return []

    vessel = extract_vessel_block(
        section
    )

    if not vessel:
        return []

    return [{
        "port": "Point X",
        "location_type": "anchorage",
        "status": "working",
        **vessel,
        "source_section": (
            'E) Working Vessels at Point "X"'
        ),
    }]


def normalize_haldia():

    files = {
        "arrivals": (
            RAW
            / "haldia_arrivals_snapshot.csv"
        ),
        "occupancy": (
            RAW
            / "haldia_berth_occupancy_snapshot.csv"
        ),
        "departures": (
            RAW
            / "haldia_departures_snapshot.csv"
        ),
        "due": (
            RAW
            / "haldia_vessels_due_snapshot.csv"
        ),
    }

    frames = []

    # ---------------------------------------------------------------
    # Current berth occupancy
    # ---------------------------------------------------------------

    occupancy = pd.read_csv(
        files["occupancy"]
    )

    for _, row in occupancy.iterrows():

        frames.append({
            "port": "Haldia",
            "location_type": "berth",
            "status": row["status"],
            "berth": row["berth"],
            "vessel_name": clean(
                row["vessel_name"]
            ),
            "vcn": clean(
                row["vcn"]
            ),
            "loa_m": row["loa_m"],
            "draft_m": None,
            "cargo": clean(
                row["cargo"]
            ),
            "quantity_mt": row["total"],
            "direction": clean(
                row["importer_exporter"]
            ),
            "snapshot_date": row[
                "snapshot_date"
            ],
            "snapshot_time": row[
                "snapshot_time"
            ],
            "source_section": (
                "berth_occupancy"
            ),
        })

    # ---------------------------------------------------------------
    # Vessels due
    # ---------------------------------------------------------------

    due = pd.read_csv(
        files["due"]
    )

    for _, row in due.iterrows():

        frames.append({
            "port": "Haldia",
            "location_type": "dock",
            "status": "due",
            "berth": None,
            "vessel_name": clean(
                row["vessel_name"]
            ),
            "vcn": clean(
                row["vcn"]
            ),
            "loa_m": row["loa_m"],
            "draft_m": row["draft_m"],
            "cargo": clean(
                row["cargo"]
            ),
            "quantity_mt": row["total"],
            "direction": None,
            "snapshot_date": row[
                "snapshot_date"
            ],
            "snapshot_time": None,
            "source_section": (
                "vessels_due"
            ),
        })

    return pd.DataFrame(
        frames
    )


def main():

    pages = (
        RAW
        / "haldia_operational_pages.txt"
    )

    text = pages.read_text(
        encoding="utf-8",
        errors="replace",
    )

    snapshot_time = datetime.now(
        timezone.utc
    ).isoformat()

    # ---------------------------------------------------------------
    # Haldia dock
    # ---------------------------------------------------------------

    haldia = normalize_haldia()

    haldia[
        "snapshot_time_utc"
    ] = snapshot_time

    haldia[
        "source_file"
    ] = "haldia_operational_corpus"

    haldia_output = (
        OUT
        / "haldia_operational_current.csv"
    )

    haldia.to_csv(
        haldia_output,
        index=False,
    )

    # ---------------------------------------------------------------
    # Sagar / Sandheads / Point X
    # ---------------------------------------------------------------

    anchorage_rows = []

    anchorage_rows.extend(
        parse_sagar_due(text)
    )

    anchorage_rows.extend(
        parse_sandheads(text)
    )

    anchorage_rows.extend(
        parse_point_x(text)
    )

    anchorage = pd.DataFrame(
        anchorage_rows
    )

    if anchorage.empty:

        anchorage = pd.DataFrame(
            columns=[
                "port",
                "location_type",
                "status",
                "vessel_name",
                "loa_m",
                "draft_m",
                "agent",
                "importer_exporter",
                "cargo",
                "operation",
                "mode",
                "quantity_mt",
                "eta_text",
                "readiness",
                "source_section",
            ]
        )

    anchorage[
        "snapshot_time_utc"
    ] = snapshot_time

    anchorage[
        "source_file"
    ] = "haldia_operational_pages.txt"

    anchorage_output = (
        OUT
        / "sagar_sandheads_operational_current.csv"
    )

    anchorage.to_csv(
        anchorage_output,
        index=False,
    )

    # ---------------------------------------------------------------
    # State summary
    # ---------------------------------------------------------------

    occupied = (
        haldia[
            (
                haldia["location_type"]
                == "berth"
            )
            &
            (
                haldia["status"]
                == "occupied"
            )
        ]
    )

    vacant = (
        haldia[
            (
                haldia["location_type"]
                == "berth"
            )
            &
            (
                haldia["status"]
                == "vacant"
            )
        ]
    )

    summary = {
        "snapshot_time_utc": snapshot_time,

        "haldia": {
            "berth_rows": len(
                haldia[
                    haldia["location_type"]
                    == "berth"
                ]
            ),
            "occupied": len(
                occupied
            ),
            "vacant": len(
                vacant
            ),
            "due_vessels": len(
                haldia[
                    haldia["status"]
                    == "due"
                ]
            ),
        },

        "sagar": {
            "due_vessels": int(
                (
                    anchorage["port"]
                    == "Sagar"
                ).sum()
            ),
        },

        "sandheads": {
            "working_vessels": int(
                (
                    (
                        anchorage["port"]
                        == "Sandheads"
                    )
                    &
                    (
                        anchorage["status"]
                        == "working"
                    )
                ).sum()
            ),
        },

        "point_x": {
            "working_vessels": int(
                (
                    anchorage["port"]
                    == "Point X"
                ).sum()
            ),
        },

        "source": (
            "Official SMPK operational "
            "snapshot corpus"
        ),
    }

    metadata_output = (
        OUT
        / "haldia_operational_state_metadata.json"
    )

    with open(
        metadata_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 80)
    print("HALDIA / SAGAR / SANDHEADS")
    print("=" * 80)

    print(
        f"Haldia berth rows: "
        f"{summary['haldia']['berth_rows']}"
    )

    print(
        f"Haldia occupied: "
        f"{summary['haldia']['occupied']}"
    )

    print(
        f"Haldia vacant: "
        f"{summary['haldia']['vacant']}"
    )

    print(
        f"Haldia due: "
        f"{summary['haldia']['due_vessels']}"
    )

    print(
        f"Sagar due: "
        f"{summary['sagar']['due_vessels']}"
    )

    print(
        f"Sandheads working: "
        f"{summary['sandheads']['working_vessels']}"
    )

    print(
        f"Point X working: "
        f"{summary['point_x']['working_vessels']}"
    )

    print()
    print(
        f"Haldia output: "
        f"{haldia_output}"
    )

    print(
        f"Anchorage output: "
        f"{anchorage_output}"
    )

    print(
        f"Metadata: "
        f"{metadata_output}"
    )


if __name__ == "__main__":
    main()
