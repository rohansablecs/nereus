from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


RAW_DIR = Path(
    "02_DATA/raw/port/paradip/daily_traffic"
)

PROCESSED_DIR = Path(
    "02_DATA/processed/paradip"
)

WAITING_FILE = (
    PROCESSED_DIR / "waiting_vessels.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "observed_waiting_delays.csv"
)

METADATA_FILE = (
    PROCESSED_DIR / "waiting_delay_metadata.json"
)


VESSEL_RE = re.compile(
    r"(?P<name>(?:MV\.|MT\.)\s+.*?)"
    r"\s+(?P<purpose>BERTHS|SAILS)\s+"
    r"(?P<loa>\d{3}\.\d{2})"
    r"\s+"
    r"(?P<rest>.*)$",
    re.I,
)


DATE_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{2})\s+"
    r"(?P<before_vessel>.*?)"
    r"(?P<vessel>(?:MV\.|MT\.).*)$"
)


def parse_datetime(
    date_string: str,
    time_string: str,
) -> datetime:

    return datetime.strptime(
        f"{date_string} {time_string}",
        "%Y-%m-%d %H:%M",
    )


def extract_page6_lines(page):
    text = page.extract_text() or ""

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def parse_movement_line(line):
    """
    Parse a Paradip page-6 movement line.

    Example:

    07-09-26 CQ-3MV. AMARYLLIS SAILS 228.99 I.ORE PELLET
    07:48 75,900 32.26 9.00

    Or:

    07-09-26 CB-1/CB-2MV. APJ ANGAD 2 BERTHS 224.94 TH.COAL
    15:00 ... 75,000 32.26 14.50
    """

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    date_match = re.match(
        r"^(?P<date>\d{2}-\d{2}-\d{2})\s*",
        line,
    )

    if not date_match:
        return None

    date_raw = date_match.group("date")

    movement_date = datetime.strptime(
        date_raw,
        "%d-%m-%y",
    ).date().isoformat()

    remainder = line[
        date_match.end():
    ]

    # ---------------------------------------------------------
    # Find vessel marker.
    #
    # The berth and vessel are frequently concatenated:
    #
    # CB-1/CB-2MV.
    # CQ-3MV.
    # ---------------------------------------------------------

    vessel_marker = re.search(
        r"(MV\.|MT\.)",
        remainder,
        re.I,
    )

    if not vessel_marker:
        return None

    berth = remainder[
        :vessel_marker.start()
    ].strip()

    vessel_text = remainder[
        vessel_marker.start():
    ].strip()

    # ---------------------------------------------------------
    # Vessel name + purpose + LOA
    # ---------------------------------------------------------

    m = VESSEL_RE.match(
        vessel_text
    )

    if not m:
        return None

    vessel_name = re.sub(
        r"\s+",
        " ",
        m.group("name").strip(),
    )

    purpose = m.group(
        "purpose"
    ).upper()

    loa = float(
        m.group("loa")
    )

    rest = m.group(
        "rest"
    ).strip()

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    time_match = re.search(
        r"\b(\d{2}:\d{2})\b",
        rest,
    )

    if not time_match:
        return None

    movement_time = (
        time_match.group(1)
    )

    movement_datetime = (
        f"{movement_date} "
        f"{movement_time}"
    )

    # ---------------------------------------------------------
    # Cargo quantity
    # ---------------------------------------------------------

    quantity = None

    quantity_matches = re.findall(
        r"\b\d{1,3}(?:,\d{3})+\b",
        rest,
    )

    if quantity_matches:
        quantity = int(
            quantity_matches[0].replace(
                ",",
                "",
            )
        )

    return {
        "movement_date": movement_date,
        "movement_time": movement_time,
        "movement_datetime": movement_datetime,
        "berth": berth,
        "vessel_name": vessel_name,
        "purpose": purpose,
        "loa_m": loa,
        "cargo_qty_mt": quantity,
        "raw_line": line,
    }


def parse_page6(page):
    lines = extract_page6_lines(page)

    records = []

    for line in lines:

        if not re.match(
            r"^\d{2}-\d{2}-\d{2}",
            line,
        ):
            continue

        record = parse_movement_line(
            line
        )

        if record is not None:
            records.append(record)

    return records


def normalise_name(name):
    name = name.upper().strip()

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name


def main():

    if not WAITING_FILE.exists():
        raise RuntimeError(
            f"Missing waiting dataset: "
            f"{WAITING_FILE}"
        )

    waiting = pd.read_csv(
        WAITING_FILE
    )

    if waiting.empty:
        raise RuntimeError(
            "Waiting dataset is empty."
        )

    print(
        f"Waiting records: {len(waiting)}"
    )

    all_movements = []

    for pdf in sorted(
        RAW_DIR.glob("*.pdf")
    ):

        with open(
            pdf,
            "rb",
        ) as f:

            if f.read(5) != b"%PDF-":
                continue

        reader = PdfReader(
            str(pdf)
        )

        if len(reader.pages) < 6:
            continue

        movements = parse_page6(
            reader.pages[5]
        )

        for movement in movements:
            movement[
                "source_pdf"
            ] = pdf.name

        all_movements.extend(
            movements
        )

        print(
            f"{pdf.name}: "
            f"movements={len(movements)}"
        )

    if not all_movements:
        raise RuntimeError(
            "No page-6 movements extracted."
        )

    movements = pd.DataFrame(
        all_movements
    )

    print(
        f"Total movements: "
        f"{len(movements)}"
    )

    # ---------------------------------------------------------
    # Only BERTHS events represent actual entry into berth.
    # ---------------------------------------------------------

    berthings = movements[
        movements["purpose"] == "BERTHS"
    ].copy()

    if berthings.empty:
        raise RuntimeError(
            "No BERTHS movements found."
        )

    # ---------------------------------------------------------
    # Normalised vessel names for matching.
    # ---------------------------------------------------------

    waiting = waiting.copy()

    waiting[
        "_vessel_key"
    ] = waiting[
        "vessel_name"
    ].map(normalise_name)

    berthings[
        "_vessel_key"
    ] = berthings[
        "vessel_name"
    ].map(normalise_name)

    # ---------------------------------------------------------
    # Build arrival datetime.
    # ---------------------------------------------------------

    waiting[
        "arrival_datetime"
    ] = pd.to_datetime(
        waiting[
            "arrival_date"
        ].astype(str)
        + " "
        + waiting[
            "arrival_time"
        ].astype(str),
        errors="coerce",
    )

    berthings[
        "berth_datetime"
    ] = pd.to_datetime(
        berthings[
            "movement_datetime"
        ],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Match waiting vessel -> actual BERTHS movement.
    #
    # If a vessel has multiple BERTHS movements, choose the first
    # one occurring at or after its reported arrival.
    # ---------------------------------------------------------

    results = []

    for _, waiting_row in waiting.iterrows():

        key = waiting_row[
            "_vessel_key"
        ]

        arrival = waiting_row[
            "arrival_datetime"
        ]

        candidates = berthings[
            berthings[
                "_vessel_key"
            ] == key
        ].copy()

        candidates = candidates[
            candidates[
                "berth_datetime"
            ] >= arrival
        ]

        if candidates.empty:
            continue

        candidates = candidates.sort_values(
            "berth_datetime"
        )

        berth = candidates.iloc[0]

        waiting_hours = (
            berth["berth_datetime"]
            - arrival
        ).total_seconds() / 3600.0

        # Sanity check.
        if waiting_hours < 0:
            continue

        result = {
            "traffic_date": waiting_row[
                "traffic_date"
            ],
            "held_on_date": waiting_row[
                "held_on_date"
            ],
            "port": "Paradip",
            "vessel_name": waiting_row[
                "vessel_name"
            ],
            "arrival_datetime": arrival,
            "berth_datetime": berth[
                "berth_datetime"
            ],
            "waiting_hours": waiting_hours,
            "waiting_days": (
                waiting_hours / 24.0
            ),
            "waiting_priority_group": waiting_row[
                "priority_group"
            ],
            "readiness": waiting_row[
                "readiness"
            ],
            "priority": waiting_row[
                "priority"
            ],
            "cargo_type": waiting_row[
                "cargo_type"
            ],
            "cargo_qty_mt": waiting_row[
                "cargo_qty_mt"
            ],
            "vessel_loa_m": waiting_row[
                "loa_m"
            ],
            "vessel_beam_m": waiting_row[
                "beam_m"
            ],
            "vessel_draft_m": waiting_row[
                "draft_m"
            ],
            "berth": berth[
                "berth"
            ],
            "movement_cargo_qty_mt": berth[
                "cargo_qty_mt"
            ],
            "source_pdf_waiting": waiting_row[
                "source_pdf"
            ],
            "source_pdf_movement": berth[
                "source_pdf"
            ],
            "movement_raw_line": berth[
                "raw_line"
            ],
        }

        results.append(result)

    if not results:
        raise RuntimeError(
            "No waiting vessels could be matched "
            "to subsequent BERTHS movements."
        )

    observed = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------
    # Remove temporary columns / sort.
    # ---------------------------------------------------------

    observed = observed.sort_values(
        [
            "traffic_date",
            "arrival_datetime",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------

    if (
        observed["waiting_hours"]
        < 0
    ).any():
        raise RuntimeError(
            "Negative waiting time detected."
        )

    if (
        observed["waiting_hours"]
        > 24 * 30
    ).any():
        raise RuntimeError(
            "Waiting time exceeds 30 days; "
            "inspect source matching."
        )

    # ---------------------------------------------------------
    # Write only after validation.
    # ---------------------------------------------------------

    observed.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metadata = {
        "source": (
            "Paradip Port Authority "
            "Daily Traffic Update"
        ),
        "method": (
            "Matched page-4 waiting vessels "
            "to page-6 BERTHS movements by "
            "vessel name and selected the first "
            "berthing at or after reported arrival."
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "waiting_records": len(waiting),
        "movement_records": len(movements),
        "berthing_records": len(berthings),
        "matched_observations": len(observed),
        "unmatched_waiting_records": (
            len(waiting) - len(observed)
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Report.
    # ---------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "PARADIP OBSERVED WAITING DELAY"
    )
    print(
        "========================================"
    )

    print(
        f"Waiting records:       {len(waiting)}"
    )

    print(
        f"BERTHS movements:      {len(berthings)}"
    )

    print(
        f"Matched observations:  {len(observed)}"
    )

    print(
        f"Unmatched waiting:     "
        f"{len(waiting) - len(observed)}"
    )

    print()

    print(
        observed[
            [
                "vessel_name",
                "arrival_datetime",
                "berth_datetime",
                "waiting_hours",
                "cargo_type",
                "cargo_qty_mt",
                "berth",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "WAITING-TIME SUMMARY"
    )

    print(
        f"Median: "
        f"{observed['waiting_hours'].median():.2f} h"
    )

    print(
        f"Mean:   "
        f"{observed['waiting_hours'].mean():.2f} h"
    )

    print(
        f"Min:    "
        f"{observed['waiting_hours'].min():.2f} h"
    )

    print(
        f"Max:    "
        f"{observed['waiting_hours'].max():.2f} h"
    )

    print()
    print(
        f"CSV: {OUTPUT_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()
