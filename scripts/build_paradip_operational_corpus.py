from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "02_DATA/raw/port/paradip/daily_traffic"
PROCESSED_DIR = ROOT / "02_DATA/processed/paradip"

INVENTORY = RAW_DIR / "report_inventory.csv"
WAITING_SOURCE = PROCESSED_DIR / "waiting_vessels.csv"

MOVEMENTS_OUT = PROCESSED_DIR / "berthing_movements_corpus_v3.csv"
DELAYS_OUT = PROCESSED_DIR / "observed_waiting_delays_corpus_v3.csv"
QUARANTINE_OUT = PROCESSED_DIR / "parser_quarantine_v3.csv"
METADATA_OUT = PROCESSED_DIR / "operational_corpus_metadata_v3.json"


DATE_RE = re.compile(
    r"^(\d{2})-(\d{2})-(\d{2})\b"
)

TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})\b"
)

PURPOSE_RE = re.compile(
    r"\b(BERTHS|SAILS)\b",
    re.IGNORECASE,
)

VESSEL_MARKER_RE = re.compile(
    r"(MV\.|MT\.)",
    re.IGNORECASE,
)

QTY_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"
)

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)


# ============================================================================
# HELPERS
# ============================================================================

def clean(text):
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def parse_short_date(text, default_year=2026):
    day, month, year = text.split("-")

    year = int(year)

    if year < 100:
        year += 2000

    return pd.Timestamp(
        year=int(year),
        month=int(month),
        day=int(day),
    )


def normalize_vessel(name):
    name = str(name).upper()

    name = re.sub(
        r"\b(MV|MT)\.?\s*",
        "",
        name,
    )

    name = re.sub(
        r"[^A-Z0-9]+",
        " ",
        name,
    )

    return clean(name)


def read_pages(path):
    reader = PdfReader(str(path))

    return [
        page.extract_text() or ""
        for page in reader.pages
    ]


# ============================================================================
# REPORT METADATA
# ============================================================================

def load_inventory():
    df = pd.read_csv(INVENTORY)

    df["filename"] = (
        df["filename"]
        .astype(str)
    )

    return df


def report_metadata(
    pdf_path,
    inventory,
):
    row = inventory[
        inventory["filename"]
        == pdf_path.name
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Inventory row not found for {pdf_path.name}"
        )

    row = row.iloc[0]

    traffic_date = pd.to_datetime(
        row["traffic_date"]
    )

    held_on = pd.to_datetime(
        row["held_on"]
    )

    return {
        "filename": pdf_path.name,
        "traffic_date": traffic_date,
        "held_on": held_on,
    }


# ============================================================================
# FIND THE ACTUAL MOVEMENT PAGE
# ============================================================================

def find_movement_page(pages):
    """
    The validated source structure is:

        D. BERTHING MOVEMENTS
        DATE NAME OF THE VESSEL PURPOSE BERTH NO TIME REMARKS
        LOA
        CARGO
        BEAM DRAFT QTY
        VESSEL SIZE

    Search by the actual section title.
    """

    for page_number, text in enumerate(
        pages,
        start=1,
    ):

        if (
            "BERTHING MOVEMENTS"
            in text.upper()
        ):
            return page_number, text

    return None, None


# ============================================================================
# MOVEMENT RECORD GROUPING
# ============================================================================

def is_movement_start(line):
    """
    Exact source pattern:

        07-09-26 CQ-3MV. AMARYLLIS SAILS ...
    """

    line = clean(line)

    if not DATE_RE.match(line):
        return False

    if not VESSEL_MARKER_RE.search(line):
        return False

    if not PURPOSE_RE.search(line):
        return False

    return True


def group_records(page_text):
    """
    Converts physical PDF lines into logical records.

    Example:

        07-09-26 CB-1/CB-2MV. SINGAPORE BULKER BERTHS ...
        VACANT FIRST
        55,000 32.26 13.30

    becomes:

        07-09-26 CB-1/CB-2MV. SINGAPORE BULKER BERTHS ...
        VACANT FIRST 55,000 32.26 13.30
    """

    physical_lines = [
        clean(x)
        for x in page_text.splitlines()
        if clean(x)
    ]

    records = []

    current = None

    for line in physical_lines:

        if is_movement_start(line):

            if current is not None:
                records.append(
                    clean(current)
                )

            current = line

        elif current is not None:

            current += " " + line

    if current is not None:
        records.append(
            clean(current)
        )

    return records


# ============================================================================
# PARSE ONE MOVEMENT
# ============================================================================

def parse_movement(
    record,
    meta,
    page_number,
):

    original = record

    # ------------------------------------------------------------------------
    # DATE
    # ------------------------------------------------------------------------

    date_match = DATE_RE.match(
        record
    )

    if not date_match:
        raise ValueError(
            "date_not_found"
        )

    movement_date = parse_short_date(
        date_match.group(0),
        meta["traffic_date"].year,
    )

    remainder = clean(
        record[
            date_match.end():
        ]
    )

    # ------------------------------------------------------------------------
    # MV./MT.
    # ------------------------------------------------------------------------

    marker = VESSEL_MARKER_RE.search(
        remainder
    )

    if not marker:
        raise ValueError(
            "vessel_marker_not_found"
        )

    berth = clean(
        remainder[
            :marker.start()
        ]
    )

    if not berth:
        raise ValueError(
            "berth_not_found"
        )

    # ------------------------------------------------------------------------
    # PURPOSE
    # ------------------------------------------------------------------------

    purpose_match = PURPOSE_RE.search(
        remainder,
        marker.end(),
    )

    if not purpose_match:
        raise ValueError(
            "purpose_not_found"
        )

    vessel_name = clean(
        remainder[
            marker.end():
            purpose_match.start()
        ]
    )

    if not vessel_name:
        raise ValueError(
            "vessel_name_not_found"
        )

    purpose = purpose_match.group(
        1
    ).upper()

    after_purpose = clean(
        remainder[
            purpose_match.end():
        ]
    )

    # ------------------------------------------------------------------------
    # LOA
    # ------------------------------------------------------------------------

    loa_match = re.match(
        r"(\d{2,3}\.\d{1,2})\b",
        after_purpose,
    )

    if not loa_match:
        raise ValueError(
            "loa_not_found"
        )

    loa = float(
        loa_match.group(1)
    )

    after_loa = clean(
        after_purpose[
            loa_match.end():
        ]
    )

    # ------------------------------------------------------------------------
    # TIME
    # ------------------------------------------------------------------------

    time_match = TIME_RE.search(
        after_loa
    )

    if not time_match:
        raise ValueError(
            "time_not_found"
        )

    movement_time = time_match.group(
        0
    )

    cargo = clean(
        after_loa[
            :time_match.start()
        ]
    )

    if not cargo:
        raise ValueError(
            "cargo_not_found"
        )

    after_time = clean(
        after_loa[
            time_match.end():
        ]
    )

    # ------------------------------------------------------------------------
    # QUANTITY
    # ------------------------------------------------------------------------

    qty_match = QTY_RE.search(
        after_time
    )

    if not qty_match:
        raise ValueError(
            "quantity_not_found"
        )

    quantity_mt = float(
        qty_match.group(0).replace(",", "")
    )

    # Text between time and quantity = remarks.
    remarks = clean(
        after_time[
            :qty_match.start()
        ]
    )

    # ------------------------------------------------------------------------
    # BEAM + DRAFT
    # ------------------------------------------------------------------------

    tail = after_time[
        qty_match.end():
    ]

    numeric = list(
        NUMBER_RE.finditer(tail)
    )

    if len(numeric) < 2:
        raise ValueError(
            "beam_draft_not_found"
        )

    beam = float(
        numeric[-2].group(0)
    )

    draft = float(
        numeric[-1].group(0)
    )

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    hour, minute = map(
        int,
        movement_time.split(":"),
    )

    timestamp = (
        movement_date
        + pd.Timedelta(
            hours=hour,
            minutes=minute,
        )
    )

    return {
        "report_filename": meta["filename"],
        "report_traffic_date": (
            meta["traffic_date"]
            .date()
            .isoformat()
        ),
        "report_held_on": (
            meta["held_on"]
            .date()
            .isoformat()
        ),
        "movement_date": (
            movement_date
            .date()
            .isoformat()
        ),
        "movement_timestamp": timestamp.isoformat(),
        "vessel_name": vessel_name,
        "vessel_name_normalized": normalize_vessel(
            vessel_name
        ),
        "purpose": purpose,
        "berth": berth,
        "movement_time": movement_time,
        "loa_m": loa,
        "beam_m": beam,
        "draft_m": draft,
        "cargo": cargo,
        "quantity_mt": quantity_mt,
        "remarks": remarks,
        "source_page": page_number,
        "source_text": original,
    }


# ============================================================================
# WAITING DATA
# ============================================================================

def load_waiting():
    """
    Reuse the already-working waiting parser output.

    We do NOT parse waiting tables again.
    """

    df = pd.read_csv(
        WAITING_SOURCE
    )

    # Normalize possible existing column naming.
    rename = {}

    if (
        "name_of_vessel" in df.columns
        and "vessel_name" not in df.columns
    ):
        rename[
            "name_of_vessel"
        ] = "vessel_name"

    if (
        "arrival_datetime" in df.columns
        and "arrival_timestamp" not in df.columns
    ):
        rename[
            "arrival_datetime"
        ] = "arrival_timestamp"

    df = df.rename(
        columns=rename
    )

    if "vessel_name" not in df.columns:
        raise RuntimeError(
            "waiting_vessels.csv has no vessel_name column"
        )

    df["vessel_name"] = (
        df["vessel_name"]
        .astype(str)
        .str.strip()
    )

    df["vessel_name_normalized"] = (
        df["vessel_name"]
        .map(normalize_vessel)
    )

    return df


# ============================================================================
# ARRIVAL TIMESTAMP
# ============================================================================

def get_arrival_timestamp(row):

    for column in (
        "arrival_timestamp",
        "arrival_datetime",
    ):

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                parsed = pd.to_datetime(
                    value,
                    errors="coerce",
                )

                if pd.notna(parsed):
                    return parsed

    date_value = None

    for column in (
        "arrival_date",
        "arrival",
        "date_of_arrival",
    ):

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                date_value = value
                break

    if date_value is None:
        return pd.NaT

    date = pd.to_datetime(
        date_value,
        errors="coerce",
    )

    if pd.isna(date):
        return pd.NaT

    time_value = None

    for column in (
        "arrival_time",
        "time",
        "time_of_arrival",
    ):

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                time_value = value
                break

    if time_value is None:
        return date

    match = TIME_RE.search(
        str(time_value)
    )

    if not match:
        return date

    hour = int(
        match.group(1)
    )

    minute = int(
        match.group(2)
    )

    return date + pd.Timedelta(
        hours=hour,
        minutes=minute,
    )


# ============================================================================
# MATCH WAITING → BERTHS
# ============================================================================

def build_delays(
    waiting,
    movements,
):

    if movements.empty:
        raise RuntimeError(
            "Cannot build delays with zero movements"
        )

    berths = movements[
        movements["purpose"]
        == "BERTHS"
    ].copy()

    berths[
        "movement_timestamp"
    ] = pd.to_datetime(
        berths[
            "movement_timestamp"
        ]
    )

    berths = berths.sort_values(
        [
            "vessel_name_normalized",
            "movement_timestamp",
        ]
    )

    output = []

    for _, row in waiting.iterrows():

        vessel = row[
            "vessel_name_normalized"
        ]

        arrival = get_arrival_timestamp(
            row
        )

        candidates = berths[
            berths[
                "vessel_name_normalized"
            ]
            == vessel
        ]

        if pd.notna(arrival):

            candidates = candidates[
                candidates[
                    "movement_timestamp"
                ] >= arrival
            ]

        if candidates.empty:

            result = row.to_dict()

            result[
                "matched_movement_timestamp"
            ] = pd.NaT

            result[
                "matched_berth"
            ] = None

            result[
                "arrival_to_berth_hours"
            ] = pd.NA

            result[
                "arrival_to_berth_days"
            ] = pd.NA

            result[
                "delay_status"
            ] = "unmatched"

            output.append(
                result
            )

            continue

        movement = candidates.iloc[0]

        berth_timestamp = movement[
            "movement_timestamp"
        ]

        result = row.to_dict()

        result[
            "matched_movement_timestamp"
        ] = berth_timestamp

        result[
            "matched_berth"
        ] = movement["berth"]

        if pd.isna(arrival):

            result[
                "arrival_to_berth_hours"
            ] = pd.NA

            result[
                "arrival_to_berth_days"
            ] = pd.NA

            result[
                "delay_status"
            ] = "matched_arrival_missing"

        else:

            hours = (
                berth_timestamp - arrival
            ).total_seconds() / 3600

            if hours < 0:

                result[
                    "arrival_to_berth_hours"
                ] = pd.NA

                result[
                    "arrival_to_berth_days"
                ] = pd.NA

                result[
                    "delay_status"
                ] = "invalid_negative_delay"

            else:

                result[
                    "arrival_to_berth_hours"
                ] = round(
                    hours,
                    4,
                )

                result[
                    "arrival_to_berth_days"
                ] = round(
                    hours / 24,
                    4,
                )

                result[
                    "delay_status"
                ] = "matched"

        output.append(
            result
        )

    return pd.DataFrame(
        output
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate(
    pdf_files,
    movements,
    delays,
    quarantine,
):

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    errors = []

    print(
        f"PDF reports discovered: {len(pdf_files)}"
    )

    if len(pdf_files) != 113:
        errors.append(
            f"Expected 113 PDFs, found {len(pdf_files)}"
        )

    print(
        f"Movement rows parsed:   {len(movements)}"
    )

    if len(movements) == 0:
        errors.append(
            "ZERO movement rows parsed"
        )

    if not movements.empty:

        berth_count = int(
            (
                movements["purpose"]
                == "BERTHS"
            ).sum()
        )

        sails_count = int(
            (
                movements["purpose"]
                == "SAILS"
            ).sum()
        )

        duplicate_count = int(
            movements.duplicated(
                subset=[
                    "report_filename",
                    "movement_timestamp",
                    "vessel_name_normalized",
                    "purpose",
                    "berth",
                ]
            ).sum()
        )

        print(
            f"BERTHS events:          {berth_count}"
        )

        print(
            f"SAILS events:           {sails_count}"
        )

        print(
            f"Duplicate movements:    {duplicate_count}"
        )

        if berth_count == 0:
            errors.append(
                "ZERO BERTHS events"
            )

        if duplicate_count:
            errors.append(
                f"{duplicate_count} duplicate movement events"
            )

    # ------------------------------------------------------------------------
    # APJ SETHU
    # ------------------------------------------------------------------------

    apj = delays[
        delays[
            "vessel_name_normalized"
        ]
        == "APJ SETHU"
    ].copy()

    if apj.empty:

        errors.append(
            "APJ SETHU not present"
        )

    else:

        matched = apj[
            apj[
                "delay_status"
            ]
            == "matched"
        ]

        if matched.empty:

            errors.append(
                "APJ SETHU exists but did not match a BERTHS event"
            )

        else:

            value = float(
                matched.iloc[0][
                    "arrival_to_berth_hours"
                ]
            )

            print(
                f"APJ SETHU delay:        {value:.2f} hours"
            )

            if abs(value - 50.2) > 1.0:

                errors.append(
                    f"APJ SETHU expected ~50.2h, got {value:.2f}h"
                )

    print(
        f"Quarantine rows:        {len(quarantine)}"
    )

    if errors:

        print()
        print("VALIDATION: FAIL")

        for error in errors:
            print(
                f"  - {error}"
            )

        raise RuntimeError(
            "Validation failed."
        )

    print()
    print("VALIDATION: PASS")


# ============================================================================
# MAIN
# ============================================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        RAW_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise RuntimeError(
            "No Paradip traffic PDFs found"
        )

    inventory = load_inventory()

    movements = []
    quarantine = []

    reports_with_section = 0

    print("=" * 80)
    print("PARADIP OPERATIONAL CORPUS V3")
    print("=" * 80)
    print(
        f"Reports discovered: {len(pdf_files)}"
    )
    print()

    # ========================================================================
    # MOVEMENT PARSING
    # ========================================================================

    for index, pdf in enumerate(
        pdf_files,
        start=1,
    ):

        try:

            pages = read_pages(
                pdf
            )

            meta = report_metadata(
                pdf,
                inventory,
            )

            page_number, page_text = (
                find_movement_page(
                    pages
                )
            )

            if page_text is None:

                quarantine.append({
                    "filename": pdf.name,
                    "section": "movement",
                    "reason": "movement_section_not_found",
                    "source_text": "",
                })

                print(
                    f"[{index:03}/{len(pdf_files):03}] "
                    f"{pdf.name} NO_SECTION"
                )

                continue

            reports_with_section += 1

            records = group_records(
                page_text
            )

            success = 0

            for record in records:

                try:

                    movement = parse_movement(
                        record,
                        meta,
                        page_number,
                    )

                    movements.append(
                        movement
                    )

                    success += 1

                except Exception as exc:

                    quarantine.append({
                        "filename": pdf.name,
                        "section": "movement",
                        "reason": str(exc),
                        "source_page": page_number,
                        "source_text": record,
                    })

            print(
                f"[{index:03}/{len(pdf_files):03}] "
                f"{pdf.name} "
                f"page={page_number} "
                f"records={len(records)} "
                f"parsed={success}"
            )

        except Exception as exc:

            quarantine.append({
                "filename": pdf.name,
                "section": "report",
                "reason": str(exc),
                "source_text": "",
            })

            print(
                f"[{index:03}/{len(pdf_files):03}] "
                f"{pdf.name} ERROR={exc}"
            )

    movement_df = pd.DataFrame(
        movements
    )

    # ========================================================================
    # WAITING
    # ========================================================================

    waiting_df = load_waiting()

    # ========================================================================
    # DEDUP
    # ========================================================================

    if not movement_df.empty:

        movement_df = movement_df.drop_duplicates(
            subset=[
                "report_filename",
                "movement_timestamp",
                "vessel_name_normalized",
                "purpose",
                "berth",
            ]
        )

        movement_df = movement_df.sort_values(
            [
                "movement_timestamp",
                "vessel_name_normalized",
            ]
        )

    # ========================================================================
    # DELAYS
    # ========================================================================

    delay_df = build_delays(
        waiting_df,
        movement_df,
    )

    quarantine_df = pd.DataFrame(
        quarantine
    )

    # ========================================================================
    # VALIDATE BEFORE ACCEPTING
    # ========================================================================

    validate(
        pdf_files,
        movement_df,
        delay_df,
        quarantine_df,
    )

    # ========================================================================
    # WRITE
    # ========================================================================

    movement_df.to_csv(
        MOVEMENTS_OUT,
        index=False,
    )

    delay_df.to_csv(
        DELAYS_OUT,
        index=False,
    )

    quarantine_df.to_csv(
        QUARANTINE_OUT,
        index=False,
    )

    matched_delays = int(
        (
            delay_df["delay_status"]
            == "matched"
        ).sum()
    )

    metadata = {
        "dataset": "Paradip operational corpus v3",
        "status": "validated",
        "generated_at_utc": (
            datetime.utcnow().isoformat()
            + "Z"
        ),
        "reports": len(pdf_files),
        "reports_with_movement_section": (
            reports_with_section
        ),
        "waiting_rows": len(waiting_df),
        "movement_rows": len(movement_df),
        "berths_events": int(
            (
                movement_df["purpose"]
                == "BERTHS"
            ).sum()
        ),
        "sails_events": int(
            (
                movement_df["purpose"]
                == "SAILS"
            ).sum()
        ),
        "delay_rows": len(delay_df),
        "matched_delays": matched_delays,
        "quarantine_rows": len(
            quarantine_df
        ),
        "delay_definition": (
            "Elapsed time between reported vessel arrival "
            "and the first subsequent BERTHS movement "
            "for the normalized vessel name."
        ),
        "interpretation": (
            "Arrival-to-berth time is an operational delay "
            "proxy and must not be interpreted as pure "
            "port congestion."
        ),
        "source_fields": [
            "report_filename",
            "report_traffic_date",
            "report_held_on",
            "source_page",
            "source_text",
        ],
    }

    with open(
        METADATA_OUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PARADIP OPERATIONAL CORPUS V3 COMPLETE")
    print("=" * 80)

    print(
        f"Reports:                {len(pdf_files)}"
    )

    print(
        f"Reports with section:   {reports_with_section}"
    )

    print(
        f"Waiting rows:           {len(waiting_df)}"
    )

    print(
        f"Movement rows:          {len(movement_df)}"
    )

    print(
        f"BERTHS events:          "
        f"{(movement_df['purpose'] == 'BERTHS').sum()}"
    )

    print(
        f"SAILS events:           "
        f"{(movement_df['purpose'] == 'SAILS').sum()}"
    )

    print(
        f"Matched delays:         {matched_delays}"
    )

    print(
        f"Unmatched waiting:      "
        f"{len(delay_df) - matched_delays}"
    )

    print(
        f"Quarantine rows:        {len(quarantine_df)}"
    )

    print()
    print("Outputs:")
    print(MOVEMENTS_OUT)
    print(DELAYS_OUT)
    print(QUARANTINE_OUT)
    print(METADATA_OUT)


if __name__ == "__main__":
    main()
