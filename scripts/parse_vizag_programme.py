from pathlib import Path
from datetime import datetime, timezone
import json
import re
import pandas as pd

RAW = Path("02_DATA/raw/port")

PROGRAMME_DATE = "2026-09-07"

TEXT_FILE = RAW / f"vizag_berthing_programme_{PROGRAMME_DATE}.txt"

text = TEXT_FILE.read_text(encoding="utf-8")
lines = [line.strip() for line in text.splitlines()]

SNAPSHOT = datetime.now(timezone.utc).isoformat()


# =========================================================
# HELPERS
# =========================================================

def clean(value):
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()

    return value or None


def number(value):
    if value is None:
        return None

    value = value.replace(",", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


def normalize_vessel(value):
    if not value:
        return None

    value = re.sub(r"\s+", " ", value)

    value = value.replace("m.v ", "")
    value = value.replace("m.t ", "")
    value = value.replace("LPG/C ", "")
    value = value.replace("LPG/C. ", "")

    return value.strip()


def extract_cargo_from_vessel_row(raw):
    """
    Extract cargo from the official Vizag programme row.

    Examples:

        I/COKING COAL 19,976 45,528 0
        I/HARD COAL 31,500 36,323 42,127
        I/M.ORE 7,480 17,453 32,077
        E/FERRO MANGANESE & STEEL CARGO 1,873 15,678 7,572

    The cargo text is terminated by the first numeric quantity.

    The source wording is preserved rather than guessing
    a normalized cargo category.
    """

    match = re.search(
        r"\b[IE]/(.+?)(?=\s+-?\d[\d,]*(?:\.\d+)?(?:\s|$))",
        raw,
        re.IGNORECASE,
    )

    if not match:
        return None

    cargo = match.group(1).strip()

    cargo = re.sub(
        r"\s+",
        " ",
        cargo,
    )

    return cargo or None


# Nationalities appearing in the programme.
# These are ONLY used to locate the boundary between
# vessel name and nationality.
NATIONALITIES = [
    "SINGAPORE",
    "LIBERIA",
    "M.ISLANDS",
    "MARSHALL ISLANDS",
    "PANAMA",
    "MALTA",
    "HONGKONG",
    "HONG KONG",
    "INDIA",
    "INDIAN",
    "KOREA",
    "SAN MARINO",
    "BAHAMAS",
    "NIGERIA",
    "USA",
]


# =========================================================
# 1. WAITING & EXPECTED VESSELS
# =========================================================

waiting = []

current_section = None

category_map = {
    "[A]": "IRON ORE EXPORTS",
    "[B]": "IRON & STEEL",
    "[C]": "OTHER MULTIPLE GENERAL CARGOES",
    "[D]": "FOOD GRAINS & OTHER EDIBLES",
    "[E]": "CRUDE & POL PRODUCTS",
    "[F]": "FERTILIZERS IMPORTS",
    "[G]": "COAL AND COKE",
    "[H]": "OTHER MULTIPLE GENERAL CARGO",
    "[I]": "CONTAINERS",
    "[J]": "CRUDE & POL PRODUCTS",
    "[K]": "FOOD GRAINS & OTHER EDIBLES",
}


for line in lines:

    section_match = re.match(
        r"^\s*(\[[A-K]\])\s*(.*)$",
        line,
    )

    if section_match:
        key = section_match.group(1)

        current_section = category_map.get(
            key,
            clean(section_match.group(2)),
        )

        continue

    if line.startswith("(a)") or line.startswith("(b)"):
        continue

    if line in {"NIL", "0", ".", ". 0"}:
        continue

    row_match = re.match(
        r"^(\d+)\s+(.*)$",
        line,
    )

    if not row_match or current_section is None:
        continue

    row_no = int(row_match.group(1))
    body = row_match.group(2)

    if body.startswith("V A C A N T"):
        continue

    if not re.search(
        r"\b(?:m\.v|m\.t|LPG/C)\b",
        body,
        re.IGNORECASE,
    ):
        continue

    raw = body

    priority = bool("" in body)

    body = body.replace("", " ")

    body = re.sub(
        r"\s+",
        " ",
        body,
    ).strip()

    vessel_match = re.search(
        r"\b(m\.v|m\.t|LPG/C\.?)\s+(.+)",
        body,
        re.IGNORECASE,
    )

    if not vessel_match:
        continue

    vessel_type = vessel_match.group(1).upper()

    remainder = vessel_match.group(2).strip()

    nationality = None
    nationality_pos = None

    for nat in NATIONALITIES:

        match = re.search(
            r"\b" + re.escape(nat) + r"\b",
            remainder,
            re.IGNORECASE,
        )

        if match:

            if (
                nationality_pos is None
                or match.start() < nationality_pos
            ):
                nationality_pos = match.start()
                nationality = match.group(0)

    if nationality_pos is not None:

        vessel = remainder[:nationality_pos].strip()

        after_nationality = remainder[
            nationality_pos + len(nationality):
        ].strip()

    else:

        vessel = remainder

        after_nationality = ""

    date_match = re.search(
        r"\b(\d{1,2})[-/]"
        r"(Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug)"
        r"(?:[-/](\d{2,4}))?\b",
        after_nationality,
        re.IGNORECASE,
    )

    expected_date = None

    if date_match:

        day = date_match.group(1)

        month = date_match.group(2)

        month_num = datetime.strptime(
            month[:3].title(),
            "%b",
        ).month

        expected_date = (
            f"2026-{month_num:02d}-{int(day):02d}"
        )

    tonnes_matches = re.findall(
        r"(?<![\d.])([\d,]{3,})\s*T\b",
        after_nationality,
        re.IGNORECASE,
    )

    tonnes_mt = None

    if tonnes_matches:
        tonnes_mt = number(tonnes_matches[-1])

    dimension_values = re.findall(
        r"(?<![\d.])(\d{1,3}\.\d{1,2})\b",
        after_nationality,
    )

    waiting.append(
        {
            "port": "Visakhapatnam",
            "programme_date": PROGRAMME_DATE,
            "snapshot_time_utc": SNAPSHOT,
            "section": current_section,
            "row_no": row_no,
            "vessel": normalize_vessel(vessel),
            "vessel_type_source": vessel_type,
            "nationality": clean(nationality),
            "expected_date": expected_date,
            "tonnes_mt": tonnes_mt,
            "priority_marker": priority,
            "dimension_candidates": (
                "|".join(dimension_values)
                or None
            ),
            "raw_row": raw,
        }
    )


waiting_df = pd.DataFrame(waiting)

waiting_df = waiting_df.drop_duplicates(
    subset=[
        "programme_date",
        "section",
        "row_no",
        "vessel",
    ],
).reset_index(drop=True)


# =========================================================
# 2. CURRENT BERTH OCCUPANCY
# =========================================================

berth_rows = []

berth_start = None

for i, line in enumerate(lines):

    if (
        "Berth Name of the Vessel" in line
        and "Berthed time" in line
    ):
        berth_start = i
        break


if berth_start is None:
    raise RuntimeError(
        "Could not locate current berth table."
    )


BERTH_PATTERN = re.compile(
    r"^(OB-1|OB\s*-\s*2|C\.TERMINAL|VGCB|SPM|OSTT|LPG|"
    r"CB-\d+|EQ-\d+[A-Z]?|Gr\.Ch\.B|WQ-\d+[A-Z]*|OR-\d+|FB)"
    r"\s+(.*)$",
    re.IGNORECASE,
)


for line in lines[berth_start + 1:]:

    if line.startswith("-" * 20):
        break

    match = BERTH_PATTERN.match(line)

    if not match:
        continue

    berth = re.sub(
        r"\s+",
        " ",
        match.group(1),
    ).strip()

    body = match.group(2).strip()

    # -----------------------------------------------------
    # VACANT
    # -----------------------------------------------------

    if body.upper().startswith("V A C A N T"):

        berth_rows.append(
            {
                "port": "Visakhapatnam",
                "programme_date": PROGRAMME_DATE,
                "snapshot_time_utc": SNAPSHOT,
                "berth": berth,
                "status": "VACANT",
                "vessel": None,
                "agent": None,
                "operation": None,
                "cargo": None,
                "worked_today_mt": None,
                "worked_upto_mt": None,
                "balance_mt": None,
                "raw_row": line,
            }
        )

        continue

    # -----------------------------------------------------
    # HANDED OVER
    # -----------------------------------------------------

    if body.upper().startswith("HANDED OVER"):

        berth_rows.append(
            {
                "port": "Visakhapatnam",
                "programme_date": PROGRAMME_DATE,
                "snapshot_time_utc": SNAPSHOT,
                "berth": berth,
                "status": "UNAVAILABLE",
                "vessel": None,
                "agent": None,
                "operation": None,
                "cargo": None,
                "worked_today_mt": None,
                "worked_upto_mt": None,
                "balance_mt": None,
                "raw_row": line,
            }
        )

        continue

    # -----------------------------------------------------
    # BERTH SPARED
    # -----------------------------------------------------

    if body.upper().startswith("BERTH SPARED"):

        berth_rows.append(
            {
                "port": "Visakhapatnam",
                "programme_date": PROGRAMME_DATE,
                "snapshot_time_utc": SNAPSHOT,
                "berth": berth,
                "status": "UNAVAILABLE",
                "vessel": None,
                "agent": None,
                "operation": None,
                "cargo": None,
                "worked_today_mt": None,
                "worked_upto_mt": None,
                "balance_mt": None,
                "raw_row": line,
            }
        )

        continue

    # -----------------------------------------------------
    # VESSEL ROW
    # -----------------------------------------------------

    vessel_match = re.search(
        r"\b(?:m\.v|m\.t)\s+(.+?)(?=\s+"
        r"(?:EVTL|Seniority|[A-Z0-9&.]+)\s+)",
        body,
        re.IGNORECASE,
    )

    vessel = None

    if vessel_match:
        vessel = vessel_match.group(1).strip()

    # -----------------------------------------------------
    # CARGO
    # -----------------------------------------------------

    cargo = extract_cargo_from_vessel_row(line)

    # -----------------------------------------------------
    # QUANTITIES
    # -----------------------------------------------------

    quantities = re.findall(
        r"(?<![\d.])([\d,]+)(?![\d.])",
        body,
    )

    numeric_values = [
        number(q)
        for q in quantities
    ]

    worked_today = None
    worked_upto = None
    balance = None

    if len(numeric_values) >= 3:

        worked_today = numeric_values[-3]

        worked_upto = numeric_values[-2]

        balance = numeric_values[-1]

    # -----------------------------------------------------
    # OPERATION
    # -----------------------------------------------------

    operation = None

    if re.search(
        r"\bI\s*&\s*E\b",
        body,
    ):

        operation = "I & E"

    elif re.search(
        r"\bI/",
        body,
    ):

        operation = "IMPORT"

    elif re.search(
        r"\bE/",
        body,
    ):

        operation = "EXPORT"

    # -----------------------------------------------------
    # SAVE OCCUPIED ROW
    # -----------------------------------------------------

    berth_rows.append(
        {
            "port": "Visakhapatnam",
            "programme_date": PROGRAMME_DATE,
            "snapshot_time_utc": SNAPSHOT,
            "berth": berth,
            "status": "OCCUPIED",
            "vessel": clean(vessel),
            "agent": None,
            "operation": operation,
            "cargo": cargo,
            "worked_today_mt": worked_today,
            "worked_upto_mt": worked_upto,
            "balance_mt": balance,
            "raw_row": line,
        }
    )


berth_df = pd.DataFrame(berth_rows)


# =========================================================
# 3. SAVE
# =========================================================

waiting_out = (
    RAW / "vizag_waiting_expected_current.csv"
)

berth_out = (
    RAW / "vizag_berth_occupancy_current.csv"
)


waiting_df.to_csv(
    waiting_out,
    index=False,
)

berth_df.to_csv(
    berth_out,
    index=False,
)


# =========================================================
# 4. SUMMARY
# =========================================================

metadata = {
    "port": "Visakhapatnam",
    "programme_date": PROGRAMME_DATE,
    "source_text": TEXT_FILE.name,
    "snapshot_time_utc": SNAPSHOT,
    "datasets": [
        waiting_out.name,
        berth_out.name,
    ],
    "waiting_expected_rows": len(waiting_df),
    "berth_rows": len(berth_df),
    "waiting_expected_tonnes_mt": (
        float(waiting_df["tonnes_mt"].sum())
        if not waiting_df.empty
        else 0
    ),
    "occupied_berths": int(
        (berth_df["status"] == "OCCUPIED").sum()
    ),
    "vacant_berths": int(
        (berth_df["status"] == "VACANT").sum()
    ),
    "unavailable_berths": int(
        (berth_df["status"] == "UNAVAILABLE").sum()
    ),
    "notes": [
        "Source is the official Visakhapatnam Port Authority berthing programme.",
        "Waiting/expected data is a point-in-time snapshot.",
        "Berth occupancy is a point-in-time operational snapshot.",
        "Ambiguous PDF columns are preserved in raw_row rather than guessed.",
        "Cargo wording is extracted directly from the source row.",
        "No historical congestion values are inferred.",
    ],
}


with open(
    RAW / "vizag_programme_operational_metadata.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("=" * 65)
print("VIZAG PROGRAMME PARSING COMPLETE")
print("=" * 65)

print(
    f"Waiting/expected rows: {len(waiting_df)}"
)

print(
    "Waiting/expected tonnes:",
    f"{metadata['waiting_expected_tonnes_mt']:,.0f} MT",
)

print()

print(
    "Current berth rows:",
    len(berth_df),
)

print(
    "Occupied:",
    metadata["occupied_berths"],
)

print(
    "Vacant:",
    metadata["vacant_berths"],
)

print(
    "Unavailable:",
    metadata["unavailable_berths"],
)

print()

print("Created:")
print(f"  {waiting_out}")
print(f"  {berth_out}")
print(
    f"  {RAW / 'vizag_programme_operational_metadata.json'}"
)

print("=" * 65)
