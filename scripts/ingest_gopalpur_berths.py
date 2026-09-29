import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PDF_PATH = Path(
    "02_DATA/raw/port/gopalpur_berthing_policy_2026-06.pdf"
)

OUTPUT_DIR = Path("02_DATA/raw/port")
CSV_PATH = OUTPUT_DIR / "gopalpur_berths.csv"
METADATA_PATH = OUTPUT_DIR / "gopalpur_berths_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

reader = PdfReader(str(PDF_PATH))

# The extracted berth table is on PDF page 21.
# pypdf uses zero-based indexing, therefore index 20.
PAGE_INDEX = 20

text = reader.pages[PAGE_INDEX].extract_text() or ""

if "Berth Parameters" not in text:
    raise RuntimeError(
        f"Could not locate berth parameter section on page index {PAGE_INDEX}."
    )


# Extract:
# B-1 300 145000 14.5 mtrs As per availability
# B-2 300 145000 14.5 mtrs Import coal
# B-3 200 75000 14.5 mtrs As per availability

pattern = re.compile(
    r"(?m)^\s*(B-\d)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+mtrs\s+"
    r"(.+?)\s*$"
)

matches = pattern.findall(text)

if len(matches) != 3:
    raise RuntimeError(
        f"Expected 3 berth records, found {len(matches)}."
    )


rows = []

for berth, loa, displacement, draft, priority in matches:
    rows.append(
        {
            "berth": berth,
            "loa_m": float(loa),
            "displacement_mt": float(displacement),
            "max_draft_m": float(draft),
            "allocation_priority": " ".join(priority.split()),
        }
    )


df = pd.DataFrame(rows)

expected_berths = ["B-1", "B-2", "B-3"]

if df["berth"].tolist() != expected_berths:
    raise RuntimeError(
        f"Unexpected berth sequence: {df['berth'].tolist()}"
    )


# Extract operational limits from the source text.
wind_match = re.search(
    r"average wind speed exceeds\s+(\d+(?:\.\d+)?)\s*Kts",
    text,
    re.IGNORECASE,
)

trim_match = re.search(
    r"Trim<\s*(\d+(?:\.\d+)?)m",
    text,
    re.IGNORECASE,
)

list_match = re.search(
    r"List\s*=\s*(\d+(?:\.\d+)?)",
    text,
    re.IGNORECASE,
)


operational_rules = {
    "maximum_permissible_draft_policy": (
        "Maximum permissible draft at each berth is "
        "promulgated on a monthly basis."
    ),
    "weather_wind_threshold_kt": (
        float(wind_match.group(1))
        if wind_match
        else None
    ),
    "maximum_trim_m": (
        float(trim_match.group(1))
        if trim_match
        else None
    ),
    "maximum_list_deg": (
        float(list_match.group(1))
        if list_match
        else None
    ),
}


df.to_csv(CSV_PATH, index=False)

metadata = {
    "source_file": str(PDF_PATH),
    "source_type": "official_port_berthing_policy",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_page": 21,
    "berth_record_count": len(df),
    "columns": list(df.columns),
    "operational_rules": operational_rules,
}

METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)


print("Gopalpur berth ingestion complete.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("CSV:", CSV_PATH)
print("Metadata:", METADATA_PATH)
print()
print(df.to_string(index=False))
print()
print("Operational rules:")
for key, value in operational_rules.items():
    print(f"  {key}: {value}")
