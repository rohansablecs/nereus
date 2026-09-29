import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL = "https://paradipport.gov.in/berth-specifications/"

OUTPUT_DIR = Path("02_DATA/raw/port")
CSV_PATH = OUTPUT_DIR / "paradip_berths.csv"
METADATA_PATH = OUTPUT_DIR / "paradip_berths_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


response = requests.get(
    URL,
    timeout=60,
    headers={"User-Agent": "NEREUS/0.1"},
)

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all("table")

if len(tables) != 1:
    raise RuntimeError(f"Expected exactly 1 table, found {len(tables)}")


table = tables[0]

rows = []

for tr in table.find_all("tr"):
    cells = tr.find_all("td")

    if not cells:
        continue

    row = {
        "sl_no": "",
        "berth_name": "",
        "commodity": "",
        "loa_m": "",
        "beam_m": "",
        "draft_m": "",
        "remarks": "",
        "berth_type": "",
    }

    for cell in cells:
        classes = cell.get("class", [])

        column_class = next(
            (c for c in classes if c.startswith("column-")),
            None,
        )

        if not column_class:
            continue

        column_number = int(column_class.split("-")[1])

        text = cell.get_text(" ", strip=True)

        mapping = {
            1: "sl_no",
            2: "berth_name",
            3: "commodity",
            4: "loa_m",
            5: "beam_m",
            6: "draft_m",
            7: "remarks",
            8: "berth_type",
        }

        if column_number in mapping:
            row[mapping[column_number]] = text

    # Only keep actual numbered berth records.
    if row["sl_no"].isdigit():
        rows.append(row)


if not rows:
    raise RuntimeError("No berth records found")


df = pd.DataFrame(rows)


# Numeric fields.
df["sl_no"] = pd.to_numeric(df["sl_no"], errors="coerce")
df["loa_m"] = pd.to_numeric(df["loa_m"], errors="coerce")
df["beam_m"] = pd.to_numeric(df["beam_m"], errors="coerce")
df["draft_m"] = pd.to_numeric(df["draft_m"], errors="coerce")


# Ensure deterministic column order.
columns = [
    "sl_no",
    "berth_name",
    "commodity",
    "loa_m",
    "beam_m",
    "draft_m",
    "remarks",
    "berth_type",
]

df = df[columns]


df.to_csv(CSV_PATH, index=False)


metadata = {
    "source_url": URL,
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_status": response.status_code,
    "table_count": len(tables),
    "row_count": len(df),
    "column_count": len(df.columns),
    "columns": columns,
}


METADATA_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)


print("Paradip berth ingestion complete.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("CSV:", CSV_PATH)
print("Metadata:", METADATA_PATH)
