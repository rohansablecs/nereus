from pathlib import Path
from datetime import datetime, timezone
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

URL = "https://smportkolkata.shipping.gov.in/smpk/hld/en/berth-position/"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

r = requests.get(URL, headers=headers, timeout=(15, 60))
r.raise_for_status()

print("HTTP:", r.status_code)
print("Bytes:", len(r.content))

soup = BeautifulSoup(r.text, "lxml")

tables = soup.find_all("table")
print("HTML tables:", len(tables))

target = None

for table in tables:
    text = table.get_text(" ", strip=True).lower()

    if (
        "berth name" in text
        and "vessel name" in text
        and "arrival date" in text
    ):
        target = table
        break

if target is None:
    raise RuntimeError(
        "Could not locate Haldia berth-position table."
    )

rows = []

for tr in target.find_all("tr"):
    cells = tr.find_all(["th", "td"])
    values = [c.get_text(" ", strip=True) for c in cells]

    if values:
        rows.append(values)

if len(rows) < 2:
    raise RuntimeError("Haldia berth-position table contains no data.")

header = rows[0]

data = [
    row for row in rows[1:]
    if len(row) == len(header)
]

df = pd.DataFrame(data, columns=header)

df.columns = [
    re.sub(
        r"[^a-z0-9]+",
        "_",
        str(c).strip().lower()
    ).strip("_")
    for c in df.columns
]

df.insert(0, "port", "Haldia")
df.insert(
    1,
    "snapshot_time_utc",
    datetime.now(timezone.utc).isoformat()
)
df.insert(2, "source_url", URL)

out = RAW / "haldia_berth_position_current.csv"
df.to_csv(out, index=False)

metadata = {
    "port": "Haldia",
    "dataset": "Current berth position",
    "source_url": URL,
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "rows": len(df),
    "columns": list(df.columns),
    "notes": [
        "Official Haldia Dock Complex berth-position page.",
        "Source page states that records are displayed from the last 30 days.",
        "This is operational activity data, not a historical congestion series.",
    ],
}

with open(
    RAW / "haldia_berth_position_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print()
print("=" * 60)
print("HALDIA BERTH POSITION INGESTION COMPLETE")
print("=" * 60)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("Columns:")
for c in df.columns:
    print(" ", c)
print()
print("Created:")
print(" ", out)
print(
    " ",
    RAW / "haldia_berth_position_metadata.json"
)
print("=" * 60)
