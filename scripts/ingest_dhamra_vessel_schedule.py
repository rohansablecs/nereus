from pathlib import Path
from datetime import datetime, timezone
import json
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.adaniports.com/ports-and-terminals/dhamra-port/vesselschedule"
RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

session = requests.Session()
session.headers.update(headers)

response = None
last_error = None

for attempt in range(1, 5):
    try:
        print(f"Attempt {attempt}/4...")

        response = session.get(
            URL,
            timeout=(15, 90),
        )

        response.raise_for_status()

        print(f"Connected: HTTP {response.status_code}")
        print(f"Downloaded: {len(response.content):,} bytes")
        break

    except requests.RequestException as exc:
        last_error = exc
        print(f"Attempt {attempt} failed: {exc}")

        if attempt < 4:
            wait = attempt * 5
            print(f"Waiting {wait}s before retry...")
            time.sleep(wait)

if response is None:
    raise RuntimeError(
        "Could not retrieve the Dhamra vessel schedule after 4 attempts."
    ) from last_error

soup = BeautifulSoup(response.text, "lxml")
tables = soup.find_all("table")

print(f"HTML tables found: {len(tables)}")

if not tables:
    raise RuntimeError(
        "The page loaded but no HTML tables were found. "
        "The schedule may be rendered dynamically."
    )

frames = []

for table_index, table in enumerate(tables, start=1):
    rows = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        values = [cell.get_text(" ", strip=True) for cell in cells]

        if values:
            rows.append(values)

    if len(rows) < 2:
        continue

    header = rows[0]
    width = len(header)

    data = [
        row for row in rows[1:]
        if len(row) == width
    ]

    if data:
        frame = pd.DataFrame(data, columns=header)
        frame.insert(0, "source_table", table_index)
        frames.append(frame)

if not frames:
    raise RuntimeError(
        "HTML tables were detected, but no usable tabular rows were found."
    )

df = pd.concat(frames, ignore_index=True)

df.columns = [
    str(column)
    .strip()
    .lower()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("-", "_")
    for column in df.columns
]

df = df.dropna(how="all").reset_index(drop=True)

snapshot_time = datetime.now(timezone.utc).isoformat()

df.insert(0, "snapshot_time_utc", snapshot_time)
df.insert(1, "port", "Dhamra")
df.insert(2, "source_url", URL)

out = RAW / "dhamra_vessel_schedule_current.csv"
df.to_csv(out, index=False)

metadata = {
    "port": "Dhamra",
    "dataset": "Current vessel schedule snapshot",
    "source_url": URL,
    "snapshot_time_utc": snapshot_time,
    "retrieved_at_runtime": True,
    "tables_found": len(tables),
    "rows_saved": len(df),
    "columns": list(df.columns),
    "purpose": [
        "current berth occupancy",
        "anchorage activity",
        "cargo mix",
        "expected completion timing",
    ],
    "important_note": (
        "This is a point-in-time operational snapshot. "
        "It is not historical congestion data."
    ),
}

with open(
    RAW / "dhamra_vessel_schedule_current_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print()
print("=" * 60)
print("Dhamra vessel schedule ingestion complete")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Snapshot: {snapshot_time}")
print()
print("Columns:")
for column in df.columns:
    print(f"  {column}")
print()
print(f"Output: {out}")
print("=" * 60)
