from pathlib import Path
import pandas as pd
import json

RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

rows = []

for day in range(1, 32):
    rows.append({
        "port": "Dhamra",
        "berth_group": "BB1 & BB2 Import Berth",
        "date": f"2017-12-{day:02d}",
        "max_sw_arrival_draft_m": 17.20,
        "status": "historical",
        "source_document": "Monthly Draft Declaration",
        "source_url": "https://www.adaniports.com/-/media/project/ports/portsandterminals/dhamra-port/documents/monthly-draft-declaration.pdf",
    })

df = pd.DataFrame(rows)

out = RAW / "dhamra_draft_declarations_historical.csv"
df.to_csv(out, index=False)

metadata = {
    "port": "Dhamra",
    "dataset": "Historical monthly draft declarations",
    "coverage_start": "2017-12-01",
    "coverage_end": "2017-12-31",
    "berth_group": "BB1 & BB2 Import Berth",
    "max_sw_arrival_draft_m": 17.20,
    "status": "historical_only",
    "warning": (
        "The currently linked official Monthly Draft Declaration "
        "document is a December 2017 declaration. It must not be "
        "used as the current permissible draft."
    ),
}

with open(
    RAW / "dhamra_draft_declarations_historical_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print("Dhamra historical draft dataset created.")
print(f"Rows: {len(df)}")
print(f"Output: {out}")
print("IMPORTANT: marked historical_only; not used as current feasibility constraint.")
