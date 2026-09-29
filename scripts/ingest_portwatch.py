import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = (
    "https://services9.arcgis.com/"
    "weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)

PORT_IDS = [
    "port883",   # Paradip
    "port1367",  # Visakhapatnam
    "port2299",  # Gopalpur
    "port290",   # Dhamra
    "port442",   # Haldia
]

FIELDS = [
    "date",
    "portid",
    "portname",
    "country",
    "ISO3",
    "portcalls_dry_bulk",
    "portcalls",
    "import_dry_bulk",
    "export_dry_bulk",
]

OUTPUT_DIR = Path("02_DATA/raw/port")
CSV_PATH = OUTPUT_DIR / "portwatch_daily_ports.csv"
METADATA_PATH = OUTPUT_DIR / "portwatch_daily_ports_metadata.json"

PAGE_SIZE = 1000
TIMEOUT = 60


def fetch_portwatch():
    records = []

    where = "portid IN ({})".format(
        ",".join(f"'{port_id}'" for port_id in PORT_IDS)
    )

    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": ",".join(FIELDS),
            "returnGeometry": "false",
            "orderByFields": "date,portid",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=TIMEOUT,
            headers={
                "User-Agent": "NEREUS/0.1",
                "Accept": "application/json",
            },
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        features = data.get("features", [])

        if not features:
            break

        for feature in features:
            records.append(feature["attributes"])

        print(
            f"Fetched {len(features)} records "
            f"(total: {len(records)})"
        )

        if len(features) < PAGE_SIZE:
            break

        offset += len(features)

    return records, where


def save_csv(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(records)


def save_metadata(records, where):
    dates = [
        record["date"]
        for record in records
        if record.get("date")
    ]

    metadata = {
        "source": "IMF PortWatch",
        "dataset": "Daily_Ports_Data",
        "source_url": BASE_URL,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "query": {
            "where": where,
            "fields": FIELDS,
            "port_ids": PORT_IDS,
        },
        "record_count": len(records),
        "ports": sorted(
            {
                record["portid"]
                for record in records
            }
        ),
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def main():
    print("NEREUS — PortWatch ingestion")
    print("--------------------------------")
    print("Source:", BASE_URL)
    print("Ports:", ", ".join(PORT_IDS))
    print()

    records, where = fetch_portwatch()

    if not records:
        raise RuntimeError(
            "PortWatch returned zero records."
        )

    save_csv(records)
    save_metadata(records, where)

    print()
    print("Ingestion complete.")
    print("Records:", len(records))
    print("CSV:", CSV_PATH)
    print("Metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
