import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = (
    "https://services9.arcgis.com/"
    "weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "portwatch_disruptions_database/FeatureServer/0/query"
)

FIELDS = [
    "eventid",
    "eventtype",
    "eventname",
    "htmlname",
    "htmldescription",
    "alertlevel",
    "country",
    "fromdate",
    "year",
    "todate",
    "severitytext",
    "lat",
    "long",
    "editdate",
    "affectedports",
    "n_affectedports",
    "affectedpopulation",
    "pageid",
]

OUTPUT_DIR = Path("02_DATA/raw/port")
CSV_PATH = OUTPUT_DIR / "portwatch_disruptions.csv"
METADATA_PATH = OUTPUT_DIR / "portwatch_disruptions_metadata.json"

TIMEOUT = 60


def fetch_disruptions():
    params = {
        "where": "1=1",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "resultRecordCount": 500,
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

    records = [
        feature["attributes"]
        for feature in data.get("features", [])
    ]

    return records


def save_csv(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(records)


def save_metadata(records):
    metadata = {
        "source": "IMF PortWatch",
        "dataset": "portwatch_disruptions_database",
        "source_url": BASE_URL,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "query": {
            "where": "1=1",
            "fields": FIELDS,
        },
        "record_count": len(records),
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
    print("NEREUS — PortWatch disruption ingestion")
    print("----------------------------------------")
    print("Source:", BASE_URL)
    print()

    records = fetch_disruptions()

    if not records:
        raise RuntimeError(
            "PortWatch returned zero disruption records."
        )

    print("Fetched records:", len(records))

    save_csv(records)
    save_metadata(records)

    print()
    print("Ingestion complete.")
    print("CSV:", CSV_PATH)
    print("Metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
