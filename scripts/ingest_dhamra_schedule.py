from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "02_DATA/raw/port"
OUT_DIR = ROOT / "02_DATA/processed/dhamra"

URL = (
    "https://www.adaniports.com/"
    "Ports-and-Terminals/dhamra-port/vesselschedule"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
}


def clean(value):
    if value is None:
        return None

    value = " ".join(str(value).split())

    if value.upper() in {
        "",
        "VACANT",
        "NAN",
        "NONE",
    }:
        return None

    return value


def parse_table(table, section):
    rows = []

    headers = [
        clean(th.get_text(" ", strip=True))
        for th in table.find_all("th")
    ]

    for tr in table.find_all("tr"):
        cells = [
            clean(td.get_text(" ", strip=True))
            for td in tr.find_all("td")
        ]

        if not cells:
            continue

        if len(cells) != len(headers):
            continue

        row = dict(zip(headers, cells))
        row["section"] = section
        rows.append(row)

    return rows


def find_table(soup, table_id):
    table = soup.find(
        "table",
        id=table_id,
    )

    if table is None:
        raise RuntimeError(
            f"Could not find table: {table_id}"
        )

    return table


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    response = requests.get(
        URL,
        timeout=30,
        headers=HEADERS,
    )

    response.raise_for_status()

    html = response.text

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # ---------------------------------------------------------------
    # Locate the four schedule tables.
    # ---------------------------------------------------------------

    tables = {
        "at_berth": "TrackingDataTableDahejBerth",
        "at_anchorage": "TrackingDataTableDahejAnchorage",
        "expected": "TrackingDataTableDahejExpected",
        "sailed_24h": "TrackingDataTableDahejSailed",
    }

    all_rows = []

    counts = {}

    for section, table_id in tables.items():

        table = find_table(
            soup,
            table_id,
        )

        rows = parse_table(
            table,
            section,
        )

        counts[section] = len(rows)

        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows
    )

    # ---------------------------------------------------------------
    # Normalize common field names.
    # ---------------------------------------------------------------

    rename = {
        "Berth no.": "berth",
        "Vessels Name": "vessel_name",
        "Imp or Exp": "direction",
        "Cargo": "cargo",
        "Expected Time of Completion (ETC)": "etc",
        "SBU Name.": "sbu_name",
        "SBU Name": "sbu_name",
        "ATA": "ata",
        "ETA": "eta",
        "Pilot on Board (POB)": "pilot_on_board",
        "Pilot Disembark (PD)": "pilot_disembark",
        "Actual Time of Unberthing (ATUB)": "actual_unberthing",
    }

    df = df.rename(
        columns=rename
    )

    # Ensure canonical columns exist.
    for column in [
        "berth",
        "vessel_name",
        "direction",
        "cargo",
        "etc",
        "sbu_name",
        "ata",
        "eta",
        "pilot_on_board",
        "pilot_disembark",
        "actual_unberthing",
    ]:
        if column not in df.columns:
            df[column] = None

    # ---------------------------------------------------------------
    # Parse timestamps.
    # ---------------------------------------------------------------

    for column in [
        "etc",
        "ata",
        "eta",
        "pilot_on_board",
        "pilot_disembark",
        "actual_unberthing",
    ]:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            dayfirst=True,
        )

    # ---------------------------------------------------------------
    # Add ingestion metadata.
    # ---------------------------------------------------------------

    snapshot = datetime.now(
        timezone.utc
    )

    df["port"] = "Dhamra"

    df["snapshot_time_utc"] = snapshot

    df["source_url"] = URL

    df["source_type"] = "official_server_rendered_schedule"

    # Stable order.
    df = df[
        [
            "port",
            "section",
            "berth",
            "sbu_name",
            "vessel_name",
            "direction",
            "cargo",
            "eta",
            "ata",
            "etc",
            "pilot_on_board",
            "pilot_disembark",
            "actual_unberthing",
            "snapshot_time_utc",
            "source_url",
            "source_type",
        ]
    ]

    output = (
        OUT_DIR
        / "dhamra_vessel_schedule_current.csv"
    )

    metadata = (
        OUT_DIR
        / "dhamra_vessel_schedule_metadata.json"
    )

    df.to_csv(
        output,
        index=False,
    )

    meta = {
        "port": "Dhamra",
        "source_url": URL,
        "source_type": "official_server_rendered_schedule",
        "snapshot_time_utc": snapshot.isoformat(),
        "http_status": response.status_code,
        "html_bytes": len(html),
        "section_counts": counts,
        "total_rows": len(df),
        "sections": list(tables.keys()),
    }

    with open(
        metadata,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
            default=str,
        )

    print("=" * 80)
    print("DHAMRA SCHEDULE INGESTION")
    print("=" * 80)

    print(
        f"HTTP status: {response.status_code}"
    )

    print(
        f"HTML bytes: {len(html):,}"
    )

    print()

    for section, count in counts.items():
        print(
            f"{section:15s}: {count}"
        )

    print(
        f"\nTotal rows: {len(df)}"
    )

    print("\nCURRENT SCHEDULE:")

    print(
        df[
            [
                "section",
                "berth",
                "sbu_name",
                "vessel_name",
                "direction",
                "cargo",
                "eta",
                "ata",
                "etc",
                "actual_unberthing",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Output: {output}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
