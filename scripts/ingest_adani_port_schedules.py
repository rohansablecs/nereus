from pathlib import Path
from datetime import datetime, timezone
import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "02_DATA/processed/adani_schedules"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
}


PORTS = {
    "dhamra": {
        "port": "Dhamra",
        "url": (
            "https://www.adaniports.com/"
            "Ports-and-Terminals/dhamra-port/vesselschedule"
        ),
        "tables": {
            "at_berth": "TrackingDataTableDahejBerth",
            "at_anchorage": "TrackingDataTableDahejAnchorage",
            "expected": "TrackingDataTableDahejExpected",
            "sailed_24h": "TrackingDataTableDahejSailed",
        },
    },

    "gopalpur": {
        "port": "Gopalpur",
        "url": (
            "https://www.adaniports.com/"
            "ports-and-terminals/gopalpur-port/vesselschedule"
        ),
        "tables": {
            "at_berth": "TrackingDataTableBerth",
            "expected": "TrackingDataTableDahejExpected",
        },
    },

    "gangavaram": {
        "port": "Gangavaram",
        "url": (
            "https://www.adaniports.com/"
            "ports-and-terminals/gangavaram-port/vesselschedule"
        ),
        "tables": {},
    },
}


def clean(value):
    if value is None:
        return None

    value = " ".join(
        str(value).split()
    )

    if value.upper() in {
        "",
        "VACANT",
        "NAN",
        "NONE",
    }:
        return None

    return value


def parse_table(table, section):

    headers = [
        clean(th.get_text(" ", strip=True))
        for th in table.find_all("th")
    ]

    rows = []

    for tr in table.find_all("tr"):

        cells = [
            clean(td.get_text(" ", strip=True))
            for td in tr.find_all("td")
        ]

        if not cells:
            continue

        if len(cells) != len(headers):
            continue

        row = dict(
            zip(headers, cells)
        )

        row["section"] = section

        rows.append(row)

    return rows


def detect_empty_state(soup, section):

    section_titles = {
        "at_anchorage": "Vessels at Anchorage",
        "sailed_24h": "Vessels sailed in last 24 hours",
    }

    title = section_titles.get(
        section
    )

    if not title:
        return False

    heading = soup.find(
        lambda tag:
        tag.name in ["h1", "h2", "h3"]
        and title.lower()
        in tag.get_text(
            " ",
            strip=True
        ).lower()
    )

    if heading:

        container = heading.parent

        text = container.get_text(
            " ",
            strip=True
        )

        if re.search(
            r"No vessels",
            text,
            re.I,
        ):
            return True

    return False


def normalize_columns(df):

    rename = {
        "Berth no.": "berth",
        "Berth Number": "berth",

        "Vessels Name": "vessel_name",
        "Vessel Name": "vessel_name",

        "Imp or Exp": "direction",

        "Cargo": "cargo",

        "Expected Time of Completion (ETC)": "etc",

        "Expected Departure": "expected_departure",
        "Arrival Time": "arrival_time",

        "SBU Name": "sbu_name",
        "SBU Name.": "sbu_name",

        "ATA": "ata",
        "ETA": "eta",

        "Pilot on Board (POB)": "pilot_on_board",
        "Pilot Disembark (PD)": "pilot_disembark",
        "Actual Time of Unberthing (ATUB)": "actual_unberthing",
    }

    return df.rename(
        columns=rename
    )


def parse_datetime(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=True,
    )


def ingest_port(key, config):

    print()
    print("=" * 80)
    print(config["port"])
    print("=" * 80)

    response = requests.get(
        config["url"],
        timeout=30,
        headers=HEADERS,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    snapshot = datetime.now(
        timezone.utc
    )

    all_rows = []

    section_counts = {}

    empty_sections = []

    # ---------------------------------------------------------------
    # Parse known tables
    # ---------------------------------------------------------------

    for section, table_id in config["tables"].items():

        table = soup.find(
            "table",
            id=table_id,
        )

        if table is None:

            if detect_empty_state(
                soup,
                section,
            ):

                section_counts[
                    section
                ] = 0

                empty_sections.append(
                    section
                )

                continue

            print(
                f"WARNING: table missing: "
                f"{table_id}"
            )

            section_counts[
                section
            ] = None

            continue

        rows = parse_table(
            table,
            section,
        )

        section_counts[
            section
        ] = len(rows)

        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows
    )

    if df.empty:

        df = pd.DataFrame(
            columns=[
                "section"
            ]
        )

    df = normalize_columns(
        df
    )

    canonical = [
        "port",
        "section",
        "berth",
        "sbu_name",
        "vessel_name",
        "direction",
        "cargo",
        "eta",
        "ata",
        "arrival_time",
        "etc",
        "expected_departure",
        "pilot_on_board",
        "pilot_disembark",
        "actual_unberthing",
    ]

    for column in canonical:

        if column not in df.columns:

            df[column] = None

    # Remove duplicate columns defensively.
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # ---------------------------------------------------------------
    # Datetimes
    # ---------------------------------------------------------------

    datetime_columns = [
        "eta",
        "ata",
        "arrival_time",
        "etc",
        "expected_departure",
        "pilot_on_board",
        "pilot_disembark",
        "actual_unberthing",
    ]

    for column in datetime_columns:

        df[column] = parse_datetime(
            df[column]
        )

    df["port"] = config["port"]

    df["snapshot_time_utc"] = snapshot

    df["source_url"] = config["url"]

    df["source_type"] = (
        "official_adani_server_rendered_schedule"
    )

    df = df[
        canonical
        + [
            "snapshot_time_utc",
            "source_url",
            "source_type",
        ]
    ]

    output = (
        OUT_DIR
        / f"{key}_current.csv"
    )

    metadata = (
        OUT_DIR
        / f"{key}_metadata.json"
    )

    df.to_csv(
        output,
        index=False,
    )

    meta = {
        "port": config["port"],
        "source_url": config["url"],
        "source_type": (
            "official_adani_server_rendered_schedule"
        ),
        "snapshot_time_utc": snapshot.isoformat(),
        "http_status": response.status_code,
        "html_bytes": len(response.text),
        "section_counts": section_counts,
        "explicit_empty_sections": empty_sections,
        "total_rows": len(df),
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
        )

    print(
        f"HTTP: {response.status_code}"
    )

    print(
        f"Rows: {len(df)}"
    )

    for section, count in section_counts.items():

        print(
            f"  {section:15s}: {count}"
        )

    if empty_sections:

        print(
            "  explicit empty: "
            + ", ".join(empty_sections)
        )

    print(
        f"Output: {output}"
    )

    return df


def main():

    # Dhamra and Gopalpur are confirmed.
    ingest_port(
        "dhamra",
        PORTS["dhamra"],
    )

    ingest_port(
        "gopalpur",
        PORTS["gopalpur"],
    )


if __name__ == "__main__":
    main()
