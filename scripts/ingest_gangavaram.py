from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = "https://www.adaniports.com"
PORT_URL = f"{BASE}/Ports-and-Terminals/Gangavaram-Port"
SCHEDULE_URL = f"{PORT_URL}/vesselschedule"
DOWNLOADS_URL = f"{BASE}/Downloads"

RAW_DIR = Path("02_DATA/raw/port")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 NEREUS/0.1",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}

TIMEOUT = 60


def get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def save_csv(rows: list[dict], filename: str) -> int:
    path = RAW_DIR / filename

    if not rows:
        print(f"[WARN] No rows for {filename}")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    print(f"[OK] {filename}: {len(df)} rows")
    return len(df)


def parse_schedule() -> dict:
    print("\n=== GANGAVARAM LIVE VESSEL SCHEDULE ===")

    response = get(SCHEDULE_URL)
    soup = BeautifulSoup(response.text, "lxml")

    tables = soup.find_all("table")
    print(f"HTML tables discovered: {len(tables)}")

    sections = {
        "berth": [],
        "anchorage": [],
        "expected": [],
        "sailed_24h": [],
    }

    # Identify tables by their headers rather than table position.
    for table in tables:
        headers = [
            clean(th.get_text(" ", strip=True)).lower()
            for th in table.find_all(["th", "td"], limit=10)
        ]

        table_text = clean(table.get_text(" ", strip=True)).lower()

        rows = []

        trs = table.find_all("tr")

        for tr in trs:
            cells = [
                clean(td.get_text(" ", strip=True))
                for td in tr.find_all(["td", "th"])
            ]

            if cells:
                rows.append(cells)

        if not rows:
            continue

        header = [x.lower() for x in rows[0]]

        # Berth table
        if (
            any("berth no" in x for x in header)
            and any("vessels name" in x for x in header)
            and any("expected time" in x for x in header)
        ):
            for row in rows[1:]:
                if len(row) >= 5:
                    sections["berth"].append(
                        {
                            "berth_no": row[0],
                            "vessel_name": row[1],
                            "imp_exp": row[2],
                            "cargo": row[3],
                            "etc": row[4],
                        }
                    )

        # Anchorage table
        elif (
            any("sbu name" in x for x in header)
            and any("ata" in x for x in header)
        ):
            for row in rows[1:]:
                if len(row) >= 4:
                    sections["anchorage"].append(
                        {
                            "sbu_name": row[0],
                            "vessel_name": row[1],
                            "imp_exp": row[2],
                            "ata": row[3],
                        }
                    )

        # Expected table
        elif (
            any("sbu name" in x for x in header)
            and any("eta" in x for x in header)
        ):
            for row in rows[1:]:
                if len(row) >= 4:
                    sections["expected"].append(
                        {
                            "sbu_name": row[0],
                            "vessel_name": row[1],
                            "imp_exp": row[2],
                            "eta": row[3],
                        }
                    )

        # Sailed table
        elif (
            any("pilot on board" in x for x in header)
            and any("actual time of unberthing" in x for x in header)
        ):
            for row in rows[1:]:
                if len(row) >= 5:
                    sections["sailed_24h"].append(
                        {
                            "sbu_name": row[0],
                            "vessel_name": row[1],
                            "pilot_on_board": row[2],
                            "pilot_disembark": row[3],
                            "actual_time_unberthing": row[4],
                        }
                    )

    save_csv(
        sections["berth"],
        "gangavaram_vessel_schedule_current.csv",
    )

    save_csv(
        sections["anchorage"],
        "gangavaram_anchorage_current.csv",
    )

    save_csv(
        sections["expected"],
        "gangavaram_expected_vessels_current.csv",
    )

    save_csv(
        sections["sailed_24h"],
        "gangavaram_sailed_24h.csv",
    )

    return {
        "schedule_url": SCHEDULE_URL,
        "table_count": len(tables),
        "counts": {
            key: len(value)
            for key, value in sections.items()
        },
    }


def discover_downloads() -> dict:
    print("\n=== GANGAVARAM OFFICIAL DOWNLOAD DISCOVERY ===")

    response = get(DOWNLOADS_URL)
    soup = BeautifulSoup(response.text, "lxml")

    links = []

    for a in soup.find_all("a", href=True):
        text = clean(a.get_text(" ", strip=True))
        href = urljoin(BASE, a["href"])

        combined = f"{text} {href}".lower()

        if "gangavaram" in combined:
            links.append(
                {
                    "name": text,
                    "url": href,
                }
            )

    # Deduplicate
    unique = {}
    for item in links:
        unique[item["url"]] = item

    links = list(unique.values())

    print(f"Gangavaram-related official links: {len(links)}")

    # Separate recurring 2026 fuel/trade notices from everything else.
    trade_2026 = []

    for item in links:
        blob = f"{item['name']} {item['url']}".lower()

        if (
            "fs_gangavaram" in blob
            or "fs_gangvaram" in blob
        ) and "2026" in blob:
            trade_2026.append(item)

    save_csv(
        links,
        "gangavaram_official_download_links.csv",
    )

    save_csv(
        trade_2026,
        "gangavaram_trade_notice_links_2026.csv",
    )

    return {
        "total_gangavaram_links": len(links),
        "trade_notice_links_2026": len(trade_2026),
        "links": links,
    }


def download_bpts(discovered_links: list[dict]) -> dict:
    print("\n=== GANGAVARAM BPTS DISCOVERY ===")

    candidates = []

    for item in discovered_links:
        blob = f"{item['name']} {item['url']}".lower()

        if (
            "berthing policy" in blob
            or "bpts" in blob
            or "tariff" in blob
        ):
            candidates.append(item)

    print(f"Possible BPTS/tariff documents: {len(candidates)}")

    # Prefer an explicit Gangavaram BPTS PDF.
    bpts = None

    for item in candidates:
        blob = f"{item['name']} {item['url']}".lower()

        if "gangavaram" in blob and "bpts" in blob and ".pdf" in blob:
            bpts = item
            break

    # Known official current BPTS fallback discovered from the official site.
    if bpts is None:
        bpts = {
            "name": "Gangavaram BPTS 2025-26",
            "url": (
                "https://www.adaniports.com/"
                "-/media/Project/Ports/PortsAndTerminals/"
                "Gangavaram-Port/Tariff/"
                "Berthing-Policy-and-Tariff-structure-2025-26.pdf"
            ),
        }

    print(f"BPTS candidate: {bpts['name']}")
    print(f"URL: {bpts['url']}")

    response = get(bpts["url"])

    pdf_path = RAW_DIR / "gangavaram_bpts.pdf"
    pdf_path.write_bytes(response.content)

    print(
        f"[OK] gangavaram_bpts.pdf: "
        f"{len(response.content):,} bytes"
    )

    return bpts


def write_metadata(schedule_meta: dict, download_meta: dict, bpts: dict):
    metadata = {
        "dataset": "Gangavaram Port raw ingestion",
        "source": "Adani Gangavaram Port / Adani Ports",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "port_page": PORT_URL,
            "live_vessel_schedule": SCHEDULE_URL,
            "official_downloads": DOWNLOADS_URL,
            "bpts": bpts["url"],
        },
        "schedule": schedule_meta,
        "downloads": {
            "total_gangavaram_links": download_meta[
                "total_gangavaram_links"
            ],
            "trade_notice_links_2026": download_meta[
                "trade_notice_links_2026"
            ],
        },
        "notes": [
            "Live schedule is current operational state.",
            "BPTS is retained as source document and is not treated as live state.",
            "Trade notices are retained as update sources and must be interpreted by effective date.",
            "No missing values are invented during ingestion.",
        ],
    }

    path = RAW_DIR / "gangavaram_dataset_metadata.json"

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] {path}")


def main():
    schedule_meta = parse_schedule()
    download_meta = discover_downloads()

    bpts = download_bpts(
        download_meta["links"]
    )

    write_metadata(
        schedule_meta,
        download_meta,
        bpts,
    )

    print("\n========================================")
    print("GANGAVARAM INGESTION COMPLETE")
    print("========================================")

    print(
        json.dumps(
            schedule_meta["counts"],
            indent=2,
        )
    )

    print(
        f"Official Gangavaram links: "
        f"{download_meta['total_gangavaram_links']}"
    )

    print(
        f"2026 trade notices discovered: "
        f"{download_meta['trade_notice_links_2026']}"
    )

    print("\nNO MODELING / NO FABRICATED VALUES.")
    print("Next step: inspect the actual extracted Gangavaram data.")
    

if __name__ == "__main__":
    main()
