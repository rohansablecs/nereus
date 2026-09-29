from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "02_DATA" / "raw" / "port" / "paradip" / "daily_traffic"
METADATA_PATH = OUT_DIR / "download_metadata.json"

SOURCE_URL = "https://paradipport.gov.in/traffic/"

HEADERS = {
    "User-Agent": "NEREUS/0.1",
    "Accept": "text/html,application/xhtml+xml",
}

TIMEOUT = 30


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.get(SOURCE_URL, timeout=TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for anchor in soup.find_all("a", href=True):
        href = urljoin(SOURCE_URL, anchor["href"])
        text = " ".join(anchor.get_text(" ", strip=True).split())

        if ".pdf" not in href.lower():
            continue

        # Only daily traffic reports.
        # Exclude monthly/major-port reports.
        if "dtr" not in href.lower():
            continue

        links.append(
            {
                "text": text,
                "url": href,
            }
        )

    # Deduplicate while preserving order.
    seen = set()
    unique_links = []

    for item in links:
        if item["url"] in seen:
            continue

        seen.add(item["url"])
        unique_links.append(item)

    print(f"Discovered daily traffic PDFs: {len(unique_links)}")

    records = []

    for index, item in enumerate(unique_links, start=1):
        url = item["url"]
        filename = Path(url.split("?")[0]).name

        # Preserve original filename.
        output_path = OUT_DIR / filename

        print(f"[{index:03d}/{len(unique_links):03d}] {filename}")

        if output_path.exists() and output_path.stat().st_size > 0:
            data = output_path.read_bytes()
            status = "already_present"
        else:
            pdf_response = session.get(url, timeout=TIMEOUT)
            pdf_response.raise_for_status()

            data = pdf_response.content
            output_path.write_bytes(data)
            status = "downloaded"

        records.append(
            {
                "filename": filename,
                "url": url,
                "source_page": SOURCE_URL,
                "status": status,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )

    metadata = {
        "source": "Paradip Port Authority",
        "source_page": SOURCE_URL,
        "description": "Official daily traffic report PDFs discovered from the Paradip Port Authority traffic page.",
        "discovered_count": len(unique_links),
        "records": records,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 50)
    print("PARADIP DAILY TRAFFIC DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Reports discovered: {len(unique_links)}")
    print(f"Directory:          {OUT_DIR}")
    print(f"Metadata:           {METADATA_PATH}")


if __name__ == "__main__":
    main()