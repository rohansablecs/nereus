from pathlib import Path
from datetime import datetime, timezone
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

BERTH_URL = "https://vizagport.com/Template/navigateTemplate/gnt/QmVydGhz"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})


def fetch(url, timeout=(15, 60)):
    last_error = None

    for attempt in range(1, 4):
        try:
            print(f"Fetching: {url}")
            print(f"Attempt {attempt}/3")

            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            print(f"HTTP {response.status_code}")
            return response

        except requests.RequestException as exc:
            last_error = exc
            print(f"Failed: {exc}")

            if attempt < 3:
                time.sleep(5)

    raise RuntimeError(f"Could not fetch {url}") from last_error


# =========================================================
# 1. CURRENT VPA BERTH PARTICULARS
# =========================================================

response = fetch(BERTH_URL)
soup = BeautifulSoup(response.text, "lxml")

tables = soup.find_all("table")

print(f"HTML tables found: {len(tables)}")

berth_table = None

for table in tables:
    text = table.get_text(" ", strip=True).lower()

    if (
        "permissible loa" in text
        and "permissible draft" in text
        and "cargo" in text
    ):
        berth_table = table
        break

if berth_table is None:
    raise RuntimeError(
        "Could not identify the VPA berth particulars table."
    )

rows = []

for tr in berth_table.find_all("tr"):
    cells = tr.find_all(["th", "td"])

    if not cells:
        continue

    values = [cell.get_text(" ", strip=True) for cell in cells]

    if any(values):
        rows.append(values)

header_index = None

for i, row in enumerate(rows):
    joined = " ".join(row).lower()

    if (
        "berth" in joined
        and "permissible" in joined
    ):
        header_index = i
        break

if header_index is None:
    raise RuntimeError(
        "Could not identify berth table header."
    )

header = rows[header_index]

data_rows = [
    row
    for row in rows[header_index + 1:]
    if len(row) == len(header)
]

berths = pd.DataFrame(
    data_rows,
    columns=header
)

berths.columns = [
    re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().lower()
    ).strip("_")
    for column in berths.columns
]

berths.insert(0, "port", "Visakhapatnam")
berths.insert(1, "source_url", BERTH_URL)
berths.insert(
    2,
    "snapshot_time_utc",
    datetime.now(timezone.utc).isoformat(),
)

berth_out = RAW / "vizag_berths_current.csv"
berths.to_csv(berth_out, index=False)

print(f"Current berth rows: {len(berths)}")


# =========================================================
# 2. FIND LATEST BERTHING PROGRAMME
# =========================================================

programme_candidates = []

for tr in soup.find_all("tr"):

    row_text = tr.get_text(" ", strip=True)

    if "Berthing Programme Dt." not in row_text:
        continue

    match = re.search(
        r"Berthing Programme Dt\.\s*(\d{2}-\d{2}-\d{4})",
        row_text,
        re.IGNORECASE,
    )

    if not match:
        continue

    date_text = match.group(1)

    try:
        programme_date = datetime.strptime(
            date_text,
            "%d-%m-%Y",
        )
    except ValueError:
        continue

    link = tr.find("a", href=True)

    if not link:
        continue

    href = link["href"].strip()

    if href.startswith("http"):
        programme_url = href

    elif href.startswith("/"):
        programme_url = "https://vizagport.com" + href

    else:
        programme_url = "https://vizagport.com/" + href.lstrip("/")

    programme_candidates.append(
        (
            programme_date,
            programme_url,
            row_text,
        )
    )


if not programme_candidates:
    raise RuntimeError(
        "VPA page contains no downloadable Berthing Programme links."
    )


programme_candidates.sort(
    key=lambda x: x[0],
    reverse=True,
)

programme_date, programme_url, programme_label = (
    programme_candidates[0]
)

print()
print("Latest VPA Berthing Programme:")
print(programme_label)
print(programme_url)


# =========================================================
# 3. DOWNLOAD PROGRAMME PDF
# =========================================================

pdf_response = fetch(
    programme_url,
    timeout=(15, 90),
)

content_type = (
    pdf_response.headers
    .get("content-type", "")
    .lower()
)

if "pdf" not in content_type and not pdf_response.content.startswith(b"%PDF"):
    raise RuntimeError(
        "Downloaded programme is not a PDF."
    )

pdf_path = RAW / (
    "vizag_berthing_programme_"
    f"{programme_date.strftime('%Y-%m-%d')}.pdf"
)

pdf_path.write_bytes(pdf_response.content)

print(
    f"Downloaded PDF: {pdf_path}"
)

print(
    f"PDF size: {len(pdf_response.content):,} bytes"
)


# =========================================================
# 4. EXTRACT PDF TEXT
# =========================================================

reader = PdfReader(str(pdf_path))

pages = []

for page in reader.pages:
    pages.append(page.extract_text() or "")

programme_text = "\n\n".join(pages)

text_path = RAW / (
    "vizag_berthing_programme_"
    f"{programme_date.strftime('%Y-%m-%d')}.txt"
)

text_path.write_text(
    programme_text,
    encoding="utf-8",
)

print(
    f"Programme pages: {len(reader.pages)}"
)

print(
    f"Extracted characters: {len(programme_text):,}"
)


# =========================================================
# 5. METADATA
# =========================================================

metadata = {
    "port": "Visakhapatnam",
    "berth_source": BERTH_URL,
    "programme_source": programme_url,
    "programme_date": programme_date.strftime(
        "%Y-%m-%d"
    ),
    "retrieved_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "berth_rows": len(berths),
    "programme_pdf_pages": len(reader.pages),
    "programme_text_characters": len(programme_text),
    "datasets": [
        "vizag_berths_current.csv",
        pdf_path.name,
        text_path.name,
    ],
    "notes": [
        "Berth particulars are current VPA website data.",
        "Berthing programme is a point-in-time operational snapshot.",
        "Original programme PDF is retained.",
        "No historical congestion values are inferred.",
    ],
}

with open(
    RAW / "vizag_dataset_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


# =========================================================
# DONE
# =========================================================

print()
print("=" * 65)
print("VISAKHAPATNAM INGESTION COMPLETE")
print("=" * 65)
print(
    f"Current berth rows:   {len(berths)}"
)
print(
    f"Programme date:       {programme_date:%Y-%m-%d}"
)
print(
    f"Programme PDF pages:  {len(reader.pages)}"
)
print(
    f"Programme text chars: {len(programme_text):,}"
)
print()
print("Created:")
print(f"  {berth_out}")
print(f"  {pdf_path}")
print(f"  {text_path}")
print(
    f"  {RAW / 'vizag_dataset_metadata.json'}"
)
print("=" * 65)
