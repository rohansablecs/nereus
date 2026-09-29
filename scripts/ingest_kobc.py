from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import requests


BASE_URL = "https://www.kobc.or.kr"
DOWNLOAD_URL = (
    f"{BASE_URL}/ebz/shippinginfoeng/kdci/"
    "excel/download.do?mId=0301000000"
)
REFERER_URL = (
    f"{BASE_URL}/ebz/shippinginfoeng/kdci/"
    "gridList.do?mId=0301000000"
)

START_DATE = "2019-01-01"
END_DATE = "2026-09-08"

OUTPUT_DIR = Path("02_DATA/raw/freight/kobc")
RAW_FILE = OUTPUT_DIR / "kobc_drybulk_daily_raw.xls"
CSV_FILE = OUTPUT_DIR / "kobc_drybulk_daily.csv"
METADATA_FILE = OUTPUT_DIR / "kobc_drybulk_daily_metadata.json"


def download():
    response = requests.post(
        DOWNLOAD_URL,
        data={
            "page": "1",
            "sDay": START_DATE,
            "eDay": END_DATE,
        },
        headers={
            "User-Agent": "NEREUS/0.1",
            "Referer": REFERER_URL,
        },
        timeout=60,
    )

    response.raise_for_status()

    if not response.content.startswith(b"\xd0\xcf\x11\xe0"):
        raise RuntimeError(
            "KOBC response does not appear to be an XLS workbook."
        )

    RAW_FILE.write_bytes(response.content)
    return response


def parse():
    df = pd.read_excel(
        RAW_FILE,
        sheet_name="sheet1",
        header=None,
        engine="xlrd",
    )

    # Row 0 = title.
    # Row 1 = actual column names.
    df = df.iloc[1:].copy()

    df.columns = [
        "record_number",
        "date",
        "kdci",
        "cape",
        "panamax",
        "supramax",
        "handy",
    ]

    df = df.reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "record_number",
        "kdci",
        "cape",
        "panamax",
        "supramax",
        "handy",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove completely invalid rows, if any.
    df = df.dropna(subset=["date"])

    # Sort chronologically for modelling.
    df = df.sort_values("date").reset_index(drop=True)

    return df


def validate(df):
    required = [
        "date",
        "kdci",
        "cape",
        "panamax",
        "supramax",
        "handy",
    ]

    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {missing_columns}"
        )

    if df["date"].duplicated().any():
        duplicates = df[df["date"].duplicated(keep=False)]
        raise RuntimeError(
            f"Duplicate dates found:\n{duplicates}"
        )

    for column in required[2:]:
        if df[column].isna().any():
            raise RuntimeError(
                f"Missing values found in {column}"
            )

        if (df[column] <= 0).any():
            raise RuntimeError(
                f"Non-positive values found in {column}"
            )

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError("Dates are not sorted.")

    print("\n=== KOBC VALIDATION ===")
    print(f"Observations : {len(df):,}")
    print(f"First date   : {df['date'].min().date()}")
    print(f"Last date    : {df['date'].max().date()}")
    print(f"Duplicates   : {df['date'].duplicated().sum()}")
    print(f"Columns      : {list(df.columns)}")

    print("\nLatest observation:")
    print(df.tail(1).to_string(index=False))


def save(df, response):
    # Store dates as ISO dates.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df.to_csv(CSV_FILE, index=False)

    metadata = {
        "dataset": "KOBC Dry Bulk Index",
        "provider": "Korea Ocean Business Corporation (KOBC)",
        "source_page": REFERER_URL,
        "download_endpoint": DOWNLOAD_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start_date": START_DATE,
        "requested_end_date": END_DATE,
        "actual_start_date": df["date"].min(),
        "actual_end_date": df["date"].max(),
        "observations": int(len(df)),
        "frequency": "business/assessment days as provided by source",
        "units": {
            "kdci": "USD/day",
            "cape": "USD/day",
            "panamax": "USD/day",
            "supramax": "USD/day",
            "handy": "USD/day",
        },
        "columns": list(df.columns),
        "raw_file": RAW_FILE.name,
        "content_disposition": response.headers.get(
            "content-disposition"
        ),
        "notes": [
            "Source workbook is preserved unchanged.",
            "No weekend/holiday observations were fabricated.",
            "Data was sorted chronologically after ingestion.",
        ],
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading KOBC historical dry bulk index...")
    response = download()

    print(f"Raw XLS saved: {RAW_FILE}")
    print(f"Bytes: {len(response.content):,}")

    df = parse()
    validate(df)
    save(df, response)

    print("\n=== OUTPUT ===")
    print(CSV_FILE)
    print(METADATA_FILE)


if __name__ == "__main__":
    main()
