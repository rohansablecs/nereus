from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT
    / "02_DATA/raw/commodity/CMO-Historical-Data-Monthly.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_DATA/raw/commodity/world_bank"
)

OUTPUT_FILE = OUTPUT_DIR / "world_bank_commodity_monthly.csv"
METADATA_FILE = OUTPUT_DIR / "world_bank_commodity_monthly_metadata.json"

SOURCE_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)


SERIES = {
    "Coal, Australian": {
        "commodity": "thermal_coal",
        "market": "Australia",
        "unit": "USD/mt",
        "description": (
            "Australian port thermal coal, FOB Newcastle/"
            "Port Kembla benchmark as defined by the World Bank."
        ),
    },
    "Coal, South African **": {
        "commodity": "thermal_coal",
        "market": "South_Africa",
        "unit": "USD/mt",
        "description": (
            "South African thermal coal, FOB Richards Bay "
            "benchmark as defined by the World Bank."
        ),
    },
    "Iron ore, cfr spot": {
        "commodity": "iron_ore",
        "market": "Global_CFR_China",
        "unit": "USD/dry_mt",
        "description": (
            "Iron ore fines, 62% Fe, CFR China benchmark "
            "as defined by the World Bank."
        ),
    },
    "Crude oil, Brent": {
        "commodity": "crude_oil",
        "market": "Brent",
        "unit": "USD/bbl",
        "description": (
            "UK Brent crude oil benchmark."
        ),
    },
}


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source workbook not found: {SOURCE_FILE}"
        )

    print("Loading World Bank Pink Sheet...")

    df = pd.read_excel(
        SOURCE_FILE,
        sheet_name="Monthly Prices",
        header=4,
    )

    period_column = df.columns[0]

    df = df.rename(
        columns={
            period_column: "period"
        }
    )

    missing = [
        column
        for column in SERIES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Expected World Bank series missing: "
            + ", ".join(missing)
        )

    selected_columns = [
        "period",
        *SERIES.keys(),
    ]

    prices = df[selected_columns].copy()

    # Keep only actual monthly observations.
    # The workbook contains additional rows after the
    # monthly series, so do not attempt to parse those.
    period_text = (
        prices["period"]
        .astype(str)
        .str.strip()
    )

    valid_period = period_text.str.fullmatch(
        r"\d{4}M\d{2}"
    )

    prices = prices.loc[
        valid_period
    ].copy()

    if prices.empty:
        raise ValueError(
            "No valid YYYY M MM monthly periods found."
        )

    prices["period"] = (
        prices["period"]
        .astype(str)
        .str.strip()
    )

    # World Bank period format:
    # 2026M08 -> 2026-08-01
    prices["date"] = pd.to_datetime(
        prices["period"]
        .str.replace(
            "M",
            "-",
            regex=False,
        )
        .add("-01"),
        format="%Y-%m-%d",
        errors="coerce",
    )

    if prices["date"].isna().any():
        raise ValueError(
            "A valid monthly period could not be converted to a date."
        )

    records = []

    for source_series, definition in SERIES.items():

        series = prices[
            [
                "period",
                "date",
                source_series,
            ]
        ].copy()

        series = series.rename(
            columns={
                source_series: "price"
            }
        )

        # World Bank uses "…" for unavailable values.
        series["price"] = pd.to_numeric(
            series["price"],
            errors="coerce",
        )

        series["commodity"] = definition["commodity"]
        series["market"] = definition["market"]
        series["unit"] = definition["unit"]
        series["source_series"] = source_series

        records.append(
            series[
                [
                    "date",
                    "period",
                    "commodity",
                    "market",
                    "price",
                    "unit",
                    "source_series",
                ]
            ]
        )

    result = pd.concat(
        records,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "date",
            "commodity",
            "market",
        ]
    ).reset_index(drop=True)

    # Integrity checks.
    duplicate_mask = result.duplicated(
        subset=[
            "date",
            "commodity",
            "market",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = result.loc[
            duplicate_mask
        ]

        raise ValueError(
            "Duplicate commodity/date/market records detected:\n"
            + duplicates.to_string(index=False)
        )

    if not result["price"].notna().any():
        raise ValueError(
            "No numeric commodity prices were extracted."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    metadata = {
        "dataset": "world_bank_commodity_monthly",
        "source": (
            "World Bank Commodity Price Data "
            "(The Pink Sheet)"
        ),
        "source_url": SOURCE_URL,
        "source_workbook": SOURCE_FILE.name,
        "source_sheet": "Monthly Prices",
        "frequency": "monthly",
        "price_basis": "nominal USD",
        "source_updated": "2026-09-02",
        "ingested_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "series": SERIES,
        "records": int(len(result)),
        "date_min": result["date"].min().strftime(
            "%Y-%m-%d"
        ),
        "date_max": result["date"].max().strftime(
            "%Y-%m-%d"
        ),
        "missing_prices": int(
            result["price"].isna().sum()
        ),
        "notes": [
            (
                "Australian coal series is thermal coal, "
                "not coking coal."
            ),
            (
                "South African coal series is thermal coal."
            ),
            (
                "Iron ore series represents 62% Fe fines "
                "CFR China from December 2008 onward."
            ),
            (
                "Raw World Bank workbook is preserved "
                "separately."
            ),
            (
                "Missing source observations are preserved "
                "as null values."
            ),
        ],
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nWORLD BANK COMMODITY INGESTION COMPLETE")
    print("Rows:", len(result))
    print("Columns:", len(result.columns))
    print(
        "Dates:",
        result["date"].min().date(),
        "→",
        result["date"].max().date(),
    )
    print("Output:", OUTPUT_FILE)
    print("Metadata:", METADATA_FILE)

    print("\nSERIES:")
    print(
        result.groupby(
            [
                "commodity",
                "market",
            ]
        )
        .agg(
            rows=("price", "size"),
            valid_prices=("price", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
