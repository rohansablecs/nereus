from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT
    / "02_DATA"
    / "raw"
    / "economic"
    / "imf"
    / "WEOApr2026all.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_DATA"
    / "raw"
    / "economic"
    / "imf"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COUNTRIES = {
    "AUS": "Australia",
    "USA": "United States",
    "MOZ": "Mozambique",
    "RUS": "Russia",
    "IDN": "Indonesia",
    "IND": "India",
}


INDICATORS = {
    "NGDP_RPCH": {
        "name": "real_gdp_growth",
        "reason": "Broad economic activity.",
    },
    "PCPIPCH": {
        "name": "cpi_inflation",
        "reason": "Macro price pressure.",
    },
    "LUR": {
        "name": "unemployment_rate",
        "reason": "Economic activity/labor-market context.",
    },
    "BCA_NGDPD": {
        "name": "current_account_pct_gdp",
        "reason": "External-balance context.",
    },
    "NGDP": {
        "name": "gdp_current_usd",
        "reason": "Economic scale.",
    },
    "TM_RPCH": {
        "name": "import_volume_growth",
        "reason": "Trade-demand context.",
    },
    "TX_RPCH": {
        "name": "export_volume_growth",
        "reason": "Export/trade-flow context.",
    },
    "TMG_RPCH": {
        "name": "import_price_growth",
        "reason": "Imported-cost pressure.",
    },
    "TXG_RPCH": {
        "name": "export_price_growth",
        "reason": "Export-price pressure.",
    },
}


def main():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"IMF source workbook not found: {SOURCE_FILE}"
        )

    print("Loading IMF April 2026 WEO workbook...")

    countries = pd.read_excel(
        SOURCE_FILE,
        sheet_name="Countries",
    )

    print(
        f"Source rows: {len(countries):,}"
    )

    required_columns = {
        "COUNTRY.ID",
        "COUNTRY",
        "INDICATOR.ID",
        "INDICATOR",
        "INDICATOR.Description",
        "FREQUENCY",
        "UNIT",
        "LATEST_ACTUAL_ANNUAL_DATA",
        "BASIS_OF_PROJECTIONS",
    }

    missing = (
        required_columns
        - set(countries.columns)
    )

    if missing:
        raise ValueError(
            "Missing expected IMF columns: "
            + ", ".join(sorted(missing))
        )

    year_columns = [
        column
        for column in countries.columns
        if isinstance(column, int)
        and 1980 <= column <= 2031
    ]

    if not year_columns:
        raise ValueError(
            "No IMF annual columns from 1980-2031 found."
        )

    selected = countries[
        countries["COUNTRY.ID"].isin(
            COUNTRIES.keys()
        )
        &
        countries["INDICATOR.ID"].isin(
            INDICATORS.keys()
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No matching country/indicator rows found."
        )

    print(
        f"Selected source rows: {len(selected):,}"
    )

    records = []

    for _, row in selected.iterrows():

        country_code = row["COUNTRY.ID"]
        indicator_code = row["INDICATOR.ID"]

        definition = INDICATORS[
            indicator_code
        ]

        for year in year_columns:

            value = row[year]

            if pd.isna(value):
                continue

            records.append(
                {
                    "year": int(year),
                    "country_code": country_code,
                    "country": COUNTRIES[
                        country_code
                    ],
                    "indicator": definition["name"],
                    "indicator_code": indicator_code,
                    "value": float(value),
                    "unit": row["UNIT"],
                    "frequency": row["FREQUENCY"],
                    "latest_actual_data": (
                        str(row["LATEST_ACTUAL_ANNUAL_DATA"])
                        if pd.notna(
                            row["LATEST_ACTUAL_ANNUAL_DATA"]
                        )
                        else None
                    ),
                    "basis_of_projections": (
                        row["BASIS_OF_PROJECTIONS"]
                        if pd.notna(
                            row["BASIS_OF_PROJECTIONS"]
                        )
                        else None
                    ),
                    "source": "IMF WEO",
                    "release": "April 2026",
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No valid economic observations were produced."
        )

    df = df.sort_values(
        [
            "country_code",
            "indicator",
            "year",
        ]
    ).reset_index(drop=True)

    duplicates = df.duplicated(
        subset=[
            "country_code",
            "indicator",
            "year",
        ]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate observations."
        )

    def classify_observation(row):
        latest = row["latest_actual_data"]

        if pd.isna(latest):
            return "unclassified"

        latest_text = str(latest).strip()

        # Calendar-year value supplied directly by IMF.
        if latest_text.isdigit():
            return (
                "historical"
                if row["year"] <= int(latest_text)
                else "projection"
            )

        # Fiscal-year metadata such as FY2024/25
        # cannot be safely converted to a calendar year.
        # Preserve it without inventing a conversion.
        if latest_text.startswith("FY"):
            return "fiscal_year_reference"

        return "unclassified"

    df["observation_type"] = (
        df.apply(
            classify_observation,
            axis=1,
        )
    )

    output_csv = (
        OUTPUT_DIR
        / "imf_weo_economic_annual.csv"
    )

    output_metadata = (
        OUTPUT_DIR
        / "imf_weo_economic_annual_metadata.json"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    metadata = {
        "dataset": "IMF World Economic Outlook",
        "release": "April 2026",
        "source_file": SOURCE_FILE.name,
        "source_url": (
            "https://data.imf.org/Datasets/WEO"
        ),
        "frequency": "annual",
        "countries": COUNTRIES,
        "indicators": INDICATORS,
        "year_range": [
            int(df["year"].min()),
            int(df["year"].max()),
        ],
        "observation_types": [
            "historical",
            "projection",
        ],
        "notes": [
            "Original IMF workbook is preserved in the raw layer.",
            "Only NEREUS-relevant countries and indicators are normalized.",
            "Missing IMF observations are preserved as missing.",
            "IMF projections are explicitly labelled and are not treated as historical observations.",
            "No synthetic or interpolated economic observations are created.",
        ],
    }

    output_metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("IMF ECONOMIC INGESTION COMPLETE")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Years: {df['year'].min()} → "
        f"{df['year'].max()}"
    )
    print(
        f"Countries: "
        f"{df['country_code'].nunique()}"
    )
    print(
        f"Indicators: "
        f"{df['indicator'].nunique()}"
    )
    print()
    print("OBSERVATION TYPES:")
    print(
        df["observation_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("COUNTRY COVERAGE:")
    print(
        df.groupby("country")
        .agg(
            rows=("value", "size"),
            indicators=("indicator", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .to_string()
    )

    print()
    print("INDICATOR COVERAGE:")
    print(
        df.groupby("indicator")
        .agg(
            rows=("value", "size"),
            countries=("country_code", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .to_string()
    )

    print()
    print("OUTPUT:")
    print(output_csv)
    print(output_metadata)


if __name__ == "__main__":
    main()
