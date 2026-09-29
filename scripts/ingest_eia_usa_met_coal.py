from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "02_DATA" / "raw" / "trade" / "usa" / "eia"
OUT = ROOT / "02_DATA" / "processed" / "trade" / "usa"

OUT.mkdir(parents=True, exist_ok=True)


FILES = {
    "exports_destination": RAW / "table_11_met_coal_exports.xlsx",
    "prices_destination": RAW / "table_12_met_coal_export_prices.xlsx",
    "exports_customs": RAW / "table_13_coal_exports_customs_district.xlsx",
    "met_exports_customs": RAW / "table_15_met_coal_exports_customs_district.xlsx",
}


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()

        if value in {"-", "—", ""}:
            return None

        value = value.replace(",", "")

        try:
            return float(value)
        except ValueError:
            return value

    return value


def read_eia_table(path):
    df = pd.read_excel(path, header=None)

    header_idx = None

    for i in range(len(df)):
        row = df.iloc[i].tolist()

        # Convert every cell to text before searching.
        row_text = [
            "" if pd.isna(x) else str(x)
            for x in row
        ]

        if any("January-March" in x for x in row_text):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(
            f"Could not find EIA header row in {path}"
        )

    header = df.iloc[header_idx].tolist()

    data = df.iloc[header_idx + 1:].copy()
    data.columns = header

    data = data.dropna(how="all")

    # First column = destination/customs district.
    first_col = data.columns[0]
    data = data.rename(columns={first_col: "entity"})

    normalized_columns = []

    for col in data.columns:
        if pd.isna(col):
            normalized_columns.append("unknown")
        else:
            normalized_columns.append(
                str(col).strip()
            )

    data.columns = normalized_columns

    return data


def parse_period_columns(df):
    result = pd.DataFrame()

    result["entity"] = (
        df["entity"]
        .astype(str)
        .str.strip()
    )

    period_map = {}

    for col in df.columns:
        text = str(col).replace("\n", " ").strip()

        if (
            text.startswith("January-March")
            and "2026" in text
        ):
            period_map["q1_2026"] = col

        elif (
            text.startswith("October-December")
            and "2025" in text
        ):
            period_map["q4_2025"] = col

        elif (
            text.startswith("January-March")
            and "2025" in text
        ):
            period_map["q1_2025"] = col

        elif text == "2026":
            period_map["ytd_2026"] = col

        elif text in {"2025", "2025.0"}:
            period_map["ytd_2025"] = col

        elif "Percent" in text:
            period_map["percent_change"] = col

    for name, col in period_map.items():
        result[name] = df[col].map(clean_value)

    return result


def finalize(df, metric, unit, geography_level):
    df["metric"] = metric
    df["unit"] = unit
    df["geography_level"] = geography_level

    df = df[
        df["entity"].notna()
        & ~df["entity"].str.lower().eq("nan")
    ]

    return df


def save_table(source_key, filename, metric, unit, geography_level):
    print()
    print(f"Reading {FILES[source_key].name}")

    df = read_eia_table(FILES[source_key])
    out = parse_period_columns(df)

    out = finalize(
        out,
        metric=metric,
        unit=unit,
        geography_level=geography_level,
    )

    path = OUT / filename
    out.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Rows: {len(out)}")
    print(f"Columns: {list(out.columns)}")


def main():
    print("=" * 80)
    print("NEREUS — EIA U.S. METALLURGICAL COAL INGESTION")
    print("=" * 80)

    for key, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"{key}: {path.name}")

    save_table(
        "exports_destination",
        "eia_met_coal_exports_by_destination.csv",
        "us_met_coal_exports",
        "short_tons",
        "destination_country",
    )

    save_table(
        "prices_destination",
        "eia_met_coal_export_prices_by_destination.csv",
        "us_met_coal_export_price",
        "usd_per_short_ton",
        "destination_country",
    )

    save_table(
        "exports_customs",
        "eia_coal_exports_by_customs_district.csv",
        "us_coal_exports",
        "short_tons",
        "customs_district",
    )

    save_table(
        "met_exports_customs",
        "eia_met_coal_exports_by_customs_district.csv",
        "us_met_coal_exports",
        "short_tons",
        "customs_district",
    )

    print()
    print("=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
