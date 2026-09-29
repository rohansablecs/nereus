from pathlib import Path
import pandas as pd


ROOT = Path("02_DATA/raw")

# Only datasets relevant to the first unified feature table.
FILES = [
    ROOT / "freight/kobc/kobc_drybulk_daily.csv",

    ROOT / "port/portwatch_daily_ports.csv",
    ROOT / "port/portwatch_disruptions.csv",

    ROOT / "port/paradip_berths.csv",

    ROOT / "port/dhamra_berths.csv",
    ROOT / "port/dhamra_bpts_berths.csv",
    ROOT / "port/dhamra_handling_capabilities.csv",
    ROOT / "port/dhamra_navigation_constraints.csv",
    ROOT / "port/dhamra_draft_declarations_historical.csv",

    ROOT / "port/gopalpur_berths.csv",

    ROOT / "port/gangavaram_berths.csv",
    ROOT / "port/gangavaram_vessel_schedule_current.csv",
    ROOT / "port/gangavaram_expected_vessels_current.csv",

    ROOT / "port/vizag_berths_current.csv",
    ROOT / "port/vizag_waiting_expected_current.csv",
    ROOT / "port/vizag_berth_occupancy_current.csv",

    ROOT / "port/haldia_berth_position_current.csv",
]


def inspect(path: Path):
    print("\n" + "=" * 90)
    print(path)

    if not path.exists():
        print("STATUS: MISSING")
        return

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"STATUS: READ ERROR — {exc}")
        return

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Columns    : {list(df.columns)}")

    date_columns = [
        c for c in df.columns
        if any(x in c.lower() for x in ["date", "time", "day", "arrival", "departure", "etc", "ata"])
    ]

    if date_columns:
        print(f"Date-like  : {date_columns}")

        for col in date_columns:
            parsed = pd.to_datetime(df[col], errors="coerce")

            valid = parsed.notna()

            if valid.any():
                print(
                    f"  {col}: "
                    f"{parsed[valid].min()} → {parsed[valid].max()} "
                    f"({valid.sum():,} parseable)"
                )

    missing = df.isna().sum()

    if missing.any():
        print("Missing:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count:,} ({count / len(df) * 100:.1f}%)")

    print("\nSample:")
    print(df.head(3).to_string(index=False))


def main():
    print("=" * 90)
    print("NEREUS — FEATURE SOURCE PROFILER")
    print("=" * 90)

    for path in FILES:
        inspect(path)


if __name__ == "__main__":
    main()
