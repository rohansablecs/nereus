from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "02_DATA/processed/forecast"

vessels = ["cape", "panamax", "supramax", "handy"]
horizons = [7, 14, 30, 60]

print("=== FORECAST HORIZON CALENDAR-TIME AUDIT ===\n")

for vessel in vessels:
    print(f"--- {vessel.upper()} ---")

    for horizon in horizons:

        path = DATA_DIR / f"{vessel}_{horizon}obs.parquet"
        df = pd.read_parquet(path)

        df["date"] = pd.to_datetime(df["date"])
        df["target_date"] = pd.to_datetime(df["target_date"])

        calendar_days = (
            df["target_date"] - df["date"]
        ).dt.total_seconds() / 86400

        print(
            f"{horizon:2d} observations | "
            f"n={len(calendar_days):4d} | "
            f"min={calendar_days.min():5.1f}d | "
            f"median={calendar_days.median():5.1f}d | "
            f"mean={calendar_days.mean():5.1f}d | "
            f"p90={calendar_days.quantile(.90):5.1f}d | "
            f"max={calendar_days.max():5.1f}d"
        )

    print()

# ============================================================
# CHECK THE ACTUAL DATE GAPS
# ============================================================

foundation = pd.read_parquet(
    ROOT / "02_DATA/processed/point_in_time_features.parquet"
)

foundation["date"] = pd.to_datetime(foundation["date"])

gaps = foundation["date"].diff().dt.days.dropna()

print("=== MARKET OBSERVATION GAP DISTRIBUTION ===")
print(f"Observations: {len(foundation):,}")
print(f"Median gap: {gaps.median():.1f} days")
print(f"Mean gap: {gaps.mean():.1f} days")
print(f"P90 gap: {gaps.quantile(.90):.1f} days")
print(f"Maximum gap: {gaps.max():.1f} days")

print("\nGap counts:")
print(
    gaps.round()
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)

# ============================================================
# CHECK TARGET RETURN DISTRIBUTION
# ============================================================

print("\n=== TARGET RETURN DISTRIBUTION ===")

for vessel in vessels:

    path = DATA_DIR / f"{vessel}_30obs.parquet"
    df = pd.read_parquet(path)

    r = df["target_return"].dropna()

    print(
        f"{vessel:9s} | "
        f"median={r.median():+.3%} | "
        f"mean={r.mean():+.3%} | "
        f"std={r.std():.3%} | "
        f"p05={r.quantile(.05):+.3%} | "
        f"p95={r.quantile(.95):+.3%}"
    )
