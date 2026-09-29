from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "02_DATA/processed/feature_foundation_daily.parquet"
OUTPUT = ROOT / "02_DATA/processed/point_in_time_features.parquet"

df = pd.read_parquet(INPUT)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ============================================================
# SOURCE GROUPS
# ============================================================

freight_cols = [
    "kdci",
    "cape",
    "panamax",
    "supramax",
    "handy",
]

commodity_cols = [
    c for c in df.columns
    if c.startswith("commodity_")
]

economic_cols = [
    c for c in df.columns
    if c.startswith("econ_")
    and c != "econ_unemployment_rate_moz"
]

port_cols = [
    c for c in df.columns
    if c.startswith("port_")
]

# ============================================================
# POINT-IN-TIME BASE
#
# The foundation contains values aligned to the freight
# observation dates.
#
# For low-frequency data, we first carry the LAST KNOWN
# observation forward, then create lags from that state.
#
# This prevents a monthly commodity observation from
# disappearing simply because the freight market has several
# observations before the next commodity update.
# ============================================================

for col in commodity_cols + economic_cols:
    df[col] = df[col].ffill()

# ============================================================
# FREIGHT FEATURES
#
# Freight itself is the target market series.
# We only use previous observations.
# ============================================================

freight_features = {}

for col in freight_cols:

    s = df[col]

    freight_features[f"{col}_lag_1"] = s.shift(1)
    freight_features[f"{col}_lag_7obs"] = s.shift(7)
    freight_features[f"{col}_lag_30obs"] = s.shift(30)

    freight_features[f"{col}_mean_7obs"] = (
        s.shift(1)
        .rolling(7, min_periods=3)
        .mean()
    )

    freight_features[f"{col}_mean_30obs"] = (
        s.shift(1)
        .rolling(30, min_periods=10)
        .mean()
    )

    freight_features[f"{col}_std_30obs"] = (
        s.shift(1)
        .rolling(30, min_periods=10)
        .std()
    )

# ============================================================
# COMMODITY FEATURES
# ============================================================

commodity_features = {}

for col in commodity_cols:

    s = df[col]

    commodity_features[f"{col}_lag_1"] = s.shift(1)

    commodity_features[f"{col}_change"] = (
        s.shift(1).pct_change()
    )

    commodity_features[f"{col}_mean_3"] = (
        s.shift(1)
        .rolling(3, min_periods=2)
        .mean()
    )

# ============================================================
# ECONOMIC FEATURES
#
# Annual values are carried forward as the latest known
# state. We do NOT create arbitrary rolling statistics over
# daily repeated values.
# ============================================================

economic_features = {}

for col in economic_cols:
    economic_features[f"{col}_lag"] = df[col].shift(1)

# ============================================================
# PORTWATCH FEATURES
#
# PortWatch activity is operational context.
# It is NOT labelled as "congestion".
# ============================================================

port_features = {}

for col in port_cols:

    s = df[col]

    port_features[f"{col}_lag_1"] = s.shift(1)

    port_features[f"{col}_mean_7"] = (
        s.shift(1)
        .rolling(7, min_periods=3)
        .mean()
    )

    port_features[f"{col}_std_30"] = (
        s.shift(1)
        .rolling(30, min_periods=10)
        .std()
    )

# ============================================================
# CALENDAR FEATURES
# ============================================================

calendar_features = {
    "day_of_week": df["date"].dt.dayofweek,
    "month": df["date"].dt.month,
    "quarter": df["date"].dt.quarter,
    "month_sin": np.sin(
        2 * np.pi * df["date"].dt.month / 12
    ),
    "month_cos": np.cos(
        2 * np.pi * df["date"].dt.month / 12
    ),
}

# ============================================================
# COMBINE FEATURES IN ONE OPERATION
#
# Avoids pandas fragmentation warnings.
# ============================================================

feature_blocks = [
    pd.DataFrame(freight_features, index=df.index),
    pd.DataFrame(commodity_features, index=df.index),
    pd.DataFrame(economic_features, index=df.index),
    pd.DataFrame(port_features, index=df.index),
    pd.DataFrame(calendar_features, index=df.index),
]

features = pd.concat(feature_blocks, axis=1)

result = pd.concat(
    [
        df[
            ["date"]
            + freight_cols
            + commodity_cols
            + economic_cols
            + port_cols
        ],
        features,
    ],
    axis=1,
)

# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result = result.copy()

result.to_parquet(
    OUTPUT,
    index=False,
)

# ============================================================
# AUDIT
# ============================================================

print("=== POINT-IN-TIME FEATURE BUILD ===")
print(f"Rows: {len(result)}")
print(f"Columns: {len(result.columns)}")
print(
    f"Date range: "
    f"{result.date.min().date()} → "
    f"{result.date.max().date()}"
)

print(f"Output: {OUTPUT}")

print("\n=== SOURCE GROUPS ===")
print(f"Freight: {len(freight_cols)}")
print(f"Commodity: {len(commodity_cols)}")
print(f"Economic: {len(economic_cols)}")
print(f"PortWatch: {len(port_cols)}")

print("\n=== TARGET NULLS ===")
for col in freight_cols:
    print(
        f"{col}: "
        f"{result[col].isna().sum()}"
    )

print("\n=== LATEST COMMODITY STATE ===")

for col in commodity_cols:

    valid = result.loc[
        result[col].notna(),
        ["date", col]
    ]

    if len(valid):

        last = valid.iloc[-1]

        print(
            f"{col}: "
            f"{last['date'].date()} = "
            f"{last[col]}"
        )

print("\n=== FINAL SAMPLE ===")

sample_cols = [
    "date",
    "cape",
    "cape_lag_1",
    "cape_mean_7obs",
]

iron_col = "commodity_iron_ore_global_cfr_china"

if iron_col in result.columns:
    sample_cols += [
        iron_col,
        f"{iron_col}_lag_1",
    ]

print(
    result[sample_cols]
    .tail(10)
    .to_string(index=False)
)
