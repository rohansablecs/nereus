from pathlib import Path
import pandas as pd
import json


RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)


SOURCE = "dhamra_port_information.pdf"


# ============================================================
# 1. PORT PROFILE
# ============================================================

port_profile = pd.DataFrame([
    {
        "port": "Dhamra",
        "channel_length_km": 18.0,
        "outer_channel_depth_m": 17.5,
        "outer_channel_width_m": 240.0,
        "inner_channel_width_m": 170.0,
        "turning_basin_diameter_m": 600.0,
        "source_document": SOURCE,
    }
])

port_profile.to_csv(
    RAW / "dhamra_port_profile.csv",
    index=False
)


# ============================================================
# 2. BERTH / TERMINAL STRUCTURE
# ============================================================

berths = pd.DataFrame([
    {
        "berth": "BB1",
        "berth_group": "Bulk Berth",
        "source_document": SOURCE,
    },
    {
        "berth": "BB2",
        "berth_group": "Bulk Berth",
        "source_document": SOURCE,
    },
    {
        "berth": "BB3",
        "berth_group": "Bulk Berth",
        "source_document": SOURCE,
    },
    {
        "berth": "BB3A",
        "berth_group": "Bulk Berth",
        "source_document": SOURCE,
    },
])

berths.to_csv(
    RAW / "dhamra_berths.csv",
    index=False
)


# ============================================================
# 3. CARGO HANDLING CAPABILITY
# ============================================================
#
# These are specifically documented cargo-handling systems.
# Do NOT treat rail loading capacity as vessel loading rate.
#

handling = pd.DataFrame([
    {
        "facility": "Rail Loading System 1",
        "cargo": "Bulk cargo",
        "equipment": "Rapid Loading Silo",
        "capacity_tph": 2000,
        "operation": "Rail loading",
        "source": "Official Dhamra Port profile",
    },
    {
        "facility": "Rail Loading System 2",
        "cargo": "Bulk cargo",
        "equipment": "Rapid Loading Silo",
        "capacity_tph": 2000,
        "operation": "Rail loading",
        "source": "Official Dhamra Port profile",
    },
    {
        "facility": "Rail Loading System 3",
        "cargo": "Bulk cargo",
        "equipment": "Rapid Loading Silo",
        "capacity_tph": 4000,
        "operation": "Rail loading",
        "source": "Official Dhamra Port profile",
    },
    {
        "facility": "Rail Loading System 4",
        "cargo": "Bulk cargo",
        "equipment": "Rapid Loading Silo",
        "capacity_tph": 4000,
        "operation": "Rail loading",
        "source": "Official Dhamra Port profile",
    },
])

handling.to_csv(
    RAW / "dhamra_handling_capabilities.csv",
    index=False
)


# ============================================================
# 4. OPERATIONAL / NAVIGATION CONSTRAINTS
# ============================================================

navigation = pd.DataFrame([
    {
        "parameter": "Outer channel depth",
        "value": 17.5,
        "unit": "m",
        "constraint_type": "navigation",
        "source_document": SOURCE,
    },
    {
        "parameter": "Outer channel width",
        "value": 240,
        "unit": "m",
        "constraint_type": "navigation",
        "source_document": SOURCE,
    },
    {
        "parameter": "Inner channel width",
        "value": 170,
        "unit": "m",
        "constraint_type": "navigation",
        "source_document": SOURCE,
    },
    {
        "parameter": "Turning basin diameter",
        "value": 600,
        "unit": "m",
        "constraint_type": "navigation",
        "source_document": SOURCE,
    },
    {
        "parameter": "Navigation authority",
        "value": "Port authority final decision on vessel movements",
        "unit": None,
        "constraint_type": "operational_rule",
        "source_document": SOURCE,
    },
])

navigation.to_csv(
    RAW / "dhamra_navigation_constraints.csv",
    index=False
)


# ============================================================
# 5. DATASET MANIFEST
# ============================================================

manifest = {
    "port": "Dhamra",
    "source_documents": [
        SOURCE,
        "official Dhamra Port profile"
    ],
    "datasets": [
        "dhamra_port_profile.csv",
        "dhamra_berths.csv",
        "dhamra_handling_capabilities.csv",
        "dhamra_navigation_constraints.csv",
    ],
    "principles": [
        "Original source PDF retained separately",
        "No unsupported values invented",
        "Rail loading capacity is not treated as vessel loading rate",
        "Historical draft declarations are kept separate from current constraints",
    ]
}

with open(
    RAW / "dhamra_dataset_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=2)


print()
print("========================================")
print("Dhamra dataset creation complete")
print("========================================")
print(f"Port profile:       {len(port_profile)} rows")
print(f"Berths:             {len(berths)} rows")
print(f"Handling:           {len(handling)} rows")
print(f"Navigation:         {len(navigation)} rows")
print()
print("Created:")
for filename in manifest["datasets"]:
    print(f"  02_DATA/raw/port/{filename}")
print("  02_DATA/raw/port/dhamra_dataset_metadata.json")
print("========================================")
