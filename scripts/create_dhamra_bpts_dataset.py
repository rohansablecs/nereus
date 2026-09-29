from pathlib import Path
import pandas as pd
import json

RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

SOURCE_URL = (
    "https://www.adaniports.com/-/media/project/ports/"
    "portsandterminals/dhamra-port/tariff/"
    "dhamra-bpts_dpc_01-wef-1-apr-2026.pdf"
)

# Current BPTS: berth parameters.
berths = [
    {
        "port": "Dhamra",
        "berth": "BB-1",
        "berth_category": "Dry Bulk",
        "loa_m": 300,
        "max_displacement_mt": 145000,
        "max_draft_m": 14.5,
        "cargo_scope": "Dry bulk",
    },
    {
        "port": "Dhamra",
        "berth": "BB-2",
        "berth_category": "Dry Bulk",
        "loa_m": 300,
        "max_displacement_mt": 145000,
        "max_draft_m": 14.5,
        "cargo_scope": "Import coal",
    },
    {
        "port": "Dhamra",
        "berth": "BB-3",
        "berth_category": "Dry Bulk",
        "loa_m": 200,
        "max_displacement_mt": 75000,
        "max_draft_m": 14.5,
        "cargo_scope": "Dry bulk",
    },
    {
        "port": "BB-3A",
        "berth_category": "Semi-mechanized",
        "loa_m": None,
        "max_displacement_mt": None,
        "max_draft_m": None,
        "cargo_scope": "Bulk / break-bulk",
    },
]

berth_df = pd.DataFrame(berths)

# Current BPTS explicitly published cargo-handling capabilities.
handling = [
    {
        "port": "Dhamra",
        "cargo": "Coal",
        "operation": "discharge",
        "capacity_mt_per_day": 50000,
        "qualification": "over",
        "source": "BPTS introduction",
    },
    {
        "port": "Dhamra",
        "cargo": "Iron ore",
        "operation": "loading",
        "capacity_mt_per_day": 40000,
        "qualification": "over",
        "source": "BPTS introduction",
    },
    {
        "port": "Dhamra",
        "cargo": "Rail",
        "operation": "loading",
        "capacity_mt_per_operation": None,
        "qualification": "1.25 hours per rake",
        "source": "BPTS introduction",
    },
    {
        "port": "Dhamra",
        "cargo": "Rail",
        "operation": "unloading",
        "capacity_mt_per_operation": None,
        "qualification": "3 hours per fully loaded rake",
        "source": "BPTS introduction",
    },
]

handling_df = pd.DataFrame(handling)

# Operational rules that materially affect feasibility/cost.
rules = [
    {
        "port": "Dhamra",
        "rule_id": "BPTS-001",
        "category": "policy",
        "parameter": "effective_date",
        "value": "2026-04-01",
        "unit": None,
        "description": "BPTS/DPC/07 effective 1 Apr 2026.",
    },
    {
        "port": "Dhamra",
        "rule_id": "BPTS-002",
        "category": "update_policy",
        "parameter": "trade_notice_precedence",
        "value": "subsequent_trade_circulars_apply",
        "unit": None,
        "description": "BPTS must be read with subsequent trade circulars.",
    },
    {
        "port": "Dhamra",
        "rule_id": "BPTS-003",
        "category": "season",
        "parameter": "monsoon_period",
        "value": "15 May - 30 September",
        "unit": None,
        "description": "Monsoon period defined by current BPTS.",
    },
    {
        "port": "Dhamra",
        "rule_id": "BPTS-004",
        "category": "berthing",
        "parameter": "berth_scheme",
        "value": "FCFS",
        "unit": None,
        "description": "First Come First Served is the standard berthing scheme.",
    },
    {
        "port": "Dhamra",
        "rule_id": "BPTS-005",
        "category": "operational",
        "parameter": "idling_definition",
        "value": "vessel alongside berth without cargo operations",
        "unit": None,
        "description": "Definition used by BPTS.",
    },
    {
        "port": "Dhamra",
        "rule_id": "BPTS-006",
        "category": "timing",
        "parameter": "per_day",
        "value": 24,
        "unit": "hours",
        "description": "Any part of a day is charged as a full day.",
    },
]

rules_df = pd.DataFrame(rules)

# Source/provenance manifest.
metadata = {
    "port": "Dhamra",
    "dataset": "Current BPTS operational dataset",
    "source_document": "Dhamra BPTS/DPC/07",
    "effective_date": "2026-04-01",
    "source_url": SOURCE_URL,
    "source_status": "official_current_bpts",
    "important_notes": [
        "Trade notices issued after this BPTS can change applicable rates/policies.",
        "Only directly useful structured facts are normalized.",
        "Original BPTS PDF remains the authoritative source.",
        "No unsupported vessel-class assumptions are introduced."
    ],
    "datasets": [
        "dhamra_bpts_berths.csv",
        "dhamra_bpts_handling.csv",
        "dhamra_bpts_rules.csv",
    ],
}

berth_df.to_csv(RAW / "dhamra_bpts_berths.csv", index=False)
handling_df.to_csv(RAW / "dhamra_bpts_handling.csv", index=False)
rules_df.to_csv(RAW / "dhamra_bpts_rules.csv", index=False)

with open(
    RAW / "dhamra_bpts_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

print("=" * 60)
print("Dhamra current BPTS dataset created")
print("=" * 60)
print(f"Berths:   {len(berth_df)}")
print(f"Handling: {len(handling_df)}")
print(f"Rules:    {len(rules_df)}")
print()
print("Created:")
for name in metadata["datasets"]:
    print(f"  02_DATA/raw/port/{name}")
print("  02_DATA/raw/port/dhamra_bpts_metadata.json")
print("=" * 60)
