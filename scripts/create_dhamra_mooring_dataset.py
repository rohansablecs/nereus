from pathlib import Path
import pandas as pd
import json

RAW = Path("02_DATA/raw/port")
RAW.mkdir(parents=True, exist_ok=True)

# Structured only from the text extracted from the official
# Dhamra Port Mooring Guidelines PDF.
rules = [
    {
        "port": "Dhamra",
        "rule_id": "MOOR-001",
        "category": "mooring_equipment",
        "parameter": "annual_brake_testing",
        "requirement": "Annual brake testing of all mooring winches, including brake holding power and rendering.",
        "value": None,
        "unit": None,
        "source_document": "Dhamra Port Mooring Guidelines",
        "source_file": "dhamra_mooring_guidelines.pdf",
        "status": "source_rule",
    },
    {
        "port": "Dhamra",
        "rule_id": "MOOR-002",
        "category": "mooring_equipment",
        "parameter": "winch_brake_rendering",
        "requirement": "Winch brakes to be set to render at 60% of MBL of the mooring rope in use.",
        "value": 60,
        "unit": "% MBL",
        "source_document": "Dhamra Port Mooring Guidelines",
        "source_file": "dhamra_mooring_guidelines.pdf",
        "status": "source_rule",
    },
    {
        "port": "Dhamra",
        "rule_id": "MOOR-003",
        "category": "mooring_equipment",
        "parameter": "residual_rope_strength",
        "requirement": "Residual strength of mooring ropes in use shall not be less than 75% of the applicable requirement stated in the guideline.",
        "value": 75,
        "unit": "% MBL",
        "source_document": "Dhamra Port Mooring Guidelines",
        "source_file": "dhamra_mooring_guidelines.pdf",
        "status": "source_rule",
    },
]

df = pd.DataFrame(rules)

out = RAW / "dhamra_mooring_rules.csv"
df.to_csv(out, index=False)

metadata = {
    "port": "Dhamra",
    "dataset": "Mooring guideline rules",
    "source_document": "Dhamra Port Mooring Guidelines",
    "source_file": "dhamra_mooring_guidelines.pdf",
    "status": "source_derived",
    "notes": [
        "Only explicitly extractable textual requirements are normalized.",
        "The PDF contains a recommended mooring pattern diagram; it is not converted into fabricated numeric data.",
        "Additional text not yet normalized remains preserved in the original PDF."
    ],
    "datasets": [
        "dhamra_mooring_rules.csv"
    ]
}

with open(
    RAW / "dhamra_mooring_rules_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

print("Dhamra mooring dataset created.")
print(f"Rows: {len(df)}")
print(f"Output: {out}")
