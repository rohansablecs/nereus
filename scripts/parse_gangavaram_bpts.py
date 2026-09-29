from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PDF = Path("02_DATA/raw/port/gangavaram_bpts.pdf")
OUT = Path("02_DATA/raw/port")
OUT.mkdir(parents=True, exist_ok=True)


def save(df, filename):
    path = OUT / filename
    df.to_csv(path, index=False)
    print(f"[OK] {filename}: {len(df)} rows")


def extract_pages():
    reader = PdfReader(PDF)
    return [
        reader.pages[i].extract_text() or ""
        for i in range(len(reader.pages))
    ]


def parse_berths(pages):
    # Page 19 of the PDF = index 18.
    text = pages[18]

    rows = [
        {
            "port": "Gangavaram",
            "berth_no": "B1",
            "designed_depth_m": 14.0,
            "permissible_draft_m": 13.0,
            "berth_length_m": 275,
            "permissible_loa_m": 230,
            "displacement_mt": 65000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B2",
            "designed_depth_m": 15.5,
            "permissible_draft_m": 14.5,
            "berth_length_m": 280,
            "permissible_loa_m": 230,
            "displacement_mt": 98000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B3",
            "designed_depth_m": 15.5,
            "permissible_draft_m": 15.0,
            "berth_length_m": 280,
            "permissible_loa_m": 230,
            "displacement_mt": 98000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B4",
            "designed_depth_m": 19.5,
            "permissible_draft_m": 18.0,
            "berth_length_m": 340,
            "permissible_loa_m": 300,
            "displacement_mt": 236000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B5",
            "designed_depth_m": 19.5,
            "permissible_draft_m": 18.0,
            "berth_length_m": 320,
            "permissible_loa_m": 292,
            "displacement_mt": 236000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B6",
            "designed_depth_m": 19.5,
            "permissible_draft_m": 18.0,
            "berth_length_m": 355,
            "permissible_loa_m": 300,
            "displacement_mt": 236000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B7",
            "designed_depth_m": 15.5,
            "permissible_draft_m": 14.5,
            "berth_length_m": 235,
            "permissible_loa_m": 200,
            "displacement_mt": 98000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B8",
            "designed_depth_m": 15.5,
            "permissible_draft_m": 14.5,
            "berth_length_m": 275,
            "permissible_loa_m": 230,
            "displacement_mt": 98000,
        },
        {
            "port": "Gangavaram",
            "berth_no": "B9",
            "designed_depth_m": 15.5,
            "permissible_draft_m": 15.5,
            "berth_length_m": 337,
            "permissible_loa_m": 310,
            "displacement_mt": 98000,
        },
    ]

    return pd.DataFrame(rows)


def parse_port_profile(pages):
    return pd.DataFrame(
        [
            {
                "port": "Gangavaram",
                "capacity_mmtpa": 60,
                "harbour_depth_m": 20.2,
                "max_vessel_dwt_mt": 200000,
                "operations": "24x7",
                "berth_count": 9,
                "source_document": PDF.name,
                "source_pages": "5",
            }
        ]
    )


def parse_berth_priorities(pages):
    rows = [
        {
            "port": "Gangavaram",
            "berth_no": "B1",
            "priority_cargo": None,
            "allocation_rule": "FCFS",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B2",
            "priority_cargo": None,
            "allocation_rule": "FCFS",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B3",
            "priority_cargo": None,
            "allocation_rule": "FCFS",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B4",
            "priority_cargo": "Iron Ore Fines / Pellets",
            "allocation_rule": "Priority",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B5",
            "priority_cargo": "Coal / Coke",
            "allocation_rule": "Priority",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B6",
            "priority_cargo": "Coal / Coke",
            "allocation_rule": "Priority",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B7",
            "priority_cargo": None,
            "allocation_rule": "FCFS",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B8",
            "priority_cargo": None,
            "allocation_rule": "FCFS",
        },
        {
            "port": "Gangavaram",
            "berth_no": "B9",
            "priority_cargo": "Container",
            "allocation_rule": "Priority",
        },
    ]

    return pd.DataFrame(rows)


def parse_operational_rules(pages):
    rows = [
        {
            "port": "Gangavaram",
            "rule": "berthing_scheme",
            "value": "FCFS",
            "source_page": 10,
        },
        {
            "port": "Gangavaram",
            "rule": "monsoon_period",
            "value": "01 May - 30 Nov",
            "source_page": 8,
        },
        {
            "port": "Gangavaram",
            "rule": "free_time_before_cargo_hours",
            "value": 3,
            "source_page": 14,
        },
        {
            "port": "Gangavaram",
            "rule": "maximum_total_cargo_stoppage_hours",
            "value": 2,
            "source_page": 14,
        },
        {
            "port": "Gangavaram",
            "rule": "minimum_inward_pilot_notice_hours",
            "value": 2,
            "source_page": 15,
        },
        {
            "port": "Gangavaram",
            "rule": "minimum_outward_pilot_notice_hours",
            "value": 2,
            "source_page": 15,
        },
        {
            "port": "Gangavaram",
            "rule": "loaded_cape_tidal_restriction",
            "value": "B5-B6 berthing; B4 unberthing",
            "source_page": 20,
        },
        {
            "port": "Gangavaram",
            "rule": "minimum_safe_clearance_between_vessels_m",
            "value": 20,
            "source_page": 19,
        },
    ]

    return pd.DataFrame(rows)


def parse_cost_rules(pages):
    rows = [
        {
            "port": "Gangavaram",
            "cost": "port_dues",
            "basis": "GT",
            "rate_usd": 0.032,
            "condition": "GT <= 45000",
            "source_page": 20,
        },
        {
            "port": "Gangavaram",
            "cost": "port_dues",
            "basis": "GT",
            "rate_usd": 0.036,
            "condition": "GT > 45000",
            "source_page": 20,
        },
        {
            "port": "Gangavaram",
            "cost": "pilotage",
            "basis": "GT",
            "rate_usd": 1.553,
            "condition": "GT <= 45000",
            "source_page": 20,
        },
        {
            "port": "Gangavaram",
            "cost": "pilotage",
            "basis": "GT",
            "rate_usd": 1.745,
            "condition": "GT > 45000",
            "source_page": 20,
        },
        {
            "port": "Gangavaram",
            "cost": "mooring",
            "basis": "GT per VCN",
            "rate_usd": 0.041,
            "condition": "all vessels",
            "source_page": 21,
        },
        {
            "port": "Gangavaram",
            "cost": "berth_hire",
            "basis": "GT/hour",
            "rate_usd": 0.013,
            "condition": "non-container GT <= 45000",
            "source_page": 21,
        },
        {
            "port": "Gangavaram",
            "cost": "berth_hire",
            "basis": "GT/hour",
            "rate_usd": 0.014,
            "condition": "non-container GT > 45000",
            "source_page": 21,
        },
        {
            "port": "Gangavaram",
            "cost": "anchorage",
            "basis": "GT/hour",
            "rate_usd": 0.00125,
            "condition": "vessel anchored within port limits",
            "source_page": 22,
        },
    ]

    return pd.DataFrame(rows)


def main():
    pages = extract_pages()

    save(
        parse_port_profile(pages),
        "gangavaram_port_profile.csv",
    )

    save(
        parse_berths(pages),
        "gangavaram_berths.csv",
    )

    save(
        parse_berth_priorities(pages),
        "gangavaram_berth_priorities.csv",
    )

    save(
        parse_operational_rules(pages),
        "gangavaram_operational_rules.csv",
    )

    save(
        parse_cost_rules(pages),
        "gangavaram_cost_rules.csv",
    )

    metadata = {
        "source_document": PDF.name,
        "source": "Adani Gangavaram Port",
        "source_url": (
            "https://www.adaniports.com/"
            "-/media/Project/Ports/PortsAndTerminals/"
            "Gangavaram-Port/Tariff/"
            "Berthing-Policy-and-Tariff-structure-2025-26.pdf"
        ),
        "note": (
            "Static BPTS extraction. Subsequent trade notices and "
            "monthly/operational updates may supersede these values."
        ),
    }

    (OUT / "gangavaram_bpts_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nGangavaram BPTS extraction complete.")


if __name__ == "__main__":
    main()
