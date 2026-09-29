from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPERATIONS_FILE = (
    PROJECT_ROOT
    / "04_OPTIMIZATION"
    / "data"
    / "port_operations.csv"
)


@dataclass
class PortOperationResult:
    port: str
    operation: str
    cargo_type: str
    cargo_mt: float

    handling_rate_mt_day: float | None
    handling_days: float | None

    handling_charge_inr_mt: float | None
    handling_charge_inr: float | None

    waiting_days: float | None

    status: str
    reason: str | None


def load_port_operations() -> pd.DataFrame:
    df = pd.read_csv(OPERATIONS_FILE)

    numeric_columns = [
        "handling_rate_mt_day",
        "handling_charge_inr_mt",
        "waiting_days",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def normalize_cargo_type(cargo_type: str) -> str:
    value = str(cargo_type).strip().lower()

    aliases = {
        "coking coal": "coking coal",
        "met coal": "coking coal",
        "metallurgical coal": "coking coal",

        "thermal coal": "thermal coal",

        "coal": "coal",

        "iron ore": "iron ore",
        "iron ore fines": "iron ore",
        "iron ore pellets": "iron ore",

        "dry bulk": "dry bulk",
    }

    return aliases.get(value, value)


def cargo_compatibility_family(cargo_type: str) -> list[str]:
    """
    Return operation-record cargo types that are compatible
    with the requested cargo.

    This is a matching rule, not a claim about port capability.

    More specific records are always preferred over generic ones.
    """

    normalized = normalize_cargo_type(cargo_type)

    if normalized in {
        "coking coal",
        "thermal coal",
        "coal",
    }:
        return [
            normalized,
            "coal",
            "dry bulk",
        ]

    if normalized == "iron ore":
        return [
            "iron ore",
            "dry bulk",
        ]

    if normalized == "dry bulk":
        return [
            "dry bulk",
        ]

    return [
        normalized,
        "dry bulk",
    ]


def find_operation(
    port: str,
    operation: str,
    cargo_type: str,
) -> pd.DataFrame:

    df = load_port_operations()

    port_mask = (
        df["port"]
        .fillna("")
        .str.strip()
        .str.lower()
        == str(port).strip().lower()
    )

    operation_mask = (
        df["operation"]
        .fillna("")
        .str.strip()
        .str.lower()
        == str(operation).strip().lower()
    )

    candidates = df[
        port_mask
        & operation_mask
    ].copy()

    if candidates.empty:
        return candidates

    compatible_types = cargo_compatibility_family(
        cargo_type
    )

    candidates["_normalized_operation_cargo"] = (
        candidates["cargo_type"]
        .fillna("")
        .map(normalize_cargo_type)
    )

    candidates["_match_priority"] = (
        candidates["_normalized_operation_cargo"]
        .map(
            {
                cargo: priority
                for priority, cargo in enumerate(
                    compatible_types
                )
            }
        )
        .fillna(999)
    )

    candidates = candidates[
        candidates["_match_priority"] < 999
    ].copy()

    if candidates.empty:
        return candidates.drop(
            columns=[
                "_normalized_operation_cargo",
                "_match_priority",
            ],
            errors="ignore",
        )

    candidates = candidates.sort_values(
        "_match_priority"
    )

    return candidates.drop(
        columns=[
            "_normalized_operation_cargo",
            "_match_priority",
        ],
        errors="ignore",
    )


def calculate_port_operation(
    port: str,
    operation: str,
    cargo_type: str,
    cargo_mt: float,
) -> PortOperationResult:

    rows = find_operation(
        port=port,
        operation=operation,
        cargo_type=cargo_type,
    )

    if rows.empty:
        return PortOperationResult(
            port=port,
            operation=operation,
            cargo_type=cargo_type,
            cargo_mt=cargo_mt,
            handling_rate_mt_day=None,
            handling_days=None,
            handling_charge_inr_mt=None,
            handling_charge_inr=None,
            waiting_days=None,
            status="unknown",
            reason=(
                "No matching port-operation record "
                "is available."
            ),
        )

    row = rows.iloc[0]

    rate = row["handling_rate_mt_day"]
    charge = row["handling_charge_inr_mt"]
    waiting = row["waiting_days"]

    handling_days = None

    if pd.notna(rate) and rate > 0:
        handling_days = cargo_mt / float(rate)

    handling_charge = None

    if pd.notna(charge):
        handling_charge = (
            cargo_mt * float(charge)
        )

    source_status = str(
        row.get("status", "")
    ).strip().lower()

    if pd.isna(rate):
        status = "partial"

        reason = (
            "Port operation exists, but no "
            "handling throughput rate is available."
        )

    elif source_status == "reference_proxy":
        status = "reference_proxy"

        reason = (
            "Handling time uses a port-wide productivity "
            "benchmark as a reference proxy; it is not "
            "a berth-specific guaranteed throughput rate."
        )

    elif source_status == "observed_reference":
        status = "observed_reference"

        reason = (
            "Handling throughput is an official port "
            "operational reference and should not be "
            "interpreted as a guaranteed vessel-specific rate."
        )

    else:
        status = "calculated"
        reason = None

    return PortOperationResult(
        port=port,
        operation=operation,
        cargo_type=cargo_type,
        cargo_mt=cargo_mt,
        handling_rate_mt_day=(
            float(rate)
            if pd.notna(rate)
            else None
        ),
        handling_days=handling_days,
        handling_charge_inr_mt=(
            float(charge)
            if pd.notna(charge)
            else None
        ),
        handling_charge_inr=handling_charge,
        waiting_days=(
            float(waiting)
            if pd.notna(waiting)
            else None
        ),
        status=status,
        reason=reason,
    )
