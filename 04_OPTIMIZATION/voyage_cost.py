from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from port_operations import calculate_port_operation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "04_OPTIMIZATION" / "data"

VESSEL_FILE = DATA_DIR / "vessels.csv"
ROUTE_FILE = DATA_DIR / "routes.csv"


@dataclass
class VoyageInput:
    route_id: str
    vessel_class: str
    cargo_mt: float
    fuel_price_usd_mt: float

    # Destination port used for the discharge operation.
    destination_port: str

    # Cargo type used by the port compatibility / operation layer.
    cargo_type: str


@dataclass
class VoyageCost:
    route_id: str
    vessel_class: str
    destination_port: str

    # ---------------------------------------------------------
    # SEA PASSAGE
    # ---------------------------------------------------------
    distance_nm: float | None

    speed_kn: float

    laden_days: float | None
    laden_fuel_mt: float | None
    bunker_cost_usd: float | None

    # ---------------------------------------------------------
    # PORT OPERATIONS
    # ---------------------------------------------------------
    discharge_handling_rate_mt_day: float | None
    discharge_handling_days: float | None

    discharge_charge_inr_mt: float | None
    discharge_charge_inr: float | None

    waiting_days: float | None

    # ---------------------------------------------------------
    # TOTALS
    # ---------------------------------------------------------
    known_voyage_days: float | None
    known_cost_usd: float | None

    status: str
    reason: str | None

    # Full breakdown for API / frontend use.
    breakdown: dict[str, Any]


def load_vessels() -> pd.DataFrame:
    df = pd.read_csv(VESSEL_FILE)

    numeric_columns = [
        "dwt_mt",
        "loa_m",
        "beam_m",
        "draft_m",
        "design_speed_kn",
        "design_consumption_laden_mt_day",
        "design_consumption_ballast_mt_day",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def load_routes() -> pd.DataFrame:
    df = pd.read_csv(
        ROUTE_FILE,
        dtype={
            "route_id": str,
            "origin": str,
            "destination": str,
            "distance_source": str,
            "distance_status": str,
        },
    )

    df["distance_nm"] = pd.to_numeric(
        df["distance_nm"],
        errors="coerce",
    )

    return df


def calculate_voyage_cost(
    voyage: VoyageInput,
) -> VoyageCost:

    vessels = load_vessels()
    routes = load_routes()

    # =========================================================
    # VESSEL
    # =========================================================

    vessel_rows = vessels[
        vessels["vessel_class"].str.lower()
        == voyage.vessel_class.lower()
    ]

    if vessel_rows.empty:
        return VoyageCost(
            route_id=voyage.route_id,
            vessel_class=voyage.vessel_class,
            destination_port=voyage.destination_port,
            distance_nm=None,
            speed_kn=0,
            laden_days=None,
            laden_fuel_mt=None,
            bunker_cost_usd=None,
            discharge_handling_rate_mt_day=None,
            discharge_handling_days=None,
            discharge_charge_inr_mt=None,
            discharge_charge_inr=None,
            waiting_days=None,
            known_voyage_days=None,
            known_cost_usd=None,
            status="unknown",
            reason="Vessel class not found.",
            breakdown={},
        )

    vessel = vessel_rows.iloc[0]

    # =========================================================
    # ROUTE
    # =========================================================

    route_rows = routes[
        routes["route_id"] == voyage.route_id
    ]

    if route_rows.empty:
        return VoyageCost(
            route_id=voyage.route_id,
            vessel_class=voyage.vessel_class,
            destination_port=voyage.destination_port,
            distance_nm=None,
            speed_kn=float(
                vessel["design_speed_kn"]
            ),
            laden_days=None,
            laden_fuel_mt=None,
            bunker_cost_usd=None,
            discharge_handling_rate_mt_day=None,
            discharge_handling_days=None,
            discharge_charge_inr_mt=None,
            discharge_charge_inr=None,
            waiting_days=None,
            known_voyage_days=None,
            known_cost_usd=None,
            status="unknown",
            reason="Route not found.",
            breakdown={},
        )

    route = route_rows.iloc[0]

    distance_nm = route["distance_nm"]

    if pd.isna(distance_nm):
        return VoyageCost(
            route_id=voyage.route_id,
            vessel_class=voyage.vessel_class,
            destination_port=voyage.destination_port,
            distance_nm=None,
            speed_kn=float(
                vessel["design_speed_kn"]
            ),
            laden_days=None,
            laden_fuel_mt=None,
            bunker_cost_usd=None,
            discharge_handling_rate_mt_day=None,
            discharge_handling_days=None,
            discharge_charge_inr_mt=None,
            discharge_charge_inr=None,
            waiting_days=None,
            known_voyage_days=None,
            known_cost_usd=None,
            status="pending_distance",
            reason=(
                "Route distance has not yet been populated "
                "from a marine routing source."
            ),
            breakdown={
                "route": {
                    "distance_nm": None,
                    "source": route.get(
                        "distance_source"
                    ),
                    "status": route.get(
                        "distance_status"
                    ),
                }
            },
        )

    # =========================================================
    # SEA PASSAGE
    # =========================================================

    speed_kn = float(
        vessel["design_speed_kn"]
    )

    laden_consumption = float(
        vessel[
            "design_consumption_laden_mt_day"
        ]
    )

    laden_days = float(distance_nm) / (
        speed_kn * 24
    )

    laden_fuel = (
        laden_days
        * laden_consumption
    )

    bunker_cost = (
        laden_fuel
        * voyage.fuel_price_usd_mt
    )

    # =========================================================
    # DESTINATION DISCHARGE
    # =========================================================

    discharge = calculate_port_operation(
        port=voyage.destination_port,
        operation="discharge",
        cargo_type=voyage.cargo_type,
        cargo_mt=voyage.cargo_mt,
    )

    # =========================================================
    # KNOWN TIME
    # =========================================================

    known_voyage_days = laden_days

    if discharge.handling_days is not None:
        known_voyage_days += (
            discharge.handling_days
        )

    if discharge.waiting_days is not None:
        known_voyage_days += (
            discharge.waiting_days
        )

    # =========================================================
    # STATUS
    # =========================================================

    missing_components = []

    if discharge.handling_days is None:
        missing_components.append(
            "discharge_handling_time"
        )

    if discharge.waiting_days is None:
        missing_components.append(
            "waiting_time"
        )

    if missing_components:
        status = "partial"

        reason = (
            "Known sea and bunker economics calculated, "
            "but some port-operation components are unavailable: "
            + ", ".join(missing_components)
        )
    else:
        status = "calculated"
        reason = None

    # =========================================================
    # COST
    # =========================================================

    # Bunker is currently the only component expressed in USD.
    #
    # Port handling charges remain in INR until we introduce
    # an explicit FX source. We do NOT silently convert them.
    known_cost_usd = bunker_cost

    # =========================================================
    # BREAKDOWN
    # =========================================================

    breakdown = {
        "route": {
            "route_id": voyage.route_id,
            "distance_nm": float(distance_nm),
            "distance_source": route.get(
                "distance_source"
            ),
            "distance_status": route.get(
                "distance_status"
            ),
        },
        "vessel": {
            "vessel_class": voyage.vessel_class,
            "speed_kn": speed_kn,
            "laden_consumption_mt_day": (
                laden_consumption
            ),
        },
        "sea_passage": {
            "laden_days": laden_days,
            "laden_fuel_mt": laden_fuel,
            "fuel_price_usd_mt": (
                voyage.fuel_price_usd_mt
            ),
            "bunker_cost_usd": bunker_cost,
        },
        "port": {
            "port": voyage.destination_port,
            "operation": "discharge",
            "cargo_type": voyage.cargo_type,
            "cargo_mt": voyage.cargo_mt,
            "handling_rate_mt_day": (
                discharge.handling_rate_mt_day
            ),
            "handling_days": (
                discharge.handling_days
            ),
            "handling_charge_inr_mt": (
                discharge.handling_charge_inr_mt
            ),
            "handling_charge_inr": (
                discharge.handling_charge_inr
            ),
            "waiting_days": (
                discharge.waiting_days
            ),
            "status": discharge.status,
            "reason": discharge.reason,
        },
        "total": {
            "known_voyage_days": (
                known_voyage_days
            ),
            "known_cost_usd": known_cost_usd,
            "cost_completeness": status,
        },
    }

    return VoyageCost(
        route_id=voyage.route_id,
        vessel_class=voyage.vessel_class,
        destination_port=voyage.destination_port,
        distance_nm=float(distance_nm),
        speed_kn=speed_kn,
        laden_days=laden_days,
        laden_fuel_mt=laden_fuel,
        bunker_cost_usd=bunker_cost,
        discharge_handling_rate_mt_day=(
            discharge.handling_rate_mt_day
        ),
        discharge_handling_days=(
            discharge.handling_days
        ),
        discharge_charge_inr_mt=(
            discharge.handling_charge_inr_mt
        ),
        discharge_charge_inr=(
            discharge.handling_charge_inr
        ),
        waiting_days=(
            discharge.waiting_days
        ),
        known_voyage_days=known_voyage_days,
        known_cost_usd=known_cost_usd,
        status=status,
        reason=reason,
        breakdown=breakdown,
    )