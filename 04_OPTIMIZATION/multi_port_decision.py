from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "04_OPTIMIZATION"),
)


from multi_port_optimizer import (
    PortOptimizationScenario,
    PORTS,
    VESSEL_CLASSES,
    build_vessel,
    build_candidate_matrix,
)

from voyage_cost import (
    VoyageInput,
    calculate_voyage_cost,
)

from port_operations import (
    calculate_port_operation,
)


# =====================================================================
# DATA PATHS
# =====================================================================

PRESSURE_FILE = (
    ROOT
    / "02_DATA"
    / "processed"
    / "port_state"
    / "port_pressure_v1.csv"
)

BUNKER_FILE = (
    ROOT
    / "04_OPTIMIZATION"
    / "data"
    / "bunker_prices.csv"
)


# =====================================================================
# DECISION RESULT
# =====================================================================

@dataclass
class DecisionOption:

    port: str
    vessel_class: str

    feasibility_status: str

    route_status: str
    distance_nm: float | None

    distance_source: str | None

    sailing_days: float | None
    laden_fuel_mt: float | None
    bunker_cost_usd: float | None

    handling_rate_mt_day: float | None
    handling_days: float | None

    handling_charge_inr_mt: float | None
    handling_charge_inr: float | None

    waiting_days: float | None

    known_voyage_days: float | None

    pressure_score: float | None
    pressure_band: str | None
    pressure_confidence: float | None
    pressure_age_hours: float | None

    delivery_days: float
    delivery_pass: bool | None

    cost_completeness: str

    decision_status: str

    limitations: list[str]


# =====================================================================
# PRESSURE
# =====================================================================

def load_pressure() -> pd.DataFrame:

    if not PRESSURE_FILE.exists():

        raise FileNotFoundError(
            f"Pressure file not found: {PRESSURE_FILE}"
        )

    df = pd.read_csv(
        PRESSURE_FILE
    )

    numeric_columns = [
        "pressure_score",
        "pressure_confidence",
        "data_age_hours",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def pressure_for_port(
    pressure: pd.DataFrame,
    port: str,
) -> dict:

    rows = pressure[
        pressure["port"]
        .astype(str)
        .str.strip()
        .str.lower()
        == port.strip().lower()
    ]

    if rows.empty:

        return {
            "pressure_score": None,
            "pressure_band": None,
            "pressure_confidence": None,
            "pressure_age_hours": None,
        }

    row = rows.iloc[0]

    return {
        "pressure_score": (
            float(row["pressure_score"])
            if pd.notna(
                row["pressure_score"]
            )
            else None
        ),
        "pressure_band": (
            str(row["pressure_band"])
            if pd.notna(
                row["pressure_band"]
            )
            else None
        ),
        "pressure_confidence": (
            float(
                row["pressure_confidence"]
            )
            if pd.notna(
                row["pressure_confidence"]
            )
            else None
        ),
        "pressure_age_hours": (
            float(
                row["data_age_hours"]
            )
            if pd.notna(
                row["data_age_hours"]
            )
            else None
        ),
    }


# =====================================================================
# BUNKER PRICE
# =====================================================================

def latest_bunker_price() -> dict:

    if not BUNKER_FILE.exists():

        raise FileNotFoundError(
            f"Bunker price file not found: {BUNKER_FILE}"
        )

    df = pd.read_csv(
        BUNKER_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["price_usd_mt"] = pd.to_numeric(
        df["price_usd_mt"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "price_usd_mt",
        ]
    )

    if df.empty:

        raise ValueError(
            "No valid bunker price observations available."
        )

    row = df.sort_values(
        "date"
    ).iloc[-1]

    return {
        "date": row["date"],
        "port": row["port"],
        "fuel_grade": row["fuel_grade"],
        "price_usd_mt": float(
            row["price_usd_mt"]
        ),
        "source": row["source"],
        "status": row["status"],
    }


# =====================================================================
# DELIVERY
# =====================================================================

def evaluate_delivery(
    known_voyage_days: float | None,
    deadline_days: float,
) -> bool | None:

    if known_voyage_days is None:

        return None

    return bool(
        known_voyage_days
        <= deadline_days
    )


# =====================================================================
# OPTION BUILDER
# =====================================================================

def build_decision_option(
    scenario: PortOptimizationScenario,
    candidate: dict,
    bunker_price: float,
    pressure: pd.DataFrame,
) -> DecisionOption:

    port = candidate[
        "port"
    ]

    vessel_class = candidate[
        "vessel_class"
    ]

    route = candidate[
        "route"
    ]

    feasibility = candidate[
        "feasibility"
    ]

    limitations = []

    feasibility_status = str(
        feasibility[
            "feasibility_status"
        ]
    )

    route_status = str(
        route[
            "status"
        ]
    )

    distance_nm = route[
        "distance_nm"
    ]

    distance_source = route[
        "distance_source"
    ]

    # ---------------------------------------------------------------
    # Pressure
    # ---------------------------------------------------------------

    pressure_state = pressure_for_port(
        pressure,
        port,
    )

    # ---------------------------------------------------------------
    # Non-optimizable candidate
    # ---------------------------------------------------------------

    if (
        not candidate[
            "eligible_for_optimization"
        ]
    ):

        if (
            route_status
            != "AVAILABLE"
        ):

            limitations.append(
                "Validated route distance is unavailable."
            )

        if feasibility_status != "FEASIBLE":

            limitations.append(
                f"Physical feasibility status is "
                f"{feasibility_status}."
            )

        return DecisionOption(
            port=port,
            vessel_class=vessel_class,

            feasibility_status=(
                feasibility_status
            ),

            route_status=route_status,
            distance_nm=(
                float(distance_nm)
                if distance_nm is not None
                else None
            ),

            distance_source=distance_source,

            sailing_days=None,
            laden_fuel_mt=None,
            bunker_cost_usd=None,

            handling_rate_mt_day=None,
            handling_days=None,

            handling_charge_inr_mt=None,
            handling_charge_inr=None,

            waiting_days=None,

            known_voyage_days=None,

            pressure_score=pressure_state[
                "pressure_score"
            ],
            pressure_band=pressure_state[
                "pressure_band"
            ],
            pressure_confidence=pressure_state[
                "pressure_confidence"
            ],
            pressure_age_hours=pressure_state[
                "pressure_age_hours"
            ],

            delivery_days=scenario.delivery_days,
            delivery_pass=None,

            cost_completeness="not_calculated",

            decision_status=(
                "NOT_CONFIRMED"
            ),

            limitations=limitations,
        )

    # ---------------------------------------------------------------
    # Voyage economics
    # ---------------------------------------------------------------

    route_id = route[
        "route_id"
    ]

    voyage = calculate_voyage_cost(
        VoyageInput(
            route_id=route_id,
            vessel_class=vessel_class,
            cargo_mt=scenario.cargo_mt,
            fuel_price_usd_mt=bunker_price,
            destination_port=port,
            cargo_type=scenario.cargo_type,
        )
    )

    # ---------------------------------------------------------------
    # Extract economics
    # ---------------------------------------------------------------

    sailing_days = voyage.laden_days
    laden_fuel_mt = voyage.laden_fuel_mt
    bunker_cost_usd = voyage.bunker_cost_usd

    handling_rate = (
        voyage.discharge_handling_rate_mt_day
    )

    handling_days = (
        voyage.discharge_handling_days
    )

    handling_charge_inr_mt = (
        voyage.discharge_charge_inr_mt
    )

    handling_charge_inr = (
        voyage.discharge_charge_inr
    )

    waiting_days = (
        voyage.waiting_days
    )

    known_voyage_days = (
        voyage.known_voyage_days
    )

    # ---------------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------------

    if voyage.status != "calculated":

        limitations.append(
            voyage.reason
            or
            "Voyage cost is incomplete."
        )

    if handling_rate is None:

        limitations.append(
            "Discharge handling throughput is unavailable."
        )

    if handling_charge_inr_mt is None:

        limitations.append(
            "Discharge handling charge is unavailable."
        )

    if waiting_days is None:

        limitations.append(
            "Waiting time is unavailable."
        )

    if (
        pressure_state[
            "pressure_score"
        ]
        is None
    ):

        limitations.append(
            "Operational pressure signal is unavailable."
        )

    # ---------------------------------------------------------------
    # Delivery
    # ---------------------------------------------------------------

    delivery_pass = evaluate_delivery(
        known_voyage_days,
        scenario.delivery_days,
    )

    if delivery_pass is False:

        limitations.append(
            "Known voyage duration exceeds the delivery deadline."
        )

    elif delivery_pass is None:

        limitations.append(
            "Delivery cannot be fully evaluated from known time components."
        )

    # ---------------------------------------------------------------
    # Decision state
    # ---------------------------------------------------------------

    if delivery_pass is False:

        decision_status = (
            "DELIVERY_RISK"
        )

    elif (
        voyage.status
        in {
            "partial",
            "reference_proxy",
        }
    ):

        decision_status = (
            "CONFIRMED_PARTIAL_COST"
        )

    else:

        decision_status = (
            "CONFIRMED"
        )

    return DecisionOption(
        port=port,
        vessel_class=vessel_class,

        feasibility_status=(
            feasibility_status
        ),

        route_status=route_status,
        distance_nm=(
            float(distance_nm)
            if distance_nm is not None
            else None
        ),

        distance_source=distance_source,

        sailing_days=sailing_days,
        laden_fuel_mt=laden_fuel_mt,
        bunker_cost_usd=bunker_cost_usd,

        handling_rate_mt_day=handling_rate,
        handling_days=handling_days,

        handling_charge_inr_mt=(
            handling_charge_inr_mt
        ),
        handling_charge_inr=(
            handling_charge_inr
        ),

        waiting_days=waiting_days,

        known_voyage_days=known_voyage_days,

        pressure_score=pressure_state[
            "pressure_score"
        ],
        pressure_band=pressure_state[
            "pressure_band"
        ],
        pressure_confidence=pressure_state[
            "pressure_confidence"
        ],
        pressure_age_hours=pressure_state[
            "pressure_age_hours"
        ],

        delivery_days=scenario.delivery_days,
        delivery_pass=delivery_pass,

        cost_completeness=voyage.status,

        decision_status=decision_status,

        limitations=limitations,
    )


# =====================================================================
# DECISION ENGINE
# =====================================================================

def build_decision_set(
    scenario: PortOptimizationScenario,
) -> dict:

    candidates = build_candidate_matrix(
        scenario
    )

    pressure = load_pressure()

    bunker = latest_bunker_price()

    bunker_price = bunker[
        "price_usd_mt"
    ]

    options = []

    for candidate in candidates:

        option = build_decision_option(
            scenario=scenario,
            candidate=candidate,
            bunker_price=bunker_price,
            pressure=pressure,
        )

        options.append(
            option
        )

    confirmed = [
        option
        for option in options
        if option.feasibility_status
        == "FEASIBLE"
        and option.route_status
        == "AVAILABLE"
    ]

    # ---------------------------------------------------------------
    # Deterministic ranking
    #
    # We intentionally do NOT invent a weighted score.
    #
    # Primary ranking:
    #
    #   1. delivery-feasible options
    #   2. lower known bunker cost
    #
    # Pressure is surfaced as a decision-risk signal, not converted
    # into an arbitrary monetary penalty.
    # ---------------------------------------------------------------

    confirmed_sorted = sorted(
        confirmed,
        key=lambda option: (
            option.delivery_pass
            is not True,
            option.bunker_cost_usd
            if option.bunker_cost_usd
            is not None
            else float("inf"),
        ),
    )

    preferred = (
        confirmed_sorted[0]
        if confirmed_sorted
        else None
    )

    fallback = (
        confirmed_sorted[1]
        if len(confirmed_sorted) > 1
        else None
    )

    return {
        "scenario": scenario,
        "bunker": bunker,
        "options": options,
        "confirmed": confirmed_sorted,
        "preferred": preferred,
        "fallback": fallback,
    }


# =====================================================================
# DISPLAY
# =====================================================================

def format_money_usd(
    value: float | None,
) -> str:

    if value is None:

        return "—"

    return f"${value:,.0f}"


def format_money_inr(
    value: float | None,
) -> str:

    if value is None:

        return "—"

    return f"₹{value:,.0f}"


def print_decision(
    decision: dict,
) -> None:

    scenario = decision[
        "scenario"
    ]

    bunker = decision[
        "bunker"
    ]

    preferred = decision[
        "preferred"
    ]

    fallback = decision[
        "fallback"
    ]

    print()
    print("=" * 110)
    print(
        "NEREUS — MULTI-PORT DECISION ENGINE"
    )
    print("=" * 110)

    print()
    print("SCENARIO")
    print("-" * 110)

    print(
        f"{scenario.cargo_mt:,.0f} MT "
        f"{scenario.cargo_type} | "
        f"{scenario.origin} → East Coast India | "
        f"deadline {scenario.delivery_days:g}d"
    )

    print()
    print("BUNKER REFERENCE")
    print("-" * 110)

    print(
        f"{bunker['date'].date()} | "
        f"{bunker['port']} | "
        f"{bunker['fuel_grade']} | "
        f"${bunker['price_usd_mt']:,.2f}/MT | "
        f"{bunker['source']}"
    )

    print()
    print("CONFIRMED OPTIONS")
    print("-" * 110)

    for index, option in enumerate(
        decision["confirmed"],
        start=1,
    ):

        print()
        print(
            f"{index}. "
            f"{option.port} / "
            f"{option.vessel_class}"
        )

        print(
            f"   Physical:       "
            f"{option.feasibility_status}"
        )

        print(
            f"   Route:          "
            f"{option.distance_nm:,.0f} NM"
            if option.distance_nm
            is not None
            else
            "   Route:          —"
        )

        print(
            f"   Sailing:        "
            f"{option.sailing_days:.2f} d"
            if option.sailing_days
            is not None
            else
            "   Sailing:        —"
        )

        print(
            f"   Laden fuel:     "
            f"{option.laden_fuel_mt:,.2f} MT"
            if option.laden_fuel_mt
            is not None
            else
            "   Laden fuel:     —"
        )

        print(
            f"   Bunker:         "
            f"{format_money_usd(option.bunker_cost_usd)}"
        )

        print(
            f"   Handling rate:  "
            f"{option.handling_rate_mt_day:,.0f} MT/day"
            if option.handling_rate_mt_day
            is not None
            else
            "   Handling rate:  —"
        )

        print(
            f"   Handling time:  "
            f"{option.handling_days:.2f} d"
            if option.handling_days
            is not None
            else
            "   Handling time:  —"
        )

        print(
            f"   Handling cost:  "
            f"{format_money_inr(option.handling_charge_inr)}"
        )

        print(
            f"   Waiting:        "
            f"{option.waiting_days:.2f} d"
            if option.waiting_days
            is not None
            else
            "   Waiting:        unknown"
        )

        print(
            f"   Known duration: "
            f"{option.known_voyage_days:.2f} d"
            if option.known_voyage_days
            is not None
            else
            "   Known duration: —"
        )

        print(
            f"   Pressure:       "
            f"{option.pressure_score:.2f} "
            f"({option.pressure_band})"
            if option.pressure_score
            is not None
            else
            "   Pressure:       unknown"
        )

        print(
            f"   Pressure conf.: "
            f"{option.pressure_confidence:.2f}"
            if option.pressure_confidence
            is not None
            else
            "   Pressure conf.: —"
        )

        print(
            f"   Pressure age:   "
            f"{option.pressure_age_hours:.2f} h"
            if option.pressure_age_hours
            is not None
            else
            "   Pressure age:   —"
        )

        print(
            f"   Delivery:       "
            f"{'PASS' if option.delivery_pass else 'RISK'}"
            if option.delivery_pass
            is not None
            else
            "   Delivery:       UNKNOWN"
        )

        print(
            f"   Cost status:    "
            f"{option.cost_completeness}"
        )

        print(
            f"   Decision state: "
            f"{option.decision_status}"
        )

        if option.limitations:

            print(
                "   Limitations:"
            )

            for limitation in (
                option.limitations
            ):

                print(
                    f"      - {limitation}"
                )

    print()
    print("=" * 110)

    if preferred is None:

        print()
        print(
            "DECISION: NO CONFIRMED OPTION"
        )

    else:

        print()
        print(
            f"DECISION: "
            f"{preferred.port} / "
            f"{preferred.vessel_class}"
        )

        print(
            f"Known bunker cost: "
            f"{format_money_usd(preferred.bunker_cost_usd)}"
        )

        print(
            f"Operational pressure: "
            f"{preferred.pressure_band} "
            f"({preferred.pressure_score:.2f})"
            if preferred.pressure_score
            is not None
            else
            "Operational pressure: unknown"
        )

        if fallback is not None:

            print()
            print(
                f"FALLBACK: "
                f"{fallback.port} / "
                f"{fallback.vessel_class}"
            )

    print()
    print("DECISION PRINCIPLES")
    print("-" * 110)

    print(
        "No arbitrary weighted score is applied."
    )

    print(
        "Bunker economics are compared in USD."
    )

    print(
        "Port handling charges remain in INR."
    )

    print(
        "Operational pressure is treated as a risk signal, "
        "not converted into a fabricated monetary penalty."
    )

    print(
        "Unknown waiting time is not treated as zero."
    )

    print(
        "Missing route distances are not estimated."
    )

    print(
        "Incomplete candidates remain visible with explicit limitations."
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    scenario = PortOptimizationScenario(
        origin="Newcastle Australia",
        cargo_type="coking coal",
        cargo_mt=60_000,
        delivery_days=30,
    )

    decision = build_decision_set(
        scenario
    )

    print_decision(
        decision
    )