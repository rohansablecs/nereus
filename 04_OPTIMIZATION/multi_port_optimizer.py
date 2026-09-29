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


from feasibility import (
    Vessel,
    evaluate_vessel,
)

from recommendation_engine import (
    load_routes,
    find_route,
    vessel_reference,
)


# =====================================================================
# PORT CONFIGURATION
# =====================================================================

PORTS = [
    {
        "name": "Paradip India",
        "canonical": "Paradip",
        "type": "conventional_berth",
        "feasibility_supported": True,
        "operational_model": "paradip_berth_cycle_ml",
    },
    {
        "name": "Dhamra India",
        "canonical": "Dhamra",
        "type": "conventional_berth",
        "feasibility_supported": True,
        "operational_model": "operational_state",
    },
    {
        "name": "Gopalpur India",
        "canonical": "Gopalpur",
        "type": "conventional_berth",
        "feasibility_supported": True,
        "operational_model": "operational_state",
    },
    {
        "name": "Visakhapatnam India",
        "canonical": "Visakhapatnam",
        "type": "conventional_berth",
        "feasibility_supported": True,
        "operational_model": "operational_state",
    },
    {
        "name": "Gangavaram India",
        "canonical": "Gangavaram",
        "type": "conventional_berth",
        "feasibility_supported": True,
        "operational_model": "operational_state",
    },
    {
        "name": "Haldia India",
        "canonical": "Haldia",
        "type": "dock",
        "feasibility_supported": False,
        "operational_model": "operational_state_limited",
    },
    {
        "name": "Sagar-Sandheads India",
        "canonical": "Sagar-Sandheads",
        "type": "anchorage_lighterage",
        "feasibility_supported": False,
        "operational_model": "anchorage_lighterage",
    },
]


VESSEL_CLASSES = [
    "Handysize",
    "Supramax",
    "Panamax",
    "Capesize",
]


# =====================================================================
# SCENARIO
# =====================================================================

@dataclass
class PortOptimizationScenario:

    origin: str
    cargo_type: str
    cargo_mt: float
    delivery_days: float


# =====================================================================
# VESSEL CONSTRUCTION
# =====================================================================

def build_vessel(
    vessel_class: str,
) -> Vessel:

    references = vessel_reference()

    if references is None:

        raise ValueError(
            "No vessel reference data available."
        )

    if not isinstance(
        references,
        pd.DataFrame,
    ):

        raise TypeError(
            "Vessel reference data must be a pandas DataFrame."
        )

    matches = references[
        references[
            "vessel_class"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        == vessel_class.strip().lower()
    ]

    if matches.empty:

        raise ValueError(
            f"No vessel reference found for {vessel_class}"
        )

    reference = matches.iloc[0]

    return Vessel(
        vessel_id=(
            f"reference-{vessel_class.lower()}"
        ),
        vessel_class=vessel_class,
        dwt_mt=float(
            reference["dwt_mt"]
        ),
        loa_m=float(
            reference["loa_m"]
        ),
        beam_m=float(
            reference["beam_m"]
        ),
        draft_m=float(
            reference["draft_m"]
        ),
    )


# =====================================================================
# ROUTE LOOKUP
# =====================================================================

def route_status(
    origin: str,
    destination: str,
    routes: pd.DataFrame,
) -> dict:

    route = find_route(
        routes,
        origin,
        destination,
    )

    if route is None:

        return {
            "status": "ROUTE_PENDING",
            "route_id": None,
            "distance_nm": None,
            "distance_source": None,
        }

    distance = route.get(
        "distance_nm"
    )

    if (
        pd.isna(distance)
        or float(distance) <= 0
    ):

        return {
            "status": "ROUTE_PENDING",
            "route_id": route.get(
                "route_id"
            ),
            "distance_nm": None,
            "distance_source": route.get(
                "distance_source"
            ),
        }

    return {
        "status": "AVAILABLE",
        "route_id": route.get(
            "route_id"
        ),
        "distance_nm": float(
            distance
        ),
        "distance_source": route.get(
            "distance_source"
        ),
    }


# =====================================================================
# PHYSICAL FEASIBILITY
# =====================================================================
#
# IMPORTANT:
#
# evaluate_vessel() is the single authoritative physical feasibility
# engine.
#
# It already distinguishes:
#
#     FEASIBLE
#     POTENTIALLY_FEASIBLE
#     INFEASIBLE
#
# This module must not duplicate berth-level constraint logic.
# =====================================================================

def evaluate_port_vessel(
    scenario: PortOptimizationScenario,
    port: dict,
    vessel: Vessel,
) -> dict:

    canonical = port[
        "canonical"
    ]

    result = {
        "port": canonical,
        "port_display": port[
            "name"
        ],
        "port_type": port[
            "type"
        ],
        "vessel_class": vessel.vessel_class,
        "cargo_mt": scenario.cargo_mt,
        "cargo_type": scenario.cargo_type,
        "operational_model": port[
            "operational_model"
        ],

        "feasibility_status": None,

        "feasible": False,
        "potentially_feasible": False,

        "feasible_berths": [],
        "potential_berths": [],
        "rejected_berths": [],

        "berth_count_evaluated": 0,

        "reason": None,

        "raw_feasibility": None,
    }

    # ---------------------------------------------------------------
    # Sagar-Sandheads
    # ---------------------------------------------------------------
    #
    # This is an anchorage/lighterage operation, not a conventional
    # berth destination.
    #
    # Do not run conventional berth geometry logic here.
    #

    if canonical == "Sagar-Sandheads":

        result[
            "feasibility_status"
        ] = "SPECIAL_OPERATION"

        result[
            "reason"
        ] = (
            "Anchorage/lighterage operation. "
            "Conventional berth feasibility is not "
            "applicable in the current engine."
        )

        return result

    # ---------------------------------------------------------------
    # Haldia
    # ---------------------------------------------------------------
    #
    # Haldia does not currently have the validated generalized berth
    # constraint dataset required by evaluate_vessel().
    #

    if canonical == "Haldia":

        result[
            "feasibility_status"
        ] = "DATA_LIMITED"

        result[
            "reason"
        ] = (
            "Haldia currently uses a dock/operational "
            "model rather than the generalized berth "
            "geometry evaluator."
        )

        return result

    # ---------------------------------------------------------------
    # Authoritative feasibility evaluation
    # ---------------------------------------------------------------

    try:

        feasibility = evaluate_vessel(
            vessel=vessel,
            destination=canonical,
            cargo_mt=scenario.cargo_mt,
            cargo_type=scenario.cargo_type,
        )

    except Exception as exc:

        result[
            "feasibility_status"
        ] = "DATA_ERROR"

        result[
            "reason"
        ] = str(exc)

        return result

    result[
        "raw_feasibility"
    ] = feasibility

    result[
        "feasibility_status"
    ] = str(
        feasibility.get(
            "status",
            "INFEASIBLE",
        )
    ).upper()

    result[
        "feasible"
    ] = bool(
        result[
            "feasibility_status"
        ]
        == "FEASIBLE"
    )

    result[
        "potentially_feasible"
    ] = bool(
        result[
            "feasibility_status"
        ]
        == "POTENTIALLY_FEASIBLE"
    )

    result[
        "berth_count_evaluated"
    ] = int(
        feasibility.get(
            "berths_evaluated",
            0,
        )
    )

    # ---------------------------------------------------------------
    # Preserve berth-level information.
    #
    # We do not re-evaluate the berths. We simply expose the
    # authoritative results already produced by feasibility.py.
    # ---------------------------------------------------------------

    berth_results = feasibility.get(
        "berths",
        [],
    )

    for berth_result in berth_results:

        if not isinstance(
            berth_result,
            dict,
        ):
            continue

        berth_name = (
            berth_result.get(
                "berth"
            )
        )

        status = str(
            berth_result.get(
                "status",
                "unknown",
            )
        ).lower()

        entry = {
            "berth": berth_name,
            "result": berth_result,
        }

        if status == "feasible":

            result[
                "feasible_berths"
            ].append(entry)

        elif status == "unknown":

            result[
                "potential_berths"
            ].append(entry)

        elif status == "infeasible":

            result[
                "rejected_berths"
            ].append(entry)

    # ---------------------------------------------------------------
    # Human-readable reason.
    # ---------------------------------------------------------------

    if result[
        "feasibility_status"
    ] == "FEASIBLE":

        result[
            "reason"
        ] = (
            f"{len(result['feasible_berths'])} berth(s) "
            "passed all currently evaluable constraints."
        )

    elif result[
        "feasibility_status"
    ] == "POTENTIALLY_FEASIBLE":

        result[
            "reason"
        ] = (
            f"{len(result['potential_berths'])} berth(s) "
            "remain potentially feasible because one or "
            "more required constraints are unknown."
        )

    elif result[
        "feasibility_status"
    ] == "INFEASIBLE":

        result[
            "reason"
        ] = (
            "No evaluated berth is currently "
            "deterministically feasible."
        )

    else:

        result[
            "reason"
        ] = feasibility.get(
            "reason",
            result["reason"],
        )

    return result


# =====================================================================
# PORT × VESSEL MATRIX
# =====================================================================

def build_candidate_matrix(
    scenario: PortOptimizationScenario,
) -> list[dict]:

    routes = load_routes()

    candidates = []

    for port in PORTS:

        route = route_status(
            scenario.origin,
            port["name"],
            routes,
        )

        for vessel_class in (
            VESSEL_CLASSES
        ):

            vessel = build_vessel(
                vessel_class
            )

            feasibility = (
                evaluate_port_vessel(
                    scenario,
                    port,
                    vessel,
                )
            )

            feasibility_status = (
                feasibility[
                    "feasibility_status"
                ]
            )

            # -------------------------------------------------------
            # Optimization eligibility
            #
            # Only:
            #
            #     FEASIBLE
            #     +
            #     AVAILABLE route
            #
            # enters confirmed optimization.
            #
            # Potential candidates remain visible but are not promoted
            # to confirmed optimization.
            # -------------------------------------------------------

            confirmed_eligible = (
                feasibility_status
                == "FEASIBLE"
                and
                route["status"]
                == "AVAILABLE"
            )

            potential_eligible = (
                feasibility_status
                == "POTENTIALLY_FEASIBLE"
                and
                route["status"]
                == "AVAILABLE"
            )

            if confirmed_eligible:

                optimization_status = (
                    "ELIGIBLE"
                )

            elif (
                feasibility_status
                == "POTENTIALLY_FEASIBLE"
            ):

                if (
                    route["status"]
                    == "AVAILABLE"
                ):

                    optimization_status = (
                        "POTENTIAL"
                    )

                else:

                    optimization_status = (
                        "POTENTIAL_ROUTE_PENDING"
                    )

            elif (
                feasibility_status
                == "INFEASIBLE"
            ):

                optimization_status = (
                    "INFEASIBLE"
                )

            elif (
                feasibility_status
                == "ROUTE_PENDING"
            ):

                optimization_status = (
                    "ROUTE_PENDING"
                )

            else:

                optimization_status = (
                    feasibility_status
                )

            candidate = {
                "port": port["canonical"],
                "port_display": port["name"],
                "port_type": port["type"],
                "vessel_class": vessel_class,

                "route": route,

                "feasibility": feasibility,

                "eligible_for_optimization": (
                    confirmed_eligible
                ),

                "potential_for_optimization": (
                    potential_eligible
                ),

                "optimization_status": (
                    optimization_status
                ),
            }

            candidates.append(
                candidate
            )

    return candidates


# =====================================================================
# CANDIDATE FILTERS
# =====================================================================

def eligible_candidates(
    candidates: list[dict],
) -> list[dict]:

    return [
        candidate
        for candidate in candidates
        if candidate[
            "eligible_for_optimization"
        ]
    ]


def potential_candidates(
    candidates: list[dict],
) -> list[dict]:

    return [
        candidate
        for candidate in candidates
        if candidate[
            "potential_for_optimization"
        ]
    ]


# =====================================================================
# MATRIX DISPLAY
# =====================================================================

def print_matrix(
    scenario: PortOptimizationScenario,
    candidates: list[dict],
) -> None:

    print()
    print("=" * 115)
    print(
        "NEREUS — MULTI-PORT OPTIMIZATION MATRIX"
    )
    print("=" * 115)

    print()
    print("SCENARIO")
    print("-" * 115)

    print(
        f"{scenario.cargo_mt:,.0f} MT "
        f"{scenario.cargo_type} | "
        f"{scenario.origin} → East Coast India | "
        f"deadline {scenario.delivery_days:g}d"
    )

    print()
    print(
        f"{'PORT':18} "
        f"{'VESSEL':11} "
        f"{'FEASIBILITY':22} "
        f"{'ROUTE':18} "
        f"{'DISTANCE':13} "
        f"{'STATUS'}"
    )

    print("-" * 115)

    for candidate in candidates:

        route = candidate[
            "route"
        ]

        feasibility = candidate[
            "feasibility"
        ]

        distance = route[
            "distance_nm"
        ]

        distance_text = (
            f"{distance:,.0f} nm"
            if distance is not None
            else "—"
        )

        print(
            f"{candidate['port']:18} "
            f"{candidate['vessel_class']:11} "
            f"{feasibility['feasibility_status']:22} "
            f"{route['status']:18} "
            f"{distance_text:13} "
            f"{candidate['optimization_status']}"
        )

    eligible = eligible_candidates(
        candidates
    )

    potential = potential_candidates(
        candidates
    )

    print()
    print("=" * 115)

    print()
    print(
        f"CONFIRMED ELIGIBLE: {len(eligible)}"
    )

    for candidate in eligible:

        print(
            f"  ✓ "
            f"{candidate['port']} / "
            f"{candidate['vessel_class']} / "
            f"{candidate['route']['distance_nm']:,.0f} NM"
        )

    print()
    print(
        f"POTENTIAL CANDIDATES: {len(potential)}"
    )

    for candidate in potential:

        print(
            f"  ? "
            f"{candidate['port']} / "
            f"{candidate['vessel_class']} / "
            f"{candidate['optimization_status']}"
        )

    print()
    print("IMPORTANT")
    print("-" * 115)

    print(
        "FEASIBLE means at least one berth passed all "
        "currently evaluable physical constraints."
    )

    print(
        "POTENTIALLY_FEASIBLE means no hard failure was "
        "proven, but one or more required constraints remain unknown."
    )

    print(
        "Missing route distances are not replaced with "
        "great-circle estimates."
    )

    print(
        "Haldia remains DATA_LIMITED and Sagar-Sandheads "
        "remains a SPECIAL_OPERATION because they are not currently "
        "represented by the generalized conventional-berth evaluator."
    )

    print(
        "Paradip is currently the only port with a validated "
        "berth-cycle ML model."
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

    candidates = build_candidate_matrix(
        scenario
    )

    print_matrix(
        scenario,
        candidates,
    )