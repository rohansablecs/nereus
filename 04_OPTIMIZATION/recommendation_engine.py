from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import pandas as pd

from feasibility import (
    Vessel,
    evaluate_vessel,
)

from voyage_cost import (
    VoyageInput,
    calculate_voyage_cost,
)

from berth_duration_predictor import (
    predict_paradip_berth_duration,
)

from port_operational_state import (
    get_paradip_operational_state,
)


ROOT = Path(__file__).resolve().parents[1]


ROUTES_PATH = (
    ROOT
    / "04_OPTIMIZATION/data/routes.csv"
)

PRESSURE_PATH = (
    ROOT
    / "02_DATA/processed/port_state/port_pressure_v1.csv"
)

VESSELS_PATH = (
    ROOT
    / "04_OPTIMIZATION/data/vessels.csv"
)


@dataclass
class Scenario:

    origin: str
    cargo_type: str
    cargo_mt: float
    destination: str
    delivery_days: float


def load_routes():

    return pd.read_csv(
        ROUTES_PATH
    )


def load_port_pressure():

    return pd.read_csv(
        PRESSURE_PATH
    )


def find_route(
    routes,
    origin,
    destination,
):

    matches = routes[
        (
            routes["origin"]
            .astype(str)
            .str.strip()
            .str.lower()
            == origin.strip().lower()
        )
        &
        (
            routes["destination"]
            .astype(str)
            .str.strip()
            .str.lower()
            == destination.strip().lower()
        )
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def canonical_port_name(
    destination,
):

    mapping = {
        "Paradip India": "Paradip",
        "Dhamra India": "Dhamra",
        "Gopalpur India": "Gopalpur",
        "Visakhapatnam India": "Visakhapatnam",
        "Gangavaram India": "Gangavaram",
        "Haldia India": "Haldia",
        "Sagar-Sandheads India": "Sagar-Sandheads",
    }

    return mapping.get(
        destination,
        destination,
    )


def pressure_for_port(
    pressure,
    destination,
):

    port_name = canonical_port_name(
        destination
    )

    matches = pressure[
        pressure["port"]
        .astype(str)
        .str.strip()
        .str.lower()
        == port_name.strip().lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def vessel_reference():

    return pd.read_csv(
        VESSELS_PATH
    )


def vessel_objects():

    df = vessel_reference()

    vessels = []

    for _, row in df.iterrows():

        vessels.append(
            Vessel(
                vessel_id=str(
                    row["vessel_class"]
                ),
                vessel_class=str(
                    row["vessel_class"]
                ),
                dwt_mt=float(
                    row["dwt_mt"]
                ),
                loa_m=float(
                    row["loa_m"]
                ),
                beam_m=float(
                    row["beam_m"]
                ),
                draft_m=float(
                    row["draft_m"]
                ),
            )
        )

    return vessels


def feasibility_summary(
    vessel,
    scenario,
):

    return evaluate_vessel(
        vessel,
        destination=canonical_port_name(
            scenario.destination
        ),
        cargo_type=scenario.cargo_type,
        cargo_mt=scenario.cargo_mt,
    )


def pressure_penalty(
    pressure_row,
):

    if pressure_row is None:
        return {
            "penalty": None,
            "confidence": 0.0,
        }

    score = pd.to_numeric(
        pressure_row.get(
            "pressure_score"
        ),
        errors="coerce",
    )

    confidence = pd.to_numeric(
        pressure_row.get(
            "pressure_confidence"
        ),
        errors="coerce",
    )

    if pd.isna(score):

        return {
            "penalty": None,
            "confidence": (
                float(confidence)
                if pd.notna(confidence)
                else 0.0
            ),
        }

    penalty = float(score) / 100.0

    return {
        "penalty": penalty,
        "confidence": (
            float(confidence)
            if pd.notna(confidence)
            else 0.0
        ),
    }


def _predict_paradip_operation(
    scenario,
):

    if canonical_port_name(
        scenario.destination
    ) != "Paradip":

        return {
            "status": "not_applicable"
        }

    state = (
        get_paradip_operational_state()
    )

    if state.get("status") != "available":

        return {
            "status": "unavailable",
            "reason": state.get(
                "reason",
                "Operational state unavailable.",
            ),
        }

    feasibility = evaluate_vessel(
        vessel_objects()[0],
        destination="Paradip",
        cargo_type=scenario.cargo_type,
        cargo_mt=scenario.cargo_mt,
    )

    return {
        "status": "state_available",
        "snapshot_date": state[
            "snapshot_date"
        ],
        "snapshot_age_days": state[
            "snapshot_age_days"
        ],
        "dry_bulk_vessels_waiting": state[
            "dry_bulk_vessels_waiting"
        ],
        "source_file": state[
            "source_file"
        ],
        "feasibility_probe": feasibility,
    }


def build_candidate(
    scenario,
    route,
    vessel,
    pressure_row,
):

    feasibility = feasibility_summary(
        vessel,
        scenario,
    )

    feasibility_status = str(
        feasibility.get(
            "status",
            "INFEASIBLE",
        )
    ).upper()

    if feasibility_status not in {
        "FEASIBLE",
        "POTENTIALLY_FEASIBLE",
        "INFEASIBLE",
    }:

        feasibility_status = "INFEASIBLE"

    feasible = (
        feasibility_status == "FEASIBLE"
    )

    candidate = {
        "vessel_class": vessel.vessel_class,

        # Backward-compatible deterministic boolean.
        "feasible": feasible,

        # Rich physical feasibility state.
        "feasibility_status": (
            feasibility_status
        ),

        "cargo_mt": scenario.cargo_mt,
        "destination": scenario.destination,

        "route_id": None,
        "route_status": None,
        "distance_nm": None,

        "sea_days": None,
        "known_voyage_days": None,

        "bunker_cost_usd": None,
        "handling_charge_inr": None,
        "cost_status": "unavailable",

        "delivery_voyage_days": None,
        "delivery_feasible": False,

        "port_pressure": None,
        "port_pressure_band": None,
        "pressure_penalty": None,
        "pressure_confidence": 0.0,

        "operational_data_age_hours": None,
        "operational_source_type": None,

        "berth_cycle_prediction": None,
        "operational_state": None,
    }

    # ===============================================================
    # PHYSICAL FEASIBILITY GATE
    # ===============================================================
    #
    # Only deterministic physical infeasibility stops processing.
    #
    # POTENTIALLY_FEASIBLE candidates continue because unresolved
    # source data is not equivalent to a physical failure.
    #

    if feasibility_status == "INFEASIBLE":

        candidate[
            "reason"
        ] = (
            "All evaluated berths have "
            "deterministic physical incompatibilities."
        )

        candidate[
            "feasibility"
        ] = feasibility

        return candidate

    # ===============================================================
    # ROUTE VALIDATION
    # ===============================================================

    if route is None:

        candidate[
            "route_status"
        ] = "missing"

        candidate[
            "reason"
        ] = (
            "No route record exists."
        )

        candidate[
            "feasibility"
        ] = feasibility

        return candidate

    candidate[
        "route_id"
    ] = route[
        "route_id"
    ]

    candidate[
        "route_status"
    ] = route[
        "distance_status"
    ]

    distance = route.get(
        "distance_nm"
    )

    if (
        pd.isna(distance)
        or float(distance) <= 0
    ):

        candidate[
            "route_status"
        ] = "incomplete"

        candidate[
            "reason"
        ] = (
            "Validated route distance unavailable."
        )

        candidate[
            "feasibility"
        ] = feasibility

        return candidate

    distance = float(
        distance
    )

    candidate[
        "distance_nm"
    ] = distance

    # ===============================================================
    # PORT PRESSURE
    # ===============================================================
    #
    # Pressure is an operational signal.
    #
    # It is NOT:
    # - a congestion probability
    # - a waiting-time prediction
    # - a calibrated risk probability
    #

    pressure_info = pressure_penalty(
        pressure_row
    )

    if pressure_row is not None:

        score = pd.to_numeric(
            pressure_row.get(
                "pressure_score"
            ),
            errors="coerce",
        )

        if pd.notna(score):

            candidate[
                "port_pressure"
            ] = float(score)

            candidate[
                "port_pressure_band"
            ] = str(
                pressure_row.get(
                    "pressure_band",
                    "UNKNOWN",
                )
            )

        candidate[
            "pressure_penalty"
        ] = pressure_info[
            "penalty"
        ]

        candidate[
            "pressure_confidence"
        ] = pressure_info[
            "confidence"
        ]

        data_age = pd.to_numeric(
            pressure_row.get(
                "data_age_hours"
            ),
            errors="coerce",
        )

        if pd.notna(data_age):

            candidate[
                "operational_data_age_hours"
            ] = float(
                data_age
            )

        source_type = (
            pressure_row.get(
                "source_type"
            )
        )

        if pd.notna(source_type):

            candidate[
                "operational_source_type"
            ] = str(
                source_type
            )

    # ===============================================================
    # VOYAGE COST
    # ===============================================================

    fuel_price = None

    try:

        bunker_path = (
            ROOT
            / "04_OPTIMIZATION/data/bunker_prices.csv"
        )

        bunker_df = pd.read_csv(
            bunker_path
        )

        if bunker_df.empty:

            raise RuntimeError(
                "Bunker price dataset is empty."
            )

        price_column = next(
            (
                column
                for column in bunker_df.columns
                if "price" in column.lower()
                and "usd" in column.lower()
            ),
            None,
        )

        if price_column is None:

            raise RuntimeError(
                "No USD bunker-price column found."
            )

        valid_prices = pd.to_numeric(
            bunker_df[
                price_column
            ],
            errors="coerce",
        ).dropna()

        if valid_prices.empty:

            raise RuntimeError(
                "No valid bunker prices found."
            )

        # Latest recorded reference price.
        #
        # This is not claimed to be the exact bunkering
        # price at the origin port.
        fuel_price = float(
            valid_prices.iloc[-1]
        )

    except Exception as exc:

        candidate[
            "cost_status"
        ] = "unavailable"

        candidate[
            "cost_error"
        ] = str(exc)

        candidate[
            "feasibility"
        ] = feasibility

        return candidate

    voyage_input = VoyageInput(
        route_id=str(
            route["route_id"]
        ),
        vessel_class=str(
            vessel.vessel_class
        ),
        cargo_mt=float(
            scenario.cargo_mt
        ),
        fuel_price_usd_mt=fuel_price,
        destination_port=canonical_port_name(
            scenario.destination
        ),
        cargo_type=str(
            scenario.cargo_type
        ),
    )

    try:

        cost = calculate_voyage_cost(
            voyage_input
        )

        candidate[
            "sea_days"
        ] = float(
            cost.laden_days
        )

        candidate[
            "known_voyage_days"
        ] = float(
            cost.known_voyage_days
        )

        candidate[
            "bunker_cost_usd"
        ] = float(
            cost.bunker_cost_usd
        )

        if pd.notna(
            cost.discharge_charge_inr
        ):

            candidate[
                "handling_charge_inr"
            ] = float(
                cost.discharge_charge_inr
            )

        candidate[
            "cost_status"
        ] = str(
            cost.status
        )

    except Exception as exc:

        candidate[
            "cost_status"
        ] = "error"

        candidate[
            "cost_error"
        ] = str(exc)

        candidate[
            "feasibility"
        ] = feasibility

        return candidate

    # ===============================================================
    # PARADIP-SPECIFIC BERTH-CYCLE MODEL
    # ===============================================================
    #
    # This model is intentionally NOT generalized to the other ports.
    #
    # It predicts berth-cycle duration only.
    # It does NOT predict anchorage waiting time.
    #

    if (
        canonical_port_name(
            scenario.destination
        )
        == "Paradip"
    ):

        state = (
            get_paradip_operational_state()
        )

        if state.get(
            "status"
        ) == "available":

            candidate[
                "operational_state"
            ] = {
                "status": "available",
                "snapshot_date": state[
                    "snapshot_date"
                ],
                "snapshot_age_days": state[
                    "snapshot_age_days"
                ],
                "dry_bulk_vessels_waiting": (
                    state[
                        "dry_bulk_vessels_waiting"
                    ]
                ),
                "source_file": state[
                    "source_file"
                ],
            }

            feasible_berths = []

            berth_results = feasibility.get(
                "berths",
                []
            )

            if isinstance(
                berth_results,
                list
            ):

                for item in berth_results:

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    if (
                        item.get(
                            "status"
                        )
                        != "feasible"
                    ):
                        continue

                    berth_name = (
                        item.get(
                            "berth"
                        )
                        or item.get(
                            "berth_name"
                        )
                    )

                    if berth_name:

                        feasible_berths.append(
                            str(
                                berth_name
                            )
                        )

            # Compatibility fallback for older feasibility
            # responses that may not contain the explicit
            # berth-level status.
            if not feasible_berths:

                from feasibility import (
                    load_port_berths,
                    evaluate_berth,
                )

                berth_df = (
                    load_port_berths(
                        "Paradip"
                    )
                )

                for _, berth_row in (
                    berth_df.iterrows()
                ):

                    berth_result = (
                        evaluate_berth(
                            vessel,
                            berth_row,
                            float(
                                scenario.cargo_mt
                            ),
                            str(
                                scenario.cargo_type
                            ),
                        )
                    )

                    if (
                        berth_result.get(
                            "status"
                        )
                        != "feasible"
                    ):
                        continue

                    berth_name = (
                        berth_result.get(
                            "berth"
                        )
                    )

                    if berth_name:

                        feasible_berths.append(
                            str(
                                berth_name
                            )
                        )

            predictions = []

            for berth_name in (
                feasible_berths
            ):

                prediction = (
                    predict_paradip_berth_duration(
                        cargo=str(
                            scenario.cargo_type
                        ),
                        berth=berth_name,
                        quantity_mt=float(
                            scenario.cargo_mt
                        ),
                        dry_bulk_vessels_waiting=(
                            state[
                                "dry_bulk_vessels_waiting"
                            ]
                        ),
                    )
                )

                if (
                    prediction.get(
                        "status"
                    )
                    != "available"
                ):
                    continue

                predictions.append(
                    {
                        "berth": berth_name,
                        "predicted_days": float(
                            prediction[
                                "predicted_berth_duration_days"
                            ]
                        ),
                    }
                )

            if predictions:

                values = [
                    item[
                        "predicted_days"
                    ]
                    for item in predictions
                ]

                scenario_prediction = float(
                    pd.Series(
                        values
                    ).median()
                )

                candidate[
                    "berth_cycle_prediction"
                ] = {
                    "status": "available",
                    "predicted_berth_cycle_days": (
                        scenario_prediction
                    ),
                    "prediction_basis": (
                        "median across feasible "
                        "Paradip berths"
                    ),
                    "berth_predictions": (
                        predictions
                    ),
                    "model_target": (
                        "berth_cycle_duration"
                    ),
                }

                candidate[
                    "delivery_voyage_days"
                ] = (
                    float(
                        candidate[
                            "sea_days"
                        ]
                    )
                    + scenario_prediction
                )

            else:

                candidate[
                    "berth_cycle_prediction"
                ] = {
                    "status": "unavailable",
                    "reason": (
                        "No feasible Paradip berth "
                        "produced a model prediction."
                    ),
                }

        else:

            candidate[
                "operational_state"
            ] = {
                "status": "unavailable",
                "reason": state.get(
                    "reason",
                    "Operational state unavailable.",
                ),
            }

    # ===============================================================
    # DELIVERY ASSESSMENT
    # ===============================================================

    if candidate.get(
        "delivery_voyage_days"
    ) is not None:

        candidate[
            "delivery_feasible"
        ] = (
            candidate[
                "delivery_voyage_days"
            ]
            <= scenario.delivery_days
        )

    elif candidate.get(
        "known_voyage_days"
    ) is not None:

        candidate[
            "delivery_feasible"
        ] = (
            candidate[
                "known_voyage_days"
            ]
            <= scenario.delivery_days
        )

    candidate[
        "feasibility"
    ] = feasibility

    return candidate


def recommend(
    scenario,
):

    routes = load_routes()

    pressure = load_port_pressure()

    route = find_route(
        routes,
        scenario.origin,
        scenario.destination,
    )

    pressure_row = pressure_for_port(
        pressure,
        scenario.destination,
    )

    candidates = []

    for vessel in vessel_objects():

        candidate = build_candidate(
            scenario,
            route,
            vessel,
            pressure_row,
        )

        candidates.append(
            candidate
        )

    # ===============================================================
    # CANDIDATE TIERS
    # ===============================================================
    #
    # CONFIRMED
    #     Deterministically physically feasible
    #     + validated route
    #     + usable voyage cost
    #
    # POTENTIAL
    #     Physical compatibility unresolved
    #     + validated route
    #     + usable voyage cost
    #
    # INELIGIBLE
    #     Physical infeasibility proven
    #
    # Potential candidates are exposed to the user but are never
    # promoted to a confirmed recommendation.
    #

    confirmed = [
        candidate
        for candidate in candidates
        if (
            candidate.get(
                "feasibility_status"
            )
            == "FEASIBLE"

            and candidate.get(
                "distance_nm"
            ) is not None

            and candidate.get(
                "cost_status"
            )
            not in {
                "unavailable",
                "error",
            }
        )
    ]

    potential = [
        candidate
        for candidate in candidates
        if (
            candidate.get(
                "feasibility_status"
            )
            == "POTENTIALLY_FEASIBLE"

            and candidate.get(
                "distance_nm"
            ) is not None

            and candidate.get(
                "cost_status"
            )
            not in {
                "unavailable",
                "error",
            }
        )
    ]

    infeasible = [
        candidate
        for candidate in candidates
        if (
            candidate.get(
                "feasibility_status"
            )
            == "INFEASIBLE"
        )
    ]

    # ===============================================================
    # DELIVERY-FEASIBLE CONFIRMED CANDIDATES
    # ===============================================================

    delivery_feasible = [
        candidate
        for candidate in confirmed
        if candidate.get(
            "delivery_feasible"
        ) is True
    ]

    # ===============================================================
    # CONFIRMED RECOMMENDATION
    # ===============================================================

    if not confirmed:

        decision = (
            "NO_CONFIRMED_CANDIDATE"
        )

        best = None

    elif delivery_feasible:

        # Known bunker cost is the primary economic ranking
        # signal currently available.
        #
        # Port pressure is only a secondary operational signal.
        best = sorted(
            delivery_feasible,
            key=lambda candidate: (
                candidate[
                    "bunker_cost_usd"
                ],
                candidate[
                    "pressure_penalty"
                ]
                if candidate.get(
                    "pressure_penalty"
                ) is not None
                else math.inf,
            ),
        )[0]

        decision = "RECOMMEND"

    else:

        best = sorted(
            confirmed,
            key=lambda candidate: (
                candidate[
                    "delivery_voyage_days"
                ]
                if candidate.get(
                    "delivery_voyage_days"
                ) is not None
                else (
                    candidate[
                        "known_voyage_days"
                    ]
                    if candidate.get(
                        "known_voyage_days"
                    ) is not None
                    else math.inf
                ),
                candidate[
                    "bunker_cost_usd"
                ],
            ),
        )[0]

        decision = "DELIVERY_RISK"

    # ===============================================================
    # POTENTIAL ALTERNATIVES
    # ===============================================================
    #
    # These are not compared directly against the confirmed winner.
    #
    # They are useful decision-support outputs because the unresolved
    # physical constraint may be resolvable with additional source
    # information.
    #

    potential_alternatives = sorted(
        potential,
        key=lambda candidate: (
            candidate.get(
                "delivery_feasible"
            ) is not True,

            candidate[
                "bunker_cost_usd"
            ]
            if candidate.get(
                "bunker_cost_usd"
            ) is not None
            else math.inf,
        ),
    )

    # ===============================================================
    # LIMITATIONS
    # ===============================================================

    limitations = [
        (
            "Only routes with observed or validated "
            "distance are cost-ranked."
        ),
        (
            "Missing route distances are not estimated."
        ),
        (
            "Port pressure is a diagnostic operational "
            "signal, not a calibrated waiting-time forecast."
        ),
        (
            "Missing operational signals are not treated "
            "as zero."
        ),
        (
            "Known voyage cost remains partial because "
            "waiting, demurrage, positioning and some "
            "charter components are not yet available."
        ),
        (
            "Potentially feasible vessels are surfaced "
            "separately and are not promoted to confirmed "
            "recommendations without resolving unknown "
            "physical constraints."
        ),
        (
            "Paradip berth-cycle prediction targets "
            "berth-cycle duration and does not represent "
            "pre-berthing anchorage waiting time."
        ),
    ]

    # ===============================================================
    # RESULT
    # ===============================================================

    result = {
        "scenario": {
            "origin": scenario.origin,
            "destination": scenario.destination,
            "cargo_type": scenario.cargo_type,
            "cargo_mt": scenario.cargo_mt,
            "delivery_days": scenario.delivery_days,
        },

        "decision": decision,

        "best_candidate": best,

        "confirmed_candidates": (
            confirmed
        ),

        "potential_alternatives": (
            potential_alternatives
        ),

        "infeasible_candidates": (
            infeasible
        ),

        # Keep the complete candidate list for compatibility
        # and dashboard/debugging.
        "candidates": candidates,

        "limitations": limitations,
    }

    return result


def main():

    scenario = Scenario(
        origin="Newcastle Australia",
        destination="Paradip India",
        cargo_type="coking coal",
        cargo_mt=60000,
        delivery_days=30,
    )

    result = recommend(
        scenario
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":

    main()