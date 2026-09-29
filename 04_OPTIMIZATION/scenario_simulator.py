from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "04_OPTIMIZATION"))

from berth_duration_predictor import predict_paradip_berth_duration
from port_operational_state import get_paradip_operational_state
from recommendation_engine import Scenario, recommend


# ---------------------------------------------------------------------------
# BASE SCENARIO
# ---------------------------------------------------------------------------

BASE_SCENARIO = Scenario(
    origin="Newcastle Australia",
    cargo_type="coking coal",
    cargo_mt=60_000,
    destination="Paradip India",
    delivery_days=30,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_panamax_candidate(result: dict) -> dict | None:
    candidates = result.get("candidates", [])

    for candidate in candidates:
        if candidate.get("vessel_class") == "Panamax":
            return candidate

    return None


def get_waiting_state() -> dict:
    state = get_paradip_operational_state()

    if not state:
        raise RuntimeError(
            "Paradip operational state is unavailable."
        )

    if state.get("status") != "available":
        raise RuntimeError(
            f"Paradip operational state unavailable: {state}"
        )

    return state


def predict_with_waiting(
    candidate: dict,
    waiting_vessels: float,
) -> dict:
    """
    Re-run the trained Paradip berth-cycle predictor using an
    explicitly supplied waiting-vessel scenario.

    This is the key what-if mechanism.

    The current real-world waiting state is NOT mutated.
    We only change the model input for the scenario.
    """

    feasibility = candidate.get("feasibility", {})

    feasible_berths = []

    for berth_result in feasibility.get(
        "berths",
        [],
    ):
        if berth_result.get("feasible") is True:
            feasible_berths.append(
                berth_result
            )

    if not feasible_berths:
        return {
            "status": "unavailable",
            "reason": "No feasible Paradip berths.",
        }

    predictions = []

    for berth_result in feasible_berths:

        berth_name = berth_result.get(
            "berth"
        )

        prediction = predict_paradip_berth_duration(
            cargo=BASE_SCENARIO.cargo_type,
            berth=str(berth_name),
            quantity_mt=BASE_SCENARIO.cargo_mt,
            dry_bulk_vessels_waiting=float(
                waiting_vessels
            ),
        )

        if prediction.get("status") != "available":
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

    if not predictions:
        return {
            "status": "unavailable",
            "reason": (
                "Berth-cycle model returned no "
                "usable predictions."
            ),
        }

    values = [
        item["predicted_days"]
        for item in predictions
    ]

    # Median across currently feasible berths.
    values_sorted = sorted(values)
    n = len(values_sorted)

    if n % 2:
        median_prediction = values_sorted[n // 2]
    else:
        median_prediction = (
            values_sorted[n // 2 - 1]
            + values_sorted[n // 2]
        ) / 2.0

    return {
        "status": "available",
        "predicted_berth_cycle_days": (
            median_prediction
        ),
        "prediction_basis": (
            "median across feasible Paradip berths"
        ),
        "berth_predictions": predictions,
        "model_target": "berth_cycle_duration",
        "waiting_vessels_input": float(
            waiting_vessels
        ),
    }


def calculate_delivery(
    candidate: dict,
    berth_cycle_days: float,
    deadline_days: float,
) -> dict:
    """
    Recalculate delivery using the shocked berth-cycle
    duration.

    We intentionally do not modify the underlying voyage
    cost or freight model here.
    """

    sea_days = candidate.get(
        "sea_days"
    )

    if sea_days is None:
        return {
            "required_days": None,
            "slack_days": None,
            "delivery_feasible": False,
        }

    sea_days = float(sea_days)

    required_days = (
        sea_days
        + berth_cycle_days
    )

    slack_days = (
        float(deadline_days)
        - required_days
    )

    return {
        "required_days": required_days,
        "slack_days": slack_days,
        "delivery_feasible": (
            slack_days >= 0
        ),
    }


def run_waiting_scenario(
    base_candidate: dict,
    base_waiting: float,
    multiplier: float,
) -> dict:

    shocked_waiting = (
        base_waiting
        * multiplier
    )

    berth_prediction = predict_with_waiting(
        candidate=base_candidate,
        waiting_vessels=shocked_waiting,
    )

    if berth_prediction.get("status") != "available":
        return {
            "waiting_multiplier": multiplier,
            "waiting_vessels": shocked_waiting,
            "status": "unavailable",
            "berth_prediction": berth_prediction,
        }

    delivery = calculate_delivery(
        candidate=base_candidate,
        berth_cycle_days=float(
            berth_prediction[
                "predicted_berth_cycle_days"
            ]
        ),
        deadline_days=BASE_SCENARIO.delivery_days,
    )

    return {
        "waiting_multiplier": multiplier,
        "waiting_vessels": shocked_waiting,
        "status": "available",
        "berth_prediction": berth_prediction,
        "required_days": delivery[
            "required_days"
        ],
        "slack_days": delivery[
            "slack_days"
        ],
        "delivery_feasible": delivery[
            "delivery_feasible"
        ],
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    print()
    print("=" * 90)
    print("NEREUS — OPERATIONAL WHAT-IF SIMULATOR")
    print("=" * 90)

    print()
    print("BASE SCENARIO")
    print("-" * 90)

    print(
        f"Origin             : "
        f"{BASE_SCENARIO.origin}"
    )

    print(
        f"Cargo              : "
        f"{BASE_SCENARIO.cargo_mt:,.0f} MT "
        f"{BASE_SCENARIO.cargo_type}"
    )

    print(
        f"Destination        : "
        f"{BASE_SCENARIO.destination}"
    )

    print(
        f"Deadline            : "
        f"{BASE_SCENARIO.delivery_days:g} days"
    )

    # -----------------------------------------------------------------------
    # Current operational state
    # -----------------------------------------------------------------------

    state = get_waiting_state()

    base_waiting = float(
        state["dry_bulk_vessels_waiting"]
    )

    print()
    print("CURRENT OPERATIONAL STATE")
    print("-" * 90)

    print(
        f"Dry-bulk vessels waiting : "
        f"{base_waiting:g}"
    )

    print(
        f"Snapshot                : "
        f"{state.get('snapshot_date')}"
    )

    print(
        f"Source                  : "
        f"{state.get('source_file')}"
    )

    print(
        f"Snapshot age            : "
        f"{state.get('snapshot_age_days')} days"
    )

    # -----------------------------------------------------------------------
    # Base recommendation
    # -----------------------------------------------------------------------

    result = recommend(
        BASE_SCENARIO
    )

    base_candidate = get_panamax_candidate(
        result
    )

    if base_candidate is None:
        raise RuntimeError(
            "Panamax candidate was not produced "
            "by the recommendation engine."
        )

    print()
    print("BASE DECISION")
    print("-" * 90)

    print(
        f"Vessel               : "
        f"{base_candidate.get('vessel_class')}"
    )

    print(
        f"Feasible             : "
        f"{base_candidate.get('feasible')}"
    )

    print(
        f"Route                : "
        f"{base_candidate.get('route_id')}"
    )

    print(
        f"Sea days             : "
        f"{base_candidate.get('sea_days')}"
    )

    print(
        f"Current berth cycle  : "
        f"{base_candidate.get('berth_cycle_prediction', {}).get('predicted_berth_cycle_days')}"
    )

    # -----------------------------------------------------------------------
    # Waiting shocks
    # -----------------------------------------------------------------------

    multipliers = [
        0.75,
        0.90,
        1.00,
        1.25,
        1.50,
        2.00,
    ]

    scenarios = []

    for multiplier in multipliers:

        scenario = run_waiting_scenario(
            base_candidate=base_candidate,
            base_waiting=base_waiting,
            multiplier=multiplier,
        )

        scenarios.append(
            scenario
        )

    print()
    print("WAITING-PRESSURE SCENARIOS")
    print("-" * 90)

    for scenario in scenarios:

        waiting = scenario[
            "waiting_vessels"
        ]

        if scenario["status"] != "available":

            print(
                f"Waiting {waiting:6.1f} | "
                f"prediction unavailable"
            )

            continue

        berth = scenario[
            "berth_prediction"
        ][
            "predicted_berth_cycle_days"
        ]

        total = scenario[
            "required_days"
        ]

        slack = scenario[
            "slack_days"
        ]

        feasible = scenario[
            "delivery_feasible"
        ]

        print(
            f"Waiting {waiting:6.1f} | "
            f"berth={berth:7.3f} d | "
            f"total={total:7.3f} d | "
            f"slack={slack:7.3f} d | "
            f"feasible={feasible}"
        )

    # -----------------------------------------------------------------------
    # Deadline sensitivity
    # -----------------------------------------------------------------------

    base_berth_prediction = predict_with_waiting(
        candidate=base_candidate,
        waiting_vessels=base_waiting,
    )

    if base_berth_prediction.get(
        "status"
    ) == "available":

        base_berth_days = float(
            base_berth_prediction[
                "predicted_berth_cycle_days"
            ]
        )

        print()
        print("DEADLINE SENSITIVITY")
        print("-" * 90)

        for deadline in [
            20,
            21,
            22,
            23,
            25,
            30,
            35,
        ]:

            delivery = calculate_delivery(
                candidate=base_candidate,
                berth_cycle_days=base_berth_days,
                deadline_days=deadline,
            )

            print(
                f"Deadline {deadline:2d}d | "
                f"required="
                f"{delivery['required_days']:7.3f} d | "
                f"slack="
                f"{delivery['slack_days']:7.3f} d | "
                f"feasible="
                f"{delivery['delivery_feasible']}"
            )

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print(
        "Waiting shocks are now passed directly into the "
        "trained Paradip berth-cycle model."
    )

    print(
        "The base operational snapshot is not modified; "
        "only the model input is perturbed."
    )

    print(
        "Sea-voyage duration is held constant while berth "
        "cycle duration responds to the waiting-pressure shock."
    )

    print(
        "This remains a what-if simulation, not a forecast "
        "of future port conditions."
    )

    print(
        "Pre-berthing anchorage waiting remains separate from "
        "the berth-cycle target."
    )


if __name__ == "__main__":
    main()
