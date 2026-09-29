import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.ai_decision import (
    AIDecisionError,
    analyze_with_ai,
)


# =========================================================
# PATHS
# =========================================================

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")

DATA_ROOT = PROJECT_ROOT / "02_DATA"
OPTIMIZATION_ROOT = PROJECT_ROOT / "04_OPTIMIZATION"


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="NEREUS",
    version="0.1.0",
    description=(
        "SIH26006 Intelligent Freight Forecasting "
        "and Vessel Chartering Decision Support"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATA HELPERS
# =========================================================

def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def optional_csv(path: Path):
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def unique_column(rows, column):
    values = []

    for row in rows:
        value = str(row.get(column, "")).strip()

        if value and value not in values:
            values.append(value)

    return values


def clean_value(value):
    """
    Convert empty strings to None while preserving
    meaningful zero/false values.
    """
    if value == "":
        return None

    return value


# =========================================================
# REGISTRY PATHS
# =========================================================

ORIGIN_MASTER = (
    DATA_ROOT
    / "raw"
    / "maritime"
    / "origins"
    / "origin_master.csv"
)

DESTINATION_MASTER = (
    DATA_ROOT
    / "raw"
    / "maritime"
    / "destinations"
    / "destination_master.csv"
)

ROUTE_MATRIX = (
    DATA_ROOT
    / "raw"
    / "maritime"
    / "searoute"
    / "route_distances.csv"
)

VESSEL_MASTER = (
    OPTIMIZATION_ROOT
    / "data"
    / "vessels.csv"
)

PORT_OPERATIONS = (
    OPTIMIZATION_ROOT
    / "data"
    / "port_operations.csv"
)


# =========================================================
# BASIC
# =========================================================

@app.get("/")
def root():
    return {
        "service": "NEREUS",
        "version": "0.1.0",
        "problem_statement": "SIH26006",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "nereus-api",
    }


# =========================================================
# DATA REGISTRY
# =========================================================

@app.get("/api/decision/metadata")
def decision_metadata():

    try:
        origins = read_csv(ORIGIN_MASTER)
        destinations = read_csv(DESTINATION_MASTER)
        routes = read_csv(ROUTE_MATRIX)
        vessels = read_csv(VESSEL_MASTER)
        operations = optional_csv(PORT_OPERATIONS)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required NEREUS data file not found: "
                f"{error}"
            ),
        )

    cargoes = unique_column(
        operations,
        "cargo_type",
    )

    return {
        "problem_statement": {
            "id": "SIH26006",
            "organization": "Ministry of Steel",
            "department": "SAIL",
            "category": "Software",
            "theme": "Transportation & Logistics",
            "title": (
                "Development of an Intelligent Freight "
                "Forecasting Model for Optimized Vessel "
                "Chartering and Bulk Cargo Procurement "
                "from overseas to East Coast of India"
            ),
        },

        "origins": origins,

        "destinations": destinations,

        "routes": routes,

        "vessels": vessels,

        "cargoes": cargoes,

        "contract_strategies": [
            {
                "id": "spot",
                "name": "Spot",
            },
            {
                "id": "short_term",
                "name": "Short term",
            },
            {
                "id": "medium_term",
                "name": "Medium term",
            },
            {
                "id": "multiple_voyage",
                "name": "Multiple voyage",
            },
        ],

        "counts": {
            "origins": len(origins),
            "destinations": len(destinations),
            "routes": len(routes),
            "vessels": len(vessels),
            "cargoes": len(cargoes),
        },
    }


# =========================================================
# DECISION REQUEST
# =========================================================

class DecisionRequest(BaseModel):
    cargo: str
    quantity_mt: float
    origin_id: str
    destination_id: str
    delivery_window_days: int
    contract_strategy: str


# =========================================================
# AI EVIDENCE BUILDER
# =========================================================

def build_ai_payload(
    request,
    origin_name,
    destination_name,
    compact_candidates,
    best_summary,
    limitations,
):
    """
    Build a clean evidence package for the AI reasoning layer.

    The AI does NOT receive authority over the deterministic
    calculations. It receives the facts already calculated
    by NEREUS and interprets them.
    """

    return {
        "scenario": {
            "cargo": request.cargo,
            "quantity_mt": request.quantity_mt,
            "origin_id": request.origin_id,
            "origin": origin_name,
            "destination_id": request.destination_id,
            "destination": destination_name,
            "delivery_window_days": (
                request.delivery_window_days
            ),
            "contract_strategy": (
                request.contract_strategy
            ),
        },

        "deterministic_decision": {
            "best_candidate": best_summary,
            "candidate_count": len(
                compact_candidates
            ),
        },

        "candidates": compact_candidates,

        "limitations": limitations,
    }


# =========================================================
# DECISION ENGINE
# =========================================================

@app.post("/api/decision/analyze")
def analyze_decision(request: DecisionRequest):

    try:

        # -------------------------------------------------
        # Validate origin
        # -------------------------------------------------

        origins = read_csv(ORIGIN_MASTER)

        origin_row = next(
            (
                row
                for row in origins
                if row.get("origin_id", "").strip()
                == request.origin_id.strip()
            ),
            None,
        )

        if origin_row is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown origin_id: "
                    f"{request.origin_id}"
                ),
            )

        # -------------------------------------------------
        # Validate destination
        # -------------------------------------------------

        destinations = read_csv(DESTINATION_MASTER)

        destination_row = next(
            (
                row
                for row in destinations
                if row.get("destination_id", "").strip()
                == request.destination_id.strip()
            ),
            None,
        )

        if destination_row is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown destination_id: "
                    f"{request.destination_id}"
                ),
            )

        origin_name = (
            origin_row.get(
                "origin_name",
                "",
            )
            .strip()
        )

        destination_name = (
            destination_row.get(
                "destination_name",
                "",
            )
            .strip()
        )

        # -------------------------------------------------
        # Load optimization engine
        # -------------------------------------------------

        optimization_path = str(
            OPTIMIZATION_ROOT
        )

        if optimization_path not in sys.path:
            sys.path.insert(
                0,
                optimization_path,
            )

        from recommendation_engine import (
            recommend,
            Scenario,
        )

        # -------------------------------------------------
        # Build scenario
        # -------------------------------------------------

        scenario = Scenario(
            origin=origin_name,
            cargo_type=request.cargo,
            cargo_mt=request.quantity_mt,
            destination=destination_name,
            delivery_days=request.delivery_window_days,
        )

        # -------------------------------------------------
        # Run deterministic recommendation engine
        # -------------------------------------------------

        raw_result = recommend(
            scenario
        )

        # -------------------------------------------------
        # Compact candidates
        # -------------------------------------------------

        compact_candidates = []

        for candidate in raw_result.get(
            "candidates",
            [],
        ):

            feasibility = candidate.get(
                "feasibility",
                {},
            )

            feasible_berths = (
                feasibility.get(
                    "feasible_berths",
                    [],
                )
                or []
            )

            unknown_berths = (
                feasibility.get(
                    "unknown_berths",
                    [],
                )
                or []
            )

            infeasible_berths = (
                feasibility.get(
                    "infeasible_berths",
                    [],
                )
                or []
            )

            # ---------------------------------------------
            # Berth cycle prediction
            # ---------------------------------------------

            berth_prediction = candidate.get(
                "berth_cycle_prediction"
            )

            prediction = None

            if berth_prediction:

                prediction = {
                    "status": berth_prediction.get(
                        "status"
                    ),
                    "predicted_berth_cycle_days": (
                        berth_prediction.get(
                            "predicted_berth_cycle_days"
                        )
                    ),
                    "prediction_basis": (
                        berth_prediction.get(
                            "prediction_basis"
                        )
                    ),
                }

            # ---------------------------------------------
            # Operational state
            # ---------------------------------------------

            operational_state = candidate.get(
                "operational_state"
            )

            compact_operational_state = None

            if operational_state:

                compact_operational_state = {
                    "status": operational_state.get(
                        "status"
                    ),
                    "snapshot_date": (
                        operational_state.get(
                            "snapshot_date"
                        )
                    ),
                    "snapshot_age_days": (
                        operational_state.get(
                            "snapshot_age_days"
                        )
                    ),
                    "dry_bulk_vessels_waiting": (
                        operational_state.get(
                            "dry_bulk_vessels_waiting"
                        )
                    ),
                }

            # ---------------------------------------------
            # Candidate
            # ---------------------------------------------

            compact_candidates.append(
                {
                    "vessel_class": candidate.get(
                        "vessel_class"
                    ),

                    "feasible": candidate.get(
                        "feasible"
                    ),

                    "feasibility_status": candidate.get(
                        "feasibility_status"
                    ),

                    "cargo_mt": candidate.get(
                        "cargo_mt"
                    ),

                    # -------------------------------------
                    # Route
                    # -------------------------------------

                    "route": {
                        "route_id": candidate.get(
                            "route_id"
                        ),
                        "status": candidate.get(
                            "route_status"
                        ),
                        "distance_nm": candidate.get(
                            "distance_nm"
                        ),
                        "sea_days": candidate.get(
                            "sea_days"
                        ),
                        "known_voyage_days": candidate.get(
                            "known_voyage_days"
                        ),
                    },

                    # -------------------------------------
                    # Economics
                    # -------------------------------------

                    "economics": {
                        "bunker_cost_usd": candidate.get(
                            "bunker_cost_usd"
                        ),
                        "handling_charge_inr": candidate.get(
                            "handling_charge_inr"
                        ),
                        "cost_status": candidate.get(
                            "cost_status"
                        ),
                    },

                    # -------------------------------------
                    # Delivery
                    # -------------------------------------

                    "delivery": {
                        "delivery_window_days": (
                            request.delivery_window_days
                        ),
                        "voyage_days": candidate.get(
                            "delivery_voyage_days"
                        ),
                        "feasible": candidate.get(
                            "delivery_feasible"
                        ),
                    },

                    # -------------------------------------
                    # Port
                    # -------------------------------------

                    "port": {
                        "name": destination_name,
                        "pressure": candidate.get(
                            "port_pressure"
                        ),
                        "pressure_band": candidate.get(
                            "port_pressure_band"
                        ),
                        "pressure_penalty": candidate.get(
                            "pressure_penalty"
                        ),
                        "pressure_confidence": candidate.get(
                            "pressure_confidence"
                        ),
                    },

                    # -------------------------------------
                    # Operations
                    # -------------------------------------

                    "operations": {
                        "data_age_hours": candidate.get(
                            "operational_data_age_hours"
                        ),
                        "source_type": candidate.get(
                            "operational_source_type"
                        ),
                        "state": (
                            compact_operational_state
                        ),
                        "berth_cycle": prediction,
                    },

                    # -------------------------------------
                    # Berths
                    # -------------------------------------

                    "berths": {
                        "evaluated": feasibility.get(
                            "berths_evaluated"
                        ),
                        "feasible_count": len(
                            feasible_berths
                        ),
                        "unknown_count": len(
                            unknown_berths
                        ),
                        "infeasible_count": len(
                            infeasible_berths
                        ),
                        "feasible": (
                            feasible_berths
                        ),
                        "unknown": (
                            unknown_berths
                        ),
                    },

                    # -------------------------------------
                    # Reason
                    # -------------------------------------

                    "reason": candidate.get(
                        "reason"
                    ),
                }
            )

        # =================================================
        # BEST CANDIDATE
        # =================================================

        best = raw_result.get(
            "best_candidate"
        )

        best_summary = None

        if best:

            best_feasibility = best.get(
                "feasibility",
                {},
            )

            best_summary = {
                "vessel_class": best.get(
                    "vessel_class"
                ),

                "feasible": best.get(
                    "feasible"
                ),

                "feasibility_status": best.get(
                    "feasibility_status"
                ),

                "route_id": best.get(
                    "route_id"
                ),

                "distance_nm": best.get(
                    "distance_nm"
                ),

                "sea_days": best.get(
                    "sea_days"
                ),

                "known_voyage_days": best.get(
                    "known_voyage_days"
                ),

                "delivery_voyage_days": best.get(
                    "delivery_voyage_days"
                ),

                "bunker_cost_usd": best.get(
                    "bunker_cost_usd"
                ),

                "handling_charge_inr": best.get(
                    "handling_charge_inr"
                ),

                "port_pressure": best.get(
                    "port_pressure"
                ),

                "port_pressure_band": best.get(
                    "port_pressure_band"
                ),

                "pressure_confidence": best.get(
                    "pressure_confidence"
                ),

                "feasible_berths": (
                    best_feasibility.get(
                        "feasible_berths",
                        [],
                    )
                ),

                "reason": best.get(
                    "reason"
                ),
            }

        # =================================================
        # LIMITATIONS
        # =================================================

        limitations = raw_result.get(
            "limitations",
            [],
        )

        # =================================================
        # AI DECISION LAYER
        # =================================================

        ai_payload = build_ai_payload(
            request=request,
            origin_name=origin_name,
            destination_name=destination_name,
            compact_candidates=compact_candidates,
            best_summary=best_summary,
            limitations=limitations,
        )

        ai_decision = None
        ai_status = "unavailable"
        ai_error = None

        try:

            ai_decision = analyze_with_ai(
                ai_payload
            )

            ai_status = "available"

        except AIDecisionError as error:

            ai_status = "unavailable"
            ai_error = str(error)

        except Exception as error:

            ai_status = "error"
            ai_error = str(error)

        # =================================================
        # RESPONSE
        # =================================================

        response = {
            "status": "ok",

            "scenario": {
                "cargo": request.cargo,
                "quantity_mt": request.quantity_mt,

                "origin_id": request.origin_id,
                "origin": origin_name,

                "destination_id": request.destination_id,
                "destination": destination_name,

                "delivery_window_days": (
                    request.delivery_window_days
                ),

                "contract_strategy": (
                    request.contract_strategy
                ),
            },

            "decision": {
                "best_candidate": best_summary,
                "candidate_count": len(
                    compact_candidates
                ),
            },

            "candidates": compact_candidates,

            "engine_status": raw_result.get(
                "status"
            ),

            "limitations": limitations,

            "ai": {
                "status": ai_status,
                "model": os.getenv(
                    "NEREUS_AI_MODEL",
                    "moonshotai/Kimi-K2-Instruct-0905",
                ),
                "decision": ai_decision,
            },
        }

        # Only expose the AI error when AI is unavailable.
        # Do not allow AI failure to break the deterministic
        # NEREUS decision engine.
        if ai_error:
            response["ai"]["error"] = ai_error

        return response

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )