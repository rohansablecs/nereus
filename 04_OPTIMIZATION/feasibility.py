from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT_DATA = PROJECT_ROOT / "02_DATA/raw/port"


@dataclass
class Vessel:
    vessel_id: str
    vessel_class: str
    dwt_mt: float
    loa_m: float
    beam_m: float
    draft_m: float


def _num(value: Any) -> float | None:
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def _contains(text: str, terms: list[str]) -> bool:
    text = text.lower()
    return any(term.lower() in text for term in terms)


def load_port_berths(destination: str) -> pd.DataFrame:
    """
    Load the canonical berth dataset for a supported destination.

    Returns a normalized dataframe with common constraint columns.
    Source-specific fields are preserved where useful.

    Missing source data is preserved as unknown rather than
    being replaced with assumptions.
    """

    destination_key = destination.strip().lower()

    # ---------------------------------------------------------
    # PARADIP
    # ---------------------------------------------------------
    if destination_key == "paradip":
        path = PORT_DATA / "paradip_berths.csv"

        df = pd.read_csv(path)

        return pd.DataFrame({
            "port": "Paradip",
            "berth": df["berth_name"],
            "loa_limit_m": pd.to_numeric(
                df["loa_m"],
                errors="coerce",
            ),
            "beam_limit_m": pd.to_numeric(
                df["beam_m"],
                errors="coerce",
            ),
            "draft_limit_m": pd.to_numeric(
                df["draft_m"],
                errors="coerce",
            ),
            "displacement_limit_mt": pd.NA,
            "cargo_constraint": df["commodity"],
            "allocation_priority": df["remarks"],
            "source_berth_type": df["berth_type"],
        })

    # ---------------------------------------------------------
    # DHAMRA
    # ---------------------------------------------------------
    if destination_key == "dhamra":
        path = PORT_DATA / "dhamra_bpts_berths.csv"

        df = pd.read_csv(path)

        return pd.DataFrame({
            "port": "Dhamra",
            "berth": df["berth"],
            "loa_limit_m": pd.to_numeric(
                df["loa_m"],
                errors="coerce",
            ),
            "beam_limit_m": pd.NA,
            "draft_limit_m": pd.to_numeric(
                df["max_draft_m"],
                errors="coerce",
            ),
            "displacement_limit_mt": pd.to_numeric(
                df["max_displacement_mt"],
                errors="coerce",
            ),
            "cargo_constraint": df["cargo_scope"],
            "allocation_priority": df["berth_category"],
            "source_berth_type": df["berth_category"],
        })

    # ---------------------------------------------------------
    # GOPALPUR
    # ---------------------------------------------------------
    if destination_key == "gopalpur":
        path = PORT_DATA / "gopalpur_berths.csv"

        df = pd.read_csv(path)

        return pd.DataFrame({
            "port": "Gopalpur",
            "berth": df["berth"],
            "loa_limit_m": pd.to_numeric(
                df["loa_m"],
                errors="coerce",
            ),
            "beam_limit_m": pd.NA,
            "draft_limit_m": pd.to_numeric(
                df["max_draft_m"],
                errors="coerce",
            ),
            "displacement_limit_mt": pd.to_numeric(
                df["displacement_mt"],
                errors="coerce",
            ),
            "cargo_constraint": df["allocation_priority"],
            "allocation_priority": df["allocation_priority"],
            "source_berth_type": pd.NA,
        })

    # ---------------------------------------------------------
    # VISAKHAPATNAM
    # ---------------------------------------------------------
    if destination_key == "visakhapatnam":
        path = PORT_DATA / "vizag_berths_current.csv"

        df = pd.read_csv(path)

        return pd.DataFrame({
            "port": "Visakhapatnam",
            "berth": df["berth"],
            "loa_limit_m": pd.to_numeric(
                df["permissible_loa_m"],
                errors="coerce",
            ),
            "beam_limit_m": pd.NA,
            "draft_limit_m": pd.to_numeric(
                df["permissible_draft_m"],
                errors="coerce",
            ),
            "displacement_limit_mt": pd.NA,
            "cargo_constraint": df["cargo"],
            "allocation_priority": pd.NA,
            "source_berth_type": pd.NA,
        })

    # ---------------------------------------------------------
    # GANGAVARAM
    # ---------------------------------------------------------
    if destination_key == "gangavaram":
        path = PORT_DATA / "gangavaram_berths.csv"

        df = pd.read_csv(path)

        return pd.DataFrame({
            "port": "Gangavaram",
            "berth": df["berth_no"],
            "loa_limit_m": pd.to_numeric(
                df["permissible_loa_m"],
                errors="coerce",
            ),
            "beam_limit_m": pd.NA,
            "draft_limit_m": pd.to_numeric(
                df["permissible_draft_m"],
                errors="coerce",
            ),
            "displacement_limit_mt": pd.to_numeric(
                df["displacement_mt"],
                errors="coerce",
            ),
            "cargo_constraint": pd.NA,
            "allocation_priority": pd.NA,
            "source_berth_type": df["constraint_variant"],
        })

    raise ValueError(
        f"Unsupported destination '{destination}'. "
        "Supported: Paradip, Dhamra, Gopalpur, "
        "Visakhapatnam, Gangavaram."
    )


def evaluate_berth(
    vessel: Vessel,
    berth: pd.Series,
    cargo_mt: float,
    cargo_type: str,
) -> dict[str, Any]:

    reasons: list[str] = []
    checks: list[dict[str, Any]] = []
    unknown_reasons: list[str] = []

    # ---------------------------------------------------------
    # LOA
    # ---------------------------------------------------------
    loa_limit = _num(berth["loa_limit_m"])

    if loa_limit is None:
        checks.append({
            "constraint": "loa",
            "status": "not_evaluated",
            "reason": "No berth LOA limit available in source dataset.",
        })
    elif vessel.loa_m <= loa_limit:
        checks.append({
            "constraint": "loa",
            "status": "pass",
            "vessel_value": vessel.loa_m,
            "limit": loa_limit,
        })
    else:
        reasons.append("loa_exceeds_limit")

        checks.append({
            "constraint": "loa",
            "status": "fail",
            "vessel_value": vessel.loa_m,
            "limit": loa_limit,
        })

    # ---------------------------------------------------------
    # BEAM
    # ---------------------------------------------------------
    beam_limit = _num(berth["beam_limit_m"])

    if beam_limit is None:
        checks.append({
            "constraint": "beam",
            "status": "not_evaluated",
            "reason": "No berth beam limit available in source dataset.",
        })
    elif vessel.beam_m <= beam_limit:
        checks.append({
            "constraint": "beam",
            "status": "pass",
            "vessel_value": vessel.beam_m,
            "limit": beam_limit,
        })
    else:
        reasons.append("beam_exceeds_limit")

        checks.append({
            "constraint": "beam",
            "status": "fail",
            "vessel_value": vessel.beam_m,
            "limit": beam_limit,
        })

    # ---------------------------------------------------------
    # DRAFT
    # ---------------------------------------------------------
    draft_limit = _num(berth["draft_limit_m"])

    if draft_limit is None:
        checks.append({
            "constraint": "draft",
            "status": "not_evaluated",
            "reason": "No berth draft limit available in source dataset.",
        })
    elif vessel.draft_m <= draft_limit:
        checks.append({
            "constraint": "draft",
            "status": "pass",
            "vessel_value": vessel.draft_m,
            "limit": draft_limit,
        })
    else:
        reasons.append("draft_exceeds_limit")

        checks.append({
            "constraint": "draft",
            "status": "fail",
            "vessel_value": vessel.draft_m,
            "limit": draft_limit,
        })

    # ---------------------------------------------------------
    # DISPLACEMENT
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # DWT and displacement are different physical quantities.
    #
    # Therefore we DO NOT compare:
    #
    #     vessel.dwt_mt <= displacement_limit
    #
    # when actual vessel displacement is unavailable.
    #
    # The constraint remains unknown instead of creating a
    # physically invalid feasibility decision.
    # ---------------------------------------------------------
    displacement_limit = _num(
        berth["displacement_limit_mt"]
    )

    if displacement_limit is None:
        checks.append({
            "constraint": "displacement",
            "status": "not_evaluated",
            "reason": (
                "No berth displacement limit available "
                "in source dataset."
            ),
        })
    else:
        checks.append({
            "constraint": "displacement",
            "status": "not_evaluated",
            "reason": (
                "Vessel displacement is not available. "
                "DWT is not used as a displacement proxy."
            ),
            "limit": displacement_limit,
        })

    # ---------------------------------------------------------
    # CARGO COMPATIBILITY
    # ---------------------------------------------------------
    cargo_constraint = _text(
        berth["cargo_constraint"]
    )

    if not cargo_constraint:

        checks.append({
            "constraint": "cargo_compatibility",
            "status": "unknown",
            "reason": (
                "No berth cargo restriction available "
                "in source dataset."
            ),
        })

        unknown_reasons.append(
            "cargo_compatibility_unknown"
        )

    else:

        cargo_specific_terms = [
            "coal",
            "iron ore",
            "ore",
            "container",
            "fertilizer",
            "pol",
            "oil",
        ]

        if not _contains(
            cargo_constraint,
            cargo_specific_terms,
        ):

            checks.append({
                "constraint": "cargo_compatibility",
                "status": "unknown",
                "reason": (
                    "Berth cargo field is not specific enough "
                    "for deterministic compatibility evaluation."
                ),
                "source_value": cargo_constraint,
            })

            unknown_reasons.append(
                "cargo_compatibility_unknown"
            )

        elif _contains(
            cargo_constraint,
            [cargo_type],
        ):

            checks.append({
                "constraint": "cargo_compatibility",
                "status": "pass",
                "source_value": cargo_constraint,
            })

        elif (
            cargo_type.lower()
            in {"coking coal", "coal"}
            and "coal" in cargo_constraint.lower()
        ):

            checks.append({
                "constraint": "cargo_compatibility",
                "status": "pass",
                "source_value": cargo_constraint,
            })

        elif (
            cargo_type.lower()
            in {
                "iron ore",
                "iron ore fines",
                "iron ore pellets",
            }
            and "iron ore" in cargo_constraint.lower()
        ):

            checks.append({
                "constraint": "cargo_compatibility",
                "status": "pass",
                "source_value": cargo_constraint,
            })

        else:

            checks.append({
                "constraint": "cargo_compatibility",
                "status": "unknown",
                "reason": (
                    "Source cargo restriction does not "
                    "match scenario explicitly."
                ),
                "source_value": cargo_constraint,
            })

            unknown_reasons.append(
                "cargo_compatibility_unknown"
            )

    # ---------------------------------------------------------
    # CARGO VS DWT SANITY
    # ---------------------------------------------------------
    if cargo_mt <= vessel.dwt_mt:

        checks.append({
            "constraint": "cargo_vs_dwt_sanity",
            "status": "pass",
            "cargo_mt": cargo_mt,
            "vessel_dwt_mt": vessel.dwt_mt,
        })

    else:

        reasons.append(
            "cargo_exceeds_vessel_dwt_sanity"
        )

        checks.append({
            "constraint": "cargo_vs_dwt_sanity",
            "status": "fail",
            "cargo_mt": cargo_mt,
            "vessel_dwt_mt": vessel.dwt_mt,
        })

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------
    #
    # Hard constraint failure always means infeasible.
    #
    # No hard failure + unresolved required information
    # means unknown.
    #
    # Only a fully evaluated berth can be called feasible.
    # ---------------------------------------------------------
    if reasons:
        status = "infeasible"
    elif unknown_reasons:
        status = "unknown"
    else:
        status = "feasible"

    return {
        "port": berth["port"],
        "berth": berth["berth"],
        "constraint_variant": _text(
            berth["source_berth_type"]
        ),
        "status": status,
        "feasible": status == "feasible",
        "reasons": reasons,
        "unknown_reasons": unknown_reasons,
        "checks": checks,
    }


def evaluate_vessel(
    vessel: Vessel,
    destination: str,
    cargo_mt: float,
    cargo_type: str,
) -> dict[str, Any]:

    berths = load_port_berths(destination)

    berth_results = [
        evaluate_berth(
            vessel=vessel,
            berth=row,
            cargo_mt=cargo_mt,
            cargo_type=cargo_type,
        )
        for _, row in berths.iterrows()
    ]

    feasible_berths = [
        result["berth"]
        for result in berth_results
        if result["status"] == "feasible"
    ]

    unknown_berths = [
        result["berth"]
        for result in berth_results
        if result["status"] == "unknown"
    ]

    infeasible_berths = [
        result["berth"]
        for result in berth_results
        if result["status"] == "infeasible"
    ]

    if feasible_berths:
        status = "FEASIBLE"
        feasible = True

    elif unknown_berths:
        status = "POTENTIALLY_FEASIBLE"
        feasible = False

    else:
        status = "INFEASIBLE"
        feasible = False

    return {
        "vessel": asdict(vessel),
        "destination": destination,
        "cargo_mt": cargo_mt,
        "cargo_type": cargo_type,

        # Backward-compatible boolean.
        #
        # True only when at least one berth is
        # deterministically feasible.
        "feasible": feasible,

        # Explicit decision status.
        "status": status,

        "feasible_berths": feasible_berths,
        "unknown_berths": unknown_berths,
        "infeasible_berths": infeasible_berths,

        "berths_evaluated": len(
            berth_results
        ),

        "berths": berth_results,
    }


def evaluate_scenario(
    cargo_mt: float,
    cargo_type: str,
    destination: str,
    vessels: list[Vessel],
) -> dict[str, Any]:

    results = [
        evaluate_vessel(
            vessel=vessel,
            destination=destination,
            cargo_mt=cargo_mt,
            cargo_type=cargo_type,
        )
        for vessel in vessels
    ]

    return {
        "scenario": {
            "cargo_mt": cargo_mt,
            "cargo_type": cargo_type,
            "destination": destination,
        },
        "vessels": results,
    }