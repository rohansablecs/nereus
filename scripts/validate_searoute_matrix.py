from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# NEREUS — SeaRoute Route Validation
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ROUTES_FILE = (
    ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "searoute"
    / "route_distances.csv"
)

OUTPUT_DIR = (
    ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "searoute"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "route_validation.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "route_validation_metadata.json"
)


# ============================================================
# EXTERNAL REFERENCE DATA
#
# These are deliberately kept separate from SeaRoute.
#
# distance_nm = reference distance
# source_type:
#   published_reference
#   observed_voyage
#
# Do NOT treat these as universally applicable ground truth.
# ============================================================

EXTERNAL_REFERENCES = [

    {
        "origin": "Newcastle",
        "destination": "Paradip",
        "reference_distance_nm": 5887.00,
        "source_type": "published_reference",
        "source_description": (
            "Published sea-distance reference for "
            "Newcastle Australia → Paradip India"
        ),
        "reference_confidence": "medium",
    },

    {
        "origin": "Newcastle",
        "destination": "Visakhapatnam",
        "reference_distance_nm": 5627.00,
        "source_type": "published_reference",
        "source_description": (
            "IEA Clean Coal Centre published sample "
            "sea-distance reference for "
            "Newcastle Australia → Visakhapatnam India"
        ),
        "reference_confidence": "medium",
    },

    {
        "origin": "Newcastle",
        "destination": "Visakhapatnam",
        "reference_distance_nm": 6065.89,
        "source_type": "observed_voyage",
        "source_description": (
            "Observed voyage reconstruction for "
            "Newcastle Australia → Visakhapatnam India; "
            "6,023.83 NM travelled plus 42.06 NM remaining"
        ),
        "reference_confidence": "medium",
    },
]


# ============================================================
# VALIDATION CLASSIFICATION
# ============================================================

def classify_difference(
    abs_difference_pct: float,
) -> str:

    """
    This classification is for model/data-quality review.

    It does NOT mean that a route is navigationally accurate.
    """

    value = abs(abs_difference_pct)

    if value <= 2:
        return "close"

    if value <= 5:
        return "acceptable_reference_range"

    if value <= 10:
        return "material_difference"

    return "large_difference"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NEREUS — SeaRoute/MARNET route validation")
    print("=" * 72)
    print()

    if not ROUTES_FILE.exists():
        raise FileNotFoundError(
            f"Route matrix not found:\n{ROUTES_FILE}"
        )

    routes = pd.read_csv(
        ROUTES_FILE
    )

    required_columns = [
        "origin",
        "destination",
        "distance_nm",
        "distance_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in routes.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in route matrix: {missing}"
        )

    # --------------------------------------------------------
    # Only compare successfully computed routes.
    # --------------------------------------------------------

    routes = routes[
        routes["distance_status"]
        == "computed_reference"
    ].copy()

    routes["distance_nm"] = pd.to_numeric(
        routes["distance_nm"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Compare every external reference
    # --------------------------------------------------------

    validation_rows = []

    for reference in EXTERNAL_REFERENCES:

        origin = reference["origin"]
        destination = reference["destination"]

        matches = routes[
            (routes["origin"] == origin)
            & (
                routes["destination"]
                == destination
            )
        ]

        if matches.empty:

            validation_rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "searoute_distance_nm": None,
                    "reference_distance_nm": (
                        reference[
                            "reference_distance_nm"
                        ]
                    ),
                    "difference_nm": None,
                    "difference_pct": None,
                    "classification": "route_not_found",
                    "source_type": (
                        reference["source_type"]
                    ),
                    "source_description": (
                        reference[
                            "source_description"
                        ]
                    ),
                    "reference_confidence": (
                        reference[
                            "reference_confidence"
                        ]
                    ),
                }
            )

            continue

        searoute_distance = float(
            matches.iloc[0]["distance_nm"]
        )

        reference_distance = float(
            reference[
                "reference_distance_nm"
            ]
        )

        difference_nm = (
            searoute_distance
            - reference_distance
        )

        difference_pct = (
            difference_nm
            / reference_distance
            * 100
        )

        classification = classify_difference(
            difference_pct
        )

        validation_rows.append(
            {
                "origin": origin,
                "destination": destination,

                "searoute_distance_nm": round(
                    searoute_distance,
                    2,
                ),

                "reference_distance_nm": round(
                    reference_distance,
                    2,
                ),

                "difference_nm": round(
                    difference_nm,
                    2,
                ),

                "difference_pct": round(
                    difference_pct,
                    2,
                ),

                "absolute_difference_pct": round(
                    abs(difference_pct),
                    2,
                ),

                "classification": classification,

                "source_type": (
                    reference[
                        "source_type"
                    ]
                ),

                "source_description": (
                    reference[
                        "source_description"
                    ]
                ),

                "reference_confidence": (
                    reference[
                        "reference_confidence"
                    ]
                ),

                "routing_engine": "searoute",

                "routing_network": "MARNET",
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    # --------------------------------------------------------
    # Save validation CSV
    # --------------------------------------------------------

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    numeric = validation[
        validation["difference_pct"]
        .notna()
    ].copy()

    summary = {}

    if not numeric.empty:

        summary = {
            "comparisons": len(numeric),

            "mean_absolute_difference_pct": round(
                numeric[
                    "absolute_difference_pct"
                ].mean(),
                2,
            ),

            "median_absolute_difference_pct": round(
                numeric[
                    "absolute_difference_pct"
                ].median(),
                2,
            ),

            "minimum_absolute_difference_pct": round(
                numeric[
                    "absolute_difference_pct"
                ].min(),
                2,
            ),

            "maximum_absolute_difference_pct": round(
                numeric[
                    "absolute_difference_pct"
                ].max(),
                2,
            ),

            "classification_counts": (
                numeric[
                    "classification"
                ]
                .value_counts()
                .to_dict()
            ),
        }

    # --------------------------------------------------------
    # Route-specific interpretation
    # --------------------------------------------------------

    interpretation = {
        "newcastle_paradip": (
            "SeaRoute is compared against a published "
            "Newcastle → Paradip distance reference. "
            "This is a direct route-level comparison."
        ),

        "newcastle_visakhapatnam": (
            "Two different external references exist: "
            "a published sample distance and an observed "
            "voyage reconstruction. They represent "
            "different concepts and should not be "
            "collapsed into a single ground-truth value."
        ),

        "general_policy": (
            "SeaRoute/MARNET should remain a reproducible "
            "computed route reference. It should not be "
            "described as navigation-grade or actual "
            "sailed distance."
        ),

        "optimization_policy": (
            "Until stronger route-distance evidence is "
            "available, optimization may use the "
            "SeaRoute/MARNET distance as a reference "
            "input while retaining source provenance "
            "and uncertainty."
        ),

        "uncertainty_policy": (
            "A route-distance uncertainty factor should "
            "not be hardcoded from these few comparisons. "
            "More independent route references are needed "
            "before calibrating a systematic distance "
            "adjustment."
        ),
    }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dataset": (
            "NEREUS SeaRoute/MARNET "
            "route validation"
        ),

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "route_matrix": str(
            ROUTES_FILE.relative_to(ROOT)
        ),

        "validation_file": str(
            VALIDATION_FILE.relative_to(ROOT)
        ),

        "external_reference_count": (
            len(EXTERNAL_REFERENCES)
        ),

        "comparison_count": (
            len(numeric)
        ),

        "summary": summary,

        "interpretation": interpretation,

        "external_references": (
            EXTERNAL_REFERENCES
        ),

        "important_warning": (
            "Published and observed voyage distances "
            "are not interchangeable. A published route "
            "distance may represent a planning/reference "
            "route, while an observed voyage distance "
            "reflects an actual voyage path and can include "
            "routing, weather, traffic, operational and "
            "other voyage-specific effects."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("VALIDATION RESULTS")
    print()

    display_columns = [
        "origin",
        "destination",
        "searoute_distance_nm",
        "reference_distance_nm",
        "difference_nm",
        "difference_pct",
        "classification",
        "source_type",
    ]

    print(
        validation[
            display_columns
        ].to_string(index=False)
    )

    print()

    print("SUMMARY")
    print()

    if summary:

        print(
            f"Comparisons: "
            f"{summary['comparisons']}"
        )

        print(
            "Mean absolute difference: "
            f"{summary['mean_absolute_difference_pct']:.2f}%"
        )

        print(
            "Median absolute difference: "
            f"{summary['median_absolute_difference_pct']:.2f}%"
        )

        print(
            "Minimum absolute difference: "
            f"{summary['minimum_absolute_difference_pct']:.2f}%"
        )

        print(
            "Maximum absolute difference: "
            f"{summary['maximum_absolute_difference_pct']:.2f}%"
        )

        print()

        print(
            "Classification counts:"
        )

        for key, value in (
            summary[
                "classification_counts"
            ].items()
        ):

            print(
                f"  {key}: {value}"
            )

    print()

    print("=" * 72)
    print("FILES")
    print("=" * 72)
    print()

    print(
        f"Validation CSV:\n"
        f"  {VALIDATION_FILE}"
    )

    print()

    print(
        f"Metadata:\n"
        f"  {METADATA_FILE}"
    )

    print()


if __name__ == "__main__":
    main()