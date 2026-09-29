from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "searoute"
    / "route_distances.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_OPTIMIZATION"
    / "data"
    / "routes.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "04_OPTIMIZATION"
    / "data"
    / "routes_metadata.json"
)


# ============================================================
# Existing route IDs
#
# Preserve the IDs already used by the optimization layer.
# New routes receive deterministic IDs.
# ============================================================

EXISTING_ROUTE_IDS = {
    ("Newcastle", "Paradip"): "AUS_NEW_PAR",
    ("Newcastle", "Dhamra"): "AUS_NEW_DHA",
    ("Newcastle", "Gopalpur"): "AUS_NEW_GOP",
    ("Newcastle", "Visakhapatnam"): "AUS_NEW_VIZ",
    ("Newcastle", "Gangavaram"): "AUS_NEW_GAN",

    ("Hay Point", "Paradip"): "AUS_HAY_PAR",
    ("Hay Point", "Dhamra"): "AUS_HAY_DHA",
    ("Hay Point", "Gopalpur"): "AUS_HAY_GOP",
    ("Hay Point", "Visakhapatnam"): "AUS_HAY_VIZ",
    ("Hay Point", "Gangavaram"): "AUS_HAY_GAN",
}


ORIGIN_CODES = {
    "Newcastle": "NEW",
    "Hay Point": "HAY",
    "Beira": "BEI",
    "Maputo": "MAP",
    "Vostochny": "VOS",
    "Taman": "TAM",
    "Samarinda": "SAM",
    "Richards Bay": "RB",
}


DESTINATION_CODES = {
    "Paradip": "PAR",
    "Dhamra": "DHA",
    "Gopalpur": "GOP",
    "Visakhapatnam": "VIZ",
    "Gangavaram": "GAN",
    "Haldia": "HAL",
}


# ============================================================
# Route ID generation
# ============================================================

def make_route_id(origin: str, destination: str) -> str:

    existing = EXISTING_ROUTE_IDS.get(
        (origin, destination)
    )

    if existing:
        return existing

    origin_code = ORIGIN_CODES.get(origin)

    destination_code = DESTINATION_CODES.get(
        destination
    )

    if not origin_code:
        raise ValueError(
            f"No origin code configured for: {origin}"
        )

    if not destination_code:
        raise ValueError(
            f"No destination code configured for: "
            f"{destination}"
        )

    return (
        f"{origin_code}_{destination_code}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("NEREUS — Build optimization route registry")
    print("=" * 72)
    print()

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"SeaRoute matrix not found:\n{SOURCE_FILE}"
        )

    source = pd.read_csv(
        SOURCE_FILE
    )

    required_columns = [
        "route_id",
        "origin",
        "destination",
        "distance_nm",
        "distance_source",
        "distance_status",
        "origin_coordinate_source",
        "destination_coordinate_source",
        "origin_coordinate_type",
        "destination_coordinate_type",
        "origin_endpoint_type",
        "destination_endpoint_type",
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
        "routing_engine",
        "routing_network",
        "units",
        "append_orig_dest",
        "algorithm",
        "restrictions",
    ]

    missing = [
        column
        for column in required_columns
        if column not in source.columns
    ]

    if missing:
        raise ValueError(
            "SeaRoute matrix is missing required "
            f"columns: {missing}"
        )

    # --------------------------------------------------------
    # Validate source rows
    # --------------------------------------------------------

    source["distance_nm"] = pd.to_numeric(
        source["distance_nm"],
        errors="coerce",
    )

    invalid_distance = source[
        source["distance_nm"].isna()
    ]

    if not invalid_distance.empty:

        raise ValueError(
            "SeaRoute matrix contains routes without "
            "a valid distance:\n"
            + invalid_distance[
                ["origin", "destination"]
            ].to_string(index=False)
        )

    invalid_status = source[
        source["distance_status"]
        != "computed_reference"
    ]

    if not invalid_status.empty:

        raise ValueError(
            "SeaRoute matrix contains routes that "
            "are not computed_reference:\n"
            + invalid_status[
                [
                    "origin",
                    "destination",
                    "distance_status",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Detect duplicate OD pairs
    # --------------------------------------------------------

    duplicates = source[
        source.duplicated(
            subset=[
                "origin",
                "destination",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:

        raise ValueError(
            "Duplicate origin/destination pairs found:\n"
            + duplicates[
                [
                    "origin",
                    "destination",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Build optimization registry
    # --------------------------------------------------------

    rows = []

    for _, row in source.iterrows():

        origin = str(
            row["origin"]
        ).strip()

        destination = str(
            row["destination"]
        ).strip()

        route_id = make_route_id(
            origin,
            destination,
        )

        rows.append(
            {
                "route_id": route_id,

                "origin": origin,

                "destination": destination,

                "distance_nm": float(
                    row["distance_nm"]
                ),

                "distance_source": str(
                    row["distance_source"]
                ),

                "distance_status": str(
                    row["distance_status"]
                ),

                "origin_coordinate_source": str(
                    row[
                        "origin_coordinate_source"
                    ]
                ),

                "destination_coordinate_source": str(
                    row[
                        "destination_coordinate_source"
                    ]
                ),

                "origin_coordinate_type": str(
                    row[
                        "origin_coordinate_type"
                    ]
                ),

                "destination_coordinate_type": str(
                    row[
                        "destination_coordinate_type"
                    ]
                ),

                "origin_endpoint_type": str(
                    row[
                        "origin_endpoint_type"
                    ]
                ),

                "destination_endpoint_type": str(
                    row[
                        "destination_endpoint_type"
                    ]
                ),

                "origin_latitude": float(
                    row["origin_latitude"]
                ),

                "origin_longitude": float(
                    row["origin_longitude"]
                ),

                "destination_latitude": float(
                    row[
                        "destination_latitude"
                    ]
                ),

                "destination_longitude": float(
                    row[
                        "destination_longitude"
                    ]
                ),

                "routing_engine": str(
                    row["routing_engine"]
                ),

                "routing_network": str(
                    row["routing_network"]
                ),

                "units": str(
                    row["units"]
                ),

                "append_orig_dest": bool(
                    row["append_orig_dest"]
                ),

                "algorithm": str(
                    row["algorithm"]
                ),

                "restrictions": str(
                    row["restrictions"]
                ),
            }
        )

    routes = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Stable ordering
    # --------------------------------------------------------

    routes = routes.sort_values(
        [
            "origin",
            "destination",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    routes.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dataset": (
            "NEREUS optimization route registry"
        ),

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_file": str(
            SOURCE_FILE.relative_to(
                PROJECT_ROOT
            )
        ),

        "output_file": str(
            OUTPUT_FILE.relative_to(
                PROJECT_ROOT
            )
        ),

        "routing_engine": "SeaRoute",

        "routing_network": "MARNET",

        "route_count": int(
            len(routes)
        ),

        "origin_count": int(
            routes["origin"].nunique()
        ),

        "destination_count": int(
            routes["destination"].nunique()
        ),

        "distance_units": "nautical_miles",

        "route_status": (
            "computed_reference"
        ),

        "route_id_policy": (
            "Existing optimization route IDs are "
            "preserved where present. Newly available "
            "origin/destination combinations receive "
            "deterministic IDs."
        ),

        "validation_policy": (
            "SeaRoute/MARNET distances are treated as "
            "reproducible planning references, not "
            "navigation-grade or actual sailed distances."
        ),

        "sagar_sandheads": {
            "status": "not_in_route_matrix",
            "action": (
                "No route is generated until a valid "
                "routing endpoint is established."
            ),
        },
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
    # Console
    # --------------------------------------------------------

    print(
        "Source routes:",
        len(source),
    )

    print(
        "Optimization routes:",
        len(routes),
    )

    print()

    print("Origins:")
    for origin in routes[
        "origin"
    ].drop_duplicates():

        count = (
            routes["origin"]
            == origin
        ).sum()

        print(
            f"  {origin:15} {count}"
        )

    print()

    print("Destinations:")
    for destination in routes[
        "destination"
    ].drop_duplicates():

        count = (
            routes["destination"]
            == destination
        ).sum()

        print(
            f"  {destination:15} {count}"
        )

    print()

    print("Routes:")
    print(
        routes[
            [
                "route_id",
                "origin",
                "destination",
                "distance_nm",
                "distance_source",
                "distance_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print()

    print(
        f"Routes:\n  {OUTPUT_FILE}"
    )

    print()

    print(
        f"Metadata:\n  {METADATA_FILE}"
    )

    print()


if __name__ == "__main__":
    main()
