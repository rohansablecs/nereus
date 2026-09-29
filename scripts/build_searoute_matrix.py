from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import searoute as sr


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ORIGIN_MASTER_FILE = (
    ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "origins"
    / "origin_master.csv"
)

DESTINATION_MASTER_FILE = (
    ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "destinations"
    / "destination_master.csv"
)

LOCATIONS_FILE = (
    ROOT
    / "02_DATA"
    / "raw"
    / "weather"
    / "open_meteo"
    / "locations.csv"
)

OUTPUT_DIR = (
    ROOT
    / "02_DATA"
    / "raw"
    / "maritime"
    / "searoute"
)

REGISTRY_FILE = OUTPUT_DIR / "endpoint_registry.csv"
ROUTES_FILE = OUTPUT_DIR / "route_distances.csv"
METADATA_FILE = OUTPUT_DIR / "route_distances_metadata.json"


# ============================================================
# ROUTING SCOPE
# ============================================================

ORIGINS = [
    "Newcastle",
    "Hay Point",
    "Beira",
    "Maputo",
    "Vostochny",
    "Taman",
    "Samarinda",
    "Richards Bay",
    "Norfolk",
    "Baltimore",
    "Mobile",
    "New Orleans",
]

DESTINATIONS = [
    "Paradip",
    "Dhamra",
    "Gopalpur",
    "Visakhapatnam",
    "Gangavaram",
    "Haldia",
]


# ============================================================
# VERIFIED MARITIME ENDPOINT OVERRIDES
#
# Coordinate order:
# longitude, latitude
# ============================================================

ENDPOINT_OVERRIDES = {

    # --------------------------------------------------------
    # Australia
    # --------------------------------------------------------

    "Newcastle": {
        "longitude": 151.856350,
        "latitude": -32.964120,
        "coordinate_source": (
            "Newcastle official Pilot Boarding Ground Alpha"
        ),
        "coordinate_source_type": "official_maritime",
        "endpoint_type": "pilot_boarding_ground",
    },

    "Hay Point": {
        "longitude": 149.305460,
        "latitude": -21.239963,
        "coordinate_source": (
            "Queensland Maritime Safety Queensland "
            "Notice 350(T) of 2026 — Hay Point Channel"
        ),
        "coordinate_source_type": "official_maritime",
        "endpoint_type": "channel_entrance",
    },

    # --------------------------------------------------------
    # USA
    # --------------------------------------------------------

    "Norfolk": {
        "longitude": -76.329056,
        "latitude": 36.918448,
        "coordinate_source": (
            "USACE Norfolk District — Norfolk International "
            "Terminals project location"
        ),
        "coordinate_source_type": "official_maritime",
        "endpoint_type": "port_terminal_reference",
    },

    "Baltimore": {
        "longitude": -76.616667,
        "latitude": 39.283333,
        "coordinate_source": (
            "UN/LOCODE US BAL — Baltimore reference coordinate"
        ),
        "coordinate_source_type": "maritime_reference",
        "endpoint_type": "port",
    },

    "Mobile": {
        "longitude": -88.043311,
        "latitude": 30.712169,
        "coordinate_source": (
            "Alabama Port Authority — Mobile port project "
            "reference coordinate"
        ),
        "coordinate_source_type": "official_port_reference",
        "endpoint_type": "port",
    },

    "New Orleans": {
        "longitude": -90.061890,
        "latitude": 29.936890,
        "coordinate_source": (
            "Port of New Orleans — port reference location"
        ),
        "coordinate_source_type": "port_reference",
        "endpoint_type": "port",
    },

    # --------------------------------------------------------
    # Indian destinations
    # --------------------------------------------------------

    "Paradip": {
        "longitude": 86.583333,
        "latitude": 20.250000,
        "coordinate_source": (
            "UN/LOCODE / Paradip Port reference coordinate"
        ),
        "coordinate_source_type": "maritime_reference",
        "endpoint_type": "port",
    },

    "Dhamra": {
        "longitude": 86.950000,
        "latitude": 20.816667,
        "coordinate_source": (
            "UN/LOCODE / Dhamra Port reference coordinate"
        ),
        "coordinate_source_type": "maritime_reference",
        "endpoint_type": "port",
    },

    "Gopalpur": {
        "longitude": 84.964444,
        "latitude": 19.303611,
        "coordinate_source": (
            "Official Gopalpur Port coordinate"
        ),
        "coordinate_source_type": "official_port",
        "endpoint_type": "port",
    },

    "Visakhapatnam": {
        "longitude": 83.300000,
        "latitude": 17.700000,
        "coordinate_source": (
            "UN/LOCODE / Visakhapatnam reference coordinate"
        ),
        "coordinate_source_type": "maritime_reference",
        "endpoint_type": "port",
    },

    "Gangavaram": {
        "longitude": 83.250000,
        "latitude": 17.650000,
        "coordinate_source": (
            "UN/LOCODE / Gangavaram reference coordinate"
        ),
        "coordinate_source_type": "maritime_reference",
        "endpoint_type": "port",
    },

    "Haldia": {
        "longitude": 88.083333,
        "latitude": 22.033333,
        "coordinate_source": (
            "Official SMPK Haldia reference coordinate"
        ),
        "coordinate_source_type": "official_port",
        "endpoint_type": "port",
    },
}


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================
# LOAD WEATHER LOCATIONS
# ============================================================

def load_locations() -> pd.DataFrame:

    df = load_csv(LOCATIONS_FILE)

    required = [
        "location_id",
        "location_name",
        "country",
        "role",
        "latitude",
        "longitude",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in locations.csv: {missing}"
        )

    df = df[required].copy()

    df["location_name"] = (
        df["location_name"]
        .astype(str)
        .str.strip()
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    return df


# ============================================================
# LOAD MASTER REGISTRIES
# ============================================================

def load_origin_master() -> pd.DataFrame:

    df = load_csv(ORIGIN_MASTER_FILE)

    required = [
        "origin_id",
        "origin_name",
        "country",
        "routing_latitude",
        "routing_longitude",
        "routing_coordinate_source",
        "routing_endpoint_type",
        "weather_location_id",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in origin_master.csv: {missing}"
        )

    df["origin_name"] = (
        df["origin_name"]
        .astype(str)
        .str.strip()
    )

    df["routing_latitude"] = pd.to_numeric(
        df["routing_latitude"],
        errors="coerce",
    )

    df["routing_longitude"] = pd.to_numeric(
        df["routing_longitude"],
        errors="coerce",
    )

    return df


def load_destination_master() -> pd.DataFrame:

    df = load_csv(DESTINATION_MASTER_FILE)

    required = [
        "destination_id",
        "destination_name",
        "state",
        "country",
        "endpoint_type",
        "weather_location_id",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in destination_master.csv: {missing}"
        )

    df["destination_name"] = (
        df["destination_name"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# BUILD ENDPOINT REGISTRY
# ============================================================

def build_endpoint_registry(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    weather_lookup = {
        row["location_name"]: row
        for _, row in locations.iterrows()
    }

    # --------------------------------------------------------
    # Origins
    # --------------------------------------------------------

    for name in ORIGINS:

        row = origins[
            origins["origin_name"] == name
        ]

        if row.empty:
            raise ValueError(
                f"Origin '{name}' not found in origin_master.csv"
            )

        source_row = row.iloc[0]

        weather_name = source_row[
            "weather_location_id"
        ]

        weather_row = None

        if pd.notna(weather_name):
            weather_matches = locations[
                locations["location_id"]
                == weather_name
            ]

            if not weather_matches.empty:
                weather_row = weather_matches.iloc[0]

        # Master registry coordinates are the default.
        latitude = source_row[
            "routing_latitude"
        ]

        longitude = source_row[
            "routing_longitude"
        ]

        coordinate_source = source_row[
            "routing_coordinate_source"
        ]

        coordinate_source_type = (
            "master_registry_reference"
        )

        endpoint_type = source_row[
            "routing_endpoint_type"
        ]

        coordinate_status = (
            "master_registry_coordinate"
        )

        # Existing verified overrides remain authoritative.
        if name in ENDPOINT_OVERRIDES:

            override = ENDPOINT_OVERRIDES[name]

            longitude = override["longitude"]
            latitude = override["latitude"]

            coordinate_source = override[
                "coordinate_source"
            ]

            coordinate_source_type = override[
                "coordinate_source_type"
            ]

            endpoint_type = override[
                "endpoint_type"
            ]

            coordinate_status = (
                "verified_override"
            )

        records.append(
            {
                "location_id": source_row[
                    "origin_id"
                ],
                "name": name,
                "role": "origin",
                "country": source_row[
                    "country"
                ],
                "longitude": longitude,
                "latitude": latitude,
                "coordinate_source": coordinate_source,
                "coordinate_source_type": (
                    coordinate_source_type
                ),
                "endpoint_type": endpoint_type,
                "coordinate_status": (
                    coordinate_status
                ),
                "weather_location_id": (
                    weather_name
                    if pd.notna(weather_name)
                    else None
                ),
                "weather_latitude": (
                    weather_row["latitude"]
                    if weather_row is not None
                    else None
                ),
                "weather_longitude": (
                    weather_row["longitude"]
                    if weather_row is not None
                    else None
                ),
            }
        )

    # --------------------------------------------------------
    # Destinations
    # --------------------------------------------------------

    for name in DESTINATIONS:

        row = destinations[
            destinations["destination_name"] == name
        ]

        if row.empty:
            raise ValueError(
                f"Destination '{name}' not found "
                f"in destination_master.csv"
            )

        source_row = row.iloc[0]

        if name not in ENDPOINT_OVERRIDES:
            raise ValueError(
                f"No verified destination endpoint: {name}"
            )

        override = ENDPOINT_OVERRIDES[name]

        weather_id = source_row[
            "weather_location_id"
        ]

        weather_row = None

        if pd.notna(weather_id):

            weather_matches = locations[
                locations["location_id"]
                == weather_id
            ]

            if not weather_matches.empty:
                weather_row = weather_matches.iloc[0]

        records.append(
            {
                "location_id": source_row[
                    "destination_id"
                ],
                "name": name,
                "role": "destination",
                "country": source_row[
                    "country"
                ],
                "longitude": override[
                    "longitude"
                ],
                "latitude": override[
                    "latitude"
                ],
                "coordinate_source": override[
                    "coordinate_source"
                ],
                "coordinate_source_type": override[
                    "coordinate_source_type"
                ],
                "endpoint_type": override[
                    "endpoint_type"
                ],
                "coordinate_status": (
                    "verified_override"
                ),
                "weather_location_id": (
                    weather_id
                    if pd.notna(weather_id)
                    else None
                ),
                "weather_latitude": (
                    weather_row["latitude"]
                    if weather_row is not None
                    else None
                ),
                "weather_longitude": (
                    weather_row["longitude"]
                    if weather_row is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# CALCULATE ROUTE
# ============================================================

def calculate_route(
    origin: pd.Series,
    destination: pd.Series,
):

    origin_coordinates = [
        float(origin["longitude"]),
        float(origin["latitude"]),
    ]

    destination_coordinates = [
        float(destination["longitude"]),
        float(destination["latitude"]),
    ]

    route = sr.searoute(
        origin_coordinates,
        destination_coordinates,
        units="naut",
        append_orig_dest=True,
    )

    distance_nm = float(
        route.properties["length"]
    )

    duration_hours = route.properties.get(
        "duration_hours"
    )

    return distance_nm, duration_hours


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("NEREUS — SeaRoute/MARNET route matrix")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    origins = load_origin_master()
    destinations = load_destination_master()
    locations = load_locations()

    print(
        f"Origin master:      {len(origins)} rows"
    )

    print(
        f"Destination master: {len(destinations)} rows"
    )

    print(
        f"Weather locations:  {len(locations)} rows"
    )

    print()

    # --------------------------------------------------------
    # Build endpoint registry
    # --------------------------------------------------------

    registry = build_endpoint_registry(
        origins,
        destinations,
        locations,
    )

    print("Routing endpoint registry:")
    print()

    print(
        registry[
            [
                "location_id",
                "name",
                "role",
                "country",
                "longitude",
                "latitude",
                "coordinate_source_type",
                "endpoint_type",
                "coordinate_status",
                "weather_location_id",
            ]
        ].to_string(index=False)
    )

    print()

    registry.to_csv(
        REGISTRY_FILE,
        index=False,
    )

    print(
        f"Endpoint registry written to:\n"
        f"{REGISTRY_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    lookup = {
        row["name"]: row
        for _, row in registry.iterrows()
    }

    # --------------------------------------------------------
    # Route matrix
    # --------------------------------------------------------

    route_records = []

    for origin_name in ORIGINS:

        origin = lookup[origin_name]

        for destination_name in DESTINATIONS:

            destination = lookup[
                destination_name
            ]

            route_id = (
                f"{origin_name}"
                f"__"
                f"{destination_name}"
            )

            print(
                f"{origin_name:20}"
                f" → "
                f"{destination_name:18}",
                end=" ",
                flush=True,
            )

            try:

                distance_nm, duration_hours = (
                    calculate_route(
                        origin,
                        destination,
                    )
                )

                print(
                    f"{distance_nm:.2f} NM"
                )

                route_records.append(
                    {
                        "route_id": route_id,
                        "origin": origin_name,
                        "destination": destination_name,
                        "distance_nm": round(
                            distance_nm,
                            4,
                        ),
                        "duration_hours": (
                            round(
                                float(duration_hours),
                                4,
                            )
                            if duration_hours is not None
                            else None
                        ),
                        "distance_source": (
                            "SeaRoute/MARNET computed"
                        ),
                        "distance_status": (
                            "computed_reference"
                        ),
                        "origin_coordinate_source": (
                            origin[
                                "coordinate_source"
                            ]
                        ),
                        "destination_coordinate_source": (
                            destination[
                                "coordinate_source"
                            ]
                        ),
                        "origin_coordinate_type": (
                            origin[
                                "coordinate_source_type"
                            ]
                        ),
                        "destination_coordinate_type": (
                            destination[
                                "coordinate_source_type"
                            ]
                        ),
                        "origin_endpoint_type": (
                            origin[
                                "endpoint_type"
                            ]
                        ),
                        "destination_endpoint_type": (
                            destination[
                                "endpoint_type"
                            ]
                        ),
                        "origin_latitude": (
                            origin["latitude"]
                        ),
                        "origin_longitude": (
                            origin["longitude"]
                        ),
                        "destination_latitude": (
                            destination["latitude"]
                        ),
                        "destination_longitude": (
                            destination["longitude"]
                        ),
                        "routing_engine": "searoute",
                        "routing_network": "MARNET",
                        "units": "nautical_miles",
                        "append_orig_dest": True,
                        "algorithm": "default",
                        "restrictions": "northwest",
                    }
                )

            except Exception as exc:

                print(
                    f"ERROR — {exc}"
                )

                route_records.append(
                    {
                        "route_id": route_id,
                        "origin": origin_name,
                        "destination": destination_name,
                        "distance_nm": None,
                        "duration_hours": None,
                        "distance_source": None,
                        "distance_status": "routing_error",
                        "origin_coordinate_source": (
                            origin[
                                "coordinate_source"
                            ]
                        ),
                        "destination_coordinate_source": (
                            destination[
                                "coordinate_source"
                            ]
                        ),
                        "origin_coordinate_type": (
                            origin[
                                "coordinate_source_type"
                            ]
                        ),
                        "destination_coordinate_type": (
                            destination[
                                "coordinate_source_type"
                            ]
                        ),
                        "origin_endpoint_type": (
                            origin[
                                "endpoint_type"
                            ]
                        ),
                        "destination_endpoint_type": (
                            destination[
                                "endpoint_type"
                            ]
                        ),
                        "origin_latitude": (
                            origin["latitude"]
                        ),
                        "origin_longitude": (
                            origin["longitude"]
                        ),
                        "destination_latitude": (
                            destination["latitude"]
                        ),
                        "destination_longitude": (
                            destination["longitude"]
                        ),
                        "routing_engine": "searoute",
                        "routing_network": "MARNET",
                        "units": "nautical_miles",
                        "append_orig_dest": True,
                        "algorithm": "default",
                        "restrictions": "northwest",
                        "error": repr(exc),
                    }
                )

    routes = pd.DataFrame(
        route_records
    )

    # --------------------------------------------------------
    # Validate matrix
    # --------------------------------------------------------

    expected_count = (
        len(ORIGINS)
        * len(DESTINATIONS)
    )

    actual_count = len(routes)

    if actual_count != expected_count:
        raise RuntimeError(
            f"Route matrix incomplete: "
            f"expected {expected_count}, "
            f"got {actual_count}"
        )

    duplicate_routes = (
        routes["route_id"]
        .duplicated()
        .sum()
    )

    if duplicate_routes:
        raise RuntimeError(
            f"Duplicate route IDs detected: "
            f"{duplicate_routes}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    routes.to_csv(
        ROUTES_FILE,
        index=False,
    )

    computed = int(
        (
            routes["distance_status"]
            == "computed_reference"
        ).sum()
    )

    errors = int(
        (
            routes["distance_status"]
            != "computed_reference"
        ).sum()
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dataset": (
            "NEREUS SeaRoute/MARNET "
            "route distance matrix"
        ),

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "routing_engine": "searoute",

        "routing_network": "MARNET",

        "units": "nautical_miles",

        "append_orig_dest": True,

        "algorithm": "default",

        "restrictions": [
            "northwest"
        ],

        "include_ports": False,

        "port_selection": False,

        "origin_count": len(
            ORIGINS
        ),

        "destination_count": len(
            DESTINATIONS
        ),

        "expected_route_count": (
            expected_count
        ),

        "computed_route_count": computed,

        "error_route_count": errors,

        "sagar_sandheads_excluded": True,

        "sagar_sandheads_reason": (
            "Sagar-Sandheads is modeled separately "
            "as an anchorage/lighterage operation "
            "rather than a conventional berth "
            "destination."
        ),

        "port_selection_note": (
            "SeaRoute include_ports is disabled. "
            "NEREUS explicitly supplies endpoint "
            "coordinates because automatic nearest "
            "port selection can select a different "
            "nearby port."
        ),

        "distance_note": (
            "Distances are reproducible "
            "SeaRoute/MARNET route references, "
            "not navigation-grade or guaranteed "
            "actual sailed-voyage distances."
        ),

        "coordinate_note": (
            "Routing coordinates come from the "
            "NEREUS maritime origin/destination "
            "registries. Verified endpoint overrides "
            "take precedence. Weather coordinates "
            "remain separate and are linked only "
            "through weather_location_id."
        ),

        "usa_origin_note": (
            "USA origins represent customs/port "
            "gateways used by the metallurgical-coal "
            "trade-flow layer. Their routing coordinates "
            "are reference points and do not represent "
            "a universal coal loading berth."
        ),

        "source_origin_master": str(
            ORIGIN_MASTER_FILE.relative_to(ROOT)
        ),

        "source_destination_master": str(
            DESTINATION_MASTER_FILE.relative_to(ROOT)
        ),

        "source_weather_locations": str(
            LOCATIONS_FILE.relative_to(ROOT)
        ),

        "origins": ORIGINS,

        "destinations": DESTINATIONS,
    }

    import json

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

    print()
    print("=" * 72)
    print("ROUTE MATRIX COMPLETE")
    print("=" * 72)
    print(
        f"Expected routes : {expected_count}"
    )
    print(
        f"Computed routes : {computed}"
    )
    print(
        f"Routing errors  : {errors}"
    )
    print(
        f"Routes file     : {ROUTES_FILE}"
    )
    print(
        f"Metadata file   : {METADATA_FILE}"
    )
    print(
        f"Registry file   : {REGISTRY_FILE}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
