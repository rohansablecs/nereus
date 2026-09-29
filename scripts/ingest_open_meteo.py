from pathlib import Path
from datetime import datetime, timezone
import json
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCATION_FILE = (
    PROJECT_ROOT
    / "02_DATA"
    / "raw"
    / "weather"
    / "open_meteo"
    / "locations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_DATA"
    / "raw"
    / "weather"
    / "open_meteo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

WEATHER_VARIABLES = [
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
    "visibility",
    "pressure_msl",
]

MARINE_VARIABLES = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "wind_wave_period",
    "swell_wave_height",
    "swell_wave_period",
    "ocean_current_velocity",
    "ocean_current_direction",
    "sea_surface_temperature",
]


def get_json(
    url: str,
    params: dict,
) -> dict:

    response = requests.get(
        url,
        params=params,
        timeout=60,
        headers={
            "User-Agent": "NEREUS/0.1",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    return response.json()


def hourly_to_records(
    payload: dict,
    location: pd.Series,
    source_type: str,
) -> list[dict]:

    hourly = payload.get("hourly")

    if not hourly:
        return []

    times = hourly.get("time", [])

    records = []

    for index, timestamp in enumerate(times):

        record = {
            "timestamp": timestamp,
            "location_id": location["location_id"],
            "location_name": location["location_name"],
            "country": location["country"],
            "role": location["role"],
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "source_type": source_type,
        }

        for variable in (
            WEATHER_VARIABLES
            if source_type == "weather"
            else MARINE_VARIABLES
        ):
            values = hourly.get(variable, [])

            record[variable] = (
                values[index]
                if index < len(values)
                else None
            )

        records.append(record)

    return records


def main():

    if not LOCATION_FILE.exists():
        raise FileNotFoundError(
            f"Location registry not found: {LOCATION_FILE}"
        )

    locations = pd.read_csv(
        LOCATION_FILE
    )

    required = {
        "location_id",
        "location_name",
        "country",
        "role",
        "latitude",
        "longitude",
    }

    missing = required - set(locations.columns)

    if missing:
        raise ValueError(
            "Missing location columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Loading {len(locations)} weather locations..."
    )

    weather_records = []
    marine_records = []

    fetched_at = datetime.now(
        timezone.utc
    ).isoformat()

    for _, location in locations.iterrows():

        print(
            f"  {location['location_id']}..."
        )

        base_params = {
            "latitude": float(
                location["latitude"]
            ),
            "longitude": float(
                location["longitude"]
            ),
            "forecast_days": 7,
            "timezone": "UTC",
        }

        weather_payload = get_json(
            WEATHER_URL,
            {
                **base_params,
                "hourly": ",".join(
                    WEATHER_VARIABLES
                ),
            },
        )

        marine_payload = get_json(
            MARINE_URL,
            {
                **base_params,
                "hourly": ",".join(
                    MARINE_VARIABLES
                ),
            },
        )

        weather_records.extend(
            hourly_to_records(
                weather_payload,
                location,
                "weather",
            )
        )

        marine_records.extend(
            hourly_to_records(
                marine_payload,
                location,
                "marine",
            )
        )

        time.sleep(0.2)

    weather_df = pd.DataFrame(
        weather_records
    )

    marine_df = pd.DataFrame(
        marine_records
    )

    if weather_df.empty:
        raise ValueError(
            "No atmospheric weather records returned."
        )

    if marine_df.empty:
        raise ValueError(
            "No marine weather records returned."
        )

    weather_df = weather_df.sort_values(
        [
            "location_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    marine_df = marine_df.sort_values(
        [
            "location_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    weather_output = (
        OUTPUT_DIR
        / "open_meteo_weather_current.csv"
    )

    marine_output = (
        OUTPUT_DIR
        / "open_meteo_marine_current.csv"
    )

    metadata_output = (
        OUTPUT_DIR
        / "open_meteo_current_metadata.json"
    )

    weather_df.to_csv(
        weather_output,
        index=False,
    )

    marine_df.to_csv(
        marine_output,
        index=False,
    )

    metadata = {
        "source": "Open-Meteo",
        "retrieved_at_utc": fetched_at,
        "weather_endpoint": WEATHER_URL,
        "marine_endpoint": MARINE_URL,
        "locations_file": str(
            LOCATION_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "locations": int(
            len(locations)
        ),
        "weather_variables": WEATHER_VARIABLES,
        "marine_variables": MARINE_VARIABLES,
        "forecast_horizon_days": 7,
        "notes": [
            "Current forecast snapshot.",
            "Weather and marine records are kept separately.",
            "Coordinates are weather sampling points, not navigation coordinates.",
            "Marine currents and sea-level data must not be used as navigation truth.",
            "No missing values are interpolated.",
        ],
    }

    metadata_output.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("OPEN-METEO INGESTION COMPLETE")
    print(
        f"Weather rows: {len(weather_df):,}"
    )
    print(
        f"Marine rows: {len(marine_df):,}"
    )
    print(
        f"Locations: {locations['location_id'].nunique()}"
    )
    print(
        f"Weather output: {weather_output}"
    )
    print(
        f"Marine output: {marine_output}"
    )
    print(
        f"Metadata: {metadata_output}"
    )

    print()
    print("WEATHER COVERAGE:")
    print(
        weather_df.groupby(
            "location_id"
        )
        .agg(
            rows=("timestamp", "size"),
            first=("timestamp", "min"),
            last=("timestamp", "max"),
        )
        .to_string()
    )

    print()
    print("MARINE COVERAGE:")
    print(
        marine_df.groupby(
            "location_id"
        )
        .agg(
            rows=("timestamp", "size"),
            first=("timestamp", "min"),
            last=("timestamp", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
