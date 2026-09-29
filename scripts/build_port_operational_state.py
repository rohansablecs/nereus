from pathlib import Path
from datetime import datetime, timezone
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "02_DATA/raw/port"
PROCESSED = ROOT / "02_DATA/processed"

OUT = PROCESSED / "port_state"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def parse_dt(value):

    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=True,
    )


def clean_text(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def dry_bulk_mask(series):

    pattern = (
        r"COAL|COKE|IRON ORE|PELLET|"
        r"ORE|LIMESTONE|BAUXITE|ALUMINA|"
        r"MANGANESE|PCI"
    )

    return series.fillna("").str.contains(
        pattern,
        case=False,
        regex=True,
    )


def build_dhamra():

    p = (
        PROCESSED
        / "adani_schedules"
        / "dhamra_current.csv"
    )

    df = pd.read_csv(p)

    at_berth = df[
        df["section"] == "at_berth"
    ].copy()

    expected = df[
        df["section"] == "expected"
    ].copy()

    sailed = df[
        df["section"] == "sailed_24h"
    ].copy()

    # Dhamra explicitly exposes vacant berth positions.
    berth_positions = (
        at_berth["berth"]
        .notna()
    )

    occupied = (
        at_berth["vessel_name"]
        .notna()
    )

    dry_at_berth = dry_bulk_mask(
        at_berth["cargo"]
    )

    dry_expected = (
        expected["sbu_name"]
        .fillna("")
        .str.contains(
            "DRY",
            case=False,
            regex=False,
        )
    )

    dry_at_berth_cargo = at_berth[
        dry_at_berth
    ]["cargo"].fillna("")

    return {
        "port": "Dhamra",

        "snapshot_time": df[
            "snapshot_time_utc"
        ].iloc[0],

        "location_model": "berth",

        "berth_position_rows": int(
            berth_positions.sum()
        ),

        "occupied_positions": int(
            occupied.sum()
        ),

        "vacant_positions": int(
            (
                berth_positions
                & ~occupied
            ).sum()
        ),

        "unavailable_positions": None,

        "occupancy_ratio": (
            float(
                occupied.sum()
                / berth_positions.sum()
            )
            if berth_positions.sum()
            else None
        ),

        "vessels_expected": len(
            expected
        ),

        "dry_bulk_vessels_at_berth": int(
            dry_at_berth.sum()
        ),

        "dry_bulk_vessels_expected": int(
            dry_expected.sum()
        ),

        "dry_bulk_cargo_mt_current": None,

        "recent_departures": len(
            sailed
        ),

        "anchorage_vessels": None,

        "anchorage_available": True,

        "data_confidence": "high",

        "source_type": (
            "official_current_schedule"
        ),
    }


def build_gopalpur():

    p = (
        PROCESSED
        / "adani_schedules"
        / "gopalpur_current.csv"
    )

    df = pd.read_csv(p)

    berth = df[
        df["section"] == "at_berth"
    ].copy()

    expected = df[
        df["section"] == "expected"
    ].copy()

    # The source explicitly reports VACANT.
    occupied = berth[
        berth["vessel_name"]
        .notna()
    ]

    vacant = berth[
        berth["vessel_name"]
        .isna()
    ]

    dry_expected = (
        expected["sbu_name"]
        .fillna("")
        .str.contains(
            "DRY",
            case=False,
            regex=False,
        )
    )

    return {
        "port": "Gopalpur",

        "snapshot_time": df[
            "snapshot_time_utc"
        ].iloc[0],

        "location_model": "single_quay_subpositions",

        "berth_position_rows": len(
            berth
        ),

        "occupied_positions": len(
            occupied
        ),

        "vacant_positions": len(
            vacant
        ),

        "unavailable_positions": None,

        "occupancy_ratio": (
            len(occupied)
            / len(berth)
            if len(berth)
            else None
        ),

        "vessels_expected": len(
            expected
        ),

        "dry_bulk_vessels_at_berth": int(
            dry_bulk_mask(
                occupied["vessel_name"]
            ).sum()
        ),

        "dry_bulk_vessels_expected": int(
            dry_expected.sum()
        ),

        "dry_bulk_cargo_mt_current": None,

        "recent_departures": 0,

        "anchorage_vessels": 0,

        "anchorage_available": True,

        "data_confidence": "medium",

        "source_type": (
            "official_current_schedule"
        ),
    }


def build_vizag():

    berth = pd.read_csv(
        RAW / "vizag_berth_occupancy_current.csv"
    )

    waiting = pd.read_csv(
        RAW / "vizag_waiting_expected_current.csv"
    )

    occupied = berth[
        berth["status"]
        .astype(str)
        .str.upper()
        == "OCCUPIED"
    ]

    vacant = berth[
        berth["status"]
        .astype(str)
        .str.upper()
        == "VACANT"
    ]

    unavailable = berth[
        berth["status"]
        .astype(str)
        .str.upper()
        == "UNAVAILABLE"
    ]

    dry_waiting = dry_bulk_mask(
        waiting["raw_row"]
    )

    dry_cargo = waiting.loc[
        dry_waiting,
        "tonnes_mt",
    ].sum(
        min_count=1
    )

    snapshot = berth[
        "snapshot_time_utc"
    ].iloc[0]

    return {
        "port": "Visakhapatnam",

        "snapshot_time": snapshot,

        "location_model": "multi_terminal",

        "berth_position_rows": len(
            berth
        ),

        "occupied_positions": len(
            occupied
        ),

        "vacant_positions": len(
            vacant
        ),

        "unavailable_positions": len(
            unavailable
        ),

        "occupancy_ratio": (
            len(occupied)
            / len(berth)
            if len(berth)
            else None
        ),

        "vessels_expected": len(
            waiting
        ),

        "dry_bulk_vessels_at_berth": int(
            dry_bulk_mask(
                occupied["raw_row"]
            ).sum()
        ),

        "dry_bulk_vessels_expected": int(
            dry_waiting.sum()
        ),

        "dry_bulk_cargo_mt_current": (
            float(dry_cargo)
            if pd.notna(dry_cargo)
            else None
        ),

        "recent_departures": None,

        "anchorage_vessels": None,

        "anchorage_available": None,

        "data_confidence": "high",

        "source_type": (
            "official_daily_berthing_programme"
        ),
    }


def build_gangavaram():

    berth = pd.read_csv(
        RAW
        / "gangavaram_vessel_schedule_current.csv"
    )

    expected = pd.read_csv(
        RAW
        / "gangavaram_expected_vessels_current.csv"
    )

    sailed = pd.read_csv(
        RAW
        / "gangavaram_sailed_24h.csv"
    )

    dry_current = dry_bulk_mask(
        berth["cargo"]
    )

    # Gangavaram's expected-vessel table identifies
    # the operational business unit ("DRY CARGO VSL
    # OPERATION") but does not provide vessel cargo.
    # Therefore dry-bulk classification is unknown.

    return {
        "port": "Gangavaram",

        "snapshot_time": None,

        "location_model": "multi_berth",

        "berth_position_rows": 9,

        "occupied_positions": 9,

        "vacant_positions": None,

        "unavailable_positions": None,

        # The current Gangavaram schedule lists vessels
        # by berth, but does not expose the complete set
        # of vacant/unavailable berth positions.
        # Therefore an occupancy ratio cannot be defensibly
        # calculated from this source.
        "occupancy_ratio": None,

        "vessels_expected": len(
            expected
        ),

        "dry_bulk_vessels_at_berth": int(
            dry_current.sum()
        ),

        # Cargo is not provided for the expected-vessel
        # records, so this metric is explicitly unknown.
        "dry_bulk_vessels_expected": None,

        "dry_bulk_cargo_mt_current": None,

        "recent_departures": len(
            sailed
        ),

        "anchorage_vessels": None,

        "anchorage_available": None,

        "data_confidence": "high",

        "source_type": (
            "official_current_schedule"
        ),
    }


def build_haldia():

    p = (
        PROCESSED
        / "haldia"
        / "haldia_operational_current.csv"
    )

    df = pd.read_csv(p)

    berth = df[
        df["location_type"]
        == "berth"
    ]

    occupied = berth[
        berth["status"]
        .astype(str)
        .str.lower()
        == "occupied"
    ]

    vacant = berth[
        berth["status"]
        .astype(str)
        .str.lower()
        == "vacant"
    ]

    due = df[
        df["status"]
        .astype(str)
        .str.lower()
        == "due"
    ]

    # The Haldia source parser can contain malformed /
    # concatenated quantity strings. Never attempt to
    # interpret those as numbers.
    quantity_numeric = pd.to_numeric(
        due["quantity_mt"],
        errors="coerce",
    )

    dry_due = dry_bulk_mask(
        due["cargo"]
    )

    dry_cargo = quantity_numeric[
        dry_due
    ].sum(
        min_count=1
    )

    return {
        "port": "Haldia",

        "snapshot_time": (
            df["snapshot_time_utc"].iloc[0]
        ),

        "location_model": "dock",

        "berth_position_rows": len(
            berth
        ),

        "occupied_positions": len(
            occupied
        ),

        "vacant_positions": len(
            vacant
        ),

        "unavailable_positions": None,

        "occupancy_ratio": (
            len(occupied)
            / len(berth)
            if len(berth)
            else None
        ),

        "vessels_expected": len(
            due
        ),

        "dry_bulk_vessels_at_berth": int(
            dry_bulk_mask(
                occupied["cargo"]
            ).sum()
        ),

        "dry_bulk_vessels_expected": int(
            dry_due.sum()
        ),

        "dry_bulk_cargo_mt_current": (
            float(dry_cargo)
            if pd.notna(dry_cargo)
            else None
        ),

        "recent_departures": None,

        "anchorage_vessels": None,

        "anchorage_available": None,

        "data_confidence": "medium",

        "source_type": (
            "official_operational_snapshot"
        ),
    }


def build_sagar():

    p = (
        PROCESSED
        / "haldia"
        / "sagar_sandheads_operational_current.csv"
    )

    df = pd.read_csv(p)

    sagar = df[
        df["port"]
        == "Sagar"
    ]

    sandheads = df[
        df["port"]
        == "Sandheads"
    ]

    point_x = df[
        df["port"]
        == "Point X"
    ]

    return {
        "port": "Sagar-Sandheads",

        "snapshot_time": (
            df["snapshot_time_utc"].iloc[0]
            if not df.empty
            else None
        ),

        "location_model": (
            "anchorage_lighterage"
        ),

        "berth_position_rows": None,

        "occupied_positions": None,

        "vacant_positions": None,

        "unavailable_positions": None,

        "occupancy_ratio": None,

        "vessels_expected": len(
            sagar
        ),

        "dry_bulk_vessels_at_berth": None,

        "dry_bulk_vessels_expected": int(
            dry_bulk_mask(
                sagar["cargo"]
            ).sum()
        ) if not sagar.empty else 0,

        "dry_bulk_cargo_mt_current": (
            float(
                sagar["quantity_mt"]
                .sum(
                    min_count=1
                )
            )
            if not sagar.empty
            else None
        ),

        "recent_departures": None,

        "anchorage_vessels": 0,

        "anchorage_available": True,

        "data_confidence": "medium",

        "source_type": (
            "official_operational_snapshot"
        ),
    }

def build_paradip():

    movement_path = (
        PROCESSED
        / "paradip"
        / "berthing_movements_corpus_v3.csv"
    )

    berth_path = (
        RAW
        / "paradip_berths.csv"
    )

    movements = pd.read_csv(movement_path)
    berths = pd.read_csv(berth_path)

    # Paradip daily traffic times are local Indian time.
    # Parse them as Asia/Kolkata first; never assume UTC.
    movements["movement_timestamp"] = pd.to_datetime(
        movements["movement_timestamp"],
        errors="coerce",
    )

    movements["report_held_on"] = pd.to_datetime(
        movements["report_held_on"],
        errors="coerce",
    )

    movements["vessel_name_normalized"] = (
        movements["vessel_name_normalized"]
        .fillna(movements["vessel_name"])
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Current wall-clock boundary in the port's local timezone.
    now_local = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    # Convert movement timestamps from naive local time to
    # timezone-aware Asia/Kolkata.
    movement_local = (
        movements["movement_timestamp"]
        .dt.tz_localize(
            "Asia/Kolkata"
        )
    )

    movements["movement_timestamp_local"] = (
        movement_local
    )

    movements["movement_timestamp_utc"] = (
        movement_local.dt.tz_convert("UTC")
    )

    # Latest report represented in the corpus.
    latest_report_date = (
        movements["report_held_on"]
        .dropna()
        .max()
    )

    if pd.isna(latest_report_date):
        raise ValueError(
            "Paradip corpus contains no usable report_held_on date."
        )

    # Only use records from the latest report date and records that
    # have actually occurred by the current local time.
    #
    # This prevents scheduled 20:00/23:00 movements and next-day
    # scheduled movements from being treated as current occupancy.
    history = movements[
        movements["report_held_on"].notna()
        &
        (
            movements["report_held_on"].dt.normalize()
            == latest_report_date.normalize()
        )
        &
        movements["movement_timestamp_local"].notna()
        &
        (
            movements["movement_timestamp_local"]
            <= now_local
        )
    ].copy()

    if history.empty:
        raise ValueError(
            "Paradip corpus contains no movement records from the "
            "latest report that have occurred by the current time."
        )

    snapshot_timestamp_local = (
        history["movement_timestamp_local"].max()
    )

    snapshot_timestamp_utc = (
        snapshot_timestamp_local
        .tz_convert("UTC")
    )

    # ---------------------------------------------------------------
    # Reconstruct latest known state for each vessel.
    # ---------------------------------------------------------------

    history = history.sort_values(
        "movement_timestamp_local"
    )

    latest_by_vessel = (
        history
        .dropna(
            subset=["vessel_name_normalized"]
        )
        .groupby(
            "vessel_name_normalized",
            as_index=False,
        )
        .tail(1)
    )

    purpose = (
        latest_by_vessel["purpose"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    active = latest_by_vessel[
        purpose.str.contains(
            r"\bBERTHS?\b|\bSHIFTS?\b",
            regex=True,
        )
    ].copy()

    active = active[
        active["berth"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Official Paradip berth universe.
    # ---------------------------------------------------------------

    berth_position_count = int(
        berths["berth_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    occupied_positions = int(
        active["berth"]
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )

    # ---------------------------------------------------------------
    # Dry-bulk workload.
    # ---------------------------------------------------------------

    dry_at_berth = dry_bulk_mask(
        active["cargo"]
    )

    dry_bulk_vessels = active[
        dry_at_berth
    ].copy()

    quantity = pd.to_numeric(
        dry_bulk_vessels["quantity_mt"],
        errors="coerce",
    )

    dry_bulk_cargo_mt = (
        float(
            quantity.sum(
                min_count=1
            )
        )
        if quantity.notna().any()
        else None
    )

    # ---------------------------------------------------------------
    # Future scheduled movements relative to the real snapshot.
    # ---------------------------------------------------------------

    future = movements[
        movements["movement_timestamp_local"].notna()
        &
        (
            movements["movement_timestamp_local"]
            > snapshot_timestamp_local
        )
    ].copy()

    future_purpose = (
        future["purpose"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    future = future[
        future_purpose.str.contains(
            r"\bBERTHS?\b|\bEXPECTED\b|\bDUE\b",
            regex=True,
        )
    ].copy()

    dry_expected = dry_bulk_mask(
        future["cargo"]
    )

    # ---------------------------------------------------------------
    # Recent departures.
    # ---------------------------------------------------------------

    recent_cutoff = (
        snapshot_timestamp_local
        - pd.Timedelta(hours=24)
    )

    history_purpose = (
        history["purpose"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    sailed = history[
        history_purpose.str.contains(
            r"\bSAILS\b",
            regex=True,
        )
        &
        (
            history["movement_timestamp_local"]
            > recent_cutoff
        )
        &
        (
            history["movement_timestamp_local"]
            <= snapshot_timestamp_local
        )
    ].copy()

    # ---------------------------------------------------------------
    # Freshness.
    # ---------------------------------------------------------------

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    data_age_hours = (
        (
            now_utc
            - snapshot_timestamp_utc
        ).total_seconds()
        / 3600
    )

    return {
        "port": "Paradip",

        "snapshot_time": (
            snapshot_timestamp_utc.isoformat()
        ),

        "location_model": "berth",

        "berth_position_rows": (
            berth_position_count
        ),

        "occupied_positions": (
            occupied_positions
        ),

        # Exact vacancy mapping is not currently reliable because
        # movement berth labels do not map one-to-one to every
        # official berth row.
        "vacant_positions": None,

        "unavailable_positions": None,

        "occupancy_ratio": (
            float(
                occupied_positions
                / berth_position_count
            )
            if berth_position_count
            else None
        ),

        "vessels_expected": len(
            future
        ),

        "dry_bulk_vessels_at_berth": int(
            len(dry_bulk_vessels)
        ),

        "dry_bulk_vessels_expected": (
            int(dry_expected.sum())
        ),

        "dry_bulk_cargo_mt_current": (
            dry_bulk_cargo_mt
        ),

        "recent_departures": len(
            sailed
        ),

        "anchorage_vessels": None,

        "anchorage_available": None,

        "data_confidence": "medium",

        "source_type": (
            "official_daily_traffic_corpus"
        ),

        "generated_at_utc": (
            now_utc.isoformat()
        ),

        "data_age_hours": round(
            max(0.0, data_age_hours),
            3,
        ),

        "source_status": "available",
    }

def main():

    builders = [
        build_dhamra,
        build_gopalpur,
        build_vizag,
        build_gangavaram,
        build_haldia,
        build_sagar,
        build_paradip,
    ]

    rows = []

    for builder in builders:
        row = builder()
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Snapshot timestamps
    #
    # Builders return ISO-8601 strings or None. Preserve those values
    # in the canonical CSV. Parse individually only when calculating
    # freshness so mixed source timestamp formats cannot destroy the
    # original snapshot.
    # ---------------------------------------------------------------

    def age_hours(value):

        if value is None or pd.isna(value):
            return None

        try:
            timestamp = pd.Timestamp(value)

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    timezone.utc
                )
            else:
                timestamp = timestamp.tz_convert(
                    timezone.utc
                )

            generated_timestamp = pd.Timestamp(
                generated
            )

            age = (
                generated_timestamp - timestamp
            ).total_seconds() / 3600

            return round(
                max(0.0, age),
                3,
            )

        except Exception:
            return None

    generated = datetime.now(
        timezone.utc
    )

    df["snapshot_time"] = (
        df["snapshot_time"]
        .apply(
            lambda value: (
                str(value)
                if pd.notna(value)
                else None
            )
        )
    )

    df["generated_at_utc"] = (
        generated.isoformat()
    )

    df["data_age_hours"] = (
        df["snapshot_time"]
        .apply(age_hours)
    )

    df["source_status"] = "available"

    # ---------------------------------------------------------------
    # Write canonical operational state
    # ---------------------------------------------------------------

    output = (
        OUT
        / "port_operational_state.csv"
    )

    metadata_output = (
        OUT
        / "port_operational_state_metadata.json"
    )

    df.to_csv(
        output,
        index=False,
    )

    metadata = {
        "generated_at_utc": generated.isoformat(),
        "rows": int(len(df)),
        "ports": df["port"].tolist(),
        "source_types": (
            df["source_type"]
            .dropna()
            .unique()
            .tolist()
        ),
        "snapshot_policy": (
            "Each builder supplies its own latest-known operational "
            "snapshot. Snapshot timestamps are preserved as returned "
            "by the builder and data_age_hours is calculated from the "
            "timestamp against generation time."
        ),
    }

    metadata_output.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
    )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    print("=" * 80)
    print("CANONICAL PORT OPERATIONAL STATE")
    print("=" * 80)

    display_cols = [
        "port",
        "location_model",
        "occupied_positions",
        "vacant_positions",
        "occupancy_ratio",
        "vessels_expected",
        "dry_bulk_vessels_at_berth",
        "dry_bulk_vessels_expected",
        "dry_bulk_cargo_mt_current",
        "recent_departures",
        "data_confidence",
        "data_age_hours",
    ]

    print(
        df[display_cols]
        .to_string(index=False)
    )

    print()
    print("Output:", output)
    print("Metadata:", metadata_output)


if __name__ == "__main__":
    main()
