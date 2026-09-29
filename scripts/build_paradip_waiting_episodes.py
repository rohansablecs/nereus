from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_DATA/processed/paradip"

WAITING = DATA / "waiting_vessels_corpus.csv"
MOVEMENTS = DATA / "berthing_movements_corpus_v3.csv"

OUTPUT = DATA / "waiting_episodes.csv"
METADATA = DATA / "waiting_episodes_metadata.json"


def norm(value):
    return (
        str(value)
        .upper()
        .replace("MV.", "")
        .replace("MT.", "")
        .strip()
    )


def main():

    waiting = pd.read_csv(WAITING)
    movements = pd.read_csv(MOVEMENTS)

    waiting["vessel_name_normalized"] = (
        waiting["vessel_name"].map(norm)
    )

    waiting["arrival_timestamp"] = pd.to_datetime(
        waiting["arrival_date"].astype(str)
        + " "
        + waiting["arrival_time"].astype(str),
        errors="coerce",
    )

    waiting["report_date"] = pd.to_datetime(
        waiting["report_traffic_date"],
        errors="coerce",
    )

    movements["movement_timestamp"] = pd.to_datetime(
        movements["movement_timestamp"],
        errors="coerce",
    )

    movements["vessel_name_normalized"] = (
        movements["vessel_name"].map(norm)
    )

    berths = movements[
        movements["purpose"].astype(str).str.upper()
        == "BERTHS"
    ].copy()

    berths = berths.sort_values(
        "movement_timestamp"
    )

    episodes = []

    # ------------------------------------------------------------------
    # Collapse repeated daily snapshots.
    #
    # Same vessel + same reported arrival timestamp = same waiting episode.
    # ------------------------------------------------------------------

    waiting = waiting.sort_values(
        [
            "vessel_name_normalized",
            "arrival_timestamp",
            "report_date",
        ]
    )

    grouped = waiting.groupby(
        [
            "vessel_name_normalized",
            "arrival_timestamp",
        ],
        dropna=False,
    )

    for (vessel, arrival), group in grouped:

        group = group.sort_values(
            "report_date"
        )

        first = group.iloc[0]
        last = group.iloc[-1]

        # First subsequent BERTHS event after arrival.
        candidates = berths[
            (
                berths["vessel_name_normalized"]
                == vessel
            )
            & (
                berths["movement_timestamp"]
                >= arrival
            )
        ]

        if candidates.empty:

            berth_timestamp = pd.NaT
            berth = None
            movement_report = None
            delay_hours = None
            status = "unmatched"

        else:

            movement = candidates.iloc[0]

            berth_timestamp = (
                movement["movement_timestamp"]
            )

            berth = movement["berth"]

            movement_report = (
                movement["report_filename"]
            )

            delay_hours = (
                berth_timestamp - arrival
            ).total_seconds() / 3600

            if delay_hours < 0:

                berth_timestamp = pd.NaT
                berth = None
                movement_report = None
                delay_hours = None
                status = "invalid_negative_delay"

            else:

                status = "matched"

        episodes.append({
            "episode_id": (
                f"{vessel}_"
                f"{arrival.strftime('%Y%m%d%H%M')}"
            ),

            "vessel_name": first["vessel_name"],

            "vessel_name_normalized": vessel,

            "arrival_timestamp": arrival,

            "first_waiting_report": first[
                "report_date"
            ],

            "last_waiting_report": last[
                "report_date"
            ],

            "waiting_snapshot_count": len(group),

            "berth_timestamp": berth_timestamp,

            "arrival_to_berth_hours": delay_hours,

            "arrival_to_berth_days": (
                delay_hours / 24
                if delay_hours is not None
                else None
            ),

            "berth": berth,

            "movement_report": movement_report,

            "loa_m": first["loa_m"],

            "beam_m": first["beam_m"],

            "draft_m": first["draft_m"],

            "cargo_type": first["cargo_type"],

            "cargo_qty_mt": first["cargo_qty_mt"],

            "priority": first["priority"],

            "priority_group": first["priority_group"],

            "readiness": first["readiness"],

            "remarks": first["remarks"],

            "source_report": first["report_filename"],

            "status": status,
        })

    episodes = pd.DataFrame(
        episodes
    )

    episodes = episodes.sort_values(
        "arrival_timestamp"
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    matched = episodes[
        episodes["status"]
        == "matched"
    ]

    print("=" * 80)
    print("PARADIP WAITING EPISODES")
    print("=" * 80)

    print(
        f"Waiting snapshots:     {len(waiting)}"
    )

    print(
        f"Unique episodes:       {len(episodes)}"
    )

    print(
        f"Matched episodes:      {len(matched)}"
    )

    print(
        f"Unmatched episodes:    "
        f"{(episodes['status'] == 'unmatched').sum()}"
    )

    print(
        f"Unique vessels:        "
        f"{episodes['vessel_name_normalized'].nunique()}"
    )

    if not matched.empty:

        print()
        print("DELAY DISTRIBUTION:")

        print(
            matched[
                "arrival_to_berth_hours"
            ].describe().to_string()
        )

    # APJ SETHU
    apj = episodes[
        episodes["vessel_name_normalized"]
        == "APJ SETHU"
    ]

    print()
    print("APJ SETHU:")

    print(
        apj[
            [
                "arrival_timestamp",
                "first_waiting_report",
                "last_waiting_report",
                "waiting_snapshot_count",
                "berth_timestamp",
                "arrival_to_berth_hours",
                "status",
            ]
        ].to_string(index=False)
    )

    if apj.empty:
        raise RuntimeError(
            "APJ SETHU missing"
        )

    apj_matched = apj[
        apj["status"]
        == "matched"
    ]

    if apj_matched.empty:
        raise RuntimeError(
            "APJ SETHU did not match"
        )

    value = float(
        apj_matched.iloc[0][
            "arrival_to_berth_hours"
        ]
    )

    if abs(value - 50.2) > 1:
        raise RuntimeError(
            f"APJ SETHU validation failed: {value}"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    episodes.to_csv(
        OUTPUT,
        index=False,
    )

    metadata = {
        "dataset": "Paradip waiting episodes",
        "status": "validated",
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "waiting_snapshots": len(waiting),
        "unique_episodes": len(episodes),
        "matched_episodes": len(matched),
        "unmatched_episodes": int(
            (
                episodes["status"]
                == "unmatched"
            ).sum()
        ),
        "unique_vessels": int(
            episodes[
                "vessel_name_normalized"
            ].nunique()
        ),
        "target": "arrival_to_berth_hours",
        "definition": (
            "Elapsed time from reported arrival "
            "to first subsequent BERTHS event."
        ),
        "episode_definition": (
            "Waiting snapshots sharing the same "
            "normalized vessel and reported arrival "
            "timestamp are treated as one episode."
        ),
        "apj_sethu_validation_hours": value,
    }

    with open(
        METADATA,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("VALIDATION: PASS")
    print("=" * 80)

    print(
        f"Output: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
