from pathlib import Path
import json
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_DATA/processed/paradip"

WAITING = DATA / "waiting_vessels_corpus.csv"
MOVEMENTS = DATA / "berthing_movements_corpus_v3.csv"

OUTPUT = DATA / "observed_waiting_delays_final.csv"
METADATA = DATA / "waiting_delay_final_metadata.json"


def normalize_name(value):
    return (
        str(value)
        .upper()
        .replace("MV.", "")
        .replace("MT.", "")
        .strip()
    )


def main():

    print("=" * 80)
    print("PARADIP WAITING → BERTH DELAY MATCHER")
    print("=" * 80)

    waiting = pd.read_csv(
        WAITING
    )

    movements = pd.read_csv(
        MOVEMENTS
    )

    print(
        f"Waiting observations: {len(waiting)}"
    )

    print(
        f"Movement observations: {len(movements)}"
    )

    # ----------------------------------------------------------------------
    # WAITING TIMESTAMP
    # ----------------------------------------------------------------------

    waiting["arrival_timestamp"] = pd.to_datetime(
        waiting["arrival_date"].astype(str)
        + " "
        + waiting["arrival_time"].astype(str),
        errors="coerce",
    )

    waiting["vessel_name_normalized"] = (
        waiting["vessel_name"]
        .map(normalize_name)
    )

    # ----------------------------------------------------------------------
    # MOVEMENT TIMESTAMP
    # ----------------------------------------------------------------------

    movements["movement_timestamp"] = pd.to_datetime(
        movements["movement_timestamp"],
        errors="coerce",
    )

    movements["vessel_name_normalized"] = (
        movements["vessel_name"]
        .map(normalize_name)
    )

    # ONLY BERTHS are valid arrival → berth completion events.
    berths = movements[
        movements["purpose"].astype(str).str.upper()
        == "BERTHS"
    ].copy()

    berths = berths.dropna(
        subset=[
            "movement_timestamp",
            "vessel_name_normalized",
        ]
    )

    berths = berths.sort_values(
        [
            "vessel_name_normalized",
            "movement_timestamp",
        ]
    )

    print(
        f"BERTHS events available: {len(berths)}"
    )

    # ----------------------------------------------------------------------
    # MATCH
    # ----------------------------------------------------------------------

    results = []

    for _, row in waiting.iterrows():

        vessel = row[
            "vessel_name_normalized"
        ]

        arrival = row[
            "arrival_timestamp"
        ]

        candidates = berths[
            berths[
                "vessel_name_normalized"
            ]
            == vessel
        ]

        # First BERTHS event at or after this reported arrival.
        candidates = candidates[
            candidates[
                "movement_timestamp"
            ] >= arrival
        ]

        if candidates.empty:

            result = row.to_dict()

            result["matched_movement_timestamp"] = pd.NaT
            result["matched_berth"] = None
            result["matched_movement_report"] = None
            result["matched_movement_source_page"] = None
            result["arrival_to_berth_hours"] = pd.NA
            result["arrival_to_berth_days"] = pd.NA
            result["delay_status"] = "unmatched"

            results.append(result)

            continue

        movement = candidates.iloc[0]

        berth_time = movement[
            "movement_timestamp"
        ]

        delay_hours = (
            berth_time - arrival
        ).total_seconds() / 3600

        result = row.to_dict()

        result[
            "matched_movement_timestamp"
        ] = berth_time

        result[
            "matched_berth"
        ] = movement["berth"]

        result[
            "matched_movement_report"
        ] = movement[
            "report_filename"
        ]

        result[
            "matched_movement_source_page"
        ] = movement[
            "source_page"
        ]

        result[
            "arrival_to_berth_hours"
        ] = round(
            delay_hours,
            4,
        )

        result[
            "arrival_to_berth_days"
        ] = round(
            delay_hours / 24,
            4,
        )

        result[
            "delay_status"
        ] = "matched"

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    # ----------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------

    matched = result_df[
        result_df["delay_status"]
        == "matched"
    ].copy()

    unmatched = result_df[
        result_df["delay_status"]
        == "unmatched"
    ].copy()

    print()
    print("=" * 80)
    print("MATCH RESULTS")
    print("=" * 80)

    print(
        f"Waiting observations: {len(result_df)}"
    )

    print(
        f"Matched:              {len(matched)}"
    )

    print(
        f"Unmatched:            {len(unmatched)}"
    )

    # ----------------------------------------------------------------------
    # APJ SETHU
    # ----------------------------------------------------------------------

    apj = result_df[
        result_df[
            "vessel_name_normalized"
        ]
        == "APJ SETHU"
    ].copy()

    print()
    print("APJ SETHU:")
    print(
        apj[
            [
                "arrival_timestamp",
                "matched_movement_timestamp",
                "matched_berth",
                "arrival_to_berth_hours",
                "delay_status",
            ]
        ].to_string(index=False)
    )

    if apj.empty:
        raise RuntimeError(
            "APJ SETHU missing from waiting corpus"
        )

    apj_matched = apj[
        apj["delay_status"]
        == "matched"
    ]

    if apj_matched.empty:
        raise RuntimeError(
            "APJ SETHU failed to match"
        )

    apj_delay = float(
        apj_matched.iloc[0][
            "arrival_to_berth_hours"
        ]
    )

    print(
        f"\nAPJ SETHU delay = {apj_delay:.2f} hours"
    )

    # Known source-derived sanity check:
    # 05-Sep 06:48 → 07-Sep 09:00 ≈ 50.2h
    if abs(apj_delay - 50.2) > 1.0:

        raise RuntimeError(
            f"APJ SETHU validation failed: "
            f"expected ~50.2h, got {apj_delay:.2f}h"
        )

    # ----------------------------------------------------------------------
    # SANITY CHECKS
    # ----------------------------------------------------------------------

    if len(result_df) != 1469:
        raise RuntimeError(
            f"Expected 1469 waiting observations, "
            f"got {len(result_df)}"
        )

    if len(matched) == 0:
        raise RuntimeError(
            "Zero waiting observations matched"
        )

    if (
        matched[
            "arrival_to_berth_hours"
        ] < 0
    ).any():

        raise RuntimeError(
            "Negative waiting delay detected"
        )

    # ----------------------------------------------------------------------
    # SUMMARY STATISTICS
    # ----------------------------------------------------------------------

    print()
    print("MATCHED DELAY DISTRIBUTION:")

    print(
        matched[
            "arrival_to_berth_hours"
        ].describe().to_string()
    )

    print()
    print("MATCHED VESSEL COUNT:")
    print(
        matched[
            "vessel_name_normalized"
        ].nunique()
    )

    print()
    print("TOP MATCHED VESSELS:")
    print(
        matched[
            "vessel_name_normalized"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )

    # ----------------------------------------------------------------------
    # WRITE
    # ----------------------------------------------------------------------

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    metadata = {
        "dataset": "Paradip observed arrival-to-berth delay",
        "status": "validated",
        "generated_at_utc": (
            datetime.utcnow().isoformat()
            + "Z"
        ),
        "waiting_observations": len(result_df),
        "berths_events_available": len(berths),
        "matched_observations": len(matched),
        "unmatched_observations": len(unmatched),
        "matched_fraction": (
            len(matched) / len(result_df)
        ),
        "unique_matched_vessels": int(
            matched[
                "vessel_name_normalized"
            ].nunique()
        ),
        "target": "arrival_to_berth_hours",
        "target_definition": (
            "Elapsed time from the waiting table's "
            "reported vessel arrival timestamp to the "
            "first subsequent BERTHS movement for the "
            "same normalized vessel name."
        ),
        "interpretation_warning": (
            "This is an observed operational delay proxy, "
            "not a pure congestion measurement. It can "
            "include berth availability, readiness, "
            "documentation, priority and other operational "
            "effects."
        ),
        "validation": {
            "reports": 113,
            "waiting_rows": 1469,
            "apj_sethu_expected_hours": 50.2,
            "apj_sethu_actual_hours": apj_delay,
            "negative_delays": int(
                (
                    matched[
                        "arrival_to_berth_hours"
                    ]
                    < 0
                ).sum()
            ),
        },
        "sources": {
            "waiting": str(
                WAITING.relative_to(ROOT)
            ),
            "movements": str(
                MOVEMENTS.relative_to(ROOT)
            ),
        },
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

    print(
        f"Metadata: {METADATA}"
    )


if __name__ == "__main__":
    main()
