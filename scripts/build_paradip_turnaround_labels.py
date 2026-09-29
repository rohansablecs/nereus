from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "02_DATA/processed/paradip"
    / "berthing_movements_corpus_v3.csv"
)

OUTPUT_DIR = (
    ROOT
    / "02_DATA/processed/paradip"
)

OUTPUT = (
    OUTPUT_DIR
    / "paradip_turnaround_labels_v1.csv"
)

METADATA = (
    OUTPUT_DIR
    / "paradip_turnaround_labels_v1_metadata.json"
)


def norm(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .upper()
        .strip()
    )


def is_berth(value):
    return norm(value) == "BERTHS"


def is_sail(value):
    return norm(value) == "SAILS"


def main():

    df = pd.read_csv(INPUT)

    required = [
        "vessel_name_normalized",
        "vessel_name",
        "purpose",
        "movement_timestamp",
        "berth",
        "loa_m",
        "beam_m",
        "draft_m",
        "cargo",
        "quantity_mt",
        "report_filename",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["movement_timestamp"] = pd.to_datetime(
        df["movement_timestamp"],
        errors="coerce",
    )

    df["vessel_name_normalized"] = (
        df["vessel_name_normalized"]
        .fillna(df["vessel_name"])
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["purpose_norm"] = (
        df["purpose"]
        .apply(norm)
    )

    df = df[
        df["movement_timestamp"].notna()
        &
        df["vessel_name_normalized"].notna()
        &
        (df["vessel_name_normalized"] != "")
    ].copy()

    df = df.sort_values(
        [
            "vessel_name_normalized",
            "movement_timestamp",
        ]
    )

    labels = []

    # ---------------------------------------------------------------
    # Construct observed berth cycles.
    #
    # A cycle begins with the first BERTHS event after a SAILS event
    # (or the first observed BERTHS event for that vessel).
    #
    # The cycle ends at the first subsequent SAILS event.
    #
    # Additional BERTHS events between them are treated as shifts,
    # not new cycles.
    # ---------------------------------------------------------------

    for vessel, group in df.groupby(
        "vessel_name_normalized",
        sort=False,
    ):

        events = group.to_dict(
            "records"
        )

        active = None
        shifts = []

        for event in events:

            purpose = event[
                "purpose_norm"
            ]

            timestamp = event[
                "movement_timestamp"
            ]

            if is_berth(purpose):

                if active is None:

                    active = {
                        "first_berth": event,
                        "last_berth": event,
                        "shifts": [],
                    }

                else:

                    active["shifts"].append(
                        event
                    )

                    active["last_berth"] = (
                        event
                    )

                continue

            if is_sail(purpose):

                if active is None:
                    continue

                first = active[
                    "first_berth"
                ]

                sail_time = timestamp

                berth_time = first[
                    "movement_timestamp"
                ]

                duration_hours = (
                    sail_time - berth_time
                ).total_seconds() / 3600.0

                # Reject impossible / corrupted cycles.
                if duration_hours < 0:
                    active = None
                    continue

                if duration_hours > 30 * 24:
                    active = None
                    continue

                quantity = pd.to_numeric(
                    first.get(
                        "quantity_mt"
                    ),
                    errors="coerce",
                )

                shift_count = len(
                    active["shifts"]
                )

                productivity = None

                if (
                    pd.notna(quantity)
                    and quantity > 0
                    and duration_hours > 0
                ):
                    productivity = (
                        quantity
                        /
                        (duration_hours / 24.0)
                    )

                labels.append(
                    {
                        "vessel_name_normalized": vessel,

                        "berth_start_timestamp": (
                            berth_time
                        ),

                        "sail_timestamp": (
                            sail_time
                        ),

                        "berth_duration_hours": (
                            duration_hours
                        ),

                        "berth_duration_days": (
                            duration_hours / 24.0
                        ),

                        "shift_count": (
                            shift_count
                        ),

                        "berth": first.get(
                            "berth"
                        ),

                        "loa_m": first.get(
                            "loa_m"
                        ),

                        "beam_m": first.get(
                            "beam_m"
                        ),

                        "draft_m": first.get(
                            "draft_m"
                        ),

                        "cargo": first.get(
                            "cargo"
                        ),

                        "quantity_mt": quantity,

                        "observed_productivity_mt_day": (
                            productivity
                        ),

                        "berth_report": first.get(
                            "report_filename"
                        ),

                        "sail_report": event.get(
                            "report_filename"
                        ),
                    }
                )

                active = None

    labels = pd.DataFrame(
        labels
    )

    if labels.empty:
        raise RuntimeError(
            "No clean BERTHS -> SAILS cycles were derived."
        )

    labels = labels.sort_values(
        "berth_start_timestamp"
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Remove exact duplicates.
    # ---------------------------------------------------------------

    duplicate_mask = labels.duplicated(
        subset=[
            "vessel_name_normalized",
            "berth_start_timestamp",
            "sail_timestamp",
        ],
        keep="first",
    )

    duplicates_removed = int(
        duplicate_mask.sum()
    )

    labels = labels[
        ~duplicate_mask
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata = {
        "source": str(
            INPUT.relative_to(ROOT)
        ),

        "output": str(
            OUTPUT.relative_to(ROOT)
        ),

        "label_definition": (
            "Observed operational berth cycle from the first BERTHS "
            "event after the previous SAILS event through the next "
            "SAILS event. Additional BERTHS events inside the cycle "
            "are treated as berth shifts."
        ),

        "rows": int(
            len(labels)
        ),

        "duplicates_removed": (
            duplicates_removed
        ),

        "date_min": (
            labels[
                "berth_start_timestamp"
            ]
            .min()
            .isoformat()
        ),

        "date_max": (
            labels[
                "berth_start_timestamp"
            ]
            .max()
            .isoformat()
        ),

        "duration_days": {
            "median": float(
                labels[
                    "berth_duration_days"
                ].median()
            ),

            "mean": float(
                labels[
                    "berth_duration_days"
                ].mean()
            ),

            "p90": float(
                labels[
                    "berth_duration_days"
                ].quantile(0.90)
            ),

            "p95": float(
                labels[
                    "berth_duration_days"
                ].quantile(0.95)
            ),

            "max": float(
                labels[
                    "berth_duration_days"
                ].max()
            ),
        },

        "productivity": {
            "valid_rows": int(
                labels[
                    "observed_productivity_mt_day"
                ].notna().sum()
            ),

            "median_mt_day": (
                float(
                    labels[
                        "observed_productivity_mt_day"
                    ].median()
                )
                if labels[
                    "observed_productivity_mt_day"
                ].notna().any()
                else None
            ),

            "p90_mt_day": (
                float(
                    labels[
                        "observed_productivity_mt_day"
                    ].quantile(0.90)
                )
                if labels[
                    "observed_productivity_mt_day"
                ].notna().any()
                else None
            ),
        },
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # ---------------------------------------------------------------
    # Console
    # ---------------------------------------------------------------

    print("=" * 80)
    print("PARADIP TURNAROUND LABELS v1")
    print("=" * 80)

    print(
        "Clean berth cycles:",
        len(labels),
    )

    print(
        "Unique vessels:",
        labels[
            "vessel_name_normalized"
        ].nunique(),
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    print()

    print(
        "Berth duration:"
    )

    print(
        labels[
            "berth_duration_days"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
        .round(2)
        .to_string()
    )

    print()

    print(
        "Shift count:"
    )

    print(
        labels[
            "shift_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    valid_productivity = labels[
        "observed_productivity_mt_day"
    ].dropna()

    print(
        "Rows with observed productivity:",
        len(valid_productivity),
    )

    if len(valid_productivity):

        print(
            "Median productivity MT/day:",
            round(
                valid_productivity.median(),
                2,
            ),
        )

        print(
            "P90 productivity MT/day:",
            round(
                valid_productivity.quantile(0.90),
                2,
            ),
        )

    print()

    print(
        "Output:",
        OUTPUT,
    )

    print(
        "Metadata:",
        METADATA,
    )


if __name__ == "__main__":
    main()
