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
    / "paradip_waiting_labels_v1.csv"
)

METADATA = (
    OUTPUT_DIR
    / "paradip_waiting_labels_v1_metadata.json"
)


DUE_TOKENS = (
    "DUE",
    "EXPECTED",
)

BERTH_TOKENS = (
    "BERTH",
    "BERTHS",
    "SHIFT",
    "SHIFTS",
)

SAIL_TOKENS = (
    "SAIL",
    "SAILS",
)


def purpose_text(value):
    if pd.isna(value):
        return ""
    return str(value).upper().strip()


def contains_token(text, tokens):
    return any(
        token in text
        for token in tokens
    )


def main():

    df = pd.read_csv(INPUT)

    required = [
        "vessel_name_normalized",
        "vessel_name",
        "purpose",
        "movement_timestamp",
        "report_filename",
        "berth",
        "loa_m",
        "beam_m",
        "draft_m",
        "cargo",
        "quantity_mt",
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
        .apply(purpose_text)
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
    # Build episodes vessel-by-vessel.
    # ---------------------------------------------------------------

    for vessel, group in df.groupby(
        "vessel_name_normalized",
        sort=False,
    ):

        events = group.to_dict(
            "records"
        )

        for i, due_event in enumerate(events):

            due_purpose = purpose_text(
                due_event["purpose"]
            )

            if not contains_token(
                due_purpose,
                DUE_TOKENS,
            ):
                continue

            due_time = due_event[
                "movement_timestamp"
            ]

            # -------------------------------------------------------
            # Find the most recent event before this DUE/EXPECTED
            # event.
            #
            # If the vessel was already BERTHED/SHIFTED and has not
            # sailed, this DUE event is not a new waiting episode.
            # -------------------------------------------------------

            already_active = False

            for previous in reversed(
                events[:i]
            ):

                previous_time = previous[
                    "movement_timestamp"
                ]

                if previous_time >= due_time:
                    continue

                previous_purpose = purpose_text(
                    previous["purpose"]
                )

                if contains_token(
                    previous_purpose,
                    SAIL_TOKENS,
                ):
                    break

                if contains_token(
                    previous_purpose,
                    BERTH_TOKENS,
                ):
                    already_active = True
                    break

            if already_active:
                continue

            # -------------------------------------------------------
            # Find the first subsequent BERTH/SHIFT event.
            #
            # If SAILS occurs first, the vessel never produced a
            # valid observed waiting-to-berth episode.
            # -------------------------------------------------------

            berth_event = None

            for later in events[i + 1:]:

                later_time = later[
                    "movement_timestamp"
                ]

                if later_time <= due_time:
                    continue

                later_purpose = purpose_text(
                    later["purpose"]
                )

                if contains_token(
                    later_purpose,
                    SAIL_TOKENS,
                ):
                    break

                if contains_token(
                    later_purpose,
                    BERTH_TOKENS,
                ):
                    berth_event = later
                    break

            if berth_event is None:
                continue

            berth_time = berth_event[
                "movement_timestamp"
            ]

            waiting_hours = (
                berth_time - due_time
            ).total_seconds() / 3600.0

            # Sanity bounds.
            if waiting_hours < 0:
                continue

            if waiting_hours > 30 * 24:
                continue

            quantity = pd.to_numeric(
                due_event.get("quantity_mt"),
                errors="coerce",
            )

            labels.append(
                {
                    "vessel_name_normalized": vessel,

                    "due_timestamp": due_time,

                    "berth_timestamp": berth_time,

                    "waiting_hours": (
                        waiting_hours
                    ),

                    "waiting_days": (
                        waiting_hours / 24.0
                    ),

                    "berth": berth_event.get(
                        "berth"
                    ),

                    "loa_m": due_event.get(
                        "loa_m"
                    ),

                    "beam_m": due_event.get(
                        "beam_m"
                    ),

                    "draft_m": due_event.get(
                        "draft_m"
                    ),

                    "cargo": due_event.get(
                        "cargo"
                    ),

                    "quantity_mt": quantity,

                    "due_purpose": due_event.get(
                        "purpose"
                    ),

                    "berth_purpose": berth_event.get(
                        "purpose"
                    ),

                    "due_report": due_event.get(
                        "report_filename"
                    ),

                    "berth_report": berth_event.get(
                        "report_filename"
                    ),
                }
            )

    labels = pd.DataFrame(
        labels
    )

    if labels.empty:
        raise RuntimeError(
            "No valid Paradip waiting episodes were derived."
        )

    labels = labels.sort_values(
        "due_timestamp"
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Basic duplicate protection.
    # ---------------------------------------------------------------

    duplicate_mask = labels.duplicated(
        subset=[
            "vessel_name_normalized",
            "due_timestamp",
            "berth_timestamp",
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

    metadata = {
        "source": str(
            INPUT.relative_to(ROOT)
        ),
        "output": str(
            OUTPUT.relative_to(ROOT)
        ),
        "label_definition": (
            "Observed waiting time from a DUE/EXPECTED movement "
            "to the next BERTH/SHIFT movement for the same vessel, "
            "excluding vessels already active before the due event "
            "and excluding episodes interrupted by SAILS."
        ),
        "rows": int(len(labels)),
        "duplicates_removed": duplicates_removed,
        "date_min": (
            labels["due_timestamp"]
            .min()
            .isoformat()
        ),
        "date_max": (
            labels["due_timestamp"]
            .max()
            .isoformat()
        ),
        "waiting_days": {
            "median": float(
                labels["waiting_days"].median()
            ),
            "mean": float(
                labels["waiting_days"].mean()
            ),
            "p90": float(
                labels["waiting_days"].quantile(0.90)
            ),
            "p95": float(
                labels["waiting_days"].quantile(0.95)
            ),
            "max": float(
                labels["waiting_days"].max()
            ),
        },
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("=" * 80)
    print("PARADIP WAITING LABELS v1")
    print("=" * 80)

    print(
        "Valid episodes:",
        len(labels),
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    print(
        "Date range:",
        labels["due_timestamp"].min(),
        "→",
        labels["due_timestamp"].max(),
    )

    print()

    print(
        "Waiting days:"
    )

    print(
        labels["waiting_days"]
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
        "Zero-wait episodes:",
        int(
            (
                labels["waiting_hours"] == 0
            ).sum()
        ),
    )

    print(
        "Episodes > 1 day:",
        int(
            (
                labels["waiting_days"] > 1
            ).sum()
        ),
    )

    print(
        "Episodes > 3 days:",
        int(
            (
                labels["waiting_days"] > 3
            ).sum()
        ),
    )

    print(
        "Episodes > 7 days:",
        int(
            (
                labels["waiting_days"] > 7
            ).sum()
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
