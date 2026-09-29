from pathlib import Path
import json
import re

import pandas as pd
import pdfplumber


PDF_PATH = Path(
    "02_DATA/raw/port/haldia_tanker_lineup_2026-06-22.pdf"
)

OUT_DIR = Path(
    "02_DATA/raw/port/haldia_operational"
)

BERTH_OUT = OUT_DIR / "haldia_berth_occupancy_snapshot.csv"
DUE_OUT = OUT_DIR / "haldia_vessels_due_snapshot.csv"
ARRIVAL_OUT = OUT_DIR / "haldia_arrivals_snapshot.csv"
DEPARTURE_OUT = OUT_DIR / "haldia_departures_snapshot.csv"
TRAFFIC_OUT = OUT_DIR / "haldia_traffic_update_snapshot.csv"
SAGAR_OUT = OUT_DIR / "haldia_sagar_operations_snapshot.csv"
META_OUT = OUT_DIR / "haldia_operational_dataset_metadata.json"

SNAPSHOT_DATE = "2026-06-22"
MORNING_TIME = "06:00"


def clean(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).replace("\n", " "),
    ).strip()


def number(value):
    value = clean(value)

    if not value:
        return None

    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def vcn_from(text):
    match = re.search(
        r"HAL\d+[A-Z]*",
        clean(text),
        re.IGNORECASE,
    )

    return (
        match.group(0).upper()
        if match
        else None
    )


def vessel_from(text):
    text = clean(text)

    if not text:
        return None

    text = re.sub(
        r"\s*\[?\s*HAL\d+[A-Z]*\s*\]?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^(?:M\.V\.|M\.T\.)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return clean(text).rstrip("[").strip()


def words_in_region(
    page,
    x0,
    x1,
    top,
    bottom,
):
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    return [
        w
        for w in words
        if (
            x0 <= w["x0"] < x1
            and top <= w["top"] < bottom
        )
    ]


def group_y(words, tolerance=3):
    """
    Group words that belong to the same printed text line.
    """

    groups = []

    for word in sorted(
        words,
        key=lambda w: (
            w["top"],
            w["x0"],
        ),
    ):
        placed = False

        for group in groups:
            if abs(
                group["top"] - word["top"]
            ) <= tolerance:
                group["words"].append(word)
                placed = True
                break

        if not placed:
            groups.append(
                {
                    "top": word["top"],
                    "words": [word],
                }
            )

    for group in groups:
        group["words"].sort(
            key=lambda w: w["x0"]
        )

    return groups


def line_text(words):
    return clean(
        " ".join(
            w["text"]
            for w in sorted(
                words,
                key=lambda w: w["x0"],
            )
        )
    )


# ============================================================
# PAGE 1 — INSIDE DOCK
# ============================================================

def parse_inside_dock(page):
    """
    Coordinate-based extraction of:

    Vessels Inside Dock and at Haldia Oil Jetties.

    Important:
    berth labels are vertically merged for some groups.
    Therefore the last explicit berth label is carried forward.
    """

    # Actual table boundaries observed in the PDF.
    TOP = 143
    BOTTOM = 493

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    table_words = [
        w
        for w in words
        if (
            TOP <= w["top"] < BOTTOM
            and 45 <= w["x0"] < 255
        )
    ]

    vessel_starts = [
        w
        for w in table_words
        if (
            w["x0"] < 115
            and w["text"] in (
                "M.V.",
                "M.T.",
            )
        )
    ]

    vessel_starts.sort(
        key=lambda w: w["top"]
    )

    # Explicit berth labels.
    berth_words = [
        w
        for w in table_words
        if 45 <= w["x0"] < 94
    ]

    berth_lines = group_y(
        berth_words,
        tolerance=3,
    )

    explicit_berths = []

    for group in berth_lines:
        text = line_text(
            group["words"]
        )

        if text in (
            "Berth",
            "No.",
        ):
            continue

        if text in (
            "2",
            "3",
            "4",
            "4A",
            "4B",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "HOJ-I",
            "HOJ-II",
            "HOJ-III",
            "TERMINAL 2",
        ):
            explicit_berths.append(
                (
                    group["top"],
                    text,
                )
            )

    explicit_berths.sort()

    def berth_at(y):
        candidates = [
            berth
            for by, berth in explicit_berths
            if by <= y + 3
        ]

        return (
            candidates[-1]
            if candidates
            else None
        )

    rows = []

    for start in vessel_starts:

        y = start["top"]

        # Words on the same printed row around vessel column.
        vessel_words = [
            w
            for w in table_words
            if (
                90 <= w["x0"] < 255
                and abs(
                    w["top"] - y
                ) <= 4
            )
        ]

        vessel_text = line_text(
            vessel_words
        )

        vcn = vcn_from(
            vessel_text
        )

        if not vcn:
            continue

        vessel_name = vessel_from(
            vessel_text
        )

        # LOA has its own narrow column.
        loa_words = words_in_region(
            page,
            257,
            289,
            y - 3,
            y + 6,
        )

        loa_text = line_text(
            loa_words
        )

        loa_m = number(
            loa_text
        )

        # Pull other columns from the same printed line.
        def col_text(x0, x1):
            ws = words_in_region(
                page,
                x0,
                x1,
                y - 3,
                y + 7,
            )

            return line_text(ws) or None

        rows.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "snapshot_time": MORNING_TIME,
                "berth": berth_at(y),
                "status": "occupied",
                "vessel_name": vessel_name,
                "vcn": vcn,
                "loa_m": loa_m,
                "steamer_agent": col_text(
                    289,
                    352,
                ),
                "cargo": col_text(
                    352,
                    440,
                ),
                "importer_exporter": col_text(
                    440,
                    520,
                ),
                "discharge_shipment": col_text(
                    520,
                    633,
                ),
                "total": col_text(
                    633,
                    737,
                ),
                "due": col_text(
                    737,
                    797,
                ),
                "raw_row": vessel_text,
            }
        )

    # Explicit vacant berth lines.
    vacancy_words = [
        w
        for w in table_words
        if (
            90 <= w["x0"] < 160
            and w["text"].lower() == "vacant"
        )
    ]

    for word in vacancy_words:
        berth = berth_at(
            word["top"]
        )

        rows.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "snapshot_time": MORNING_TIME,
                "berth": berth,
                "status": "vacant",
                "vessel_name": None,
                "vcn": None,
                "loa_m": None,
                "steamer_agent": None,
                "cargo": None,
                "importer_exporter": None,
                "discharge_shipment": None,
                "total": None,
                "due": None,
                "raw_row": "Vacant",
            }
        )

    # Deduplicate.
    unique = {}

    for row in rows:
        key = (
            row["berth"],
            row["vcn"],
            row["status"],
        )

        unique[key] = row

    return sorted(
        unique.values(),
        key=lambda r: (
            r["berth"] or "",
            r["vessel_name"] or "",
        ),
    )


# ============================================================
# PAGES 1–2 — VESSELS DUE
# ============================================================

def parse_due_page(
    page,
    top,
    bottom,
):
    """
    Each vessel starts with M.V. or M.T.

    We build a block from one vessel marker to the next.
    Columns are then recovered using X-coordinate regions.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    table_words = [
        w
        for w in words
        if (
            top <= w["top"] < bottom
            and 45 <= w["x0"] < 680
        )
    ]

    starts = [
        w
        for w in table_words
        if (
            w["x0"] < 65
            and w["text"] in (
                "M.V.",
                "M.T.",
            )
        )
    ]

    starts.sort(
        key=lambda w: w["top"]
    )

    rows = []

    for i, start in enumerate(starts):

        y0 = start["top"] - 2

        y1 = (
            starts[i + 1]["top"] - 2
            if i + 1 < len(starts)
            else bottom
        )

        block = [
            w
            for w in table_words
            if (
                y0 <= w["top"] < y1
            )
        ]

        # Vessel column.
        vessel_words = [
            w
            for w in block
            if w["x0"] < 180
        ]

        vessel_text = line_text(
            vessel_words
        )

        vcn = vcn_from(
            vessel_text
        )

        if not vcn:
            continue

        vessel_name = vessel_from(
            vessel_text
        )

        # Dimension columns.
        loa_words = [
            w
            for w in block
            if 180 <= w["x0"] < 213
        ]

        draft_words = [
            w
            for w in block
            if 213 <= w["x0"] < 240
        ]

        loa_values = [
            w["text"]
            for w in sorted(
                loa_words,
                key=lambda w: w["top"],
            )
            if re.fullmatch(
                r"\d+(?:\.\d+)?",
                w["text"],
            )
        ]

        draft_values = [
            w["text"]
            for w in sorted(
                draft_words,
                key=lambda w: w["top"],
            )
            if re.fullmatch(
                r"\d+(?:\.\d+)?",
                w["text"],
            )
        ]

        loa_m = (
            number(loa_values[0])
            if loa_values
            else None
        )

        draft_m = (
            number(draft_values[0])
            if draft_values
            else None
        )

        def block_col(x0, x1):
            ws = [
                w
                for w in block
                if x0 <= w["x0"] < x1
            ]

            return line_text(ws) or None

        rows.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "vessel_name": vessel_name,
                "vcn": vcn,
                "loa_m": loa_m,
                "draft_m": draft_m,
                "cargo": block_col(
                    240,
                    352,
                ),
                "total": block_col(
                    352,
                    383,
                ),
                "steamer_agent": block_col(
                    383,
                    475,
                ),
                "declaration_datetime": block_col(
                    475,
                    514,
                ),
                "importer_exporter": block_col(
                    514,
                    595,
                ),
                "origin_destination": block_col(
                    595,
                    691,
                ),
                "ready": block_col(
                    691,
                    728,
                ),
                "raw_row": vessel_text,
            }
        )

    return rows


def parse_vessels_due(
    page1,
    page2,
):
    """
    Page 1 + continuation on page 2.

    The source has 30 due-vessel starts on page 1 and
    22 on page 2 in this snapshot.
    """

    page1_rows = parse_due_page(
        page1,
        500,
        1110,
    )

    page2_rows = parse_due_page(
        page2,
        20,
        490,
    )

    rows = (
        page1_rows
        + page2_rows
    )

    unique = {}

    for row in rows:
        unique[row["vcn"]] = row

    return list(
        unique.values()
    )


# ============================================================
# PAGE 2 — ARRIVALS
# ============================================================

def parse_arrivals(page):
    """
    Arrivals is the lower-left table on page 2.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    # Actual table area observed from PDF coordinates.
    table_words = [
        w
        for w in words
        if (
            530 <= w["top"] < 650
            and 45 <= w["x0"] < 420
        )
    ]

    starts = [
        w
        for w in table_words
        if (
            w["x0"] < 65
            and w["text"] in (
                "M.V.",
                "M.T.",
            )
        )
    ]

    starts.sort(
        key=lambda w: w["top"]
    )

    rows = []

    for i, start in enumerate(starts):

        y0 = start["top"] - 2
        y1 = (
            starts[i + 1]["top"] - 2
            if i + 1 < len(starts)
            else 650
        )

        block = [
            w
            for w in table_words
            if y0 <= w["top"] < y1
        ]

        vessel_text = line_text(
            [
                w
                for w in block
                if w["x0"] < 260
            ]
        )

        vcn = vcn_from(
            vessel_text
        )

        if not vcn:
            continue

        rows.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "vessel_name": vessel_from(
                    vessel_text
                ),
                "vcn": vcn,
                "berth": line_text(
                    [
                        w
                        for w in block
                        if 260 <= w["x0"] < 295
                    ]
                ) or None,
                "haul_in_date": line_text(
                    [
                        w
                        for w in block
                        if 295 <= w["x0"] < 333
                    ]
                ) or None,
                "haul_in_time": line_text(
                    [
                        w
                        for w in block
                        if 333 <= w["x0"] < 357
                    ]
                ) or None,
                "commence_work_date": line_text(
                    [
                        w
                        for w in block
                        if 357 <= w["x0"] < 392
                    ]
                ) or None,
                "commence_work_time": line_text(
                    [
                        w
                        for w in block
                        if 392 <= w["x0"] < 420
                    ]
                ) or None,
                "raw_row": line_text(block),
            }
        )

    return rows


# ============================================================
# PAGE 2 — COMPLETION / DEPARTURE
# ============================================================

def parse_departures(page):
    """
    Completion and Departure is the lower-right table.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    table_words = [
        w
        for w in words
        if (
            530 <= w["top"] < 700
            and 420 <= w["x0"] < 800
        )
    ]

    starts = [
        w
        for w in table_words
        if (
            w["x0"] < 445
            and w["text"] in (
                "M.V.",
                "M.T.",
            )
        )
    ]

    starts.sort(
        key=lambda w: w["top"]
    )

    rows = []

    for i, start in enumerate(starts):

        y0 = start["top"] - 2
        y1 = (
            starts[i + 1]["top"] - 2
            if i + 1 < len(starts)
            else 700
        )

        block = [
            w
            for w in table_words
            if y0 <= w["top"] < y1
        ]

        vessel_text = line_text(
            [
                w
                for w in block
                if w["x0"] < 625
            ]
        )

        vcn = vcn_from(
            vessel_text
        )

        if not vcn:
            continue

        rows.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "vessel_name": vessel_from(
                    vessel_text
                ),
                "vcn": vcn,
                "berth": line_text(
                    [
                        w
                        for w in block
                        if 625 <= w["x0"] < 665
                    ]
                ) or None,
                "finish_work_date": line_text(
                    [
                        w
                        for w in block
                        if 665 <= w["x0"] < 700
                    ]
                ) or None,
                "finish_work_time": line_text(
                    [
                        w
                        for w in block
                        if 700 <= w["x0"] < 733
                    ]
                ) or None,
                "haul_out_date": line_text(
                    [
                        w
                        for w in block
                        if 733 <= w["x0"] < 770
                    ]
                ) or None,
                "haul_out_time": line_text(
                    [
                        w
                        for w in block
                        if 770 <= w["x0"] < 800
                    ]
                ) or None,
                "raw_row": line_text(block),
            }
        )

    return rows


# ============================================================
# PAGE 2 — TRAFFIC
# ============================================================

def parse_traffic(page):
    """
    Preserve the actual traffic-update rows.

    No derived congestion score is created here.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    table_words = [
        w
        for w in words
        if (
            545 <= w["top"] < 660
            and 420 <= w["x0"] < 800
        )
    ]

    groups = group_y(
        table_words,
        tolerance=3,
    )

    rows = []

    for group in groups:
        text = line_text(
            group["words"]
        )

        if not text:
            continue

        if text.lower() in (
            "commodity",
            "cumulative",
        ):
            continue

        if re.search(
            r"\d",
            text,
        ):
            rows.append(
                {
                    "snapshot_date": SNAPSHOT_DATE,
                    "raw_row": text,
                }
            )

    return rows


# ============================================================
# PAGE 5 — SAGAR / SANDHEADS
# ============================================================

def parse_sagar(page):
    """
    Extract explicit Sagar and Sandheads operational observations.

    Uses the page text because the Sagar section is not a regular
    ruled table.
    """

    text = page.extract_text(
        x_tolerance=2,
        y_tolerance=3,
    ) or ""

    rows = []

    # ---------------------------------------------------------
    # Sagar working vessel
    # ---------------------------------------------------------

    working_match = re.search(
        r"Working\s+Vessels\s+at\s+Sagar"
        r"(.*?)"
        r"(?:Working\s+Vessels\s+at\s+Sandheads|"
        r"Vessels\s+due\s+at\s+Sagar|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if working_match:
        block = clean(
            working_match.group(1)
        )

        vessel_match = re.search(
            r"(?:M\.V\.\s*)?"
            r"([A-Z][A-Z0-9 .'-]+?)"
            r"\s+LOA:\s*"
            r"(\d+(?:\.\d+)?)"
            r"\s+Draft:\s*"
            r"(\d+(?:\.\d+)?)",
            block,
            flags=re.IGNORECASE,
        )

        if vessel_match:
            rows.append(
                {
                    "snapshot_date": SNAPSHOT_DATE,
                    "location": "Sagar",
                    "status": "working",
                    "vessel_name": clean(
                        vessel_match.group(1)
                    ),
                    "loa_m": number(
                        vessel_match.group(2)
                    ),
                    "draft_m": number(
                        vessel_match.group(3)
                    ),
                    "raw_row": block,
                }
            )

    # ---------------------------------------------------------
    # Sagar due vessel
    # ---------------------------------------------------------

    due_match = re.search(
        r"Vessels\s+due\s+at\s+Sagar"
        r"(.*?)"
        r"(?:Vessels\s+due\s+at\s+New\s+Inner\s+Anchorage|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if due_match:
        block = clean(
            due_match.group(1)
        )

        vessel_match = re.search(
            r"(?:M\.V\.\s*)?"
            r"([A-Z][A-Z0-9 .'-]+?)"
            r"\s+LOA:\s*"
            r"(\d+(?:\.\d+)?)"
            r"\s+Draft:\s*"
            r"(\d+(?:\.\d+)?)",
            block,
            flags=re.IGNORECASE,
        )

        if vessel_match:
            rows.append(
                {
                    "snapshot_date": SNAPSHOT_DATE,
                    "location": "Sagar",
                    "status": "due",
                    "vessel_name": clean(
                        vessel_match.group(1)
                    ),
                    "loa_m": number(
                        vessel_match.group(2)
                    ),
                    "draft_m": number(
                        vessel_match.group(3)
                    ),
                    "raw_row": block,
                }
            )

    # ---------------------------------------------------------
    # Sandheads
    # ---------------------------------------------------------

    sandheads_match = re.search(
        r"Working\s+Vessels\s+at\s+Sandheads"
        r"(.*?)"
        r"(?:Vessels\s+due\s+at\s+Sandheads|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if sandheads_match:
        block = clean(
            sandheads_match.group(1)
        )

        if re.search(
            r"\bNO\s+VESSEL\b",
            block,
            flags=re.IGNORECASE,
        ):
            rows.append(
                {
                    "snapshot_date": SNAPSHOT_DATE,
                    "location": "Sandheads",
                    "status": "no_vessel",
                    "vessel_name": None,
                    "loa_m": None,
                    "draft_m": None,
                    "raw_row": block,
                }
            )

    return rows


# ============================================================
# MAIN
# ============================================================

def write_csv(rows, path):
    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    return df


def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            PDF_PATH
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pdfplumber.open(
        PDF_PATH
    ) as pdf:

        page1 = pdf.pages[0]
        page2 = pdf.pages[1]
        page5 = pdf.pages[4]

        berth_rows = parse_inside_dock(
            page1
        )

        due_rows = parse_vessels_due(
            page1,
            page2,
        )

        arrival_rows = parse_arrivals(
            page2
        )

        departure_rows = parse_departures(
            page2
        )

        traffic_rows = parse_traffic(
            page2
        )

        sagar_rows = parse_sagar(
            page5
        )

    berth_df = write_csv(
        berth_rows,
        BERTH_OUT,
    )

    due_df = write_csv(
        due_rows,
        DUE_OUT,
    )

    arrival_df = write_csv(
        arrival_rows,
        ARRIVAL_OUT,
    )

    departure_df = write_csv(
        departure_rows,
        DEPARTURE_OUT,
    )

    traffic_df = write_csv(
        traffic_rows,
        TRAFFIC_OUT,
    )

    sagar_df = write_csv(
        sagar_rows,
        SAGAR_OUT,
    )

    metadata = {
        "source_file": str(PDF_PATH),
        "snapshot_date": SNAPSHOT_DATE,
        "morning_position": MORNING_TIME,
        "pages_used": [1, 2, 5],
        "parser_method": (
            "coordinate-based PDF word extraction"
        ),
        "datasets": {
            "berth_occupancy": len(berth_df),
            "vessels_due": len(due_df),
            "arrivals": len(arrival_df),
            "departures": len(departure_df),
            "traffic_update": len(traffic_df),
            "sagar_sandheads": len(sagar_df),
        },
        "notes": [
            "Coordinates are used because source tables are "
            "visually ruled and vertically merged.",
            "Raw source text is preserved.",
            "Missing values are not fabricated.",
            "VCN is the vessel identity key.",
            "No congestion score is generated at ingestion.",
            "Sagar and Sandheads are distinct locations.",
        ],
    }

    META_OUT.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Haldia operational parsing complete.")
    print()

    print(
        f"Berth occupancy rows: {len(berth_df)}"
    )

    print(
        f"Vessels due rows: {len(due_df)}"
    )

    print(
        f"Arrivals rows: {len(arrival_df)}"
    )

    print(
        f"Departure rows: {len(departure_df)}"
    )

    print(
        f"Traffic rows: {len(traffic_df)}"
    )

    print(
        f"Sagar/Sandheads rows: {len(sagar_df)}"
    )

    print()

    if not berth_df.empty:
        print("BERTH OCCUPANCY")
        print(
            berth_df[
                [
                    "berth",
                    "status",
                    "vessel_name",
                    "vcn",
                    "loa_m",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    if not due_df.empty:
        print("VESSELS DUE")
        print(
            due_df[
                [
                    "vessel_name",
                    "vcn",
                    "loa_m",
                    "draft_m",
                ]
            ].head(60).to_string(
                index=False
            )
        )

    print()

    if not arrival_df.empty:
        print("ARRIVALS")
        print(
            arrival_df[
                [
                    "vessel_name",
                    "vcn",
                    "berth",
                    "haul_in_date",
                    "haul_in_time",
                    "commence_work_date",
                    "commence_work_time",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    if not departure_df.empty:
        print("DEPARTURES")
        print(
            departure_df[
                [
                    "vessel_name",
                    "vcn",
                    "berth",
                    "finish_work_date",
                    "finish_work_time",
                    "haul_out_date",
                    "haul_out_time",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    if not sagar_df.empty:
        print("SAGAR / SANDHEADS")
        print(
            sagar_df[
                [
                    "location",
                    "status",
                    "vessel_name",
                    "loa_m",
                    "draft_m",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
