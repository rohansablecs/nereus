from pathlib import Path
import json
import re

import pandas as pd
from pypdf import PdfReader


PDF_PATH = Path(
    "02_DATA/raw/port/haldia_tanker_lineup_2026-06-22.pdf"
)

OUT_CSV = Path(
    "02_DATA/raw/port/haldia_tanker_lineup_2026-06-22.csv"
)

OUT_META = Path(
    "02_DATA/raw/port/haldia_tanker_lineup_2026-06-22_metadata.json"
)


def clean(value):
    if value is None:
        return None

    value = re.sub(r"\s+", " ", str(value)).strip()

    return value or None


def get_lineup_page():
    reader = PdfReader(PDF_PATH)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if "Tanker Line up and Seniority" in text:
            return page_number, text

    raise RuntimeError(
        "Tanker lineup page not found."
    )


def parse_vessel_blocks(text):
    """
    Detect tanker records from the Haldia tanker lineup.

    A real vessel record has this structure:

        Sl.No
        Vessel Name
        HAL VCN
        Draft
        LOA

    Vessel names can span multiple extracted-text lines.

    The PDF table header can also begin with numbers, so we
    explicitly reject candidates containing table-header text.
    """

    pattern = re.compile(
        r"(?ms)^"
        r"(?P<sl_no>\d+)"
        r"\s+"
        r"(?P<name>[^0-9]{1,100}?)"
        r"\s+"
        r"(?P<vcn>HAL\d+[A-Z]*)"
        r"\s+"
        r"(?P<draft>\d+(?:\.\d+)?)"
        r"\s+"
        r"(?P<loa>\d+(?:\.\d+)?)"
        r"(?=\s|$)"
    )

    matches = []

    for match in pattern.finditer(text):
        name = clean(match.group("name"))

        if not name:
            continue

        # Reject the PDF table header accidentally matching as
        # a vessel record.
        header_markers = [
            "Sl. No",
            "Vessels Name",
            "VCN",
            "Draft",
            "LOA",
            "Cargo",
            "Anchoring Time",
            "Document Readiness",
            "Final Readiness",
            "Estimated Completion",
            "Agents",
            "Importer",
            "Exporter",
            "Terminal",
            "Compatible Berth",
            "Remarks",
        ]

        if any(
            marker.lower() in name.lower()
            for marker in header_markers
        ):
            continue

        # Vessel names should contain alphabetic characters.
        if not re.search(r"[A-Za-z]", name):
            continue

        matches.append(match)

    rows = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        raw_block = text[start:end].strip()

        rows.append(
            {
                "sl_no": int(match.group("sl_no")),
                "vessel_name": clean(match.group("name")),
                "vcn": match.group("vcn"),
                "draft_m": float(match.group("draft")),
                "loa_m": float(match.group("loa")),
                "raw_row": raw_block,
            }
        )

    return rows


def extract_fields(row):
    """
    Extract only fields that can be recovered safely.

    The original PDF has multiline cargo and remarks, so the
    complete raw vessel block is always preserved.
    """

    raw = row["raw_row"]

    # Remove the vessel identity section.
    prefix = re.compile(
        r"(?ms)^"
        r"\d+\s+"
        r".*?"
        r"HAL\d+[A-Z]*\s+"
        r"\d+(?:\.\d+)?\s+"
        r"\d+(?:\.\d+)?\s*"
    )

    remainder = prefix.sub(
        "",
        raw,
        count=1,
    )

    # ---------------------------------------------------------
    # Readiness
    # ---------------------------------------------------------

    readiness_matches = list(
        re.finditer(
            r"\b(Not Ready|Ready)\b",
            remainder,
            flags=re.IGNORECASE,
        )
    )

    document_readiness = None
    final_readiness = None

    if len(readiness_matches) >= 1:
        document_readiness = (
            readiness_matches[0]
            .group(1)
            .title()
        )

    if len(readiness_matches) >= 2:
        final_readiness = (
            readiness_matches[1]
            .group(1)
            .title()
        )

    # ---------------------------------------------------------
    # ETA
    # ---------------------------------------------------------

    eta_match = re.search(
        r"\bETA[- ]?"
        r"([0-9]{1,2}"
        r"(?:\.[0-9]{2}\.[0-9]{4})?"
        r"(?:\([^)]+\))?)",
        remainder,
        flags=re.IGNORECASE,
    )

    eta_raw = None

    if eta_match:
        eta_raw = (
            "ETA-"
            + eta_match.group(1)
        )

    # ---------------------------------------------------------
    # Anchoring time
    # Example: 21(17:30)
    # ---------------------------------------------------------

    anchoring_match = re.search(
        r"\b\d{1,2}\(\d{2}:\d{2}\)\b",
        remainder,
    )

    anchoring_time_raw = (
        anchoring_match.group(0)
        if anchoring_match
        else None
    )

    # ---------------------------------------------------------
    # Cargo
    # ---------------------------------------------------------

    cargo_raw = None

    cargo_end_positions = []

    for expression in [
        r"\bETA[- ]?\d",
        r"\b\d{1,2}\(\d{2}:\d{2}\)\b",
        r"\bNot Ready\b",
        r"\bReady\b",
    ]:
        match = re.search(
            expression,
            remainder,
            flags=re.IGNORECASE,
        )

        if match:
            cargo_end_positions.append(
                match.start()
            )

    if cargo_end_positions:
        cargo_raw = clean(
            remainder[
                :min(cargo_end_positions)
            ]
        )

    # ---------------------------------------------------------
    # Estimated completion hours
    # ---------------------------------------------------------

    estimated_completion_hours = None

    if len(readiness_matches) >= 2:
        second_readiness = readiness_matches[1]

        after_readiness = (
            remainder[
                second_readiness.end():
            ]
        )

        completion_match = re.match(
            r"\s*(\d{1,3})\b",
            after_readiness,
        )

        if completion_match:
            value = int(
                completion_match.group(1)
            )

            if 0 < value <= 999:
                estimated_completion_hours = value

    row.update(
        {
            "cargo_raw": cargo_raw,
            "anchoring_time_raw": anchoring_time_raw,
            "eta_raw": eta_raw,
            "document_readiness": document_readiness,
            "final_readiness": final_readiness,
            "estimated_completion_hours": (
                estimated_completion_hours
            ),
        }
    )

    return row


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            PDF_PATH
        )

    page_number, text = (
        get_lineup_page()
    )

    rows = parse_vessel_blocks(
        text
    )

    if not rows:
        raise RuntimeError(
            "No tanker vessel records found."
        )

    rows = [
        extract_fields(row)
        for row in rows
    ]

    df = pd.DataFrame(rows)

    # VCN is the stable source identity.
    df = df.drop_duplicates(
        subset=["vcn"],
        keep="first",
    ).reset_index(drop=True)

    # The document states the tanker lineup is as of
    # 22/06/2026 at 11:00 Hrs.
    df["snapshot_date"] = (
        "2026-06-22"
    )

    df["snapshot_time"] = (
        "11:00"
    )

    df["source_page"] = (
        page_number
    )

    df["source_document"] = (
        PDF_PATH.name
    )

    columns = [
        "snapshot_date",
        "snapshot_time",
        "sl_no",
        "vessel_name",
        "vcn",
        "draft_m",
        "loa_m",
        "cargo_raw",
        "anchoring_time_raw",
        "eta_raw",
        "document_readiness",
        "final_readiness",
        "estimated_completion_hours",
        "source_page",
        "source_document",
        "raw_row",
    ]

    df = df[columns]

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "source_file": str(
            PDF_PATH
        ),
        "source_page": page_number,
        "snapshot_date": "2026-06-22",
        "snapshot_time": "11:00",
        "rows_parsed": len(df),
        "columns": list(df.columns),
        "parser_notes": [
            "Tanker lineup is on page 6.",
            "Priority sections restart Sl. No numbering.",
            "VCN is used as the stable vessel identity.",
            "Vessel names may span multiple PDF text lines.",
            "PDF table headers are explicitly rejected.",
            "Multiline source blocks are preserved in raw_row.",
            "Cargo is preserved as cargo_raw.",
            "Source missingness is preserved.",
            "No vessel-level values are invented.",
            "This is an operational snapshot, not a live feed.",
        ],
    }

    OUT_META.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Haldia tanker lineup parsing complete."
    )

    print(
        f"Source page: {page_number}"
    )

    print(
        f"Rows parsed: {len(df)}"
    )

    print(
        f"CSV: {OUT_CSV}"
    )

    print(
        f"Metadata: {OUT_META}"
    )

    print("\nVessels:")

    print(
        df[
            [
                "sl_no",
                "vessel_name",
                "vcn",
                "draft_m",
                "loa_m",
                "document_readiness",
                "final_readiness",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
