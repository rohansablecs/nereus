from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


RAW_DIR = Path("02_DATA/raw/port/paradip/daily_traffic")
OUT_DIR = Path("02_DATA/processed/paradip")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SECTION_NAMES = {
    "FOR CB-1 & CB-2",
    "FOR OJ",
    "FOR SPM",
    "FOR PICT",
    "Others",
}

VESSEL_RE = re.compile(
    r"(MV\.|MT\.)\s+(.+?)\s+(\d{3}\.\d{2})\s+(\d+)\s+"
)


def get_dates(text):
    text = " ".join(text.split())

    m = re.search(
        r"DAILY TRAFFIC UPDATE FOR\s+"
        r"(\d{2}-\d{2}-\d{4})\s+"
        r"HELD ON DT:\s*"
        r"(\d{2}-\d{2}-\d{4})",
        text,
        re.I,
    )

    if not m:
        raise RuntimeError("Could not extract report dates.")

    return (
        datetime.strptime(
            m.group(1), "%d-%m-%Y"
        ).date().isoformat(),
        datetime.strptime(
            m.group(2), "%d-%m-%Y"
        ).date().isoformat(),
    )


def get_page_lines(page):
    text = page.extract_text() or ""
    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


def parse_line(line):
    """
    Parse the actual flattened pypdf line.

    Example source:

    32.26 0.00 05.09 0648TNPGCLL 75000 SEAPORTMV. APJ SETHU
    224.86 1 TH.COAL J M BAXIMSPLR - P
    """

    vessel_match = VESSEL_RE.search(line)

    if not vessel_match:
        raise RuntimeError(
            f"Cannot locate vessel/LOA in line:\n{line}"
        )

    vessel_name = (
        vessel_match.group(1)
        + " "
        + vessel_match.group(2)
    ).strip()

    loa = float(vessel_match.group(3))
    priority = int(vessel_match.group(4))

    before = line[:vessel_match.start()].strip()
    after = line[vessel_match.end():].strip()

    # ---------------------------------------------------------
    # Beam + draft
    # ---------------------------------------------------------

    first = re.match(
        r"^(\d+\.\d+)\s+(\d+\.\d+)\s+",
        before,
    )

    if not first:
        raise RuntimeError(
            f"Cannot parse beam/draft:\n{line}"
        )

    beam = float(first.group(1))
    draft = float(first.group(2))

    # ---------------------------------------------------------
    # Arrival date/time
    #
    # Search independently because pypdf can concatenate the
    # following token onto the time.
    # ---------------------------------------------------------

    arrival = re.search(
        r"(\d{2})\.(\d{2})\s+(\d{4})",
        before,
    )

    if not arrival:
        raise RuntimeError(
            f"Cannot parse arrival:\n{line}"
        )

    day = int(arrival.group(1))
    month = int(arrival.group(2))
    hhmm = arrival.group(3)

    arrival_date = (
        f"2026-{month:02d}-{day:02d}"
    )

    arrival_time = (
        f"{hhmm[:2]}:{hhmm[2:]}"
    )

    # ---------------------------------------------------------
    # Cargo quantity
    #
    # It is the first 4-6 digit number before the vessel name.
    # ---------------------------------------------------------

    qty_matches = re.findall(
        r"\b\d{4,6}\b",
        before,
    )

    if not qty_matches:
        raise RuntimeError(
            f"Cannot parse quantity:\n{line}"
        )

    cargo_qty_mt = int(qty_matches[-1])

    # ---------------------------------------------------------
    # Cargo
    # ---------------------------------------------------------

    cargo = None

    for value in [
        "TH.COAL",
        "C.COAL",
        "PCI COAL",
        "I.ORE",
        "P.COKE",
        "S.COIL",
        "PIG IRON",
        "UREA",
        "CRUDE OIL",
        "BUTANE/PROP",
        "NAPTHA",
    ]:
        if re.search(
            rf"(?<!\S){re.escape(value)}(?!\S)",
            after,
            re.I,
        ):
            cargo = value
            break

    # ---------------------------------------------------------
    # Vessel size
    # ---------------------------------------------------------

    vessel_size = None

    if "ICT/NR" in after:
        vessel_size = "ICT/NR"
    elif re.search(r"\bNR\b", after):
        vessel_size = "NR"

    # ---------------------------------------------------------
    # Remarks
    # ---------------------------------------------------------

    remark = re.search(
        r"\([A-Z]\)",
        after,
    )

    remarks = (
        remark.group(0)
        if remark
        else None
    )

    return {
        "vessel_name": vessel_name,
        "loa_m": loa,
        "beam_m": beam,
        "draft_m": draft,
        "arrival_date": arrival_date,
        "arrival_time": arrival_time,
        "cargo_type": cargo,
        "cargo_qty_mt": cargo_qty_mt,
        "priority": priority,
        "vessel_size": vessel_size,
        "remarks": remarks,
        "raw_line": line,
    }


def parse_page4(lines):
    section = None
    readiness = None
    rows = []

    for line in lines:

        # Section
        if line in SECTION_NAMES:
            section = line
            readiness = None
            continue

        # Readiness
        if line in {"READY", "NOT READY"}:
            readiness = line
            continue

        # Ignore headings / footnotes
        if (
            line == "PRIORITY"
            or line.startswith("B. VESSELS")
            or line.startswith("NAME OF THE VESSEL")
            or line.startswith("*A-Inadequate")
        ):
            continue

        # Every actual vessel row starts with beam + draft +
        # arrival date/time.
        if not re.match(
            r"^\d+\.\d+\s+\d+\.\d+\s+\d{2}\.\d{2}\s+\d{4}",
            line,
        ):
            continue

        if section is None:
            raise RuntimeError(
                f"Vessel encountered without section:\n{line}"
            )

        row = parse_line(line)

        row["priority_group"] = section
        row["readiness"] = readiness

        rows.append(row)

    return rows


def validate(df):

    if len(df) != 13:
        raise RuntimeError(
            f"Expected 13 waiting vessels, got {len(df)}."
        )

    expected_sections = {
        "FOR CB-1 & CB-2": 3,
        "FOR OJ": 3,
        "FOR SPM": 2,
        "FOR PICT": 4,
        "Others": 1,
    }

    actual = (
        df["priority_group"]
        .value_counts()
        .to_dict()
    )

    if actual != expected_sections:
        raise RuntimeError(
            f"Section mismatch.\n"
            f"Expected: {expected_sections}\n"
            f"Actual: {actual}"
        )

    expected_rows = {
        "MV. APJ SETHU": {
            "loa_m": 224.86,
            "beam_m": 32.26,
            "draft_m": 0.00,
            "cargo_qty_mt": 75000,
            "cargo_type": "TH.COAL",
            "priority_group": "FOR CB-1 & CB-2",
            "readiness": "READY",
        },
        "MV. SINGAPORE BULKER": {
            "loa_m": 189.99,
            "beam_m": 32.26,
            "draft_m": 13.30,
            "cargo_qty_mt": 55000,
            "cargo_type": "TH.COAL",
            "priority_group": "FOR CB-1 & CB-2",
            "readiness": "READY",
        },
        "MV. APJ ANGAD 2": {
            "loa_m": 224.94,
            "beam_m": 32.26,
            "draft_m": 14.50,
            "cargo_qty_mt": 75000,
            "cargo_type": "TH.COAL",
            "priority_group": "FOR CB-1 & CB-2",
            "readiness": "READY",
        },
    }

    for vessel, expected in expected_rows.items():

        match = df[
            df["vessel_name"] == vessel
        ]

        if len(match) != 1:
            raise RuntimeError(
                f"Missing/duplicate vessel: {vessel}"
            )

        row = match.iloc[0]

        for field, expected_value in expected.items():

            actual_value = row[field]

            if isinstance(
                expected_value,
                float,
            ):
                if pd.isna(actual_value):
                    raise RuntimeError(
                        f"{vessel}: {field} missing."
                    )

                if abs(
                    float(actual_value)
                    - expected_value
                ) > 0.01:
                    raise RuntimeError(
                        f"{vessel}: {field}="
                        f"{actual_value}; expected "
                        f"{expected_value}"
                    )

            elif actual_value != expected_value:
                raise RuntimeError(
                    f"{vessel}: {field}="
                    f"{actual_value!r}; expected "
                    f"{expected_value!r}"
                )

    required = [
        "vessel_name",
        "loa_m",
        "beam_m",
        "draft_m",
        "arrival_date",
        "arrival_time",
        "cargo_qty_mt",
        "priority_group",
        "readiness",
    ]

    for field in required:
        if df[field].isna().any():
            bad = df[
                df[field].isna()
            ]["vessel_name"].tolist()

            raise RuntimeError(
                f"{field} missing for: {bad}"
            )

    print("VALIDATION: PASS")


def main():

    pdfs = sorted(
        RAW_DIR.glob("*.pdf")
    )

    if not pdfs:
        raise RuntimeError(
            f"No PDFs found in {RAW_DIR}"
        )

    all_rows = []
    metadata = []

    for pdf in pdfs:

        with open(pdf, "rb") as f:
            header = f.read(5)

        if header != b"%PDF-":
            print(
                f"SKIP non-PDF: {pdf.name}"
            )
            continue

        reader = PdfReader(
            str(pdf)
        )

        if len(reader.pages) < 6:
            print(
                f"SKIP invalid PDF: {pdf.name}"
            )
            continue

        first_page = (
            reader.pages[0].extract_text()
            or ""
        )

        traffic_date, held_on_date = (
            get_dates(first_page)
        )

        lines = get_page_lines(
            reader.pages[3]
        )

        rows = parse_page4(lines)

        if not rows:
            raise RuntimeError(
                f"No waiting vessels found in {pdf.name}"
            )

        for row in rows:
            row["traffic_date"] = traffic_date
            row["held_on_date"] = held_on_date
            row["port"] = "Paradip"
            row["source_pdf"] = pdf.name

        all_rows.extend(rows)

        metadata.append(
            {
                "source_pdf": pdf.name,
                "traffic_date": traffic_date,
                "held_on_date": held_on_date,
                "waiting_rows": len(rows),
            }
        )

        print(
            f"{pdf.name}: "
            f"traffic_date={traffic_date}, "
            f"held_on={held_on_date}, "
            f"waiting={len(rows)}"
        )

    df = pd.DataFrame(all_rows)

    columns = [
        "traffic_date",
        "held_on_date",
        "port",
        "priority_group",
        "readiness",
        "vessel_name",
        "loa_m",
        "beam_m",
        "draft_m",
        "arrival_date",
        "arrival_time",
        "cargo_type",
        "cargo_qty_mt",
        "priority",
        "vessel_size",
        "remarks",
        "raw_line",
        "source_pdf",
    ]

    df = df[columns]

    validate(df)

    output = (
        OUT_DIR
        / "waiting_vessels.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    metadata_path = (
        OUT_DIR
        / "daily_traffic_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "source": (
                    "Paradip Port Authority "
                    "Daily Traffic Update"
                ),
                "parser_version": "v9_raw_line",
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "files": metadata,
                "total_rows": len(df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print("PARADIP DAILY TRAFFIC INGESTION COMPLETE")
    print("========================================")
    print(f"Rows: {len(df)}")
    print(f"CSV:  {output}")
    print(f"JSON: {metadata_path}")


if __name__ == "__main__":
    main()
