from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "02_DATA/raw/port/paradip/daily_traffic"
OUT_DIR = ROOT / "02_DATA/processed/paradip"

OUT_CSV = OUT_DIR / "paradip_operational_history_daily.csv"
OUT_META = OUT_DIR / "paradip_operational_history_daily_metadata.json"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Report date
# ---------------------------------------------------------------------------

def parse_report_date_from_filename(path: Path) -> pd.Timestamp | None:
    """
    Paradip Daily Traffic Report filenames follow:

        dtrDDMM.pdf

    Examples:

        dtr0710.pdf  -> 07 October
        dtr2206.pdf  -> 22 June
        dtr3008.pdf  -> 30 August
        DTR0107.pdf  -> 01 July

    Some files have a suffix because of duplicate downloads:

        dtr2607-2.pdf
        dtr1408-2.pdf
        dtr1508-2.pdf

    The suffix does not change the report date.

    The year is inferred from the filename date and the corpus:
    - October/November 2025 files are 2025
    - June/July/August/September 2026 files are 2026
    """

    name = path.stem.upper()

    match = re.search(
        r"DTR\s*(\d{2})(\d{2})(?:[-_]\d+)?$",
        name,
    )

    if not match:
        return None

    day = int(match.group(1))
    month = int(match.group(2))

    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None

    # Corpus spans Oct/Nov 2025 and Jun/Sep 2026.
    # The filename alone gives DD/MM, so use the month to resolve the year.
    if month in {10, 11, 12}:
        year = 2025
    else:
        year = 2026

    try:
        return pd.Timestamp(
            datetime(year, month, day)
        )
    except ValueError:
        return None


# Keep the old function name available for audit/import compatibility.
def parse_report_date(text: str) -> pd.Timestamp | None:
    """
    Deprecated text-based date parser.

    Historical operational ingestion now derives the canonical report date
    from the filename because the PDF contains multiple operational dates.
    This function remains only for compatibility with older tooling.
    """
    patterns = [
        r"\b(\d{2})[-/](\d{2})[-/](20\d{2})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = map(int, match.groups())

            try:
                return pd.Timestamp(
                    datetime(year, month, day)
                )
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def parse_number(value) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def section_text(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
) -> str:

    lines = text.splitlines()

    start_idx = None

    for i, line in enumerate(lines):
        upper = line.upper()

        if any(
            re.search(pattern, upper)
            for pattern in start_patterns
        ):
            start_idx = i
            break

    if start_idx is None:
        return ""

    end_idx = len(lines)

    for i in range(start_idx + 1, len(lines)):
        upper = lines[i].upper()

        if any(
            re.search(pattern, upper)
            for pattern in end_patterns
        ):
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


# ---------------------------------------------------------------------------
# Waiting / anchorage
# ---------------------------------------------------------------------------

def parse_waiting_state(text: str) -> dict:

    section = section_text(
        text,
        start_patterns=[
            r"VESSELS\s+WAITING\s+AT\s+ANCHORAGE",
            r"VESSEL[S]?\s+WAITING\s+AT\s+ANCHORAGE",
        ],
        end_patterns=[
            r"VESSELS\s+EXPECTED",
            r"EXPECTED\s+VESSELS",
            r"BERTHING\s+MOVEMENTS",
            r"VESSELS\s+SAILED",
        ],
    )

    if not section:
        return {
            "vessels_waiting_anchorage": None,
            "dry_bulk_vessels_waiting": None,
            "dry_bulk_cargo_mt_waiting": None,
        }

    lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip()
    ]

    vessel_count = 0
    dry_bulk_count = 0
    dry_bulk_cargo = 0.0

    dry_bulk_keywords = [
        "COAL",
        "COKE",
        "ORE",
        "PELLET",
        "GYPSUM",
        "LIMESTONE",
        "FERTILIZER",
        "UREA",
        "CLINKER",
        "PETCOKE",
        "SULPHUR",
        "BAUXITE",
        "MANGANESE",
    ]

    for line in lines:

        upper = line.upper()

        # Avoid headers and obvious non-vessel lines.
        if any(
            x in upper
            for x in [
                "VESSELS WAITING",
                "VESSEL NAME",
                "TOTAL",
                "ANCHORAGE",
            ]
        ):
            continue

        # A vessel row usually has an LOA / draft / numeric structure.
        # We use presence of a vessel-like name + dimensions/cargo.
        if re.search(r"\b\d{2,3}(?:\.\d+)?\b", line):

            vessel_count += 1

            if any(
                keyword in upper
                for keyword in dry_bulk_keywords
            ):
                dry_bulk_count += 1

                numbers = re.findall(
                    r"\b\d[\d,]*(?:\.\d+)?\b",
                    line,
                )

                numeric_values = []

                for value in numbers:
                    parsed = parse_number(value)

                    if parsed is not None:
                        numeric_values.append(parsed)

                # Cargo quantity is normally one of the larger numeric
                # values in the row. Avoid blindly treating LOA/draft
                # as cargo. Use plausible MT values.
                cargo_candidates = [
                    x
                    for x in numeric_values
                    if 500 <= x <= 500000
                ]

                if cargo_candidates:
                    dry_bulk_cargo += max(
                        cargo_candidates
                    )

    return {
        "vessels_waiting_anchorage": vessel_count,
        "dry_bulk_vessels_waiting": dry_bulk_count,
        "dry_bulk_cargo_mt_waiting": (
            dry_bulk_cargo
            if dry_bulk_count > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Berthing movement counts
# ---------------------------------------------------------------------------

def parse_movement_state(text: str) -> dict:

    section = section_text(
        text,
        start_patterns=[
            r"D\.?\s*BERTHING\s+MOVEMENTS",
            r"BERTHING\s+MOVEMENTS",
        ],
        end_patterns=[
            r"VESSEL[S]?\s+WAITING",
            r"VESSEL[S]?\s+EXPECTED",
            r"IMPORTANT\s+NOTES",
            r"PORT\s+STATISTICS",
        ],
    )

    if not section:
        return {
            "berth_events": None,
            "sail_events": None,
            "shift_events": None,
            "dry_bulk_berth_events": None,
            "dry_bulk_sail_events": None,
            "unique_berth_events": None,
        }

    berth_events = 0
    sail_events = 0
    shift_events = 0
    dry_bulk_berth_events = 0
    dry_bulk_sail_events = 0

    berth_names = set()

    dry_bulk_keywords = [
        "COAL",
        "COKE",
        "ORE",
        "PELLET",
        "GYPSUM",
        "LIMESTONE",
        "FERTILIZER",
        "UREA",
        "CLINKER",
        "PETCOKE",
        "SULPHUR",
        "BAUXITE",
        "MANGANESE",
    ]

    for raw_line in section.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if "BERTHS" in upper:
            berth_events += 1

            if any(
                keyword in upper
                for keyword in dry_bulk_keywords
            ):
                dry_bulk_berth_events += 1

        elif "SAILS" in upper:
            sail_events += 1

            if any(
                keyword in upper
                for keyword in dry_bulk_keywords
            ):
                dry_bulk_sail_events += 1

        elif "SHIFT" in upper:
            shift_events += 1

        # Known Paradip berth naming patterns.
        berth_patterns = [
            r"\b(?:CB|EQ|NQ|CQ|KICT|SPM|SOJ|NQ|MPT|GC|SOUTH|NORTH|IOHP)"
            r"[- ]?\d+[A-Z0-9-]*\b",
        ]

        for pattern in berth_patterns:
            matches = re.findall(pattern, upper)

            for match in matches:
                berth_names.add(
                    re.sub(r"\s+", " ", match.strip())
                )

    return {
        "berth_events": berth_events,
        "sail_events": sail_events,
        "shift_events": shift_events,
        "dry_bulk_berth_events": dry_bulk_berth_events,
        "dry_bulk_sail_events": dry_bulk_sail_events,
        "unique_berth_events": (
            len(berth_names)
            if berth_events > 0
            else 0
        ),
    }


# ---------------------------------------------------------------------------
# Expected vessels
# ---------------------------------------------------------------------------

def parse_expected_state(text: str) -> dict:

    section = section_text(
        text,
        start_patterns=[
            r"VESSELS\s+EXPECTED",
            r"EXPECTED\s+VESSELS",
            r"VESSELS\s+DUE",
        ],
        end_patterns=[
            r"VESSELS\s+WAITING",
            r"BERTHING\s+MOVEMENTS",
            r"IMPORTANT\s+NOTES",
        ],
    )

    if not section:
        return {
            "vessels_expected": None,
            "dry_bulk_vessels_expected": None,
            "dry_bulk_cargo_mt_expected": None,
        }

    vessel_count = 0
    dry_bulk_count = 0
    dry_bulk_cargo = 0.0

    dry_bulk_keywords = [
        "COAL",
        "COKE",
        "ORE",
        "PELLET",
        "GYPSUM",
        "LIMESTONE",
        "FERTILIZER",
        "UREA",
        "CLINKER",
        "PETCOKE",
        "SULPHUR",
        "BAUXITE",
        "MANGANESE",
    ]

    for raw_line in section.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if any(
            x in upper
            for x in [
                "VESSELS EXPECTED",
                "EXPECTED VESSELS",
                "VESSEL NAME",
                "TOTAL",
            ]
        ):
            continue

        if re.search(r"\b\d{2,3}(?:\.\d+)?\b", line):

            vessel_count += 1

            if any(
                keyword in upper
                for keyword in dry_bulk_keywords
            ):
                dry_bulk_count += 1

                numbers = re.findall(
                    r"\b\d[\d,]*(?:\.\d+)?\b",
                    line,
                )

                candidates = []

                for value in numbers:
                    parsed = parse_number(value)

                    if parsed is not None and 500 <= parsed <= 500000:
                        candidates.append(parsed)

                if candidates:
                    dry_bulk_cargo += max(candidates)

    return {
        "vessels_expected": vessel_count,
        "dry_bulk_vessels_expected": dry_bulk_count,
        "dry_bulk_cargo_mt_expected": (
            dry_bulk_cargo
            if dry_bulk_count > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Build one report
# ---------------------------------------------------------------------------

def parse_report(path: Path) -> dict | None:

    report_date = parse_report_date_from_filename(path)

    if report_date is None:
        raise ValueError(
            f"Could not parse report date from filename: {path.name}"
        )

    text = extract_pdf_text(path)

    if not text.strip():
        raise ValueError(
            f"No extractable text: {path.name}"
        )

    waiting = parse_waiting_state(text)
    expected = parse_expected_state(text)
    movements = parse_movement_state(text)

    return {
        "date": report_date,
        "source_file": path.name,
        **waiting,
        **expected,
        **movements,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DIR.glob("*"),
        key=lambda p: p.name.lower(),
    )

    pdf_files = [
        path
        for path in files
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    ]

    rows = []
    failures = []

    for path in pdf_files:

        try:
            row = parse_report(path)

            if row is not None:
                rows.append(row)

        except Exception as exc:

            failures.append(
                {
                    "file": path.name,
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No operational reports were successfully parsed."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    # -----------------------------------------------------------------------
    # Important:
    #
    # Multiple physical files can exist for the same report date because
    # downloads may have suffixes such as "-2".
    #
    # We do NOT silently throw one away.
    #
    # If multiple files exist for the same date, combine their values using
    # the maximum for count/cargo signals. This is appropriate only for the
    # operational summary signals here because the duplicate files are
    # alternate copies of the same daily report.
    # -----------------------------------------------------------------------

    numeric_columns = [
        "vessels_waiting_anchorage",
        "dry_bulk_vessels_waiting",
        "dry_bulk_cargo_mt_waiting",
        "vessels_expected",
        "dry_bulk_vessels_expected",
        "dry_bulk_cargo_mt_expected",
        "berth_events",
        "sail_events",
        "shift_events",
        "dry_bulk_berth_events",
        "dry_bulk_sail_events",
        "unique_berth_events",
    ]

    duplicate_counts = (
        df.groupby("date")
        .size()
        .sort_values(ascending=False)
    )

    # Combine same-date alternate downloads.
    aggregations = {}

    for column in numeric_columns:
        aggregations[column] = "max"

    aggregations["source_file"] = lambda values: "|".join(
        sorted(
            str(value)
            for value in values
        )
    )

    df = (
        df.groupby(
            "date",
            as_index=False,
        )
        .agg(aggregations)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = {
        "dataset": "Paradip daily operational history",
        "source_directory": str(RAW_DIR),
        "date_source": "filename",
        "filename_date_pattern": "DTRDDMM",
        "source_files": len(pdf_files),
        "successful_files": len(rows),
        "failed_files": len(failures),
        "unique_report_dates": len(df),
        "duplicate_date_groups": int(
            (duplicate_counts > 1).sum()
        ),
        "failed_files_detail": failures,
        "date_min": (
            df["date"].min().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
        "date_max": (
            df["date"].max().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
        "columns": list(df.columns),
        "notes": [
            "Canonical report date is derived from the DTRDDMM filename because PDF text contains multiple operational dates.",
            "Files with suffixes such as -2 are treated as alternate copies of the same report date.",
            "Duplicate same-date files are combined conservatively using maximum available operational signal values.",
            "The dataset contains daily operational snapshots, not vessel-level waiting-time labels.",
            "vessels_waiting_anchorage represents vessels reported waiting at anchorage in the source report.",
            "unique_berth_events counts distinct berth labels observed in the report's BERTHING MOVEMENTS section and is not simultaneous berth occupancy.",
        ],
    }

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    OUT_META.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PARADIP OPERATIONAL HISTORY")
    print("=" * 100)
    print(f"Source files:       {len(pdf_files)}")
    print(f"Parsed files:       {len(rows)}")
    print(f"Failed files:       {len(failures)}")
    print(f"Unique dates:       {len(df)}")
    print(
        f"Duplicate groups:   {int((duplicate_counts > 1).sum())}"
    )

    if not df.empty:
        print(
            f"Date range:         "
            f"{df['date'].min().date()} → {df['date'].max().date()}"
        )

    print()
    print("Coverage:")

    for column in numeric_columns:

        available = int(
            df[column].notna().sum()
        )

        print(
            f"  {column:<32} "
            f"{available}/{len(df)}"
        )

    if failures:

        print()
        print("FAILED FILES:")

        for failure in failures:
            print(
                f"  {failure['file']}: "
                f"{failure['error']}"
            )

    print()
    print("LATEST SNAPSHOTS:")

    display_columns = [
        "date",
        "vessels_waiting_anchorage",
        "dry_bulk_vessels_waiting",
        "dry_bulk_cargo_mt_waiting",
        "vessels_expected",
        "dry_bulk_vessels_expected",
        "dry_bulk_cargo_mt_expected",
        "berth_events",
        "sail_events",
        "shift_events",
        "dry_bulk_berth_events",
        "dry_bulk_sail_events",
        "unique_berth_events",
    ]

    print(
        df[
            display_columns
        ]
        .tail(10)
        .to_string(index=False)
    )

    print()
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_META}")


if __name__ == "__main__":
    main()
