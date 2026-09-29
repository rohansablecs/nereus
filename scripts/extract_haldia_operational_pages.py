from pathlib import Path
import json
import re

from pypdf import PdfReader


PDF_PATH = Path(
    "02_DATA/raw/port/haldia_tanker_lineup_2026-06-22.pdf"
)

OUT_DIR = Path(
    "02_DATA/raw/port/haldia_operational"
)

OUT_TEXT = OUT_DIR / "haldia_operational_pages.txt"
OUT_META = OUT_DIR / "haldia_operational_metadata.json"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reader = PdfReader(PDF_PATH)

    pages = []

    for page_number in range(1, 6):
        text = reader.pages[page_number - 1].extract_text() or ""

        pages.append(
            {
                "page": page_number,
                "characters": len(text),
                "text": text,
            }
        )

    with OUT_TEXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        for page in pages:
            f.write(
                f"\n{'=' * 80}\n"
            )
            f.write(
                f"PAGE {page['page']}\n"
            )
            f.write(
                f"{'=' * 80}\n\n"
            )
            f.write(
                page["text"]
            )
            f.write("\n")

    metadata = {
        "source_file": str(PDF_PATH),
        "source_pages": "1-5",
        "snapshot_date": "2026-06-22",
        "snapshot_times": {
            "morning_position": "06:00",
            "vessel_lineup": "11:00",
            "printed": "09:32",
        },
        "pages": [
            {
                "page": p["page"],
                "characters": p["characters"],
            }
            for p in pages
        ],
        "parser_notes": [
            "Pages 1-5 are preserved before normalization.",
            "No values are inferred from flattened PDF columns.",
            "Raw page text is retained for provenance.",
            "Structured extraction will be performed from these source blocks.",
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
        "Haldia operational page extraction complete."
    )
    print(
        f"Pages extracted: {len(pages)}"
    )
    print(
        f"Text: {OUT_TEXT}"
    )
    print(
        f"Metadata: {OUT_META}"
    )

    for page in pages:
        print(
            f"Page {page['page']}: "
            f"{page['characters']} characters"
        )


if __name__ == "__main__":
    main()
