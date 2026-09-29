from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "02_DATA"
    / "processed"
    / "trade"
    / "usa"
)

ORIGINS = DATA / "usa_met_coal_origins.csv"

MET_EXPORTS = (
    DATA
    / "eia_met_coal_exports_by_customs_district.csv"
)

TOTAL_EXPORTS = (
    DATA
    / "eia_coal_exports_by_customs_district.csv"
)


def load_data():
    origins = pd.read_csv(ORIGINS)
    met = pd.read_csv(MET_EXPORTS)
    total = pd.read_csv(TOTAL_EXPORTS)

    return origins, met, total


def build_origin_state():
    origins, met, total = load_data()

    met = met[
        [
            "entity",
            "q1_2026",
            "q4_2025",
            "q1_2025",
            "ytd_2026",
            "ytd_2025",
            "percent_change",
        ]
    ].copy()

    met = met.rename(
        columns={
            "entity": "eia_customs_name",
            "q1_2026": "met_q1_2026",
            "q4_2025": "met_q4_2025",
            "q1_2025": "met_q1_2025",
            "ytd_2026": "met_ytd_2026",
            "ytd_2025": "met_ytd_2025",
            "percent_change": "met_yoy_percent",
        }
    )

    total = total[
        [
            "entity",
            "q1_2026",
            "q4_2025",
            "q1_2025",
            "ytd_2026",
            "ytd_2025",
            "percent_change",
        ]
    ].copy()

    total = total.rename(
        columns={
            "entity": "eia_customs_name",
            "q1_2026": "coal_q1_2026",
            "q4_2025": "coal_q4_2025",
            "q1_2025": "coal_q1_2025",
            "ytd_2026": "coal_ytd_2026",
            "ytd_2025": "coal_ytd_2025",
            "percent_change": "coal_yoy_percent",
        }
    )

    result = origins.merge(
        met,
        on="eia_customs_name",
        how="left",
    )

    result = result.merge(
        total,
        on="eia_customs_name",
        how="left",
    )

    # Share of total U.S. coal exports represented by metallurgical coal
    # at the customs district.
    result["met_coal_share_q1_2026"] = (
        result["met_q1_2026"]
        / result["coal_q1_2026"]
    )

    result["met_coal_share_ytd_2026"] = (
        result["met_ytd_2026"]
        / result["coal_ytd_2026"]
    )

    # Simple activity classification.
    # This is an observed trade-flow classification, NOT congestion.
    def classify(row):
        value = row["met_yoy_percent"]

        if pd.isna(value):
            return "unknown"

        if value >= 20:
            return "increasing"

        if value <= -20:
            return "decreasing"

        return "stable"

    result["trade_flow_state"] = result.apply(
        classify,
        axis=1,
    )

    return result


def main():
    result = build_origin_state()

    output = DATA / "usa_met_coal_origin_state.csv"

    result.to_csv(
        output,
        index=False,
    )

    print("=" * 90)
    print("NEREUS — USA MET COAL ORIGIN STATE")
    print("=" * 90)
    print()

    columns = [
        "origin_id",
        "origin_name",
        "eia_customs_name",
        "met_q1_2026",
        "met_yoy_percent",
        "met_coal_share_q1_2026",
        "trade_flow_state",
    ]

    print(
        result[columns]
        .to_string(index=False)
    )

    print()
    print(f"Saved: {output}")
    print(f"Rows: {len(result)}")


if __name__ == "__main__":
    main()
