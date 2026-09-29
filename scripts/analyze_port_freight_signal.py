from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DATA = Path(
    "02_DATA/processed/freight_port_features.parquet"
)

TARGETS = [
    "cape",
    "panamax",
    "supramax",
    "handy",
]

PORT_FEATURES = [
    "dhamra_portcalls_dry_bulk",
    "gopalpur_portcalls_dry_bulk",
    "haldia_portcalls_dry_bulk",
    "paradip_portcalls_dry_bulk",
    "vizag_portcalls_dry_bulk",
    "all_ports_portcalls_dry_bulk",

    "dhamra_import_dry_bulk",
    "gopalpur_import_dry_bulk",
    "haldia_import_dry_bulk",
    "paradip_import_dry_bulk",
    "vizag_import_dry_bulk",
    "all_ports_import_dry_bulk",

    "dhamra_export_dry_bulk",
    "gopalpur_export_dry_bulk",
    "haldia_export_dry_bulk",
    "paradip_export_dry_bulk",
    "vizag_export_dry_bulk",
    "all_ports_export_dry_bulk",
]

LAGS = [1, 3, 7, 14, 30]


def pct_change(series, periods=1):
    return series.pct_change(periods=periods)


def safe_corr(x, y):
    mask = x.notna() & y.notna()

    x = x[mask]
    y = y[mask]

    if len(x) < 30:
        return np.nan, np.nan, len(x)

    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, len(x)

    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    return pearson, spearman, len(x)


def main():

    df = pd.read_parquet(DATA)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    print("=" * 100)
    print("NEREUS — PORTWATCH → FREIGHT LEAD/LAG ANALYSIS")
    print("=" * 100)

    results = []

    # ------------------------------------------------------------
    # Convert freight to daily percentage changes.
    #
    # KOBC only publishes assessment days, so this is calculated
    # over successive KOBC observations.
    # ------------------------------------------------------------

    freight_returns = {}

    for target in TARGETS:
        freight_returns[target] = (
            df[target]
            .pct_change()
        )

    # ------------------------------------------------------------
    # Test whether PORT activity at t-k predicts FREIGHT movement
    # at t.
    # ------------------------------------------------------------

    for port_feature in PORT_FEATURES:

        if port_feature not in df.columns:
            continue

        port_change = (
            df[port_feature]
            .pct_change()
        )

        # Raw level is also tested separately.
        port_level = df[port_feature]

        for lag in LAGS:

            # Activity known LAG observations before the
            # freight observation.
            lagged_change = port_change.shift(lag)
            lagged_level = port_level.shift(lag)

            for target in TARGETS:

                freight_change = freight_returns[target]

                # Activity CHANGE → freight CHANGE
                p, s, n = safe_corr(
                    lagged_change,
                    freight_change,
                )

                results.append(
                    {
                        "feature": port_feature,
                        "target": target,
                        "lag": lag,
                        "signal": "change_to_change",
                        "pearson": p,
                        "spearman": s,
                        "n": n,
                    }
                )

                # Activity LEVEL → freight CHANGE
                p, s, n = safe_corr(
                    lagged_level,
                    freight_change,
                )

                results.append(
                    {
                        "feature": port_feature,
                        "target": target,
                        "lag": lag,
                        "signal": "level_to_change",
                        "pearson": p,
                        "spearman": s,
                        "n": n,
                    }
                )

    results = pd.DataFrame(results)

    # ------------------------------------------------------------
    # Rank strongest relationships.
    # ------------------------------------------------------------

    for signal_type in [
        "change_to_change",
        "level_to_change",
    ]:

        subset = results[
            results["signal"] == signal_type
        ].copy()

        subset["abs_spearman"] = (
            subset["spearman"].abs()
        )

        print("\n" + "=" * 100)
        print(
            f"TOP 30 — {signal_type.upper()}"
        )
        print("=" * 100)

        top = (
            subset
            .sort_values(
                "abs_spearman",
                ascending=False,
            )
            .head(30)
        )

        print(
            top[
                [
                    "feature",
                    "target",
                    "lag",
                    "pearson",
                    "spearman",
                    "n",
                ]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------
    # Aggregate by vessel class.
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("BEST PORTWATCH SIGNAL BY VESSEL CLASS")
    print("=" * 100)

    subset = results[
        results["signal"] == "change_to_change"
    ].copy()

    subset["abs_spearman"] = (
        subset["spearman"].abs()
    )

    for target in TARGETS:

        top = (
            subset[
                subset["target"] == target
            ]
            .sort_values(
                "abs_spearman",
                ascending=False,
            )
            .head(10)
        )

        print(
            f"\n### {target.upper()}"
        )

        print(
            top[
                [
                    "feature",
                    "lag",
                    "pearson",
                    "spearman",
                    "n",
                ]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------
    # Save complete results.
    # ------------------------------------------------------------

    output = Path(
        "02_DATA/processed/"
        "port_freight_signal_analysis.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
