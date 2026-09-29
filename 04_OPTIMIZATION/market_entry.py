from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FREIGHT_PATH = (
    ROOT
    / "02_DATA/raw/freight/kobc/"
    / "kobc_drybulk_daily.csv"
)

sys.path.insert(
    0,
    str(ROOT / "04_OPTIMIZATION"),
)

from recommendation_engine import (
    Scenario,
    recommend,
)


# =====================================================================
# SCENARIO
# =====================================================================

@dataclass
class MarketEntryScenario:

    origin: str
    destination: str
    cargo_type: str
    cargo_mt: float
    delivery_days: float


# =====================================================================
# FREIGHT DATA
# =====================================================================

VESSEL_FREIGHT_COLUMN = {
    "Handysize": "handy",
    "Supramax": "supramax",
    "Panamax": "panamax",
    "Capesize": "cape",
}


def load_freight_history() -> pd.DataFrame:

    if not FREIGHT_PATH.exists():
        raise FileNotFoundError(
            f"Freight dataset not found: {FREIGHT_PATH}"
        )

    df = pd.read_csv(
        FREIGHT_PATH,
        parse_dates=["date"],
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


def freight_market_state(
    df: pd.DataFrame,
    vessel_class: str,
) -> dict:

    if vessel_class not in VESSEL_FREIGHT_COLUMN:

        return {
            "status": "unavailable",
            "vessel_class": vessel_class,
            "reason": "Unknown vessel class.",
        }

    column = VESSEL_FREIGHT_COLUMN[
        vessel_class
    ]

    if column not in df.columns:

        return {
            "status": "unavailable",
            "vessel_class": vessel_class,
            "reason": (
                f"Freight column '{column}' "
                "not found."
            ),
        }

    series = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if series.empty:

        return {
            "status": "unavailable",
            "vessel_class": vessel_class,
        }

    current = float(
        series.iloc[-1]
    )

    recent_30 = series.tail(30)
    recent_90 = series.tail(90)
    recent_365 = series.tail(365)

    volatility_30 = float(
        recent_30.pct_change()
        .dropna()
        .std()
    )

    volatility_90 = float(
        recent_90.pct_change()
        .dropna()
        .std()
    )

    mean_90 = float(
        recent_90.mean()
    )

    std_90 = float(
        recent_90.std()
    )

    if std_90 > 0:

        z_90 = (
            current - mean_90
        ) / std_90

    else:

        z_90 = 0.0

    percentile_365 = float(
        (
            recent_365 <= current
        ).mean()
        * 100.0
    )

    return {
        "status": "available",
        "vessel_class": vessel_class,
        "index_column": column,
        "observation_date": (
            df["date"].iloc[-1]
            .strftime("%Y-%m-%d")
        ),
        "current_freight_usd_day": current,
        "mean_90d_usd_day": mean_90,
        "std_90d_usd_day": std_90,
        "z_score_90d": z_90,
        "percentile_365d": percentile_365,
        "volatility_30d": volatility_30,
        "volatility_90d": volatility_90,
        "history_start": (
            df["date"].min()
            .strftime("%Y-%m-%d")
        ),
        "history_end": (
            df["date"].max()
            .strftime("%Y-%m-%d")
        ),
    }


# =====================================================================
# PERSISTENCE
# =====================================================================

def persistence_forecast(
    current_value: float,
    horizon_days: int,
) -> dict:

    return {
        "horizon_days": horizon_days,
        "forecast_method": "persistence",
        "forecast_freight_usd_day": (
            float(current_value)
        ),
        "forecast_change_pct": 0.0,
    }


# =====================================================================
# MARKET INTERPRETATION
# =====================================================================

def interpret_market(
    market_state: dict,
) -> dict:

    if market_state.get(
        "status"
    ) != "available":

        return {
            "signal": "UNAVAILABLE",
            "reason": (
                "Freight history unavailable."
            ),
        }

    percentile = float(
        market_state[
            "percentile_365d"
        ]
    )

    volatility = float(
        market_state[
            "volatility_30d"
        ]
    )

    if percentile <= 25:

        level = "FAVORABLE"

    elif percentile >= 75:

        level = "EXPENSIVE"

    else:

        level = "NEUTRAL"

    if volatility >= 0.08:

        volatility_band = "HIGH"

    elif volatility >= 0.04:

        volatility_band = "MODERATE"

    else:

        volatility_band = "LOW"

    return {
        "signal": level,
        "volatility_band": volatility_band,
        "interpretation_basis": (
            "Historical percentile + realized "
            "freight volatility."
        ),
        "not_a_directional_forecast": True,
        "z_score_90d": float(
            market_state["z_score_90d"]
        ),
    }


# =====================================================================
# SCHEDULE BANDS
# =====================================================================

def classify_slack(
    slack_days: float,
) -> str:

    if slack_days < 0:

        return "INFEASIBLE"

    if slack_days < 2:

        return "CRITICAL"

    if slack_days < 5:

        return "TIGHT"

    return "SAFE"


# =====================================================================
# ENTRY WINDOW
# =====================================================================

def evaluate_entry_window(
    offset_days: int,
    delivery_days: float,
    required_days: float,
) -> dict:

    remaining_delivery_window = (
        float(delivery_days)
        - float(offset_days)
    )

    slack_days = (
        remaining_delivery_window
        - float(required_days)
    )

    return {
        "entry_offset_days": int(
            offset_days
        ),
        "remaining_delivery_window_days": (
            remaining_delivery_window
        ),
        "required_transit_plus_port_days": (
            float(required_days)
        ),
        "delivery_slack_days": (
            slack_days
        ),
        "schedule_band": classify_slack(
            slack_days
        ),
        "delivery_feasible": (
            slack_days >= 0
        ),
    }


# =====================================================================
# LATEST FEASIBLE ENTRY
# =====================================================================

def find_latest_feasible_entry(
    windows: list[dict],
) -> dict | None:

    feasible = [
        window
        for window in windows
        if window.get(
            "delivery_feasible",
            False,
        )
    ]

    if not feasible:
        return None

    return max(
        feasible,
        key=lambda row: row[
            "entry_offset_days"
        ],
    )


# =====================================================================
# ENTRY DECISION CONTEXT
# =====================================================================

def build_entry_decision(
    market_signal: str,
    latest_feasible: dict | None,
    delivery_days: float,
) -> dict:

    if latest_feasible is None:

        return {
            "decision": "NO_FEASIBLE_ENTRY",
            "reason": (
                "No tested entry point can satisfy "
                "the delivery requirement."
            ),
        }

    latest_offset = int(
        latest_feasible[
            "entry_offset_days"
        ]
    )

    slack = float(
        latest_feasible[
            "delivery_slack_days"
        ]
    )

    # -----------------------------------------------------------------
    # This is deliberately NOT a trading recommendation.
    #
    # We combine:
    #   1. market context
    #   2. schedule flexibility
    #
    # A historically expensive market with abundant schedule slack
    # means there is room to monitor rather than claiming an immediate
    # charter decision.
    # -----------------------------------------------------------------

    if latest_offset <= 0:

        schedule_position = (
            "NO_DELAY_BUFFER"
        )

    elif latest_offset <= 2:

        schedule_position = (
            "VERY_LITTLE_DELAY_BUFFER"
        )

    elif latest_offset <= 7:

        schedule_position = (
            "LIMITED_DELAY_BUFFER"
        )

    else:

        schedule_position = (
            "MEANINGFUL_DELAY_BUFFER"
        )

    if market_signal == "FAVORABLE":

        if latest_offset >= 7:

            decision = "MONITOR_FOR_ENTRY"

        else:

            decision = "ENTRY_WINDOW_TIGHT"

    elif market_signal == "EXPENSIVE":

        if latest_offset >= 7:

            decision = "DELAY_OPTION_AVAILABLE"

        elif latest_offset > 0:

            decision = "LIMITED_DELAY_OPTION"

        else:

            decision = "SCHEDULE_CONSTRAINED"

    elif market_signal == "NEUTRAL":

        if latest_offset >= 7:

            decision = "MONITOR"

        else:

            decision = "SCHEDULE_CONSTRAINED"

    else:

        decision = "MARKET_SIGNAL_UNAVAILABLE"

    return {
        "decision": decision,
        "market_signal": market_signal,
        "latest_feasible_entry_offset_days": (
            latest_offset
        ),
        "latest_feasible_slack_days": slack,
        "schedule_position": schedule_position,
        "delivery_days": float(
            delivery_days
        ),
        "is_autonomous_trading_advice": False,
    }


# =====================================================================
# MAIN ENGINE
# =====================================================================

def run_market_entry(
    scenario: MarketEntryScenario,
) -> dict:

    freight = load_freight_history()

    base_scenario = Scenario(
        origin=scenario.origin,
        destination=scenario.destination,
        cargo_type=scenario.cargo_type,
        cargo_mt=scenario.cargo_mt,
        delivery_days=scenario.delivery_days,
    )

    recommendation = recommend(
        base_scenario
    )

    candidates = []

    for candidate in recommendation.get(
        "candidates",
        [],
    ):

        if not candidate.get(
            "feasible",
            False,
        ):
            continue

        vessel_class = candidate[
            "vessel_class"
        ]

        market = freight_market_state(
            freight,
            vessel_class,
        )

        if market.get(
            "status"
        ) != "available":

            continue

        interpretation = interpret_market(
            market
        )

        # -------------------------------------------------------------
        # Required duration.
        #
        # This is the current validated planning requirement.
        # We do NOT pretend to know future port conditions.
        # -------------------------------------------------------------

        required_days = candidate.get(
            "delivery_voyage_days"
        )

        if required_days is None:

            required_days = candidate.get(
                "known_voyage_days"
            )

        if required_days is None:
            continue

        required_days = float(
            required_days
        )

        # -------------------------------------------------------------
        # Evaluate EVERY possible integer entry offset.
        #
        # This gives the dashboard an actual entry window rather than
        # four arbitrary points.
        # -------------------------------------------------------------

        windows = []

        max_offset = int(
            max(
                0,
                scenario.delivery_days,
            )
        )

        for offset in range(
            0,
            max_offset + 1,
        ):

            window = evaluate_entry_window(
                offset_days=offset,
                delivery_days=(
                    scenario.delivery_days
                ),
                required_days=required_days,
            )

            forecast = persistence_forecast(
                market[
                    "current_freight_usd_day"
                ],
                offset,
            )

            window.update(
                {
                    "freight_forecast": forecast,
                    "market_signal": (
                        interpretation[
                            "signal"
                        ]
                    ),
                    "freight_percentile_365d": (
                        market[
                            "percentile_365d"
                        ]
                    ),
                    "freight_volatility_30d": (
                        market[
                            "volatility_30d"
                        ]
                    ),
                }
            )

            windows.append(
                window
            )

        latest_feasible = (
            find_latest_feasible_entry(
                windows
            )
        )

        decision = build_entry_decision(
            market_signal=interpretation[
                "signal"
            ],
            latest_feasible=latest_feasible,
            delivery_days=scenario.delivery_days,
        )

        candidates.append(
            {
                "vessel_class": vessel_class,
                "market_state": market,
                "market_interpretation": (
                    interpretation
                ),
                "required_transit_plus_port_days": (
                    required_days
                ),
                "entry_windows": windows,
                "latest_feasible_entry": (
                    latest_feasible
                ),
                "decision_context": decision,
            }
        )

    # -----------------------------------------------------------------
    # Rank candidate vessel classes.
    #
    # Current ranking remains deliberately conservative:
    #
    # 1. feasible
    # 2. validated route / operationally usable candidate
    # 3. larger entry flexibility
    #
    # We do NOT rank by a fabricated total delivered cost.
    # -----------------------------------------------------------------

    candidates.sort(
        key=lambda candidate: (
            candidate[
                "latest_feasible_entry"
            ][
                "entry_offset_days"
            ]
            if candidate[
                "latest_feasible_entry"
            ]
            is not None
            else -1
        ),
        reverse=True,
    )

    best = (
        candidates[0]
        if candidates
        else None
    )

    return {
        "status": (
            "available"
            if candidates
            else "unavailable"
        ),
        "scenario": {
            "origin": scenario.origin,
            "destination": scenario.destination,
            "cargo_type": scenario.cargo_type,
            "cargo_mt": scenario.cargo_mt,
            "delivery_days": scenario.delivery_days,
        },
        "freight_observation_date": (
            freight["date"].iloc[-1]
            .strftime("%Y-%m-%d")
        ),
        "candidates": candidates,
        "best_candidate": best,
        "limitations": [
            (
                "Future freight uses the validated "
                "persistence baseline because current "
                "directional ML models did not outperform "
                "persistence."
            ),
            (
                "Future port conditions are not forecast. "
                "The current operational state and berth-cycle "
                "prediction are used as the present-state "
                "planning reference."
            ),
            (
                "Historical freight percentile is market "
                "context, not a probability of future freight "
                "movement."
            ),
            (
                "No charter hire, demurrage, positioning or "
                "complete delivered-cost model is claimed."
            ),
            (
                "Entry windows are timing-feasibility and "
                "market-context scenarios, not autonomous "
                "trading advice."
            ),
        ],
    }


# =====================================================================
# CLI
# =====================================================================

def print_summary(
    result: dict,
) -> None:

    print()
    print("=" * 100)
    print("NEREUS — MARKET ENTRY ENGINE")
    print("=" * 100)

    scenario = result[
        "scenario"
    ]

    print()
    print("SCENARIO")
    print("-" * 100)

    print(
        f"{scenario['cargo_mt']:,.0f} MT "
        f"{scenario['cargo_type']} | "
        f"{scenario['origin']} → "
        f"{scenario['destination']} | "
        f"deadline {scenario['delivery_days']:g}d"
    )

    print(
        f"Freight observation: "
        f"{result['freight_observation_date']}"
    )

    if result["status"] != "available":

        print()
        print(
            "No feasible market-entry candidate."
        )

        return

    for candidate in result[
        "candidates"
    ]:

        market = candidate[
            "market_state"
        ]

        interpretation = candidate[
            "market_interpretation"
        ]

        latest = candidate[
            "latest_feasible_entry"
        ]

        decision = candidate[
            "decision_context"
        ]

        print()
        print(
            f"VESSEL CLASS: "
            f"{candidate['vessel_class']}"
        )

        print("-" * 100)

        print(
            f"Current freight      : "
            f"${market['current_freight_usd_day']:,.0f}/day"
        )

        print(
            f"365d percentile      : "
            f"{market['percentile_365d']:.1f}%"
        )

        print(
            f"30d volatility       : "
            f"{market['volatility_30d']:.4f}"
        )

        print(
            f"Market signal        : "
            f"{interpretation['signal']}"
        )

        print(
            f"Required duration    : "
            f"{candidate['required_transit_plus_port_days']:.3f}d"
        )

        if latest is not None:

            print(
                f"Latest feasible entry: "
                f"+{latest['entry_offset_days']}d"
            )

            print(
                f"Remaining window     : "
                f"{latest['remaining_delivery_window_days']:.3f}d"
            )

            print(
                f"Latest slack         : "
                f"{latest['delivery_slack_days']:.3f}d"
            )

            print(
                f"Schedule band        : "
                f"{latest['schedule_band']}"
            )

        else:

            print(
                "Latest feasible entry: NONE"
            )

        print(
            f"Decision context     : "
            f"{decision['decision']}"
        )

        print(
            f"Schedule position    : "
            f"{decision['schedule_position']}"
        )

        print()
        print("ENTRY WINDOW")
        print("-" * 100)

        # Print useful checkpoints rather than all 31 rows.
        for offset in [
            0,
            3,
            5,
            7,
            10,
            14,
            21,
            int(
                scenario["delivery_days"]
            ),
        ]:

            matching = [
                row
                for row in candidate[
                    "entry_windows"
                ]
                if row[
                    "entry_offset_days"
                ] == offset
            ]

            if not matching:
                continue

            row = matching[0]

            print(
                f"+{offset:2d}d | "
                f"remaining={row['remaining_delivery_window_days']:6.2f}d | "
                f"slack={row['delivery_slack_days']:7.2f}d | "
                f"{row['schedule_band']:11} | "
                f"freight=${row['freight_forecast']['forecast_freight_usd_day']:,.0f}/d"
            )

    print()
    print("=" * 100)
    print("BEST CURRENT CANDIDATE")
    print("=" * 100)

    best = result[
        "best_candidate"
    ]

    if best is None:

        print(
            "No candidate."
        )

    else:

        latest = best[
            "latest_feasible_entry"
        ]

        print(
            f"Vessel class         : "
            f"{best['vessel_class']}"
        )

        print(
            f"Market signal        : "
            f"{best['market_interpretation']['signal']}"
        )

        if latest:

            print(
                f"Latest feasible entry: "
                f"+{latest['entry_offset_days']} days"
            )

            print(
                f"Schedule slack       : "
                f"{latest['delivery_slack_days']:.3f} days"
            )

        print(
            f"Decision              : "
            f"{best['decision_context']['decision']}"
        )

    print()
    print("LIMITATIONS")
    print("-" * 100)

    for limitation in result[
        "limitations"
    ]:

        print(
            f"- {limitation}"
        )


if __name__ == "__main__":

    scenario = MarketEntryScenario(
        origin="Newcastle Australia",
        destination="Paradip India",
        cargo_type="coking coal",
        cargo_mt=60_000,
        delivery_days=30,
    )

    result = run_market_entry(
        scenario
    )

    print_summary(
        result
    )
