from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "03_ML/risk/paradip/"
    / "paradip_berth_duration_model.joblib"
)


def predict_paradip_berth_duration(
    cargo: str,
    berth: str,
    quantity_mt: float,
    dry_bulk_vessels_waiting: float,
) -> dict:

    if not MODEL_PATH.exists():
        return {
            "status": "unavailable",
            "reason": (
                f"Model artifact not found: "
                f"{MODEL_PATH}"
            ),
        }

    model = joblib.load(
        MODEL_PATH
    )

    X = pd.DataFrame(
        [
            {
                "cargo": cargo,
                "berth": berth,
                "quantity_mt": quantity_mt,
                "dry_bulk_vessels_waiting": (
                    dry_bulk_vessels_waiting
                ),
            }
        ]
    )

    prediction = float(
        model.predict(X)[0]
    )

    # The model predicts berth-cycle duration.
    # Never return a negative physical duration.
    prediction = max(
        0.0,
        prediction,
    )

    return {
        "status": "available",
        "predicted_berth_duration_days": prediction,
        "target": "berth_cycle_duration",
    }
