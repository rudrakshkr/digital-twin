from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "synthetic"
MODEL_PATH = ROOT / "ml" / "artifacts" / "glucose_spike_model.joblib"


def add_features(wearable: pd.DataFrame) -> pd.DataFrame:
    df = wearable.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("patient_id", group_keys=False)

    # Glucose history
    for periods, name in [
        (1, "15m"),
        (2, "30m"),
        (4, "60m"),
        (8, "120m"),
    ]:
        df[f"glucose_lag_{name}"] = grouped["glucose"].shift(periods)

    df["glucose_change_15m"] = (
        df["glucose"] - df["glucose_lag_15m"]
    )

    df["glucose_change_60m"] = (
        df["glucose"] - df["glucose_lag_60m"]
    )

    df["glucose_mean_60m"] = (
        grouped["glucose"]
        .rolling(4, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["glucose_std_60m"] = (
        grouped["glucose"]
        .rolling(4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["glucose_mean_120m"] = (
        grouped["glucose"]
        .rolling(8, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["glucose_std_120m"] = (
        grouped["glucose"]
        .rolling(8, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Activity / autonomic features
    df["steps_60m"] = (
        grouped["steps"]
        .rolling(4, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["steps_120m"] = (
        grouped["steps"]
        .rolling(8, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["heart_rate_mean_60m"] = (
        grouped["heart_rate"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["hrv_mean_60m"] = (
        grouped["hrv"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Time features
    hour = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
    )

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Categorical encoding
    df["activity"] = df["activity"].map(
        {
            "sleep": 0,
            "sedentary": 1,
            "walking": 2,
        }
    )

    df["sleep_stage"] = df["sleep_stage"].map(
        {
            "awake": 0,
            "light": 1,
            "deep": 2,
            "rem": 3,
        }
    )

    return df


def prepare_dataset() -> pd.DataFrame:
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    wearable = pd.read_csv(DATA_DIR / "wearable_data.csv")

    df = wearable.merge(
        patients,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    df = add_features(df)

    df["family_history_diabetes"] = (
        df["family_history_diabetes"].astype(int)
    )

    medication_dummies = pd.get_dummies(
        df["medication"],
        prefix="medication",
        dtype=int,
    )

    df = pd.concat(
        [df, medication_dummies],
        axis=1,
    )

    return df


def get_prediction(
    patient_id: str,
    at_timestamp: str | None = None,
) -> dict:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = artifact["features"]
    threshold = artifact["threshold"]

    df = prepare_dataset()

    patient_df = df[
        df["patient_id"] == patient_id
    ].sort_values("timestamp")

    if at_timestamp:
        requested_time = pd.to_datetime(
            at_timestamp,
            utc=True,
        )

        patient_df = patient_df[
            patient_df["timestamp"] <= requested_time
        ]

    if patient_df.empty:
        raise ValueError(
            f"Patient '{patient_id}' was not found."
        )

    # Use the latest available state.
    current = patient_df.iloc[-1]

    # Build feature vector in exactly the same order
    # used during training.
    X = pd.DataFrame(
        [[
            current.get(feature)
            for feature in feature_columns
        ]],
        columns=feature_columns,
    )

    if X.isna().any().any():
        missing = X.columns[
            X.isna().iloc[0]
        ].tolist()

        raise ValueError(
            f"Missing features for prediction: {missing}"
        )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    if probability >= 0.70:
        risk_level = "high"
    elif probability >= 0.40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    result = {
        "patient_id": patient_id,
        "timestamp": current["timestamp"].isoformat(),
        "prediction_horizon_minutes": 120,
        "glucose_spike_probability": round(
            probability,
            4,
        ),
        "glucose_spike_probability_percent": round(
            probability * 100,
            1,
        ),
        "predicted_spike": prediction,
        "risk_level": risk_level,
        "current_state": {
            "glucose": float(current["glucose"]),
            "heart_rate": float(current["heart_rate"]),
            "hrv": float(current["hrv"]),
            "steps": int(current["steps"]),
            "activity": {
                0: "sleep",
                1: "sedentary",
                2: "walking",
            }[int(current["activity"])],
            "sleep_stage": {
                0: "awake",
                1: "light",
                2: "deep",
                3: "rem",
            }[int(current["sleep_stage"])],
            "hba1c": float(current["hba1c"]),
            "bmi": float(current["bmi"]),
        },
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient-id",
        required=True,
        help="Patient ID, e.g. P0001",
    )

    parser.add_argument(
        "--timestamp",
        required=False,
        help="Optional ISO timestamp for historical state.",
    )

    args = parser.parse_args()

    try:
        result = get_prediction(
            args.patient_id,
            args.timestamp,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc)
                },
                indent=2,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()