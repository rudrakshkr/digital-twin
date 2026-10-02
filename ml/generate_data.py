from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_PATIENTS = 300
DAYS = 7
READING_INTERVAL_MINUTES = 15

rng = np.random.default_rng(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return float(np.clip(value, minimum, maximum))


def generate_patients() -> pd.DataFrame:
    patients = []

    medications = [
        "metformin",
        "metformin",
        "metformin",
        "metformin_glimepiride",
        "none",
    ]

    for index in range(NUM_PATIENTS):
        age = int(rng.integers(35, 76))
        sex = rng.choice(["M", "F"])

        bmi = round(clamp(rng.normal(28.5, 4.5), 20, 42), 1)

        family_history = bool(rng.random() < 0.55)

        hba1c = round(
            clamp(
                6.0
                + (age - 35) * 0.025
                + (bmi - 25) * 0.08
                + rng.normal(0, 0.45),
                5.5,
                10.5,
            ),
            1,
        )

        fasting_glucose = round(
            clamp(
                85
                + (hba1c - 5.5) * 20
                + rng.normal(0, 10),
                75,
                190,
            ),
            0,
        )

        systolic_bp = int(
            clamp(
                rng.normal(118, 12)
                + max(age - 45, 0) * 0.3
                + max(bmi - 25, 0) * 0.8,
                95,
                175,
            )
        )

        diastolic_bp = int(
            clamp(
                rng.normal(76, 8)
                + max(bmi - 25, 0) * 0.3,
                55,
                110,
            )
        )

        cholesterol = int(
            clamp(
                rng.normal(195, 30)
                + max(age - 45, 0) * 0.4,
                120,
                320,
            )
        )

        medication = rng.choice(medications)

        patients.append(
            {
                "patient_id": f"P{index + 1:04d}",
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "family_history_diabetes": family_history,
                "has_type2_diabetes": True,
                "hba1c": hba1c,
                "fasting_glucose": fasting_glucose,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "cholesterol": cholesterol,
                "medication": medication,
            }
        )

    return pd.DataFrame(patients)


def generate_wearable_data(patients: pd.DataFrame) -> pd.DataFrame:
    rows = []

    start_time = datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc)

    readings_per_day = (24 * 60) // READING_INTERVAL_MINUTES
    total_readings = DAYS * readings_per_day

    for patient in patients.itertuples(index=False):
        baseline = (
            patient.fasting_glucose
            + max(patient.hba1c - 6.0, 0) * 5
        )

        insulin_resistance = clamp(
            0.5
            + (patient.bmi - 25) * 0.025
            + (patient.hba1c - 6) * 0.08,
            0.2,
            1.4,
        )

        for reading_index in range(total_readings):
            timestamp = start_time + timedelta(
                minutes=reading_index * READING_INTERVAL_MINUTES
            )

            hour = timestamp.hour + timestamp.minute / 60

            # Approximate sleep period.
            is_sleeping = hour >= 23 or hour < 7

            # Daily activity pattern.
            if is_sleeping:
                activity = "sleep"
                steps = 0
                heart_rate = rng.normal(58, 5)
                hrv = rng.normal(58, 10)
            elif 7 <= hour < 9:
                activity = "walking"
                steps = int(max(0, rng.normal(60, 25)))
                heart_rate = rng.normal(82, 8)
                hrv = rng.normal(42, 8)
            elif 12 <= hour < 14:
                activity = "walking"
                steps = int(max(0, rng.normal(45, 20)))
                heart_rate = rng.normal(80, 8)
                hrv = rng.normal(40, 8)
            elif 17 <= hour < 20:
                activity = rng.choice(["walking", "sedentary"])
                steps = int(
                    max(
                        0,
                        rng.normal(
                            100 if activity == "walking" else 15,
                            35,
                        ),
                    )
                )
                heart_rate = rng.normal(
                    88 if activity == "walking" else 72,
                    9,
                )
                hrv = rng.normal(
                    36 if activity == "walking" else 46,
                    8,
                )
            else:
                activity = "sedentary"
                steps = int(max(0, rng.normal(18, 10)))
                heart_rate = rng.normal(70, 6)
                hrv = rng.normal(46, 9)

            # Simulated daily meal effects.
            meal_effect = 0.0

            if 7.5 <= hour < 9.5:
                meal_effect += 25 * insulin_resistance
            elif 12 <= hour < 14:
                meal_effect += 35 * insulin_resistance
            elif 18.5 <= hour < 20.5:
                meal_effect += 40 * insulin_resistance

            # Activity reduces short-term glucose rise.
            activity_effect = -min(steps / 80, 12)

            noise = rng.normal(0, 4)

            glucose = (
                baseline
                + meal_effect
                + activity_effect
                + noise
            )

            # Add occasional stronger post-meal spikes.
            if (
                (7.5 <= hour < 10)
                or (12 <= hour < 15)
                or (18.5 <= hour < 21)
            ) and rng.random() < 0.12:
                glucose += rng.uniform(20, 55) * insulin_resistance

            glucose = round(clamp(glucose, 65, 320), 1)

            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "patient_id": patient.patient_id,
                    "glucose": glucose,
                    "heart_rate": round(clamp(heart_rate, 45, 150), 1),
                    "hrv": round(clamp(hrv, 10, 100), 1),
                    "steps": steps,
                    "sleep_stage": (
                        rng.choice(
                            ["light", "deep", "rem"],
                            p=[0.5, 0.25, 0.25],
                        )
                        if is_sleeping
                        else "awake"
                    ),
                    "activity": activity,
                }
            )

    wearable = pd.DataFrame(rows)
    wearable["timestamp"] = pd.to_datetime(wearable["timestamp"], utc=True)

    return wearable


def add_prediction_target(wearable: pd.DataFrame) -> pd.DataFrame:
    wearable = wearable.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    # 2-hour future glucose maximum.
    future_columns = []

    steps_ahead = 120 // READING_INTERVAL_MINUTES

    for offset in range(1, steps_ahead + 1):
        column = wearable.groupby("patient_id")["glucose"].shift(-offset)
        future_columns.append(column)

    future_max = pd.concat(future_columns, axis=1).max(axis=1)

    wearable["future_2h_max_glucose"] = future_max

    # Prototype prediction target:
    # significant future glucose elevation above 180 mg/dL.
    wearable["glucose_spike_next_2h"] = (
        wearable["future_2h_max_glucose"] > 180
    ).astype(int)

    wearable = wearable.dropna(
        subset=["future_2h_max_glucose"]
    )

    return wearable


def main() -> None:
    print("Generating synthetic EHR data...")

    patients = generate_patients()

    print("Generating synthetic wearable time-series data...")

    wearable = generate_wearable_data(patients)

    print("Creating 2-hour prediction labels...")

    wearable = add_prediction_target(wearable)

    patients_path = OUTPUT_DIR / "patients.csv"
    wearable_path = OUTPUT_DIR / "wearable_data.csv"

    patients.to_csv(patients_path, index=False)
    wearable.to_csv(wearable_path, index=False)

    print()
    print("Generation complete.")
    print(f"Patients: {len(patients):,}")
    print(f"Wearable rows: {len(wearable):,}")
    print()
    print(f"Saved: {patients_path}")
    print(f"Saved: {wearable_path}")
    print()
    print("Spike distribution:")
    print(
        wearable["glucose_spike_next_2h"]
        .value_counts(normalize=True)
        .sort_index()
    )


if __name__ == "__main__":
    main()
