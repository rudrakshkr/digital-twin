from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
MODEL_DIR = ROOT / "ml" / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def add_features(wearable: pd.DataFrame) -> pd.DataFrame:
    df = wearable.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("patient_id", group_keys=False)

    # Historical glucose features.
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

    # Recent activity.
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

    # Cyclical time features.
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Encode categorical variables.
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


def main() -> None:
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    wearable = pd.read_csv(DATA_DIR / "wearable_data.csv")

    print("Loaded:")
    print(f"  Patients: {len(patients):,}")
    print(f"  Wearable rows: {len(wearable):,}")

    df = wearable.merge(
        patients,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    df = add_features(df)

    feature_columns = [
        # Current sensor state
        "glucose",
        "heart_rate",
        "hrv",
        "steps",

        # Recent glucose trajectory
        "glucose_lag_15m",
        "glucose_lag_30m",
        "glucose_lag_60m",
        "glucose_lag_120m",
        "glucose_change_15m",
        "glucose_change_60m",
        "glucose_mean_60m",
        "glucose_std_60m",
        "glucose_mean_120m",
        "glucose_std_120m",

        # Recent activity/autonomic state
        "steps_60m",
        "steps_120m",
        "heart_rate_mean_60m",
        "hrv_mean_60m",

        # Time context
        "hour_sin",
        "hour_cos",
        "day_of_week",

        # EHR
        "age",
        "bmi",
        "hba1c",
        "fasting_glucose",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "family_history_diabetes",
    ]

    # Convert boolean to numeric.
    df["family_history_diabetes"] = (
        df["family_history_diabetes"].astype(int)
    )

    # One-hot encode medication.
    medication_dummies = pd.get_dummies(
        df["medication"],
        prefix="medication",
        dtype=int,
    )

    df = pd.concat([df, medication_dummies], axis=1)

    medication_columns = list(medication_dummies.columns)

    feature_columns.extend(medication_columns)

    target = "glucose_spike_next_2h"

    # Never use future_2h_max_glucose as a feature.
    forbidden_columns = {
        "future_2h_max_glucose",
        target,
    }

    feature_columns = [
        column
        for column in feature_columns
        if column not in forbidden_columns
    ]

    model_df = df[
        ["patient_id", target, *feature_columns]
    ].dropna()

    print(f"Rows after feature engineering: {len(model_df):,}")

    # ------------------------------------------------------------
    # Patient-level split.
    # A patient's records must not appear in both train and test.
    # ------------------------------------------------------------

    patient_labels = (
        model_df.groupby("patient_id")[target]
        .max()
        .reset_index()
    )

    train_patients, test_patients = train_test_split(
        patient_labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=patient_labels[target],
    )

    train_ids = set(train_patients["patient_id"])
    test_ids = set(test_patients["patient_id"])

    train_df = model_df[
        model_df["patient_id"].isin(train_ids)
    ]

    test_df = model_df[
        model_df["patient_id"].isin(test_ids)
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    X_test = test_df[feature_columns]
    y_test = test_df[target]

    print()
    print(f"Train patients: {len(train_ids)}")
    print(f"Test patients:  {len(test_ids)}")
    print(f"Train rows:     {len(train_df):,}")
    print(f"Test rows:      {len(test_df):,}")

    print()
    print("Training class distribution:")
    print(y_train.value_counts(normalize=True))

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print()
    print("Training Random Forest...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print()
    print("===== MODEL RESULTS =====")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Feature importance.
    importance = (
        pd.Series(
            model.feature_importances_,
            index=feature_columns,
        )
        .sort_values(ascending=False)
        .head(15)
    )

    print()
    print("Top features:")
    print(importance)

    # ------------------------------------------------------------
    # Save model + metadata
    # ------------------------------------------------------------

    artifact = {
        "model": model,
        "features": feature_columns,
        "threshold": 0.5,
    }

    model_path = MODEL_DIR / "glucose_spike_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(artifact, model_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "train_patients": len(train_ids),
        "test_patients": len(test_ids),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": 0.5,
        "positive_rate_test": float(y_test.mean()),
        "features": feature_columns,
        "top_features": [
            {
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in importance.items()
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print()
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()