"""Load the trained pipeline once and produce scored, explained predictions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import FEATURE_COLUMNS, FEATURE_LABELS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


class ModelNotReadyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise ModelNotReadyError("Train the model first: python -m src.train")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise ModelNotReadyError("Train the model first: python -m src.train")
    return joblib.load(MODEL_PATH)


def score_band(score: float) -> tuple[str, str]:
    if score < 50:
        return "At risk", "This profile is likely to miss the passing band without a change in study habits."
    if score < 70:
        return "Needs support", "The predicted score is passing-adjacent. Targeted support should lift outcomes."
    if score < 85:
        return "On track", "Performance is solid. Small, consistent gains in study quality can push this higher."
    return "Excellent", "This profile is in the high-performing band. Keep the current rhythm and recover well."


def recommendations(row: dict, predicted: float) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if row["hours_studied"] < 10:
        items.append(
            {
                "title": "Increase focused study time",
                "detail": "Weekly study hours are below 10. Aim for 12–16 hours of planned, distraction-free work.",
                "priority": "high",
            }
        )
    if row["attendance_pct"] < 80:
        items.append(
            {
                "title": "Protect attendance",
                "detail": "Attendance under 80% is a strong drag on predicted scores. Treat missed classes as recoverable with same-week notes.",
                "priority": "high",
            }
        )
    if row["sleep_hours"] < 6.5:
        items.append(
            {
                "title": "Stabilize sleep",
                "detail": "Sleep below 6.5 hours correlates with weaker recall. Target 7–8 hours on school nights.",
                "priority": "medium",
            }
        )
    if row["assignments_completed_pct"] < 75:
        items.append(
            {
                "title": "Close assignment gaps",
                "detail": "Incomplete coursework leaves holes that show up on exams. Prioritize overdue work this week.",
                "priority": "high",
            }
        )
    if row["tutoring_sessions"] == 0 and predicted < 75:
        items.append(
            {
                "title": "Add a tutoring loop",
                "detail": "Even one or two weekly sessions can recover weak topics faster than solo revision.",
                "priority": "medium",
            }
        )
    if row["internet_access"] == "No":
        items.append(
            {
                "title": "Secure study connectivity",
                "detail": "No home internet access is a structural barrier. Use campus labs or a library block for digital work.",
                "priority": "high",
            }
        )
    if row["motivation_level"] == "Low":
        items.append(
            {
                "title": "Rebuild motivation with small wins",
                "detail": "Set a 25-minute daily goal and track completion. Momentum usually returns before motivation does.",
                "priority": "medium",
            }
        )
    if not items:
        items.append(
            {
                "title": "Keep the current system",
                "detail": "Inputs are already in a healthy range. Maintain consistency and review one weak topic each week.",
                "priority": "low",
            }
        )
    return items[:4]


def predict_student(payload: dict) -> dict:
    model = load_model()
    metrics = load_metrics()
    frame = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])
    predicted = float(model.predict(frame)[0])
    predicted = max(0.0, min(100.0, predicted))
    residual_std = float(metrics.get("residual_std", 5.0))
    low = max(0.0, round(predicted - 1.64 * residual_std, 1))
    high = min(100.0, round(predicted + 1.64 * residual_std, 1))
    band, copy = score_band(predicted)
    return {
        "predicted_score": round(predicted, 1),
        "band": band,
        "band_copy": copy,
        "confidence_interval": [low, high],
        "recommendations": recommendations(payload, predicted),
        "model_name": metrics.get("best_model", "unknown"),
        "feature_labels": FEATURE_LABELS,
    }
