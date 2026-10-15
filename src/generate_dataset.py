"""Generate a reproducible synthetic student-performance dataset.

The generator encodes well-studied academic relationships (prior achievement,
study time, attendance, sleep, support) plus noise so the project is fully
offline and license-clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import CATEGORICAL_CHOICES, FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"


def generate_dataset(n_rows: int = 2800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hours_studied = np.clip(rng.normal(14, 6, n_rows), 1, 40)
    attendance_pct = np.clip(rng.normal(82, 12, n_rows), 40, 100)
    previous_score = np.clip(rng.normal(68, 14, n_rows), 25, 98)
    sleep_hours = np.clip(rng.normal(7.0, 1.3, n_rows), 3.5, 10.5)
    tutoring_sessions = rng.integers(0, 8, n_rows)
    physical_activity_days = rng.integers(0, 8, n_rows)
    assignments_completed_pct = np.clip(rng.normal(78, 16, n_rows), 20, 100)
    commute_minutes = np.clip(rng.normal(28, 16, n_rows), 5, 90)

    parental_involvement = rng.choice(
        CATEGORICAL_CHOICES["parental_involvement"], n_rows, p=[0.22, 0.48, 0.30]
    )
    internet_access = rng.choice(CATEGORICAL_CHOICES["internet_access"], n_rows, p=[0.86, 0.14])
    extracurricular = rng.choice(CATEGORICAL_CHOICES["extracurricular"], n_rows, p=[0.58, 0.42])
    motivation_level = rng.choice(
        CATEGORICAL_CHOICES["motivation_level"], n_rows, p=[0.20, 0.50, 0.30]
    )
    learning_style = rng.choice(CATEGORICAL_CHOICES["learning_style"], n_rows)

    parental_map = {"Low": -4.5, "Medium": 0.8, "High": 4.2}
    internet_map = {"Yes": 2.4, "No": -3.1}
    extra_map = {"Yes": 1.6, "No": -0.4}
    motivation_map = {"Low": -5.0, "Medium": 1.0, "High": 4.8}

    sleep_penalty = np.where(sleep_hours < 6.0, (6.0 - sleep_hours) * 2.8, 0)
    sleep_penalty += np.where(sleep_hours > 9.2, (sleep_hours - 9.2) * 1.4, 0)
    study_boost = hours_studied * 1.15 + 0.035 * hours_studied * attendance_pct / 10
    prior = 0.38 * previous_score
    attendance = 0.18 * attendance_pct
    tutoring = tutoring_sessions * 1.35
    assignments = 0.09 * assignments_completed_pct
    activity = physical_activity_days * 0.45
    commute_penalty = commute_minutes * 0.045

    parental = np.vectorize(parental_map.get)(parental_involvement)
    internet = np.vectorize(internet_map.get)(internet_access)
    extra = np.vectorize(extra_map.get)(extracurricular)
    motivation = np.vectorize(motivation_map.get)(motivation_level)

    noise = rng.normal(0, 4.2, n_rows)
    exam_score = (
        8
        + prior
        + study_boost
        + attendance
        + tutoring
        + assignments
        + activity
        + parental
        + internet
        + extra
        + motivation
        - sleep_penalty
        - commute_penalty
        + noise
    )
    exam_score = np.clip(exam_score, 18, 100)

    frame = pd.DataFrame(
        {
            "hours_studied": np.round(hours_studied, 1),
            "attendance_pct": np.round(attendance_pct, 1),
            "previous_score": np.round(previous_score, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "tutoring_sessions": tutoring_sessions,
            "physical_activity_days": physical_activity_days,
            "assignments_completed_pct": np.round(assignments_completed_pct, 1),
            "commute_minutes": np.round(commute_minutes, 0).astype(int),
            "parental_involvement": parental_involvement,
            "internet_access": internet_access,
            "extracurricular": extracurricular,
            "motivation_level": motivation_level,
            "learning_style": learning_style,
            TARGET_COLUMN: np.round(exam_score, 1),
        }
    )
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    frame.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(frame)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
