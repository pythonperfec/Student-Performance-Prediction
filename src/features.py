"""Shared feature contracts used by training and inference."""

from __future__ import annotations

NUMERIC_FEATURES = [
    "hours_studied",
    "attendance_pct",
    "previous_score",
    "sleep_hours",
    "tutoring_sessions",
    "physical_activity_days",
    "assignments_completed_pct",
    "commute_minutes",
]

CATEGORICAL_FEATURES = [
    "parental_involvement",
    "internet_access",
    "extracurricular",
    "motivation_level",
    "learning_style",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "exam_score"

FEATURE_LABELS = {
    "hours_studied": "Weekly study hours",
    "attendance_pct": "Attendance",
    "previous_score": "Previous exam score",
    "sleep_hours": "Sleep hours",
    "tutoring_sessions": "Tutoring sessions",
    "physical_activity_days": "Physical activity",
    "assignments_completed_pct": "Assignments completed",
    "commute_minutes": "Commute time",
    "parental_involvement": "Parental involvement",
    "internet_access": "Internet access",
    "extracurricular": "Extracurriculars",
    "motivation_level": "Motivation",
    "learning_style": "Learning style",
}

CATEGORICAL_CHOICES = {
    "parental_involvement": ["Low", "Medium", "High"],
    "internet_access": ["Yes", "No"],
    "extracurricular": ["Yes", "No"],
    "motivation_level": ["Low", "Medium", "High"],
    "learning_style": ["Visual", "Auditory", "Kinesthetic", "Reading"],
}
