"""Pydantic contracts for JSON inference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    hours_studied: float = Field(ge=0, le=60)
    attendance_pct: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=3, le=12)
    tutoring_sessions: int = Field(ge=0, le=20)
    physical_activity_days: int = Field(ge=0, le=7)
    assignments_completed_pct: float = Field(ge=0, le=100)
    commute_minutes: int = Field(ge=0, le=180)
    parental_involvement: Literal["Low", "Medium", "High"]
    internet_access: Literal["Yes", "No"]
    extracurricular: Literal["Yes", "No"]
    motivation_level: Literal["Low", "Medium", "High"]
    learning_style: Literal["Visual", "Auditory", "Kinesthetic", "Reading"]


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: str


class PredictionResponse(BaseModel):
    predicted_score: float
    band: str
    band_copy: str
    confidence_interval: list[float]
    recommendations: list[Recommendation]
    model_name: str
