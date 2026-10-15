"""ScholarSight: student performance prediction served by FastAPI + Jinja."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.inference import ModelNotReadyError, load_metrics, predict_student
from app.schemas import PredictionResponse, StudentFeatures
from src.features import CATEGORICAL_CHOICES

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="ScholarSight",
    description="Predict student exam performance from academic and lifestyle signals.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _page_context(request: Request, **extra) -> dict:
    metrics = None
    error = None
    try:
        metrics = load_metrics()
    except ModelNotReadyError as exc:
        error = str(exc)
    context = {
        "request": request,
        "choices": CATEGORICAL_CHOICES,
        "metrics": metrics,
        "model_error": error,
    }
    context.update(extra)
    return context


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _page_context(request, active="predict"))


@app.get("/insights", response_class=HTMLResponse)
def insights(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "insights.html", _page_context(request, active="insights"))


@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    hours_studied: float = Form(...),
    attendance_pct: float = Form(...),
    previous_score: float = Form(...),
    sleep_hours: float = Form(...),
    tutoring_sessions: int = Form(...),
    physical_activity_days: int = Form(...),
    assignments_completed_pct: float = Form(...),
    commute_minutes: int = Form(...),
    parental_involvement: str = Form(...),
    internet_access: str = Form(...),
    extracurricular: str = Form(...),
    motivation_level: str = Form(...),
    learning_style: str = Form(...),
) -> HTMLResponse:
    payload = {
        "hours_studied": hours_studied,
        "attendance_pct": attendance_pct,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "tutoring_sessions": tutoring_sessions,
        "physical_activity_days": physical_activity_days,
        "assignments_completed_pct": assignments_completed_pct,
        "commute_minutes": commute_minutes,
        "parental_involvement": parental_involvement,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "motivation_level": motivation_level,
        "learning_style": learning_style,
    }
    try:
        StudentFeatures.model_validate(payload)
        result = predict_student(payload)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return templates.TemplateResponse(request, "partials/result.html", {"result": result})


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_api(features: StudentFeatures) -> PredictionResponse:
    try:
        result = predict_student(features.model_dump())
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.get("/api/v1/health")
def health() -> dict:
    try:
        metrics = load_metrics()
        ready = True
        model = metrics.get("best_model")
    except ModelNotReadyError:
        ready = False
        model = None
    return {"status": "ok" if ready else "model_missing", "model": model}


@app.get("/api/v1/metrics")
def metrics_api() -> dict:
    try:
        return load_metrics()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
