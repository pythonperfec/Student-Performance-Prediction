from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "model_missing"}


def test_home_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "ScholarSight" in response.text


SAMPLE = {
    "hours_studied": 16,
    "attendance_pct": 90,
    "previous_score": 78,
    "sleep_hours": 7.5,
    "tutoring_sessions": 2,
    "physical_activity_days": 4,
    "assignments_completed_pct": 88,
    "commute_minutes": 20,
    "parental_involvement": "High",
    "internet_access": "Yes",
    "extracurricular": "Yes",
    "motivation_level": "High",
    "learning_style": "Visual",
}


def test_json_predict_when_model_ready():
    health = client.get("/api/v1/health").json()
    if health["status"] != "ok":
        return
    response = client.post("/api/v1/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_score"] <= 100
    assert body["band"]
