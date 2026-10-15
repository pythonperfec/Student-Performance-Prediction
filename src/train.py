"""Train candidate regressors, keep the best sklearn Pipeline, and persist metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.generate_dataset import DATA_PATH, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=280,
            l2_regularization=0.1,
            random_state=42,
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(rmse, 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def feature_importances(pipeline: Pipeline, sample: pd.DataFrame) -> list[dict[str, float | str]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    transformed = preprocessor.transform(sample)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float))
    else:
        rng = np.random.default_rng(0)
        baseline = model.predict(transformed)
        values = np.zeros(transformed.shape[1], dtype=float)
        for idx in range(transformed.shape[1]):
            shuffled = transformed.copy()
            rng.shuffle(shuffled[:, idx])
            values[idx] = float(np.mean(np.abs(baseline - model.predict(shuffled))))

    values = values / values.sum() if values.sum() else values
    ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
    cleaned = []
    for name, score in ranked[:12]:
        label = str(name).replace("num__", "").replace("cat__", "")
        cleaned.append({"feature": label, "importance": round(float(score), 4)})
    return cleaned


def load_or_create_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset()
    frame.to_csv(DATA_PATH, index=False)
    return frame


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame = load_or_create_data()
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    leaderboard: list[dict] = []
    best_name = ""
    best_cv = -np.inf
    best_pipeline: Pipeline | None = None

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )
        cv_r2 = float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1).mean())
        pipeline.fit(X_train, y_train)
        holdout = evaluate(y_test.to_numpy(), pipeline.predict(X_test))
        row = {"name": name, "cv_r2": round(cv_r2, 4), **holdout}
        leaderboard.append(row)
        if cv_r2 > best_cv:
            best_cv = cv_r2
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None
    residuals = y_test.to_numpy() - best_pipeline.predict(X_test)
    residual_std = float(np.std(residuals))
    importances = feature_importances(best_pipeline, X_train.sample(min(400, len(X_train)), random_state=42))

    joblib.dump(best_pipeline, MODEL_PATH)
    payload = {
        "best_model": best_name,
        "n_rows": int(len(frame)),
        "n_features": len(FEATURE_COLUMNS),
        "test_size": 0.2,
        "cv_folds": 5,
        "leaderboard": sorted(leaderboard, key=lambda row: row["cv_r2"], reverse=True),
        "residual_std": round(residual_std, 3),
        "feature_importances": importances,
        "feature_means": {
            col: round(float(frame[col].mean()), 2) for col in NUMERIC_FEATURES
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["leaderboard"], indent=2))
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
