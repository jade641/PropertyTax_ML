from pathlib import Path
from typing import Any, Dict, Optional

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# locate model
model_files = list(MODELS_DIR.glob("*_propertytax_model.pkl"))
if not model_files:
    raise RuntimeError(f"No model file found in {MODELS_DIR}")
MODEL_PATH = model_files[0]
MODEL = joblib.load(MODEL_PATH)

# load feature info
FEATURE_INFO_PATH = MODELS_DIR / "propertytax_feature_info.json"
if not FEATURE_INFO_PATH.exists():
    raise RuntimeError(f"Missing feature info at {FEATURE_INFO_PATH}")

# Robustly load feature info whether it was saved as a JSON object, list, or
# pandas DataFrame-serializable JSON. Normalize into a dict `FEATURE_INFO_DICT`.
text = FEATURE_INFO_PATH.read_text(encoding='utf-8')
FEATURE_INFO_DICT = None
try:
    parsed = json.loads(text)
except Exception:
    # try pandas as a last resort
    try:
        parsed = pd.read_json(FEATURE_INFO_PATH)
    except Exception as e:
        raise RuntimeError(f"Could not parse feature info: {e}")

if isinstance(parsed, dict):
    FEATURE_INFO_DICT = parsed
elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
    FEATURE_INFO_DICT = parsed[0]
elif isinstance(parsed, pd.DataFrame):
    FEATURE_INFO_DICT = parsed.to_dict(orient='records')[0]
else:
    raise RuntimeError("Unrecognized feature info format; must be dict or list of dicts")

ALL_FEATURES = FEATURE_INFO_DICT.get("all_features", [])
NUMERIC_FEATURES = set(FEATURE_INFO_DICT.get("numeric_features", []))
CATEGORICAL_FEATURES = set(FEATURE_INFO_DICT.get("categorical_features", []))
TARGET_NAME = FEATURE_INFO_DICT.get("target", "is_late_payment")
MODEL_NAME = FEATURE_INFO_DICT.get("best_model_name", MODEL_PATH.stem)
TRAINING_DATA_PATH = ROOT / "datasets" / "PropertyTax_model_ready.csv"
CHART_SAMPLE_SIZE = 200


def _load_training_frame() -> pd.DataFrame:
    if TRAINING_DATA_PATH.exists():
        try:
            return pd.read_csv(TRAINING_DATA_PATH, usecols=lambda column: column in set(ALL_FEATURES))
        except Exception:
            pass

    return pd.DataFrame({feature: [np.nan] for feature in ALL_FEATURES})


def _build_chart_sample() -> pd.DataFrame:
    training_frame = _load_training_frame()
    rng = np.random.default_rng(42)
    sample = pd.DataFrame(index=range(CHART_SAMPLE_SIZE))

    for feature in ALL_FEATURES:
        source = training_frame[feature] if feature in training_frame.columns else pd.Series([np.nan])
        non_null = source.dropna()

        if feature in CATEGORICAL_FEATURES or not pd.api.types.is_numeric_dtype(source):
            if non_null.empty:
                sample[feature] = ["unknown"] * CHART_SAMPLE_SIZE
                continue

            distribution = non_null.astype(str).value_counts(normalize=True)
            sample[feature] = rng.choice(distribution.index.to_list(), size=CHART_SAMPLE_SIZE, p=distribution.to_numpy())
            continue

        numeric_values = pd.to_numeric(source, errors="coerce").dropna()
        baseline = float(numeric_values.median()) if not numeric_values.empty else 0.0
        spread = float(numeric_values.std(ddof=0)) if len(numeric_values) > 1 else 0.0
        noise_scale = max(abs(spread) * 0.05, abs(baseline) * 0.02, 0.1)
        values = baseline + rng.normal(0.0, noise_scale, size=CHART_SAMPLE_SIZE)

        if feature in {"assessment_year", "prior_assessments", "prior_late_payments", "prior_unpaid_payments", "due_month", "due_quarter"}:
            values = np.rint(values)

        if feature == "assessment_year":
            values = np.clip(values, 2000, 2100)
        elif feature == "due_month":
            values = np.clip(values, 1, 12)
        elif feature == "due_quarter":
            values = np.clip(values, 1, 4)
        elif feature in {"assessment_level", "tax_rate"}:
            values = np.clip(values, 0.0, None)
        elif feature.startswith("log_"):
            values = np.clip(values, -50.0, None)

        sample[feature] = values

    return sample[ALL_FEATURES]


CHART_SAMPLE = _build_chart_sample()
PIPELINE_PREPROCESS = MODEL.named_steps["preprocess"]
PIPELINE_MODEL = MODEL.named_steps["model"]

if hasattr(PIPELINE_PREPROCESS, "get_feature_names_out"):
    TRANSFORMED_FEATURE_NAMES = list(PIPELINE_PREPROCESS.get_feature_names_out())
else:
    TRANSFORMED_FEATURE_NAMES = list(ALL_FEATURES)

if hasattr(PIPELINE_MODEL, "coef_"):
    TRANSFORMED_COEFFICIENTS = np.asarray(PIPELINE_MODEL.coef_[0], dtype=float)
elif hasattr(PIPELINE_MODEL, "feature_importances_"):
    TRANSFORMED_COEFFICIENTS = np.asarray(PIPELINE_MODEL.feature_importances_, dtype=float)
else:
    TRANSFORMED_COEFFICIENTS = np.zeros(len(TRANSFORMED_FEATURE_NAMES))


def _normalize_feature_name(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def _build_feature_importance() -> list[dict[str, float | str]]:
    if len(TRANSFORMED_FEATURE_NAMES) != len(TRANSFORMED_COEFFICIENTS):
        return [{"name": _normalize_feature_name(f), "importance": 0.0} for f in ALL_FEATURES[:15]]

    ranked = sorted(
        (
            {
                "name": _normalize_feature_name(feature_name),
                "importance": float(coefficient),
                "absolute_importance": abs(float(coefficient)),
            }
            for feature_name, coefficient in zip(TRANSFORMED_FEATURE_NAMES, TRANSFORMED_COEFFICIENTS)
        ),
        key=lambda item: item["absolute_importance"],
        reverse=True,
    )

    return [{"name": item["name"], "importance": round(abs(float(item["importance"])), 4)} for item in ranked[:15]]


FEATURE_IMPORTANCE = _build_feature_importance()

app = FastAPI(title="PropertyTax ML Inference")


class PredictRequest(BaseModel):
    data: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "PropertyTax ML Inference",
        "model": str(MODEL_PATH.name),
        "endpoints": ["/health", "/predict", "/docs", "/chart/feature-importance", "/chart/risk-distribution", "/chart/probability-histogram"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


@app.post("/predict")
def predict(payload: PredictRequest, threshold: float = 0.5):
    # build dataframe
    row = payload.features if payload.features is not None else payload.data
    if row is None:
        raise HTTPException(status_code=400, detail="`data` or `features` must be provided")

    df = pd.DataFrame([row])
    # ensure all expected columns exist
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    # drop unexpected columns
    df = df[ALL_FEATURES]

    # predict probability
    try:
        proba = MODEL.predict_proba(df)[0, 1]
    except Exception:
        # fallback to decision_function if available
        try:
            score = MODEL.decision_function(df)[0]
            proba = 1 / (1 + np.exp(-score))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model doesn't support probability output: {e}")

    label = int(proba >= threshold)
    
    # risk level mapping
    if proba >= 0.75:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "model": MODEL_NAME,
        "probability": float(proba),
        "prediction": label,
        "predictedLabel": label,  # for C# backend compatibility
        "threshold": float(threshold),
        "riskLevel": risk,        # for C# backend compatibility
        "confidence": round(float(proba) * 100.0, 2),  # for C# backend compatibility
        "topFeatures": [{"feature": name, "importance": 0.0} for name in ALL_FEATURES[:5]], # fallback
    }


@app.get("/chart/feature-importance")
def chart_feature_importance():
    return {"features": FEATURE_IMPORTANCE}


def _find_uploaded_dataset(filename: str) -> Optional[Path]:
    """Search for an uploaded dataset file under parent folders named 'uploads/ml-datasets' or 'uploads'."""
    if not filename:
        return None

    current = Path(__file__).resolve().parent
    for _ in range(0, 8):
        candidate1 = current / "uploads" / "ml-datasets" / filename
        candidate2 = current / "uploads" / filename
        if candidate1.exists():
            return candidate1
        if candidate2.exists():
            return candidate2
        if current.parent == current:
            break
        current = current.parent

    candidate1 = ROOT / "uploads" / "ml-datasets" / filename
    candidate2 = ROOT / "uploads" / filename
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2

    return None


@app.get("/chart/risk-distribution")
def chart_risk_distribution(dataset: Optional[str] = None):
    # If a dataset is provided, try to load it and compute distribution from real data
    if dataset:
        try:
            path = _find_uploaded_dataset(dataset)
            if path and path.exists():
                df = pd.read_csv(path)
                # ensure expected columns exist and in right order
                for c in ALL_FEATURES:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[ALL_FEATURES]
                if df.shape[0] > 0:
                    try:
                        probabilities = MODEL.predict_proba(df)[:, 1]
                        low = int(np.sum(probabilities < 0.3))
                        medium = int(np.sum((probabilities >= 0.3) & (probabilities <= 0.6)))
                        high = int(np.sum(probabilities > 0.6))
                        return {"low": low, "medium": medium, "high": high}
                    except Exception:
                        # fall through to sample-based response
                        pass
        except Exception:
            # ignore and fall back to CHART_SAMPLE
            pass

    # fallback to deterministic sample
    probabilities = MODEL.predict_proba(CHART_SAMPLE)[:, 1]
    low = int(np.sum(probabilities < 0.3))
    medium = int(np.sum((probabilities >= 0.3) & (probabilities <= 0.6)))
    high = int(np.sum(probabilities > 0.6))

    return {"low": low, "medium": medium, "high": high}


@app.get("/chart/probability-histogram")
def chart_probability_histogram(dataset: Optional[str] = None):
    # If a dataset is provided, try to load and compute histogram
    if dataset:
        try:
            path = _find_uploaded_dataset(dataset)
            if path and path.exists():
                df = pd.read_csv(path)
                for c in ALL_FEATURES:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[ALL_FEATURES]
                if df.shape[0] > 0:
                    try:
                        probabilities = MODEL.predict_proba(df)[:, 1] * 100.0
                        bins = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
                        counts = [
                            int(np.sum(probabilities <= 20)),
                            int(np.sum((probabilities > 20) & (probabilities <= 40))),
                            int(np.sum((probabilities > 40) & (probabilities <= 60))),
                            int(np.sum((probabilities > 60) & (probabilities <= 80))),
                            int(np.sum(probabilities > 80)),
                        ]
                        return {"bins": bins, "counts": counts}
                    except Exception:
                        pass
        except Exception:
            pass

    # fallback to deterministic sample
    probabilities = MODEL.predict_proba(CHART_SAMPLE)[:, 1] * 100.0
    bins = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    counts = [
        int(np.sum(probabilities <= 20)),
        int(np.sum((probabilities > 20) & (probabilities <= 40))),
        int(np.sum((probabilities > 40) & (probabilities <= 60))),
        int(np.sum((probabilities > 60) & (probabilities <= 80))),
        int(np.sum(probabilities > 80)),
    ]

    return {"bins": bins, "counts": counts}
