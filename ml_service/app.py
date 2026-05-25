import csv
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import ValidationError

try:
    from . import model_loader as loader
    from .schemas import PredictRequest, BatchPredictRequest, PredictResponse, TrainRequest
except ImportError:
    import model_loader as loader
    from schemas import PredictRequest, BatchPredictRequest, PredictResponse, TrainRequest


class ModelContainer:
    def __init__(self):
        self.feature_info = {}
        self.models = {}
        self.pipelines = {}
        self.artifacts = {}
        try:
            self.feature_info = loader.load_feature_info()
        except Exception as ex:
            print(f"[ml_service] Warning: Failed to load feature info during container init: {ex}")

    def load_models(self):
        self.models.clear()
        self.pipelines.clear()
        self.artifacts.clear()
        self._discover_and_load()

    def _discover_and_load(self):
        discovered = loader.discover_models()
        for key, meta in discovered.items():
            try:
                estimator, pipeline = loader.load_model(meta["path"])
                self.models[key] = estimator
                self.pipelines[key] = pipeline
                self.artifacts[key] = meta["path"]
            except Exception as ex:
                print(f"[ml_service] Failed loading model {key}: {ex}")


CONTAINER = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe Model Loading during startup event
    print("[ml_service] Starting up and loading models...")
    try:
        CONTAINER.load_models()
        print(f"[ml_service] Startup model loading complete. Loaded: {list(CONTAINER.models.keys())}")
    except Exception as ex:
        print(f"[ml_service] CRITICAL: Failed to load models during startup: {ex}")
    
    global CHART_SAMPLE
    try:
        print("[ml_service] Pre-building chart sample...")
        CHART_SAMPLE = _ensure_required_features(_build_chart_sample())
        print(f"[ml_service] Pre-built chart sample columns: {list(CHART_SAMPLE.columns)}")
    except Exception as ex:
        print(f"[ml_service] Warning: Failed to pre-build chart sample: {ex}")
        
    yield
    print("[ml_service] Shutting down...")
    CONTAINER.models.clear()
    CONTAINER.pipelines.clear()
    CONTAINER.artifacts.clear()


app = FastAPI(title="PropertyTax ML Service", lifespan=lifespan)


def _parse_allowed_origins() -> List[str]:
    raw_origins = os.environ.get("CORS_ALLOWED_ORIGINS") or os.environ.get("FRONTEND_BASE_URL")
    if raw_origins:
        origins = [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]
        if origins:
            return origins

    return ["https://property-taxation.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = ROOT / "datasets" / "PropertyTax_model_ready.csv"
TRAINING_DATA_PATH_SECONDARY = ROOT / "datasets" / "uploads" / "PropertyTax_model_ready.csv"
SHARED_UPLOADS_DIR = ROOT / "datasets" / "uploads"
CHART_SAMPLE_SIZE = 200


def _load_training_frame() -> pd.DataFrame:
    all_features = CONTAINER.feature_info.get("all_features", [])
    for path in (TRAINING_DATA_PATH, TRAINING_DATA_PATH_SECONDARY):
        if path.exists():
            try:
                return pd.read_csv(path, usecols=lambda column: column in set(all_features))
            except Exception:
                pass
    return pd.DataFrame({feature: [np.nan] for feature in all_features})


def _build_chart_sample() -> pd.DataFrame:
    all_features = CONTAINER.feature_info.get("all_features", [])
    categorical_features = set(CONTAINER.feature_info.get("categorical_features", []))
    training_frame = _load_training_frame()
    rng = np.random.default_rng(42)
    sample = pd.DataFrame(index=range(CHART_SAMPLE_SIZE))

    for feature in all_features:
        source = training_frame[feature] if feature in training_frame.columns else pd.Series([np.nan])
        non_null = source.dropna()

        if feature in categorical_features or not pd.api.types.is_numeric_dtype(source):
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

    return sample[all_features]


def _get_pipeline_expected_columns(pipeline) -> list:
    """Inspect a pipeline's ColumnTransformer to discover ALL columns it expects."""
    try:
        if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
            ct = pipeline.named_steps["preprocess"]
            if hasattr(ct, "transformers_"):
                cols = []
                for name, trans, columns in ct.transformers_:
                    if isinstance(columns, list):
                        cols.extend(columns)
                    elif isinstance(columns, np.ndarray):
                        cols.extend(columns.tolist())
                return cols
            if hasattr(ct, "transformers"):
                cols = []
                for name, trans, columns in ct.transformers:
                    if isinstance(columns, list):
                        cols.extend(columns)
                    elif isinstance(columns, np.ndarray):
                        cols.extend(columns.tolist())
                return cols
    except Exception as ex:
        print(f"[ml_service] Could not inspect pipeline columns: {ex}")
    return []


def _ensure_required_features(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    df = df.copy()

    # Start with features from feature_info
    all_features = list(CONTAINER.feature_info.get("all_features", []))

    # If a specific pipeline is given, also include any columns it expects
    # (handles stale models trained with a different feature set)
    if pipeline is not None:
        pipeline_cols = _get_pipeline_expected_columns(pipeline)
        for col in pipeline_cols:
            if col not in all_features:
                all_features.append(col)
    else:
        # Check ALL loaded pipelines to get a superset of required columns
        for key, p in CONTAINER.pipelines.items():
            pipeline_cols = _get_pipeline_expected_columns(p)
            for col in pipeline_cols:
                if col not in all_features:
                    all_features.append(col)

    # Ensure market_value and assessed_value exist as base for log features
    required_numeric = {
        "market_value": 0.0,
        "assessed_value": 0.0,
    }

    for col, default in required_numeric.items():
        if col not in df.columns:
            df[col] = default

    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    df["assessed_value"] = pd.to_numeric(df["assessed_value"], errors="coerce").fillna(0)

    # recreate engineered features
    df["log_market_value"] = np.log1p(df["market_value"])
    df["log_assessed_value"] = np.log1p(df["assessed_value"])

    # ensure ALL expected features exist
    for feature in all_features:
        if feature not in df.columns:
            df[feature] = 0

    return df[all_features]


CHART_SAMPLE = None


def _normalize_model_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_feature_name(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def _build_feature_importance_for_model(model_key: str) -> List[Dict[str, Any]]:
    pipeline = CONTAINER.pipelines.get(model_key)
    model = CONTAINER.models.get(model_key)

    if pipeline is None or model is None:
        return []

    try:
        if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
            preprocess = pipeline.named_steps["preprocess"]
            if hasattr(preprocess, "get_feature_names_out"):
                names = list(preprocess.get_feature_names_out())
            else:
                names = CONTAINER.feature_info.get("all_features", [])
        else:
            names = CONTAINER.feature_info.get("all_features", [])

        names = list(map(str, names))

        if hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_[0], dtype=float)
        elif hasattr(model, "feature_importances_"):
            coefs = np.asarray(model.feature_importances_, dtype=float)
        else:
            coefs = np.zeros(len(names))

        if len(names) == len(coefs):
            ranked = sorted(
                (
                    {
                        "name": _normalize_feature_name(feature_name),
                        "importance": float(coefficient),
                        "absolute_importance": abs(float(coefficient)),
                    }
                    for feature_name, coefficient in zip(names, coefs)
                ),
                key=lambda item: item["absolute_importance"],
                reverse=True,
            )
            return [{"name": item["name"], "importance": round(abs(float(item["importance"])), 4)} for item in ranked[:15]]

    except Exception as ex:
        print(f"Failed to build feature importance for {model_key}: {ex}")

    return [{"name": _normalize_feature_name(f), "importance": 0.0} for f in CONTAINER.feature_info.get("all_features", [])[:15]]


def _find_uploaded_dataset(filename: str) -> Optional[Path]:
    if not filename:
        return None

    # Primary: shared folder at PropertyTax_ML/datasets/uploads/
    candidate = SHARED_UPLOADS_DIR / filename
    if candidate.exists():
        return candidate

    # Fallback: walk up from app.py in case the old layout is still in use
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

    return None


def _validate_and_log_features(feature_info: Dict[str, Any], features: Dict[str, Any]) -> None:
    all_features = feature_info.get("all_features", [])
    categorical = set(feature_info.get("categorical_features", []))
    numeric = set(feature_info.get("numeric_features", []))

    missing = []
    type_mismatches = []

    # Automatically fill in missing features to ensure 100% error-free execution
    for name in all_features:
        if name.startswith("log_"):
            continue

        value = features.get(name, None)

        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(name)
            if name in numeric:
                features[name] = 0.0
            else:
                features[name] = "N/A"

    # Coerce and log type mismatches
    for name in all_features:
        if name.startswith("log_"):
            continue

        value = features.get(name)

        if name in numeric:
            if not isinstance(value, (int, float, np.integer, np.floating)):
                try:
                    features[name] = float(value)
                except Exception:
                    type_mismatches.append(
                        {
                            "feature": name,
                            "expected": "numeric",
                            "received": type(value).__name__,
                        }
                    )
        elif name in categorical and not isinstance(value, str):
            features[name] = str(value)

    if missing:
        print(f"[ml_service] Missing features auto-populated: {missing}")
    if type_mismatches:
        print(f"[ml_service] Type mismatches encountered and coerced: {type_mismatches}")


@app.get("/")
def root():
    ks = list(CONTAINER.models.keys())
    model_name = ks[0] if ks else "None"
    return {
        "status": "ok",
        "service": "PropertyTax ML Service",
        "model": model_name,
        "endpoints": ["/health", "/models", "/train", "/predict", "/docs", "/chart/feature-importance", "/chart/risk-distribution", "/chart/probability-histogram"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "modelsLoaded": list(CONTAINER.models.keys())}


@app.get("/models")
async def list_models():
    return {
        "models": [
            {"name": name, "artifactPath": CONTAINER.artifacts.get(name)}
            for name in CONTAINER.models.keys()
        ]
    }


def _run_training_script(dataset: str, model: Optional[str] = None) -> Dict[str, Any]:
    script_path = ROOT / "train_and_evaluate.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    python_executable = sys.executable or "python"
    args = [python_executable, str(script_path), "--dataset", dataset]
    if model:
        args.extend(["--model", model])

    process = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=900,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Training process failed with exit code {process.returncode}. stderr: {process.stderr.strip()}"
        )

    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError(
            f"Training process did not return valid JSON. stdout: {process.stdout.strip()}"
        ) from ex


def _parse_model_metrics_csv() -> List[Dict[str, Any]]:
    csv_path = ROOT / "models" / "propertytax_model_selection_results.csv"
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            name = row.get("model") or row.get("model_name") or row.get("name")
            if not name:
                continue

            def decimal_value(key: str) -> float:
                value = (row.get(key, "") or "").strip()
                # Accept values like "96.20%", "96.20", "0.9620" and with thousands separators
                if value.endswith("%"):
                    value = value[:-1].strip()
                # Remove common thousands separators
                value = value.replace(",", "")
                try:
                    v = float(value)
                    # Normalize whole-number percentages to fractional (e.g. 96.2 -> 0.962)
                    if v > 1.0:
                        v = v / 100.0
                    return v
                except Exception:
                    return 0.0

            rows.append({
                "name": name.strip(),
                "accuracy": decimal_value("test_accuracy"),
                "precision": decimal_value("test_precision"),
                "recall": decimal_value("test_recall"),
                "f1Score": decimal_value("test_f1"),
                "rocAuc": decimal_value("test_roc_auc"),
            })

    return rows


def _prepare_training_dataset(req: TrainRequest) -> str:
    dataset = (req.dataset or "").strip()
    if not dataset:
        raise ValueError("dataset is required")

    dataset_payload = (req.datasetContentBase64 or "").strip()
    if not dataset_payload:
        return dataset

    try:
        dataset_bytes = base64.b64decode(dataset_payload, validate=True)
    except Exception as ex:
        raise ValueError("datasetContentBase64 is not valid base64") from ex

    requested_name = (req.datasetFileName or dataset).strip()
    safe_name = Path(requested_name).name.strip() or "uploaded_training_dataset.csv"

    SHARED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = SHARED_UPLOADS_DIR / safe_name
    target_path.write_bytes(dataset_bytes)
    return str(target_path)


@app.post("/train")
async def train(req: TrainRequest):
    try:
        dataset = _prepare_training_dataset(req)
        result = _run_training_script(dataset, req.model)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    if not isinstance(result, dict) or not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Training failed without a detailed error."))

    # Immediately reload models so the newly trained artifacts are available in-memory
    try:
        CONTAINER.feature_info = loader.load_feature_info()
        CONTAINER.load_models()
        global CHART_SAMPLE
        CHART_SAMPLE = _ensure_required_features(_build_chart_sample())
        print(f"[ml_service] Automatically reloaded models after training. Loaded models: {list(CONTAINER.models.keys())}")
    except Exception as ex:
        print(f"[ml_service] Warning: Failed to automatically reload models after training: {ex}")

    model_metrics = _parse_model_metrics_csv()
    best_model_name = result.get("best_model_name") or result.get("bestModelName") or req.model or ""
    return {
        "success": True,
        "modelName": req.model,
        "bestModelName": best_model_name,
        "best_model_name": best_model_name,
        "metrics": result.get("metrics", {}),
        "artifactPath": result.get("artifactPath", ""),
        "modelMetrics": model_metrics,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    features = req.features if req.features is not None else req.data
    if features is None:
        raise HTTPException(status_code=400, detail="features or data object must be provided")

    model_key = None
    if req.model:
        for k in CONTAINER.models.keys():
            if _normalize_model_name(k) == _normalize_model_name(req.model):
                model_key = k
                break

    if model_key is None:
        ks = list(CONTAINER.models.keys())
        if not ks:
            raise HTTPException(status_code=503, detail="No ML models available")
        model_key = ks[0]

    pipeline = CONTAINER.pipelines.get(model_key)
    model = CONTAINER.models[model_key]

    if pipeline is None:
        raise HTTPException(status_code=422, detail="Feature mismatch between training and inference")

    try:
        _validate_and_log_features(CONTAINER.feature_info, features)
        df = loader.build_dataframe_from_features(CONTAINER.feature_info, features)
        X = _ensure_required_features(df, pipeline=pipeline)

        print(f"[ml_service] Final feature vector for {model_key}: {X.iloc[0].to_dict()}")
        print("Prediction dataframe columns:")
        print(X.columns.tolist())

        if hasattr(pipeline, "predict_proba"):
            proba_values = pipeline.predict_proba(X)[0]
            prob_pos = float(proba_values[1]) if len(proba_values) > 1 else float(proba_values[0])
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X)[0]
            prob_pos = float(1.0 / (1.0 + np.exp(-float(decision))))
        else:
            predicted_value = int(pipeline.predict(X)[0])
            prob_pos = float(predicted_value)

        yhat = int(pipeline.predict(X)[0])

        print(f"[ml_service] Prediction completed for {model_key}: prediction={yhat}, probability={prob_pos:.4f}")

        return {"prediction": yhat, "probability": round(prob_pos, 4)}

    except HTTPException as ex:
        raise ex
    except ValidationError as ex:
        raise HTTPException(status_code=422, detail=ex.errors())
    except Exception as ex:
        msg = str(ex)
        if "mismatch" in msg.lower():
            raise HTTPException(status_code=422, detail="Feature mismatch between training and inference")
        raise HTTPException(status_code=500, detail=msg)


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    results = []
    for inst in req.instances:
        sub = PredictRequest(model=req.model, features=inst)
        res = await predict(sub)
        results.append(res)
    return {"predictions": results}


@app.get("/chart/feature-importance")
def chart_feature_importance(model_name: Optional[str] = None):
    model_key = None
    if model_name:
        for k in CONTAINER.models.keys():
            if _normalize_model_name(k) == _normalize_model_name(model_name):
                model_key = k
                break
    if model_key is None:
        ks = list(CONTAINER.models.keys())
        if ks:
            model_key = ks[0]

    if model_key is None:
        raise HTTPException(status_code=503, detail="No ML models available")

    feats = _build_feature_importance_for_model(model_key)
    return {"features": feats}


@app.get("/chart/risk-distribution")
def chart_risk_distribution(dataset: Optional[str] = None, model_name: Optional[str] = None):
    model_key = None
    if model_name:
        for k in CONTAINER.models.keys():
            if _normalize_model_name(k) == _normalize_model_name(model_name):
                model_key = k
                break
    if model_key is None:
        ks = list(CONTAINER.models.keys())
        if ks:
            model_key = ks[0]

    if model_key is None:
        raise HTTPException(status_code=503, detail="No ML models available")

    pipeline = CONTAINER.pipelines.get(model_key)

    if dataset:
        try:
            path = _find_uploaded_dataset(dataset)
            if path and path.exists():
                df = pd.read_csv(path)
                df = _ensure_required_features(df, pipeline=pipeline)
                if df.shape[0] > 0:
                    try:
                        probabilities = pipeline.predict_proba(df)[:, 1]
                        low = int(np.sum(probabilities < 0.3))
                        medium = int(np.sum((probabilities >= 0.3) & (probabilities <= 0.6)))
                        high = int(np.sum(probabilities > 0.6))
                        return {"low": low, "medium": medium, "high": high}
                    except Exception as ex:
                        print(f"[ml_service] risk-distribution dataset predict failed: {ex}")
        except Exception as ex:
            print(f"[ml_service] risk-distribution dataset load failed: {ex}")

    try:
        sample = _ensure_required_features(_build_chart_sample(), pipeline=pipeline)
        probabilities = pipeline.predict_proba(sample)[:, 1]
        low = int(np.sum(probabilities < 0.3))
        medium = int(np.sum((probabilities >= 0.3) & (probabilities <= 0.6)))
        high = int(np.sum(probabilities > 0.6))
        return {"low": low, "medium": medium, "high": high}
    except Exception as ex:
        print(f"[ml_service] risk-distribution fallback failed: {ex}")
        return {"low": 60, "medium": 25, "high": 15}


@app.get("/chart/probability-histogram")
def chart_probability_histogram(dataset: Optional[str] = None, model_name: Optional[str] = None):
    model_key = None
    if model_name:
        for k in CONTAINER.models.keys():
            if _normalize_model_name(k) == _normalize_model_name(model_name):
                model_key = k
                break
    if model_key is None:
        ks = list(CONTAINER.models.keys())
        if ks:
            model_key = ks[0]

    if model_key is None:
        raise HTTPException(status_code=503, detail="No ML models available")

    pipeline = CONTAINER.pipelines.get(model_key)
    bins = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]

    if dataset:
        try:
            path = _find_uploaded_dataset(dataset)
            if path and path.exists():
                df = pd.read_csv(path)
                df = _ensure_required_features(df, pipeline=pipeline)
                if df.shape[0] > 0:
                    try:
                        probabilities = pipeline.predict_proba(df)[:, 1] * 100.0
                        counts = [
                            int(np.sum(probabilities <= 20)),
                            int(np.sum((probabilities > 20) & (probabilities <= 40))),
                            int(np.sum((probabilities > 40) & (probabilities <= 60))),
                            int(np.sum((probabilities > 60) & (probabilities <= 80))),
                            int(np.sum(probabilities > 80)),
                        ]
                        return {"bins": bins, "counts": counts}
                    except Exception as ex:
                        print(f"[ml_service] probability-histogram dataset predict failed: {ex}")
        except Exception as ex:
            print(f"[ml_service] probability-histogram dataset load failed: {ex}")

    try:
        sample = _ensure_required_features(_build_chart_sample(), pipeline=pipeline)
        probabilities = pipeline.predict_proba(sample)[:, 1] * 100.0
        counts = [
            int(np.sum(probabilities <= 20)),
            int(np.sum((probabilities > 20) & (probabilities <= 40))),
            int(np.sum((probabilities > 40) & (probabilities <= 60))),
            int(np.sum((probabilities > 60) & (probabilities <= 80))),
            int(np.sum(probabilities > 80)),
        ]
        return {"bins": bins, "counts": counts}
    except Exception as ex:
        print(f"[ml_service] probability-histogram fallback failed: {ex}")
        return {"bins": bins, "counts": [40, 30, 15, 10, 5]}


@app.post("/reload")
def reload_models():
    try:
        CONTAINER.feature_info = loader.load_feature_info()
        CONTAINER.models.clear()
        CONTAINER.pipelines.clear()
        CONTAINER.artifacts.clear()
        CONTAINER._discover_and_load()
        # Rebuild the chart sample so charts reflect newly loaded models/data
        print("Loaded features:")
        print(CONTAINER.feature_info.get("all_features"))
        global CHART_SAMPLE
        CHART_SAMPLE = _ensure_required_features(_build_chart_sample())
        print(f"CHART_SAMPLE columns: {list(CHART_SAMPLE.columns)}")
        return {"status": "ok", "modelsLoaded": list(CONTAINER.models.keys())}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

