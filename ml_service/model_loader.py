import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    v = str(s).strip()
    return v if v != "" else None


def _safe_categorical(raw: Any) -> str:
    value = _norm(raw)
    return value if value is not None else "N/A"


def _safe_numeric(raw: Any) -> float:
    try:
        if raw is None:
            return 0.0
        if isinstance(raw, str) and raw.strip() == "":
            return 0.0
        return float(raw)
    except Exception:
        return 0.0


def load_feature_info() -> Dict[str, Any]:
    path = os.path.join(BASE, "propertytax_feature_info.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def discover_models() -> Dict[str, Dict[str, Any]]:
    """Discover .pkl artifacts and produce a mapping of friendly model names -> artifact path.

    Uses propertytax_model_selection_results.csv (if present) to map display names to files.
    Falls back to filename-based token matching.
    """
    # list .pkl files
    files = [f for f in os.listdir(BASE) if f.endswith(".pkl")]
    mapping: Dict[str, Dict[str, Any]] = {}

    # read selection CSV to get canonical model names (best-effort)
    selection_path = os.path.join(BASE, "propertytax_model_selection_results.csv")
    canonical_names: List[str] = []
    if os.path.exists(selection_path):
        try:
            with open(selection_path, "r", encoding="utf-8") as fh:
                # first column 'model'
                for i, line in enumerate(fh):
                    if i == 0:
                        continue
                    parts = line.strip().split(",")
                    if parts:
                        canonical_names.append(parts[0].strip())
        except Exception:
            canonical_names = []

    # try to match files to canonical names by normalized tokens
    for fname in files:
        base = os.path.splitext(fname)[0]
        norm_base = _normalize_name(base)
        chosen = None
        for cname in canonical_names:
            if _normalize_name(cname) in norm_base or norm_base in _normalize_name(cname):
                chosen = cname
                break

        if chosen is None:
            # fallback: use human-friendly from filename
            chosen = base.replace("_", " ").title()

        mapping[chosen] = {"path": os.path.join(BASE, fname), "filename": fname}

    return mapping


def _apply_estimator_compatibility_fixes(estimator: Any) -> Any:
    # Older serialized LogisticRegression artifacts can miss multi_class after sklearn upgrades.
    if estimator is not None and estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
        setattr(estimator, "multi_class", "auto")

    return estimator


def load_model(path: str) -> Tuple[Any, Optional[Any]]:
    """Load a saved artifact. Return (estimator, pipeline).

    If artifact is not a sklearn Pipeline, raise an error so inference cannot silently bypass preprocessing.
    """
    artifact = joblib.load(path)

    if not hasattr(artifact, "named_steps"):
        filename = os.path.basename(path)
        raise ValueError(
            f"Model must be trained as Pipeline (preprocessor + estimator): {filename}. "
            f"Expected a sklearn Pipeline artifact at {path}."
        )

    pipeline = artifact
    try:
        estimator = pipeline.steps[-1][1]
    except Exception as exc:
        raise ValueError(f"Unable to resolve final estimator from pipeline: {path}") from exc

    estimator = _apply_estimator_compatibility_fixes(estimator)

    return estimator, pipeline


def build_dataframe_from_features(feature_info: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
    """Create a DataFrame matching training ordering and apply minimal cleaning.

    Rules:
    - categorical: strings only; empty -> 'N/A'
    - numeric: float only; missing/invalid -> 0.0
    - log features: derived once here from the base numeric values
    """
    cols = feature_info.get("all_features", [])
    categorical = set(feature_info.get("categorical_features", []))
    numeric = set(feature_info.get("numeric_features", []))

    row: Dict[str, Any] = {}

    for c in cols:
        raw = features.get(c, None)

        if c in categorical:
            row[c] = _safe_categorical(raw)
        elif c in numeric:
            row[c] = _safe_numeric(raw)
        else:
            row[c] = _safe_categorical(raw)

    def safe_log(x: Any) -> float:
        try:
            xv = float(x)
            return float(np.log(xv)) if xv > 0 else 0.0
        except Exception:
            return 0.0

    # Derive log features once in Python from the base numeric inputs.
    log_base_map = {
        "log_market_value": "market_value",
        "log_assessed_value": "assessed_value",
        "log_outstanding_balance": "outstanding_balance",
    }

    for log_feature, base_feature in log_base_map.items():
        if log_feature in cols:
            base_value = row.get(base_feature)
            if base_value is None or not isinstance(base_value, (int, float, np.integer, np.floating)) or float(base_value) <= 0:
                row[log_feature] = 0.0
            else:
                row[log_feature] = safe_log(base_value)

    # Final pass: ensure exact column order and hard type consistency.
    ordered_row: Dict[str, Any] = {}
    for c in cols:
        value = row.get(c, None)
        if c in numeric:
            ordered_row[c] = _safe_numeric(value)
        else:
            ordered_row[c] = _safe_categorical(value)

    df = pd.DataFrame([ordered_row], columns=cols)

    return df


def explain_prediction(model, pipeline, X: pd.DataFrame, feature_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Attempt to compute top features based on model type
    try:
        # Try to get transformed feature names if pipeline exists
        if pipeline is not None and hasattr(pipeline, "get_feature_names_out"):
            try:
                names = pipeline.get_feature_names_out()
            except Exception:
                names = X.columns
        else:
            names = X.columns

        names = list(map(str, names))

        # If model has feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_)
            idx = np.argsort(importances)[::-1][:5]
            return [{"feature": names[i] if i < len(names) else str(i), "importance": float(importances[i])} for i in idx]

        # If linear model
        if hasattr(model, "coef_"):
            coefs = np.abs(np.array(model.coef_).ravel())
            idx = np.argsort(coefs)[::-1][:5]
            return [{"feature": names[i] if i < len(names) else str(i), "importance": float(coefs[i])} for i in idx]

    except Exception:
        pass

    # fallback: use feature_info ordering and return first features
    out = []
    for f in feature_info.get("all_features", [])[:5]:
        out.append({"feature": f, "importance": 0.0})
    return out
