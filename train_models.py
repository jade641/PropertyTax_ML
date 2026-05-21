#!/usr/bin/env python3
"""
Train and save three model pipelines for PropertyTax ML service.

Creates sklearn Pipeline artifacts (preprocessor + estimator) in the
`PropertyTax_ML/models` folder with the filenames expected by the
inference service. By default it will not overwrite existing artifacts
unless `--force` is provided.

Run from the PropertyTax_ML folder:
    python train_models.py

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"
TRAINING_CSV = DATASETS_DIR / "PropertyTax_model_ready.csv"
FEATURE_INFO_PATH = MODELS_DIR / "propertytax_feature_info.json"


def load_feature_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Create OneHotEncoder while handling sklearn API differences ('sparse' vs 'sparse_output')
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)  # older sklearn

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="N/A")),
        ("onehot", onehot),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features),
    ], remainder="drop", verbose_feature_names_out=False)

    return preprocessor


def train_and_save(X: pd.DataFrame, y: pd.Series, preprocessor, estimator, dest: Path):
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, dest)


def main(force: bool = False) -> int:
    if not FEATURE_INFO_PATH.exists():
        print(f"Missing feature info: {FEATURE_INFO_PATH}")
        return 2

    if not TRAINING_CSV.exists():
        print(f"Training CSV not found: {TRAINING_CSV}")
        return 2

    feature_info = load_feature_info(FEATURE_INFO_PATH)
    all_features = feature_info.get("all_features", [])
    categorical = feature_info.get("categorical_features", [])
    numeric = feature_info.get("numeric_features", [])
    target = feature_info.get("target", "is_late_payment")

    print("Loading training data (this may take a moment)...")
    df = pd.read_csv(TRAINING_CSV)

    # Ensure all expected feature columns exist in the training frame
    for c in all_features:
        if c not in df.columns:
            df[c] = np.nan

    if target not in df.columns:
        print(f"Target column '{target}' not found in training CSV")
        return 2

    X = df[all_features]
    y = df[target].fillna(0).astype(int)

    preprocessor = build_preprocessor(categorical, numeric)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_train = [
        ("Logistic Regression", "logistic_regression_propertytax_model.pkl", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")),
        ("Random Forest", "random_forest_propertytax_model.pkl", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("Extra Trees", "extra_trees_propertytax_model.pkl", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]

    for display_name, filename, estimator in models_to_train:
        dest = MODELS_DIR / filename
        if dest.exists() and not force:
            print(f"Skipping existing model artifact: {dest.name}")
            continue

        print(f"Training {display_name}...")
        try:
            train_and_save(X, y, preprocessor, estimator, dest)
            print(f"Saved model artifact: {dest}")
        except Exception as exc:
            print(f"Failed training {display_name}: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save PropertyTax ML pipelines")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing model artifacts")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
