"""
Small example to run a single prediction locally using the saved model and the processed dataset (if available).
"""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"

model_files = list(MODELS_DIR.glob("*_propertytax_model.pkl"))
if not model_files:
    raise SystemExit(f"No model found in {MODELS_DIR}")
model_path = model_files[0]
model = joblib.load(model_path)

feature_info_path = MODELS_DIR / "propertytax_feature_info.json"
if not feature_info_path.exists():
    raise SystemExit(f"Missing feature info at {feature_info_path}")

feature_info = json.loads(feature_info_path.read_text(encoding='utf-8'))
all_features = feature_info.get('all_features', [])

processed_csv = DATASETS_DIR / "PropertyTax_model_ready.csv"
if not processed_csv.exists():
    print(f"Processed dataset not found at {processed_csv}. You can still call the model by providing a dict of features.")
    sample = {k: None for k in all_features}
else:
    df = pd.read_csv(processed_csv)
    sample_row = df.sample(n=1, random_state=42).iloc[0]
    sample = {k: (sample_row[k] if k in df.columns else None) for k in all_features}

# prepare dataframe
import numpy as np
import pandas as pd
row_df = pd.DataFrame([sample])
for c in all_features:
    if c not in row_df.columns:
        row_df[c] = np.nan
row_df = row_df[all_features]

try:
    proba = model.predict_proba(row_df)[0, 1]
except Exception:
    score = model.decision_function(row_df)[0]
    proba = 1 / (1 + np.exp(-score))

print('Model:', model_path.name)
print('Probability (late):', proba)
print('Prediction (0/1) at 0.5 threshold:', int(proba >= 0.5))

