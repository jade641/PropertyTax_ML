#!/usr/bin/env python3
"""
Thin wrapper around notebooks/propertytax.py for C# StartTrainingWorkflow().

All training, feature-engineering, and model-selection logic lives in
propertytax.main().  This script simply:

  1. Redirects ALL non-JSON output (print, logging, matplotlib) to stderr
     so that only the final JSON result appears on stdout.
  2. Calls propertytax.main(dataset_cli=...) to run the real training.
  3. Reads the saved artifacts (propertytax_feature_info.json) and emits
     the exact JSON structure C# expects on stdout.

Expected JSON on stdout (success):
    {
      "success": true,
      "best_model_name": "Logistic Regression",
      "metrics": { "accuracy": ..., "precision": ..., "recall": ..., "f1Score": ..., "rocAuc": ... },
      "artifactPath": "C:\\...\\models\\logistic_regression_propertytax_model.pkl"
    }

Expected JSON on stdout (failure):
    { "success": false, "error": "..." }
"""
import argparse
import io
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Redirect ALL stdout-bound output to stderr BEFORE importing anything that
# might print or configure logging (matplotlib, sklearn, propertytax, etc.).
# We capture a reference to the real stdout first so we can write the final
# JSON there at the very end.
# ---------------------------------------------------------------------------
_real_stdout = sys.stdout
sys.stdout = sys.stderr  # everything that calls print() goes to stderr now

# Force matplotlib to use a non-interactive backend (no GUI windows)
import matplotlib
matplotlib.use("Agg")

# Redirect ALL loggers to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
    force=True,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"
SHARED_UPLOADS_DIR = DATASETS_DIR / "uploads"
FEATURE_INFO_PATH = MODELS_DIR / "propertytax_feature_info.json"


def _resolve_dataset_path(filename: str) -> str | None:
    if not filename:
        return None

    # If already an absolute path that exists, use it directly
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return str(p)

    # All directories to search, in priority order
    search_dirs = [
        # 1. Shared uploads folder (datasets/uploads/)
        DATASETS_DIR / "uploads",
        # 2. datasets/ root
        DATASETS_DIR,
        # 3. C# backend bin/Debug output
        ROOT.parent / "backend" / "PropertyTax.API" / "PropertyTax.API" / "bin" / "Debug" / "net8.0" / "uploads" / "ml-datasets",
        # 4. C# backend bin/Release output
        ROOT.parent / "backend" / "PropertyTax.API" / "PropertyTax.API" / "bin" / "Release" / "net8.0" / "uploads" / "ml-datasets",
        # 5. Walk up from ROOT and check uploads/ml-datasets at each level
    ]

    # Check fixed dirs first
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return str(candidate)

    # Walk up 8 levels from ROOT
    current = ROOT
    for _ in range(8):
        for sub in (
            Path("uploads") / "ml-datasets" / filename,
            Path("uploads") / filename,
            Path("datasets") / "uploads" / filename,
            Path("bin") / "Debug" / "net8.0" / "uploads" / "ml-datasets" / filename,
            Path("bin") / "Release" / "net8.0" / "uploads" / "ml-datasets" / filename,
        ):
            c = current / sub
            if c.exists():
                return str(c)
        if current.parent == current:
            break
        current = current.parent

    # Log all searched paths for debugging
    logging.warning(
        "Dataset '%s' not found in any search path. "
        "Falling back to propertytax.find_dataset().",
        filename,
    )
    return filename


def _emit_json(obj: dict) -> None:
    """Write *obj* as a single JSON line to the REAL stdout (fd 1)."""
    _real_stdout.write(json.dumps(obj))
    _real_stdout.write("\n")
    _real_stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PropertyTax ML model (wrapper)"
    )
    parser.add_argument(
        "--model", type=str, default="", help="Model display name (ignored — propertytax.py auto-selects the best)"
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="Dataset filename to train on"
    )
    # Keep legacy flags so old invocations don't crash
    parser.add_argument("--all", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--retrain", action="store_true", help="(legacy, ignored)")
    args = parser.parse_args()

    try:
        # ---------------------------------------------------------------
        # Import propertytax AFTER stdout has been redirected so its
        # module-level prints and logging go to stderr.
        # ---------------------------------------------------------------
        # Add notebooks/ to the import path
        notebooks_dir = str(ROOT / "notebooks")
        if notebooks_dir not in sys.path:
            sys.path.insert(0, notebooks_dir)

        import propertytax  # type: ignore

        # Resolve the dataset path
        dataset_path = _resolve_dataset_path(args.dataset) if args.dataset else None
        logging.info("Resolved dataset path: %s", dataset_path)

        # ------- Run the real training -------
        propertytax.main(dataset_cli=dataset_path, train_all_models=True)

        # ---------------------------------------------------------------
        # Read back the saved artifacts to build the JSON C# expects.
        # ---------------------------------------------------------------
        if not FEATURE_INFO_PATH.exists():
            raise FileNotFoundError(
                f"propertytax.main() completed but {FEATURE_INFO_PATH} was not created."
            )

        with FEATURE_INFO_PATH.open("r", encoding="utf-8") as fh:
            feature_info = json.load(fh)

        best_model_name: str = feature_info.get("best_model_name", "")
        if not best_model_name:
            raise ValueError("best_model_name is empty in feature_info.json")

        # Metrics — prefer the already-saved dict in feature_info
        saved_metrics = feature_info.get("best_model_metrics", {})
        metrics = {
            "accuracy": saved_metrics.get("accuracy", 0.0),
            "precision": saved_metrics.get("precision", 0.0),
            "recall": saved_metrics.get("recall", 0.0),
            "f1Score": saved_metrics.get("f1Score", 0.0),
            "rocAuc": saved_metrics.get("rocAuc", 0.0),
        }

        # Artifact path
        model_slug = best_model_name.lower().replace(" ", "_")
        pkl_path = MODELS_DIR / f"{model_slug}_propertytax_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Expected model artifact not found: {pkl_path}")

        artifact_path = str(pkl_path.resolve())

        # Read and parse selection results CSV if it exists
        model_metrics = []
        csv_path = MODELS_DIR / "propertytax_model_selection_results.csv"
        if csv_path.exists():
            try:
                import csv
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSV headers: model, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc
                        name = row.get("model", "").strip()
                        if not name:
                            continue
                        
                        def clean_pct(val):
                            if not val:
                                return 0.0
                            val_str = str(val).strip()
                            if val_str.endswith("%"):
                                val_str = val_str[:-1].strip()
                            try:
                                parsed = float(val_str)
                                if parsed > 1.0:
                                    parsed /= 100.0
                                return parsed
                            except Exception:
                                return 0.0

                        model_metrics.append({
                            "name": name,
                            "accuracy": clean_pct(row.get("test_accuracy")),
                            "precision": clean_pct(row.get("test_precision")),
                            "recall": clean_pct(row.get("test_recall")),
                            "f1Score": clean_pct(row.get("test_f1")),
                            "rocAuc": clean_pct(row.get("test_roc_auc")),
                        })
            except Exception as e:
                logging.warning("Failed to parse selection CSV in Python: %s", e)

        # Trigger hot-reload in running FastAPI (best-effort)
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/reload", method="POST")
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception:
            pass

        _emit_json({
            "success": True,
            "best_model_name": best_model_name,
            "metrics": metrics,
            "artifactPath": artifact_path,
            "modelMetrics": model_metrics,
        })
        sys.exit(0)

    except SystemExit:
        raise  # let sys.exit() propagate
    except Exception as ex:
        _emit_json({
            "success": False,
            "error": str(ex),
        })
        # Also log the full traceback to stderr for debugging
        logging.exception("train_and_evaluate.py failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
