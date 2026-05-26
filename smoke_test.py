"""Minimal smoke test for the PropertyTax ML inference app.

Run from the PropertyTax_ML folder:
    python smoke_test.py
"""

from json import dumps, loads
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


import os

BASE_URL = os.environ.get(
    "ML_BASE_URL",
    "https://propertytax-ml.onrender.com"
)


def get_json(path: str):
    with urlopen(f"{BASE_URL}{path}") as response:
        return loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict):
    request = Request(
        f"{BASE_URL}{path}",
        data=dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return loads(response.read().decode("utf-8"))


def main():
    root = get_json("/")
    health = get_json("/health")
    models = get_json("/models")
    manifest = get_json("/artifacts/manifest")
    prediction = post_json(
        "/predict?threshold=0.5",
        {"data": {"payment_compliance_score": 0.8, "assessed_value": 100000}},
    )

    assert root.get("status") == "ok", root
    assert health.get("status") == "ok", health
    assert isinstance(models.get("models"), list), models
    assert isinstance(manifest.get("artifacts"), list), manifest
    assert "prediction" in prediction, prediction
    assert isinstance(prediction.get("prediction"), int), prediction

    print("Smoke test passed")
    print("root:", root)
    print("health:", health)
    print("models:", models)
    print("manifest:", manifest)
    print("prediction:", prediction)


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Smoke test failed: {exc}")