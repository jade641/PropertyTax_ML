import urllib.request, json

endpoints = [
    "http://127.0.0.1:8000/health",
    "http://127.0.0.1:8000/chart/feature-importance",
    "http://127.0.0.1:8000/chart/risk-distribution",
    "http://127.0.0.1:8000/chart/probability-histogram",
]

for url in endpoints:
    name = url.split("/")[-1] or "root"
    try:
        r = urllib.request.urlopen(url)
        print(f"  {name}: {r.status} OK")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

# Test predict
data = json.dumps({"features": {"taxpayer_type": "Individual", "property_type": "Residential", "tax_amount": 5000, "prior_late_payments": 2, "outstanding_balance": 1000, "payment_compliance_score": 0.7}}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/predict", data=data, headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req)
    print(f"  predict: {r.status} OK")
except Exception as e:
    print(f"  predict: FAILED - {e}")

print("\nAll tests done!")
