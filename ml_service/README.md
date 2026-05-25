PropertyTax ML Service
======================

Production deploy:

1. Build command:

   pip install -r requirements.txt

2. Start command:

   uvicorn ml_service.app:app --host 0.0.0.0 --port $PORT

3. Health check path:

   /health

Run locally:

1. Create virtualenv and install requirements:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2. Start server:

   uvicorn ml_service.app:app --host 0.0.0.0 --port 8000

Endpoints:

- GET /health
- GET /models
- POST /predict  (body: {"model": "OptionalModelName", "features": { ... }})
- POST /predict/batch

Notes:
- The service loads any .pkl files found in ../models
- Ensure the training pipeline saved the preprocessing pipeline together with the estimator so that inference will use the same preprocessing and feature ordering.
- Set `FRONTEND_BASE_URL` or `CORS_ALLOWED_ORIGINS` in production if the frontend URL changes.
