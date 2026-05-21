Run the inference app locally:

Install the environment (from the `PropertyTax_ML` folder):

```bash
pip install -r requirements.txt
```

Start the FastAPI app with uvicorn:

```bash
uvicorn inference.app:app --reload --host 127.0.0.1 --port 8000
```

Health: `GET /health`
Predict: `POST /predict` with JSON body `{ "data": { <feature:value pairs> } }` and optional `threshold` query param.
