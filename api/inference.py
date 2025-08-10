from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError
import joblib
import sqlite3
from datetime import datetime
import threading
import os

from prometheus_client import Counter, Histogram, generate_latest

# Define input schema
class IrisInput(BaseModel):
    features: list

app = FastAPI(title="ML Inference API")

# Prometheus indicators demo checkin
REQUEST_LAG= Histogram('api_request_latency_seconds', 'Request latency in seconds')
VALIDATION_ERRORS = Counter('validation_errors_total', 'Total number of validation errors')
MODEL_READY_SUCCESS = Counter('model_load_success', 'Model loaded')
MODEL_FAILED= Counter('model_load_failured', 'Model load failed')

# Load the best model from the "model" folder
model_path = os.path.join("model", "Logistic_Regression_best_model.pkl")
try:
    model = joblib.load(model_path)
    MODEL_READY_SUCCESS.inc()
except Exception:
    MODEL_FAILED.inc()
    raise

# Create SQLite DB (in-memory or use a file like 'logs.db')
conn = sqlite3.connect('logs.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    input_data TEXT,
    prediction TEXT
)
""")
conn.commit()
db_lock = threading.Lock()

@app.post("/predict")
async def predict(input_data: IrisInput, request: Request):
    start_time = request.state.start_time if hasattr(request.state, "start_time") else None
    try:
        features = input_data.features
        prediction = model.predict([features])[0]
        # Log to DB
        with db_lock:
            cursor.execute(
                "INSERT INTO logs (timestamp, input_data, prediction) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), str(features), str(prediction))
            )
            conn.commit()
        return {"prediction": int(prediction)}
    except ValidationError as ve:
        VALIDATION_ERRORS.inc()
        return JSONResponse(
            status_code=422,
            content={"validation_error": ve.errors()}
        )
    except Exception as db_error:
        raise HTTPException(status_code=500, detail=str(db_error))
    finally:
        if start_time:
            REQUEST_LAG.observe(time() - start_time)

@app.get("/metrics")
async def metrics():
    with db_lock:
        cursor.execute("SELECT COUNT(*) FROM logs")
        total_requests = cursor.fetchone()[0]
        cursor.execute("SELECT timestamp, input_data, prediction FROM logs ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
    table_html = "<table border='1'><tr><th>Timestamp</th><th>Input</th><th>Prediction</th></tr>"
    for row in rows:
        ts = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%f")
        formatted_ts = ts.strftime("%d %b %Y, %I:%M:%S %p")
        table_html += f"<tr><td>{formatted_ts}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    table_html += "</table>"
    html = f"""
    <h2>Model Inference Metrics</h2>
    <p><strong>Total Requests:</strong> {total_requests}</p>
    <h3>Last 10 Prediction Logs</h3>
    {table_html}
    """
    return HTMLResponse(content=html)

@app.get("/prometheusmetrics")
async def prometheus_metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain; version=0.0.4")

@app.get("/")
async def home():
    return "ML Inference API is running!"

# Optional: Middleware to track request start time for latency
from time import time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request.state.start_time = time()
    response = await call_next(request)
    return response

# To run: uvicorn api.inference:app --reload --port 7000