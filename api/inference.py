from flask import Flask, request, jsonify
from schemas import IrisInput
from pydantic import ValidationError
import joblib
import sqlite3
from datetime import datetime
import threading
import os

from prometheus_client import Counter, Histogram, generate_latest
from time import time

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total number of prediction requests')
#REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency')
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions made', ['pclass'])

REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds')
VALIDATION_ERRORS = Counter('validation_errors_total', 'Total number of validation errors')
MODEL_LOAD_SUCCESS = Counter('model_load_success_total', 'Model load succeeded')
MODEL_LOAD_FAILURE = Counter('model_load_failure_total', 'Model load failed')
DB_INSERT_FAILURES = Counter('db_insert_failures_total', 'DB insert failures during logging')

# Load the best model from the "model" folder
model_path = os.path.join("model", "Logistic_Regression_best_model.pkl")

try:
    model = joblib.load(model_path)
    MODEL_LOAD_SUCCESS.inc()
except:
    MODEL_LOAD_FAILURE.inc()
    raise

# Create SQLite DB (in-memory or use a file like 'logs.db')
conn = sqlite3.connect('logs.db', check_same_thread=False)
cursor = conn.cursor()

# Initialize log table
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    input_data TEXT,
    prediction TEXT
)
""")
conn.commit()

# Lock for thread-safe writes
db_lock = threading.Lock()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time()
        input_data = request.get_json()
        REQUEST_COUNT.inc()
        validated_input = IrisInput(**input_data)
        features = validated_input.features

        prediction = model.predict([features])[0]
        PREDICTION_COUNTER.labels(pclass=str(prediction)).inc()
        # Log to DB
        with db_lock:
            cursor.execute(
                "INSERT INTO logs (timestamp, input_data, prediction) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), str(features), str(prediction))
            )
            conn.commit()

        return jsonify({"prediction": int(prediction)})
    except ValidationError as ve:
        VALIDATION_ERRORS.inc()
        errors = ve.errors()
        formatted_errors = [
            {
                "field": err["loc"][0],
                "message": err["msg"]
            } for err in errors
        ]
        return jsonify({"validation_error": formatted_errors}), 422
    except Exception as db_error:
        DB_INSERT_FAILURES.inc()
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        REQUEST_LATENCY.observe(time() - start_time)

@app.route('/metrics', methods=["GET"])
def metrics():
    with db_lock:
        cursor.execute("SELECT COUNT(*) FROM logs")
        total_requests = cursor.fetchone()[0]

        cursor.execute("SELECT timestamp, input_data, prediction FROM logs ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()

    # Build HTML table
    table_html = "<table border='1'><tr><th>Timestamp</th><th>Input</th><th>Prediction</th></tr>"
    for row in rows:
         ts = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%f")
         formatted_ts = ts.strftime("%d %b %Y, %I:%M:%S %p")
         table_html += f"<tr><td>{formatted_ts}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    table_html += "</table>"

    return f"""
    <h2>Model Inference Metrics</h2>
    <p><strong>Total Requests:</strong> {total_requests}</p>
    <h3>Last 10 Prediction Logs</h3>
    {table_html}
    """
@app.route('/prometheusmetrics', methods=["GET"])
def metrics1():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}

@app.route("/", methods=["GET"])
def home():
    return "ML Inference API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)

