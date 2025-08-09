From python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# Set working directory to the API folder

RUN pip install --no-cache-dir -r requirements.txt
# Expose the port your Flask app uses
EXPOSE 7000
COPY . .
#CMD ["python", "api/inference.py"]
CMD ["uvicorn", "api.inference.app", "--host", "0.0.0.0", "--port", "8000"]