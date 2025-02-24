import os
import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Configure MLflow tracking
os.environ.update({
    'MLFLOW_TRACKING_USERNAME': 'mlflow_admin',
    'MLFLOW_TRACKING_PASSWORD': 'mlflow_admin',
    'MLFLOW_TRACKING_URI': 'http://localhost:5000',
    'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
    'AWS_ACCESS_KEY_ID': 'minio',
    'AWS_SECRET_ACCESS_KEY': 'minio123'
})

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Load the trained model from MLflow
model_uri = "models:/hotel_booking_cancelation_detector/latest"  # Use the latest version
model = mlflow.sklearn.load_model(model_uri)

# Initialize FastAPI
app = FastAPI(title="Hotel Booking Cancellation Predictor")

# Define request schema
class PredictionRequest(BaseModel):
    features: List[float]  # Ensure the input is a list of float values

# Define a prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input into Pandas DataFrame
        input_data = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(input_data)[0]  # 0 or 1

        return {"prediction": int(prediction)}  # Convert NumPy int to standard Python int

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def home():
    return {"message": "Hotel Booking Cancellation Prediction API is running 🚀"}
