import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

os.environ.update({
    'MLFLOW_TRACKING_USERNAME': 'mlflow_admin',
    'MLFLOW_TRACKING_PASSWORD': 'mlflow_admin',
    'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
    'AWS_ACCESS_KEY_ID': 'minio',
    'AWS_SECRET_ACCESS_KEY': 'minio123'
})

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Start MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hotel_booking_cancellation_detector")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.set_tags({
        "project": "hotel_booking",
        "author": "Mohamed AZGHARI",
        "model_type": "RandomForest",
    })

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest.pkl"
    joblib.dump(model, model_path)
    
    # Log model in MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

print("✅ Model training complete. Accuracy:", accuracy)
