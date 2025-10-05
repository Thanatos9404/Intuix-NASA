import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import joblib

from model import ExoplanetModel

app = FastAPI(title="Exoplanet Detection API", version="1.0.0")


class PredictionInput(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    top_features: Dict[str, float]


def load_latest_model() -> ExoplanetModel:
    models_dir = Path("models")
    model_files = list(models_dir.glob("model_*.joblib"))

    if not model_files:
        raise FileNotFoundError("No trained model found")

    latest_model = max(model_files, key=lambda p: p.stem.split("_")[-1])

    model = ExoplanetModel(str(latest_model))
    return model


model = None


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_latest_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model/metrics")
async def get_model_metrics():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    models_dir = Path("models")
    metric_files = list(models_dir.glob("metrics_*.json"))

    if not metric_files:
        raise HTTPException(status_code=404, detail="No metrics found")

    latest_metrics = max(metric_files, key=lambda p: p.stem.split("_")[-1])

    with open(latest_metrics, "r") as f:
        metrics = json.load(f)

    return metrics


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([input_data.features])

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        predicted_class = model.label_encoder.inverse_transform([pred])[0]

        probabilities = {
            model.label_encoder.classes_[i]: float(proba[i])
            for i in range(len(proba))
        }

        explanations = model.explain(df, top_k=5)
        top_features = explanations[0]

        return PredictionResponse(
            predicted_class=predicted_class,
            probabilities=probabilities,
            top_features=top_features
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        df = pd.read_csv(file.file)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        df["predicted_class"] = model.label_encoder.inverse_transform(predictions)

        for i, class_name in enumerate(model.label_encoder.classes_):
            df[f"prob_{class_name}"] = probabilities[:, i]

        output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)

        from fastapi.responses import FileResponse
        return FileResponse(output_path, filename=output_path, media_type="text/csv")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
