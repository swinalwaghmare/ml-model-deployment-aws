import json
import time
from pathlib import Path

import uvicorn

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Iris Classifier API",
    description="ML inference service - part of the AWS CI/CD pipeline project",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"
META_PATH  = BASE_DIR / "model" / "model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["classes"]
START_TIME = time.time()

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", json_schema_extra={"example": 5.1}) 
    sepal_width:  float = Field(..., description="Sepal width in cm", json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., description="Petal length in cm", json_schema_extra={"example": 1.4})
    petal_width:  float = Field(..., description="Petal width in cm", json_schema_extra={"example": 0.2})

class PredictResponse(BaseModel):
    prediction:    str
    class_index:   int
    probabilities: dict[str, float]
    model_version: str

@app.get("/health")
def health():
    """Health check used by load balancer and kubernetes liveness probes"""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model": metadata["model_type"],
        "accuracy": metadata["accuracy"],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run inference and return the predicted iris species."""
    features = np.array([[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width,
    ]])

    try:
        class_index   = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        prediction = CLASS_NAMES[class_index],
        class_index = class_index,
        probabilities= {
            name: round(float(prob), 4)
            for name, prob in zip(CLASS_NAMES, probabilities)
        },
        model_version= metadata["model_type"],
    )

@app.get("/model-info")
def model_info():
    """Return metadata abouth the loaded model."""
    return metadata

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)