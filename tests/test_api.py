import pytest
from fastapi.testclient import TestClient

from app.main import app 

client = TestClient(app)

def test_health_status_okk():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health_has_uptime():
    r = client.get("/health")
    assert "uptime_seconds" in r.json()

SETOSA_SAMPLE = {           # Clearly a setosa
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2,
}
 
VIRGINICA_SAMPLE = {        # Clearly a virginica
    "sepal_length": 6.7,
    "sepal_width":  3.0,
    "petal_length": 5.2,
    "petal_width":  2.3,
}

def test_predict_setosa():
    r = client.post("/predict", json=SETOSA_SAMPLE)
    assert r.status_code == 200
    assert r.json()['prediction'] == "setosa"


def test_predict_virginica():
    r = client.post("/predict", json=VIRGINICA_SAMPLE)
    assert r.status_code == 200
    assert r.json()['prediction'] == "virginica"

def test_predict_returns_probabilities():
    r = client.post("/predict", json=SETOSA_SAMPLE)
    probs = r.json()["probabilities"]
    assert set(probs.keys()) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(probs.values()) - 1.0) < 1e-4

def test_predict_missing_field():
    bad_payload = {"sepal_length" : 5.1}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422 

def test_model_info_has_accuracy():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "accuracy" in r.json()

def test_model_info_accuracy_above_threshold():
    r = client.get("/model-info")
    assert r.json()["accuracy"] >= 0.90