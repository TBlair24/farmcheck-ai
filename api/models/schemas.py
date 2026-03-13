from pydantic import BaseModel
from typing import Optional

class PredictionResult(BaseModel):
    indicator:       str
    confidence:      float
    compliant:       bool
    binary_score:    int
    domain:          str

class PredictionResponse(BaseModel):
    status:          str
    filename:        str
    prediction:      PredictionResult
    all_scores:      dict[str, float]
    model_version:   str = "yolov8n-cls-v1"
    inference_ms:    float

class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    model_version:   str
    classes:         list[str]