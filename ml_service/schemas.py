from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    model: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None


class BatchPredictRequest(BaseModel):
    model: Optional[str] = None
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    model: str
    probability: float
    predictedLabel: int
    prediction: Optional[int] = None
    threshold: Optional[float] = None
    riskLevel: str
    confidence: float
    topFeatures: List[Dict[str, Any]]
