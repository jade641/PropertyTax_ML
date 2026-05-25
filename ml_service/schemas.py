from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_payload(self):
        payload = self.features if self.features is not None else self.data
        if payload is None:
            raise ValueError("features or data object must be provided")
        if not isinstance(payload, dict) or len(payload) == 0:
            raise ValueError("features or data object must be a non-empty object")
        return self


class BatchPredictRequest(BaseModel):
    model: Optional[str] = None
    instances: List[Dict[str, Any]]


class TrainRequest(BaseModel):
    model: Optional[str] = None
    dataset: str
    datasetFileName: Optional[str] = None
    datasetContentBase64: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    model: Optional[str] = None
    predictedLabel: Optional[int] = None
    threshold: Optional[float] = None
    riskLevel: Optional[str] = None
    confidence: Optional[float] = None
    topFeatures: List[Dict[str, Any]] = Field(default_factory=list)
