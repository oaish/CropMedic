from typing import List, Optional
from pydantic import BaseModel, Field


class TreatmentDetail(BaseModel):
    crop: str = Field(..., example="Tomato")
    status: str = Field(..., example="Infected")
    severity: str = Field(..., example="Moderate")
    recommendation: str = Field(..., example="Prune infected leaves and apply fungicide.")
    preventative_measures: List[str] = Field(
        default_factory=list,
        example=["Water at base", "Crop rotation"]
    )
    organic_options: Optional[str] = Field(
        None, 
        example="Apply organic copper spray."
    )


class PredictionResponse(BaseModel):
    filename: str = Field(..., example="leaf_sample.jpg")
    disease: str = Field(..., example="Tomato Early Blight")
    confidence: float = Field(..., example=0.94, description="Confidence score between 0.0 and 1.0")
    treatment: TreatmentDetail


class HealthResponse(BaseModel):
    status: str = "ok"
    classifier_type: str = "MockDiseaseClassifier"
    version: str = "1.0.0"
