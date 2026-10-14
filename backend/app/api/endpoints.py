from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any

from app.models.schemas import PredictionResponse, HealthResponse, TreatmentDetail
from app.services.disease_classifier import BaseDiseaseClassifier, get_classifier
from app.services.treatment_service import TreatmentService, get_treatment_service

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit


@router.get("/health", response_model=HealthResponse)
async def health_check(classifier: BaseDiseaseClassifier = Depends(get_classifier)):
    """Health check endpoint returning backend status and active model."""
    return HealthResponse(
        status="ok",
        classifier_type=classifier.__class__.__name__,
        version="1.0.0"
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    classifier: BaseDiseaseClassifier = Depends(get_classifier),
    treatment_service: TreatmentService = Depends(get_treatment_service)
):
    """
    Accepts an uploaded crop/leaf image, runs the CNN classifier,
    and returns predicted disease with treatment recommendations.
    """
    # 1. Basic validation of uploaded file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    filename_lower = file.filename.lower()
    has_valid_ext = any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    
    # Also check content-type if provided
    is_image_type = file.content_type and file.content_type.startswith("image/")

    if not has_valid_ext and not is_image_type:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an image file (JPG, PNG, WEBP)."
        )

    # 2. Read image bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image file: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    # 3. Predict disease using classifier abstraction
    try:
        disease_name, confidence = classifier.predict(image_bytes, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")

    # 4. Fetch treatment recommendation from treatment service
    treatment_data = treatment_service.get_recommendation(disease_name)

    treatment = TreatmentDetail(
        crop=treatment_data.get("crop", "Unknown Crop"),
        status=treatment_data.get("status", "Infected"),
        severity=treatment_data.get("severity", "Moderate"),
        recommendation=treatment_data.get("recommendation", "Follow standard disease management."),
        preventative_measures=treatment_data.get("preventative_measures", []),
        organic_options=treatment_data.get("organic_options")
    )

    return PredictionResponse(
        filename=file.filename,
        disease=disease_name,
        confidence=confidence,
        treatment=treatment
    )
