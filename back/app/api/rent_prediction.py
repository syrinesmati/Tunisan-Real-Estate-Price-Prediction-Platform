"""
Rent Prediction API Endpoints (train5 model)
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import RentPredictionRequest, RentPredictionResponse
from app.ml.rent_model_service import rent_model_service

router = APIRouter()


@router.post("/predict", response_model=RentPredictionResponse)
async def predict_rent(request: RentPredictionRequest):
    """Predict rent price using the train5 RandomForest pipeline"""
    try:
        if not rent_model_service.loaded:
            rent_model_service.load()

        result = rent_model_service.predict(request.features.model_dump())
        return RentPredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@router.get("/model-info")
async def get_rent_model_info():
    """Get info about the rent model"""
    return {
        "loaded": rent_model_service.loaded,
        "error": rent_model_service.last_error,
        "model": "rent_random_forest_train5",
    }
