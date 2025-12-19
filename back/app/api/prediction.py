"""
Prediction API Endpoints
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    UserRole
)
from app.ml.predictor import PricePredictor
from app.ml.model_manager import model_manager

router = APIRouter()
predictor = PricePredictor()


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict property price based on features
    
    - **For Sellers**: Returns predicted optimal price
    - **For Buyers**: Compares found price with prediction
    """
    try:
        # Get appropriate model based on transaction type
        model = model_manager.get_model(request.property_features.transaction_type.value)
        
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model for {request.property_features.transaction_type.value} not available"
            )
        
        # Make prediction
        prediction_result = predictor.predict(
            model=model,
            features=request.property_features
        )
        
        # Build response
        response_data = {
            "predicted_price": prediction_result["predicted_price"],
            "confidence_interval": prediction_result["confidence_interval"],
            "market_insights": prediction_result["insights"]
        }
        
        # For buyers: add price comparison
        if request.user_role == UserRole.BUYER and request.property_features.found_price:
            found_price = request.property_features.found_price
            predicted_price = prediction_result["predicted_price"]
            
            price_diff = found_price - predicted_price
            price_diff_pct = (price_diff / predicted_price) * 100
            
            # Good deal if found price is <= 10% below predicted
            is_good_deal = price_diff_pct <= 10
            
            response_data.update({
                "is_good_deal": is_good_deal,
                "price_difference": price_diff,
                "price_difference_percentage": price_diff_pct
            })
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "models": model_manager.get_models_info(),
        "status": "loaded" if model_manager.is_loaded() else "not_loaded"
    }
