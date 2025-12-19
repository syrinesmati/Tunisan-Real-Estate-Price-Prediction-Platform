"""
Recommendations API Endpoints
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse
)
from app.ml.recommender import PropertyRecommender

router = APIRouter()
recommender = PropertyRecommender()


@router.post("/similar", response_model=RecommendationResponse)
async def get_similar_properties(request: RecommendationRequest):
    """
    Find similar properties using KNN
    
    - **For Sellers**: Show similar listings to compare
    - **For Buyers**: Discover alternative options
    """
    try:
        recommendations = recommender.find_similar(
            property_features=request.property_features,
            n_recommendations=request.n_recommendations
        )
        
        return RecommendationResponse(
            similar_properties=recommendations["properties"],
            similarity_scores=recommendations["scores"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {str(e)}"
        )


@router.get("/stats")
async def get_recommendation_stats():
    """Get statistics about available properties for recommendations"""
    try:
        stats = recommender.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
