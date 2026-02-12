"""
Main FastAPI Application Entry Point
Tunisian Real Estate Price Prediction Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import recommendations, rent_prediction
from app.core.config import settings
from app.ml.rent_model_service import rent_model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Load ML models
    print("🚀 Loading ML models...")

    # Load train5 rent model
    try:
        rent_model_service.load()
        print("✅ Train5 rent model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load train5 rent model: {e}")
    
    
    yield
    
    # Shutdown: Cleanup
    print("👋 Shutting down...")


app = FastAPI(
    title="Tunisian Real Estate Price Prediction API",
    description="AI-powered platform for predicting property prices and finding similar listings",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(rent_prediction.router, prefix="/api/v1/rent", tags=["Rent Model (train5)"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Tunisian Real Estate Price Prediction API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rent_model_loaded": rent_model_service.loaded,
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
