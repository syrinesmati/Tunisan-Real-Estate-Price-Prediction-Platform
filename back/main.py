"""
Main FastAPI Application Entry Point
Tunisian Real Estate Price Prediction Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import prediction, recommendations, scraper
from app.core.config import settings
from app.ml.model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Load ML models
    print("🚀 Loading ML models...")
    model_manager.load_models()
    print("✅ Models loaded successfully!")
    
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
app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["Prediction"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(scraper.router, prefix="/api/v1/scraper", tags=["Scraper"])


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
        "models_loaded": model_manager.is_loaded(),
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
