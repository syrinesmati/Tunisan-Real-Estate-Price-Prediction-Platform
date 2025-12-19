"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "Tunisian Real Estate API"
    DEBUG: bool = True
    API_VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "tunisian_real_estate_prediction"
    
    # Database
    DATABASE_URL: str = "sqlite:///./real_estate.db"
    
    # Model Paths
    RENT_MODEL_PATH: str = "./models/rent_model"
    SALE_MODEL_PATH: str = "./models/sale_model"
    
    # Scraping
    SCRAPING_USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    SCRAPING_DELAY: int = 2
    MAX_SCRAPING_PAGES: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
