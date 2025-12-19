"""
Pydantic Models for Request/Response Validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


class TransactionType(str, Enum):
    """Transaction type enum"""
    RENT = "rent"
    SALE = "sale"


class UserRole(str, Enum):
    """User role enum"""
    SELLER = "seller"
    BUYER = "buyer"


class PropertyFeatures(BaseModel):
    """Property features for prediction"""
    
    # Location
    governorate: str = Field(..., description="Tunisian governorate/region")
    city: str = Field(..., description="City name")
    delegation: Optional[str] = Field(None, description="Delegation/municipality")
    
    # Basic Info
    property_type: str = Field(..., description="apartment, house, villa, etc.")
    transaction_type: TransactionType
    
    # Size & Rooms
    area: float = Field(..., gt=0, description="Area in square meters")
    rooms: Optional[int] = Field(None, ge=0, description="Number of rooms")
    bedrooms: Optional[int] = Field(None, ge=0, description="Number of bedrooms")
    bathrooms: Optional[int] = Field(None, ge=0, description="Number of bathrooms")
    
    # Features
    floor: Optional[int] = Field(None, ge=0, description="Floor number")
    has_elevator: Optional[bool] = None
    has_parking: Optional[bool] = None
    has_garden: Optional[bool] = None
    has_pool: Optional[bool] = None
    is_furnished: Optional[bool] = None
    
    # Condition
    construction_year: Optional[int] = Field(None, ge=1900, le=2030)
    condition: Optional[str] = Field(None, description="new, good, needs_renovation")
    
    # For buyers: their found price
    found_price: Optional[float] = Field(None, gt=0, description="Price found by buyer")


class PredictionRequest(BaseModel):
    """Request for price prediction"""
    user_role: UserRole
    property_features: PropertyFeatures


class PredictionResponse(BaseModel):
    """Response from price prediction"""
    predicted_price: float
    confidence_interval: dict[str, float]
    is_good_deal: Optional[bool] = None  # For buyers
    price_difference: Optional[float] = None  # Difference between found and predicted
    price_difference_percentage: Optional[float] = None
    market_insights: dict


class Property(BaseModel):
    """Property listing from scraper"""
    id: Optional[str] = None
    title: str
    price: float
    governorate: str
    city: str
    property_type: str
    transaction_type: TransactionType
    area: Optional[float] = None
    rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    url: str
    image_url: Optional[str] = None
    description: Optional[str] = None
    posted_date: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Request for similar properties"""
    property_features: PropertyFeatures
    n_recommendations: int = Field(5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    """Response with similar properties"""
    similar_properties: List[Property]
    similarity_scores: List[float]


class ScrapeRequest(BaseModel):
    """Request to trigger scraping"""
    governorates: Optional[List[str]] = None
    transaction_type: Optional[TransactionType] = None
    max_pages: int = Field(10, ge=1, le=50)


class ScrapeResponse(BaseModel):
    """Response from scraping operation"""
    status: str
    properties_scraped: int
    timestamp: str
