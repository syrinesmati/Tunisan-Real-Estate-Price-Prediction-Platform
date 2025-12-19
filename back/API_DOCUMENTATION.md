# Backend API Documentation

## Overview
FastAPI-based REST API for Tunisian Real Estate Price Prediction platform.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required (add JWT if needed).

## Endpoints

### Health Check

#### GET `/`
Basic health check

**Response:**
```json
{
  "status": "online",
  "message": "Tunisian Real Estate Price Prediction API",
  "version": "1.0.0"
}
```

#### GET `/health`
Detailed health check

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "api_version": "1.0.0"
}
```

---

### Prediction Endpoints

#### POST `/api/v1/prediction/predict`
Predict property price based on features.

**Request Body:**
```json
{
  "user_role": "seller" | "buyer",
  "property_features": {
    "governorate": "Tunis",
    "city": "La Marsa",
    "delegation": "La Marsa",
    "property_type": "apartment" | "house" | "villa" | "studio" | "duplex",
    "transaction_type": "sale" | "rent",
    "area": 120.5,
    "rooms": 4,
    "bedrooms": 3,
    "bathrooms": 2,
    "floor": 3,
    "has_elevator": true,
    "has_parking": true,
    "has_garden": false,
    "has_pool": false,
    "is_furnished": true,
    "construction_year": 2020,
    "condition": "new" | "good" | "needs_renovation",
    "found_price": 250000  // Required for buyers
  }
}
```

**Response:**
```json
{
  "predicted_price": 245000.50,
  "confidence_interval": {
    "lower": 230000.00,
    "upper": 260000.00
  },
  "is_good_deal": true,  // Only for buyers
  "price_difference": -5000.50,  // Only for buyers
  "price_difference_percentage": -2.04,  // Only for buyers
  "market_insights": {
    "price_per_sqm": 2033.75,
    "location": "La Marsa, Tunis",
    "property_type": "apartment",
    "transaction_type": "sale",
    "premium_features_count": 3
  }
}
```

#### GET `/api/v1/prediction/model-info`
Get information about loaded models.

**Response:**
```json
{
  "models": {
    "rent": {
      "loaded": true,
      "path": "./models/rent_model"
    },
    "sale": {
      "loaded": true,
      "path": "./models/sale_model"
    }
  },
  "status": "loaded"
}
```

---

### Recommendation Endpoints

#### POST `/api/v1/recommendations/similar`
Find similar properties using KNN.

**Request Body:**
```json
{
  "property_features": {
    // Same as prediction request
  },
  "n_recommendations": 5
}
```

**Response:**
```json
{
  "similar_properties": [
    {
      "id": "prop_123",
      "title": "Appartement S+3 à La Marsa",
      "price": 240000,
      "governorate": "Tunis",
      "city": "La Marsa",
      "property_type": "apartment",
      "transaction_type": "sale",
      "area": 115,
      "rooms": 4,
      "bedrooms": 3,
      "url": "https://...",
      "image_url": "https://...",
      "description": "...",
      "posted_date": "2024-01-15"
    }
  ],
  "similarity_scores": [0.95, 0.92, 0.89, 0.87, 0.85]
}
```

#### GET `/api/v1/recommendations/stats`
Get statistics about available properties.

**Response:**
```json
{
  "total_properties": 1250,
  "by_type": {
    "apartment": 800,
    "house": 300,
    "villa": 150
  },
  "by_governorate": {
    "Tunis": 500,
    "Sfax": 300,
    "Sousse": 250
  },
  "status": "ready"
}
```

---

### Scraper Endpoints

#### POST `/api/v1/scraper/scrape`
Trigger web scraping (runs in background).

**Request Body:**
```json
{
  "governorates": ["Tunis", "Sfax"],  // Optional
  "transaction_type": "sale" | "rent",  // Optional
  "max_pages": 10
}
```

**Response:**
```json
{
  "status": "scraping_started",
  "properties_scraped": 0,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### GET `/api/v1/scraper/status`
Get scraping status.

**Response:**
```json
{
  "is_scraping": false,
  "last_scrape": "2024-01-15T10:30:00",
  "total_scraped": 150
}
```

#### GET `/api/v1/scraper/data-stats`
Get statistics about scraped data.

**Response:**
```json
{
  "total_properties": 150,
  "by_transaction_type": {
    "sale": 100,
    "rent": 50
  },
  "by_governorate": {
    "Tunis": 80,
    "Sfax": 70
  },
  "last_updated": "2024-01-15T10:30:00"
}
```

---

## Error Responses

All endpoints return standard error responses:

**400 Bad Request:**
```json
{
  "detail": "Invalid request parameters"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error message"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Model for sale not available"
}
```

---

## Rate Limiting
Currently no rate limiting (implement if needed for production).

## CORS
Configured to allow requests from:
- http://localhost:5173 (Vite dev server)
- http://localhost:3000
- Your production domain

---

## Development

### Run API Locally
```bash
cd back
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Environment Variables
See `.env.example` for required configurations.
