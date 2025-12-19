# Tunisian Real Estate Price Prediction Platform 🏠

An intelligent end-to-end ML platform that predicts fair prices for Tunisian rental and sale properties, helping both sellers set competitive prices and buyers identify good opportunities through real-time market analysis.

## 🎯 Project Vision

Build a web platform powered by AI that:
- **For Sellers/Landlords**: Provides optimal pricing recommendations based on property features and market data
- **For Buyers/Renters**: Evaluates if a property price is fair and discovers similar opportunities
- **For Everyone**: Offers real-time market insights from live-scraped Tunisian real estate listings

## 🏗️ Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│   Frontend      │─────▶│   Backend       │─────▶│   ML Models     │
│   (React)       │      │   (FastAPI)     │      │   (MLflow)      │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                │                           │
                                │                           │
                         ┌──────▼──────┐           ┌───────▼────────┐
                         │             │           │                │
                         │  Scraping   │           │  Training      │
                         │  Module     │           │  Pipeline      │
                         │             │           │                │
                         └─────────────┘           └────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 20+
- Git

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ML-project
```

### 2. Environment Setup

**Backend:**
```bash
cd back
cp .env.example .env
# Edit .env with your configurations
```

**Frontend:**
```bash
cd front
cp .env.example .env
# Set VITE_API_URL=http://localhost:8000
```

**ML Pipeline:**
```bash
cd ML
cp .env.example .env
```

### 3. Run with Docker Compose

```bash
# From project root
docker-compose up -d
```

This will start:
- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **PostgreSQL**: localhost:5432

### 4. Manual Setup (Development)

**Backend:**
```bash
cd back
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd front
npm install
npm run dev
```

**ML Training:**
```bash
cd ML
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

## 📊 Data Setup

### Option 1: Kaggle Dataset
1. Download Tunisian real estate dataset from Kaggle
2. Place CSV file in `ML/data/raw/tunisia_real_estate.csv`

### Option 2: Web Scraping
```bash
# Start backend first, then trigger scraping via API
curl -X POST http://localhost:8000/api/v1/scraper/scrape \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 10}'
```

## 🤖 ML Pipeline

### Training Models
```bash
cd ML
python train.py
```

This will:
- Load and preprocess data
- Apply clustering-based missing value imputation
- Engineer features
- Train multiple models (Linear Regression, Random Forest, XGBoost, LightGBM)
- Log experiments to MLflow
- Save best models

### View Experiments
Open MLflow UI: http://localhost:5000

### Retraining
```bash
python retrain.py
```

Automatically checks for new scraped data and retrains models if needed.

## 🎨 Frontend Features

- **Role Selection**: Choose between Seller or Buyer
- **Transaction Type**: Select Rent or Sale
- **Property Form**: Enter comprehensive property details
- **Price Prediction**: Get AI-powered price estimates
- **Deal Assessment**: For buyers, evaluate if a price is fair
- **Recommendations**: View 5 similar properties using KNN

## 🔌 API Endpoints

### Prediction
```
POST /api/v1/prediction/predict
Body: {
  "user_role": "seller",
  "property_features": {
    "governorate": "Tunis",
    "city": "La Marsa",
    "property_type": "apartment",
    "transaction_type": "sale",
    "area": 120,
    "rooms": 4,
    "bedrooms": 3,
    ...
  }
}
```

### Recommendations
```
POST /api/v1/recommendations/similar
Body: {
  "property_features": {...},
  "n_recommendations": 5
}
```

### Scraping
```
POST /api/v1/scraper/scrape
Body: {
  "governorates": ["Tunis", "Sfax"],
  "transaction_type": "sale",
  "max_pages": 10
}
```

## 🛠️ Technology Stack

### Frontend
- **React 18** with Vite
- **TailwindCSS** for styling
- **React Router** for navigation
- **React Hook Form** for form handling
- **TanStack Query** for data fetching
- **Axios** for API calls

### Backend
- **FastAPI** - High-performance Python web framework
- **Pydantic** - Data validation
- **Beautiful Soup** + **Selenium** - Web scraping
- **SQLAlchemy** - Database ORM
- **PostgreSQL** - Database

### ML Pipeline
- **scikit-learn** - Core ML library
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **MLflow** - Experiment tracking
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Frontend web server
- **Azure** - Cloud deployment

## 📁 Project Structure

```
ML-project/
├── back/                      # Backend (FastAPI)
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Configuration
│   │   ├── ml/               # ML model serving
│   │   ├── models/           # Pydantic schemas
│   │   └── scraping/         # Web scrapers
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── front/                     # Frontend (React)
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   └── services/         # API services
│   ├── package.json
│   └── Dockerfile
│
├── ML/                        # ML Pipeline
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── model_evaluation.py
│   ├── train.py
│   ├── retrain.py
│   └── requirements.txt
│
├── docker-compose.yml
└── README.md
```

## 🔄 Workflow

### For Sellers
1. Select "Seller" role
2. Choose transaction type (Rent/Sale)
3. Fill property details
4. **Get predicted optimal price**
5. View similar properties for comparison

### For Buyers
1. Select "Buyer" role
2. Choose transaction type
3. Fill property details + found price
4. **Get price evaluation** (Good Deal or Not)
5. View similar alternative properties

## 🚢 Deployment to Azure

### Using Azure VM

1. **Create Azure VM**
```bash
az vm create \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys
```

2. **Install Docker on VM**
```bash
ssh azureuser@<vm-ip>
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

3. **Deploy Application**
```bash
git clone <your-repo>
cd ML-project
docker-compose up -d
```

4. **Configure Firewall**
- Open ports: 80 (frontend), 8000 (backend), 5000 (MLflow)

## 📈 Performance Monitoring

- **MLflow**: Track model performance over time
- **Backend Logs**: Monitor API usage
- **Scraping Stats**: Track data collection

## 🧪 Testing

```bash
# Backend tests
cd back
pytest

# Frontend tests
cd front
npm test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Kaggle for Tunisian housing dataset
- Tunisian real estate platforms (Tayara, Mubawab)
- MLflow for experiment tracking
- FastAPI for the excellent framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ for the Tunisian real estate market
