import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

const api = {
  // Health check
  healthCheck: () => apiClient.get('/health'),

  // Prediction
  predictPrice: (data) => apiClient.post('/api/v1/prediction/predict', data),
  
  getModelInfo: () => apiClient.get('/api/v1/prediction/model-info'),

  // Recommendations
  getRecommendations: (data) =>
    apiClient.post('/api/v1/recommendations/similar', data),
  
  getRecommendationStats: () => apiClient.get('/api/v1/recommendations/stats'),

  // Scraper
  triggerScraping: (data) => apiClient.post('/api/v1/scraper/scrape', data),
  
  getScrapingStatus: () => apiClient.get('/api/v1/scraper/status'),
  
  getScrapedDataStats: () => apiClient.get('/api/v1/scraper/data-stats'),
}

export default api
