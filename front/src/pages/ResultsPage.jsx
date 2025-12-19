import { useLocation, Link } from 'react-router-dom'
import { CheckCircle, XCircle, TrendingUp, Home } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'
import PropertyCard from '../components/PropertyCard'

export default function ResultsPage() {
  const location = useLocation()
  const { prediction, propertyData, userRole } = location.state || {}

  // Fetch similar properties
  const { data: recommendations, isLoading: recsLoading } = useQuery({
    queryKey: ['recommendations', propertyData],
    queryFn: () =>
      api.getRecommendations({
        property_features: propertyData,
        n_recommendations: 5,
      }),
    enabled: !!propertyData,
  })

  if (!prediction) {
    return (
      <div className="text-center py-16">
        <p className="text-gray-600 mb-4">No prediction data available</p>
        <Link to="/predict" className="text-primary-600 hover:underline">
          Go to Prediction Page
        </Link>
      </div>
    )
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
    }).format(price)
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Prediction Result */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Prediction Results</h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-4">Predicted Price</h3>
            <div className="bg-primary-50 p-6 rounded-lg">
              <div className="text-4xl font-bold text-primary-600">
                {formatPrice(prediction.predicted_price)}
              </div>
              <div className="text-sm text-gray-600 mt-2">
                Range: {formatPrice(prediction.confidence_interval.lower)} -{' '}
                {formatPrice(prediction.confidence_interval.upper)}
              </div>
            </div>
          </div>

          {userRole === 'buyer' && prediction.is_good_deal !== undefined && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Deal Assessment</h3>
              <div
                className={`p-6 rounded-lg ${
                  prediction.is_good_deal
                    ? 'bg-green-50 border-2 border-green-200'
                    : 'bg-orange-50 border-2 border-orange-200'
                }`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  {prediction.is_good_deal ? (
                    <CheckCircle className="w-8 h-8 text-green-600" />
                  ) : (
                    <XCircle className="w-8 h-8 text-orange-600" />
                  )}
                  <span className="text-2xl font-bold">
                    {prediction.is_good_deal ? 'Good Deal!' : 'Consider Negotiating'}
                  </span>
                </div>
                <div className="text-sm">
                  Price difference:{' '}
                  <span className="font-semibold">
                    {formatPrice(prediction.price_difference)} (
                    {prediction.price_difference_percentage.toFixed(1)}%)
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Market Insights */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold mb-3">Market Insights</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Price per m²:</span>
              <div className="font-semibold">
                {formatPrice(prediction.market_insights.price_per_sqm)}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Location:</span>
              <div className="font-semibold">{prediction.market_insights.location}</div>
            </div>
            <div>
              <span className="text-gray-600">Type:</span>
              <div className="font-semibold">{prediction.market_insights.property_type}</div>
            </div>
            <div>
              <span className="text-gray-600">Premium Features:</span>
              <div className="font-semibold">
                {prediction.market_insights.premium_features_count}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Similar Properties */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Similar Properties</h2>

        {recsLoading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
            <p className="text-gray-600 mt-4">Finding similar properties...</p>
          </div>
        ) : recommendations?.data?.similar_properties?.length > 0 ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.data.similar_properties.map((property, index) => (
              <PropertyCard
                key={property.id || index}
                property={property}
                similarity={recommendations.data.similarity_scores[index]}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-600">
            <p>No similar properties found at the moment.</p>
            <p className="text-sm mt-2">
              Try scraping more data or adjusting your search criteria.
            </p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex justify-center space-x-4">
        <Link to="/predict" className="btn btn-primary">
          New Prediction
        </Link>
        <Link to="/" className="btn btn-secondary">
          Back to Home
        </Link>
      </div>
    </div>
  )
}
