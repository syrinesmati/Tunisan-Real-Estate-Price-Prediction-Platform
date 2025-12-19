import { ExternalLink, MapPin, Home, Maximize } from 'lucide-react'

export default function PropertyCard({ property, similarity }) {
  const formatPrice = (price) => {
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
    }).format(price)
  }

  const similarityPercentage = similarity ? (similarity * 100).toFixed(0) : null

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      {/* Image Placeholder */}
      <div className="h-48 bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center">
        <Home className="w-16 h-16 text-white opacity-50" />
      </div>

      <div className="p-4">
        {/* Similarity Badge */}
        {similarityPercentage && (
          <div className="inline-block bg-primary-100 text-primary-700 px-2 py-1 rounded text-xs font-semibold mb-2">
            {similarityPercentage}% Match
          </div>
        )}

        {/* Title */}
        <h3 className="font-bold text-lg mb-2 line-clamp-2">{property.title}</h3>

        {/* Price */}
        <div className="text-2xl font-bold text-primary-600 mb-3">
          {formatPrice(property.price)}
        </div>

        {/* Details */}
        <div className="space-y-2 text-sm text-gray-600">
          <div className="flex items-center space-x-2">
            <MapPin className="w-4 h-4" />
            <span>
              {property.city}, {property.governorate}
            </span>
          </div>

          {property.area && (
            <div className="flex items-center space-x-2">
              <Maximize className="w-4 h-4" />
              <span>{property.area} m²</span>
            </div>
          )}

          {property.bedrooms && (
            <div className="flex items-center space-x-2">
              <Home className="w-4 h-4" />
              <span>
                {property.bedrooms} bedroom{property.bedrooms > 1 ? 's' : ''}
              </span>
            </div>
          )}
        </div>

        {/* Link */}
        {property.url && (
          <a
            href={property.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 flex items-center justify-center space-x-2 text-primary-600 hover:text-primary-700 font-medium"
          >
            <span>View Listing</span>
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>
    </div>
  )
}
