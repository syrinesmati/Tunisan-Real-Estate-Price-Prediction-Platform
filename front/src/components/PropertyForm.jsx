import { useForm } from 'react-hook-form'

const tunisianGovernorates = [
  'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan', 'Bizerte',
  'Béja', 'Jendouba', 'Kef', 'Siliana', 'Sousse', 'Monastir', 'Mahdia',
  'Sfax', 'Kairouan', 'Kasserine', 'Sidi Bouzid', 'Gabès', 'Medenine',
  'Tataouine', 'Gafsa', 'Tozeur', 'Kebili'
]

const propertyTypes = [
  'apartment', 'house', 'villa', 'studio', 'duplex', 'land', 'commercial'
]

export default function PropertyForm({ onSubmit, isLoading, userRole }) {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm()

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="card space-y-6">
      <h2 className="text-xl font-bold">Property Details</h2>

      {/* Location */}
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Governorate <span className="text-red-500">*</span>
          </label>
          <select
            {...register('governorate', { required: 'Governorate is required' })}
            className="input"
          >
            <option value="">Select governorate</option>
            {tunisianGovernorates.map((gov) => (
              <option key={gov} value={gov}>
                {gov}
              </option>
            ))}
          </select>
          {errors.governorate && (
            <p className="text-red-500 text-sm mt-1">{errors.governorate.message}</p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            City <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            {...register('city', { required: 'City is required' })}
            className="input"
            placeholder="Enter city"
          />
          {errors.city && (
            <p className="text-red-500 text-sm mt-1">{errors.city.message}</p>
          )}
        </div>
      </div>

      {/* Property Type & Area */}
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Property Type <span className="text-red-500">*</span>
          </label>
          <select
            {...register('property_type', { required: 'Property type is required' })}
            className="input"
          >
            <option value="">Select type</option>
            {propertyTypes.map((type) => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </option>
            ))}
          </select>
          {errors.property_type && (
            <p className="text-red-500 text-sm mt-1">{errors.property_type.message}</p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Area (m²) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            step="0.1"
            {...register('area', {
              required: 'Area is required',
              min: { value: 1, message: 'Area must be positive' },
            })}
            className="input"
            placeholder="e.g., 120"
          />
          {errors.area && (
            <p className="text-red-500 text-sm mt-1">{errors.area.message}</p>
          )}
        </div>
      </div>

      {/* Rooms */}
      <div className="grid md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Total Rooms</label>
          <input
            type="number"
            {...register('rooms', { min: 0 })}
            className="input"
            placeholder="e.g., 4"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Bedrooms</label>
          <input
            type="number"
            {...register('bedrooms', { min: 0 })}
            className="input"
            placeholder="e.g., 2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Bathrooms</label>
          <input
            type="number"
            {...register('bathrooms', { min: 0 })}
            className="input"
            placeholder="e.g., 1"
          />
        </div>
      </div>

      {/* Additional Details */}
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Floor Number</label>
          <input
            type="number"
            {...register('floor', { min: 0 })}
            className="input"
            placeholder="e.g., 3"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Construction Year</label>
          <input
            type="number"
            {...register('construction_year', { min: 1900, max: 2030 })}
            className="input"
            placeholder="e.g., 2020"
          />
        </div>
      </div>

      {/* Features Checkboxes */}
      <div>
        <label className="block text-sm font-medium mb-3">Features</label>
        <div className="grid md:grid-cols-3 gap-4">
          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_elevator')} className="rounded" />
            <span>Elevator</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_parking')} className="rounded" />
            <span>Parking</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_garden')} className="rounded" />
            <span>Garden</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_pool')} className="rounded" />
            <span>Pool</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('is_furnished')} className="rounded" />
            <span>Furnished</span>
          </label>
        </div>
      </div>

      {/* For Buyers: Found Price */}
      {userRole === 'buyer' && (
        <div>
          <label className="block text-sm font-medium mb-1">
            Price You Found (TND)
          </label>
          <input
            type="number"
            step="0.01"
            {...register('found_price', { min: 0 })}
            className="input"
            placeholder="Enter the price you found"
          />
          <p className="text-sm text-gray-500 mt-1">
            We'll compare this with our prediction to assess if it's a good deal
          </p>
        </div>
      )}

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading}
        className="w-full btn btn-primary py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? 'Predicting...' : 'Get Price Prediction'}
      </button>
    </form>
  )
}
