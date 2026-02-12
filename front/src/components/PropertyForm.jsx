import { useEffect, useMemo } from 'react'
import { useForm } from 'react-hook-form'
import cityByRegion from '../data/cityByRegion.json'
import priceSegmentByRegion from '../data/priceSegmentByRegion.json'

const regionOptions = Object.keys(priceSegmentByRegion).sort((a, b) =>
  a.localeCompare(b, 'fr', { sensitivity: 'base' })
)

const propertyTypes = [
  'apartment', 'house', 'villa', 'studio', 'duplex', 'land', 'commercial'
]

const computePriceSegment = (region) => {
  if (region && priceSegmentByRegion[region]) {
    return priceSegmentByRegion[region]
  }

  return 'Medium'
}

export default function PropertyForm({ onSubmit, isLoading, userRole }) {
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
  } = useForm()

  const region = watch('region')
  const priceSegment = useMemo(() => computePriceSegment(region), [region])
  const cityValue = useMemo(() => {
    if (!region) {
      return ''
    }

    return cityByRegion[region] || ''
  }, [region])

  useEffect(() => {
    setValue('price_segment', priceSegment, { shouldValidate: true })
  }, [priceSegment, setValue])

  useEffect(() => {
    setValue('city', cityValue, { shouldValidate: true })
  }, [cityValue, setValue])

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="card space-y-6">
      <h2 className="text-xl font-bold">Property Details</h2>

      {/* Location */}
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Region <span className="text-red-500">*</span>
          </label>
          <select
            {...register('region', { required: 'Region is required' })}
            className="input"
          >
            <option value="">Select region</option>
            {regionOptions.map((regionOption) => (
              <option key={regionOption} value={regionOption}>
                {regionOption}
              </option>
            ))}
          </select>
          {errors.region && (
            <p className="text-red-500 text-sm mt-1">{errors.region.message}</p>
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
            placeholder="City auto-selected"
            value={cityValue}
            readOnly
          />
          {errors.city && (
            <p className="text-red-500 text-sm mt-1">{errors.city.message}</p>
          )}
        </div>
      </div>

      {/* Property Type & Surface */}
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
            Surface (m²) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            step="0.1"
            {...register('surface', {
              required: 'Surface is required',
              min: { value: 1, message: 'Surface must be positive' },
            })}
            className="input"
            placeholder="e.g., 120"
          />
          {errors.surface && (
            <p className="text-red-500 text-sm mt-1">{errors.surface.message}</p>
          )}
        </div>
      </div>

      {/* Rooms */}
      <div className="grid md:grid-cols-2 gap-4">
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
          <label className="block text-sm font-medium mb-1">Bathrooms</label>
          <input
            type="number"
            {...register('bathrooms', { min: 0 })}
            className="input"
            placeholder="e.g., 1"
          />
        </div>
      </div>

      {/* Price Segment (Auto) */}
      <div>
        <label className="block text-sm font-medium mb-1">Price Segment</label>
        <input
          type="text"
          value={priceSegment}
          readOnly
          className="input bg-gray-50 text-gray-600 cursor-not-allowed"
        />
        <input type="hidden" {...register('price_segment')} />
        <p className="text-xs text-gray-500 mt-1">
          Calculated automatically from the selected region.
        </p>
      </div>

      {/* Features Checkboxes */}
      <div>
        <label className="block text-sm font-medium mb-3">Features</label>
        <div className="grid md:grid-cols-3 gap-4">
          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_ascenseur')} className="rounded" />
            <span>Ascenseur</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_garage')} className="rounded" />
            <span>Garage</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_jardin')} className="rounded" />
            <span>Jardin</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_piscine')} className="rounded" />
            <span>Piscine</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('is_meuble')} className="rounded" />
            <span>Meuble</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_terrasse')} className="rounded" />
            <span>Terrasse</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_chauffage')} className="rounded" />
            <span>Chauffage</span>
          </label>

          <label className="flex items-center space-x-2">
            <input type="checkbox" {...register('has_climatisation')} className="rounded" />
            <span>Climatisation</span>
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
