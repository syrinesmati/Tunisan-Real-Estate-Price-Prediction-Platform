import { Link } from 'react-router-dom'
import { TrendingUp, Search, BarChart3, ShieldCheck } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-16">
        <h1 className="text-5xl font-bold text-gray-900 mb-6">
          Predict Fair Property Prices in Tunisia
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          AI-powered platform helping sellers set competitive prices and buyers find the best deals
          in the Tunisian real estate market.
        </p>
        <Link
          to="/predict"
          className="inline-block bg-primary-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-primary-700 transition-colors"
        >
          Get Started
        </Link>
      </section>

      {/* Features */}
      <section className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-primary-100 p-3 rounded-lg">
              <TrendingUp className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="text-xl font-bold">For Sellers</h3>
          </div>
          <p className="text-gray-600">
            Get AI-predicted optimal pricing for your property based on real market data
            and comparable listings in your area.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-green-100 p-3 rounded-lg">
              <Search className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-xl font-bold">For Buyers</h3>
          </div>
          <p className="text-gray-600">
            Evaluate if a property price is fair and discover similar opportunities
            to make informed purchasing decisions.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-purple-100 p-3 rounded-lg">
              <BarChart3 className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-xl font-bold">Smart Recommendations</h3>
          </div>
          <p className="text-gray-600">
            Discover 5 similar properties from live-scraped listings to compare
            and find the best match for your needs.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-orange-100 p-3 rounded-lg">
              <ShieldCheck className="w-6 h-6 text-orange-600" />
            </div>
            <h3 className="text-xl font-bold">Real-Time Market Data</h3>
          </div>
          <p className="text-gray-600">
            Powered by continuously updated data from major Tunisian real estate
            platforms for accurate predictions.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
        <div className="space-y-6">
          <div className="flex items-start space-x-4">
            <div className="bg-primary-600 text-white w-10 h-10 rounded-full flex items-center justify-center font-bold flex-shrink-0">
              1
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2">Select Your Role</h4>
              <p className="text-gray-600">
                Choose whether you're a seller looking to price your property or a buyer evaluating a deal.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="bg-primary-600 text-white w-10 h-10 rounded-full flex items-center justify-center font-bold flex-shrink-0">
              2
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2">Enter Property Details</h4>
              <p className="text-gray-600">
                Fill in information about location, size, features, and condition.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="bg-primary-600 text-white w-10 h-10 rounded-full flex items-center justify-center font-bold flex-shrink-0">
              3
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2">Get AI Prediction</h4>
              <p className="text-gray-600">
                Receive an accurate price prediction with confidence intervals and market insights.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="bg-primary-600 text-white w-10 h-10 rounded-full flex items-center justify-center font-bold flex-shrink-0">
              4
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2">Discover Similar Properties</h4>
              <p className="text-gray-600">
                View 5 recommended similar properties to compare and make better decisions.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
