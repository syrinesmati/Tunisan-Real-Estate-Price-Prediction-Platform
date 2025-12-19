import { Link } from 'react-router-dom'
import { Home, TrendingUp } from 'lucide-react'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <nav className="bg-white shadow-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center space-x-2">
              <Home className="w-6 h-6 text-primary-600" />
              <span className="text-xl font-bold text-gray-800">
                Tunisian Real Estate Predictor
              </span>
            </Link>
            <div className="flex items-center space-x-4">
              <Link
                to="/predict"
                className="flex items-center space-x-1 text-gray-700 hover:text-primary-600"
              >
                <TrendingUp className="w-5 h-5" />
                <span>Get Prediction</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-8">
        {children}
      </main>

      <footer className="bg-white border-t mt-16">
        <div className="container mx-auto px-4 py-6 text-center text-gray-600">
          <p>&copy; 2025 Tunisian Real Estate Price Predictor. AI-Powered Property Valuation.</p>
        </div>
      </footer>
    </div>
  )
}
