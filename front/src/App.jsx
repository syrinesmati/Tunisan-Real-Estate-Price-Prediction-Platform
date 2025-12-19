import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import PredictionPage from './pages/PredictionPage'
import ResultsPage from './pages/ResultsPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<PredictionPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Layout>
  )
}

export default App
