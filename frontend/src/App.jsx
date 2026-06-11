import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Summarizer from './pages/Summarizer'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/summarizer" element={<Summarizer />} />
      </Routes>
    </Layout>
  )
}

export default App
