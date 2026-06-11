import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'

// Placeholders for now
const Landing = () => <div>Landing Page</div>
const Summarizer = () => <div>Summarizer Page</div>

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
