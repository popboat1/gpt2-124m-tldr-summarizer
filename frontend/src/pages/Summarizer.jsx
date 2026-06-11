import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, FileText, Zap } from 'lucide-react'

export default function Summarizer() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [metrics, setMetrics] = useState({ tokensPerSec: 0, temp: 0.8 })

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-4xl mx-auto py-8 flex flex-col gap-8"
    >
      <header className="flex justify-between items-center pb-4 border-b border-gray-800">
        <h2 className="text-2xl font-light">New Summary</h2>
        <button 
          aria-label="Settings"
          className="p-2 text-gray-400 hover:text-white transition-colors rounded-md hover:bg-gray-800"
        >
          <Settings size={20} />
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="flex flex-col gap-2">
          <label htmlFor="input-text" className="text-sm text-gray-500 font-medium flex items-center gap-2">
            <FileText size={16} /> Input Document
          </label>
          <textarea 
            id="input-text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to summarize..."
            className="w-full h-80 bg-gray-900 border border-gray-800 rounded-xl p-4 text-gray-200 focus:outline-none focus:border-gray-600 focus:ring-1 focus:ring-gray-600 transition-all resize-none"
          />
        </div>
        
        <div className="flex flex-col gap-2">
          <label id="output-label" className="text-sm text-gray-500 font-medium flex items-center gap-2">
            <Zap size={16} /> Summary Output
          </label>
          <div 
            role="region"
            tabIndex={0}
            aria-labelledby="output-label"
            className="w-full h-80 bg-gray-900 border border-gray-800 rounded-xl p-4 text-gray-200 overflow-y-auto whitespace-pre-wrap"
          >
            {outputText || <span className="text-gray-600 italic">Summary will appear here...</span>}
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center">
        <button 
          className="px-6 py-2.5 bg-white text-black rounded-lg font-medium hover:bg-gray-200 transition-colors"
        >
          Summarize
        </button>
        <div className="text-sm text-gray-500 flex gap-4">
          <span>{metrics.tokensPerSec.toFixed(1)} tokens/sec</span>
          <span>Temp: {metrics.temp}</span>
        </div>
      </div>
    </motion.div>
  )
}
