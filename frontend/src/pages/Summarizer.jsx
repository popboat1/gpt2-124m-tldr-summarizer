import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, FileText, Zap, MessageSquare, CheckCircle } from 'lucide-react'
import RedditFetcher from '../components/RedditFetcher'

export default function Summarizer() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [metrics, setMetrics] = useState({ tokensPerSec: 0 })
  const [inferenceSettings, setInferenceSettings] = useState({ temp: 0.7, topK: 40 })
  
  const [selectedModel, setSelectedModel] = useState('PPO Aligned')

  const handleGenerate = () => {
    setOutputText('Summary will appear here...')
    setMetrics({ tokensPerSec: 64.2 })
  }

  return (
    <>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex-grow flex flex-col md:flex-row w-full gap-gutter h-full min-h-[600px] pt-8 pb-4"
      >
        {/* Left Panel: Subreddit Buckets */}
        <aside className="w-full md:w-64 flex flex-col gap-md border-r border-outline-variant pr-md flex-shrink-0 h-full overflow-y-auto">
          <div className="font-headline-md text-headline-md text-on-background flex justify-between items-center mb-sm">
            Data Buckets
          </div>
          
          <RedditFetcher onFetch={(text) => setInputText(text)} />
          
          <div className="flex flex-col gap-unit">
            {['r/relationships', 'r/tifu', 'r/running', 'r/AskReddit'].map((sub, i) => (
              <button 
                key={sub} 
                onClick={async () => {
                  try {
                    const res = await fetch(`/api/reddit/${sub.replace('r/', '')}`)
                    const data = await res.json()
                    if (data.text) setInputText(data.text)
                  } catch (e) {
                    console.error(e)
                  }
                }}
                className={`text-left px-sm py-sm rounded border transition-colors cursor-pointer ${i === 0 ? 'bg-surface-container-low border-outline-variant text-primary font-bold' : 'hover:bg-surface-container-lowest border-transparent hover:border-outline-variant text-on-surface-variant'}`}
              >
                <span className="font-mono-label text-mono-label">{sub}</span>
              </button>
            ))}
          </div>
          <div className="mt-lg border-t border-outline-variant pt-md flex flex-col gap-sm pb-8">
            <span className="font-mono-label text-mono-label text-secondary uppercase tracking-wider">Samples</span>
            {/* Prompt Card 1 */}
            <div 
              onClick={() => setInputText("My (28F) boyfriend (30M) refuses to eat vegetables. It's starting to affect his health and we fight about it constantly. How do I approach this without sounding like his mother?")}
              className="p-sm bg-surface-container-lowest border border-outline-variant rounded cursor-pointer hover:bg-surface-container-low transition-colors shadow-stroke"
            >
              <p className="font-body-md text-body-md text-on-surface line-clamp-3">
                My (28F) boyfriend (30M) refuses to eat vegetables. It's starting to affect his health and we fight about it constantly. How do I approach this without sounding like his mother?
              </p>
              <div className="mt-unit flex justify-between items-center text-secondary">
                <span className="font-mono-label text-mono-label text-[11px]">ID: REL-089</span>
                <MessageSquare size={16} />
              </div>
            </div>
            {/* Prompt Card 2 */}
            <div 
              onClick={() => setInputText("I found out my sister has been secretly reading my diary. I confronted her and she denied it, but I set a trap. Now things are incredibly tense at home.")}
              className="p-sm bg-surface-container-lowest border border-outline-variant rounded cursor-pointer hover:bg-surface-container-low transition-colors shadow-stroke opacity-70 hover:opacity-100"
            >
              <p className="font-body-md text-body-md text-on-surface line-clamp-3">
                I found out my sister has been secretly reading my diary. I confronted her and she denied it, but I set a trap. Now things are incredibly tense at home.
              </p>
              <div className="mt-unit flex justify-between items-center text-secondary">
                <span className="font-mono-label text-mono-label text-[11px]">ID: REL-102</span>
                <MessageSquare size={16} />
              </div>
            </div>
          </div>
        </aside>

        {/* Center & Right Panel Container (Flexible area) */}
        <div className="flex-grow flex flex-col h-full gap-lg overflow-hidden">
          {/* Source Input Section */}
          <section className="flex flex-col gap-sm h-1/3 flex-shrink-0">
            <div className="flex justify-between items-end">
              <h2 className="font-headline-md text-headline-md text-on-background flex items-center gap-2">
                <FileText size={20} />
                Source Input
              </h2>
              <span className="font-mono-label text-mono-label text-secondary px-sm py-unit bg-[#F0F0EB] rounded">
                Length: {inputText.length} chars
              </span>
            </div>
            <div className="flex-grow relative">
              <textarea 
                aria-label="Input Document"
                className="w-full h-full p-md bg-surface-container-lowest border border-outline-variant rounded font-body-lg text-body-lg text-on-surface resize-none focus:outline-none focus:border-[#C27D38] focus:ring-0 transition-colors shadow-stroke" 
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to summarize..."
              />
            </div>
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-md">
                <div className="flex bg-surface-container-low rounded-lg p-1 border border-outline-variant">
                  <button 
                    onClick={() => setSelectedModel('PPO Aligned')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer ${selectedModel === 'PPO Aligned' ? 'bg-primary shadow-sm text-on-primary border-transparent' : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high'}`}
                  >
                    PPO Aligned
                  </button>
                  <button 
                    onClick={() => setSelectedModel('SFT Baseline')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer ${selectedModel === 'SFT Baseline' ? 'bg-primary shadow-sm text-on-primary border-transparent' : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high'}`}
                  >
                    SFT Baseline
                  </button>
                </div>
                
                <div className="flex items-center gap-sm">
                  <label className="text-body-md text-on-surface-variant whitespace-nowrap">Temp: {inferenceSettings.temp}</label>
                  <input type="range" min="0" max="2" step="0.1" value={inferenceSettings.temp} onChange={(e) => setInferenceSettings({...inferenceSettings, temp: parseFloat(e.target.value)})} className="w-24 accent-primary" />
                  
                  <label className="text-body-md text-on-surface-variant whitespace-nowrap ml-sm">Top K: {inferenceSettings.topK}</label>
                  <input type="range" min="1" max="100" value={inferenceSettings.topK} onChange={(e) => setInferenceSettings({...inferenceSettings, topK: parseInt(e.target.value)})} className="w-24 accent-primary" />
                </div>
              </div>
              <button 
                onClick={handleGenerate}
                className="bg-primary text-on-primary px-lg py-sm rounded hover:bg-surface-tint transition-colors font-body-md text-body-md flex items-center gap-sm cursor-pointer"
              >
                <Zap size={18} />
                Summarize
              </button>
            </div>
          </section>

          {/* Comparison/Output Section */}
          <section className="flex-grow flex flex-col gap-gutter h-2/3 overflow-hidden pb-4">
            <div className={`flex-1 flex flex-col border rounded bg-surface-container-lowest overflow-hidden shadow-stroke relative ${selectedModel === 'PPO Aligned' ? 'border-[#C27D38]' : 'border-outline-variant'}`}>
              <div className="bg-surface-container-low px-md py-sm border-b border-outline-variant flex justify-between items-center">
                <div className="flex items-center gap-xs">
                  <span className={`font-mono-label text-mono-label font-bold ${selectedModel === 'PPO Aligned' ? 'text-primary' : 'text-on-surface'}`}>
                    {selectedModel}
                  </span>
                  {selectedModel === 'PPO Aligned' && <CheckCircle className="text-primary" size={14} />}
                </div>
                <span className="font-mono-label text-mono-label text-secondary text-[11px]">
                  {selectedModel === 'PPO Aligned' ? 'Reward Model: Helpful+Harmless' : `Temperature: ${inferenceSettings.temp}`}
                </span>
              </div>
              <div 
                role="region"
                aria-label="Summary Output"
                className="p-md flex-grow overflow-y-auto font-body-md text-body-md text-on-surface leading-relaxed whitespace-pre-wrap"
              >
                {outputText || <span className="text-on-surface-variant italic">Summary will appear here...</span>}
              </div>

              {/* Inline Metrics Dashboard */}
              <div className="bg-surface-container-low border-t border-outline-variant py-sm px-md flex flex-wrap justify-between items-center text-[12px] font-mono-metric">
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wider opacity-70">Velocity:</span>
                  <span className="text-primary font-medium">{metrics.tokensPerSec ? `${metrics.tokensPerSec.toFixed(1)} t/s` : '0.0 t/s'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wider opacity-70">Time:</span>
                  <span className="font-medium">1.42s</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wider opacity-70">Length:</span>
                  <span className="font-medium">128</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wider opacity-70">Drift:</span>
                  <span className="text-tertiary font-medium">0.045 KL</span>
                </div>
              </div>
            </div>
          </section>
        </div>
      </motion.div>
    </>
  )
}
