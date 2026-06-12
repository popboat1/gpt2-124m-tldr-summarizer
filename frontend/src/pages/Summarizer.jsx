import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, FileText, Zap, MessageSquare, CheckCircle } from 'lucide-react'

const preset_prompts = [
    "I have a roommate who keeps eating my food without asking. Every single time I buy groceries, half of it disappears within 48 hours. I tried talking to him politely but he just laughs it off and says he will replace it, but he never does. Should I get a mini-fridge for my room or confront him one last time more aggressively?",
    "My boss asked me to work over the weekend for a major client launch, but it is completely short notice and I already booked non-refundable tickets to visit my parents. If I say no, I am worried it will impact my performance review next month, but if I go, I disappoint my family. How do I navigate this professionally?",
    "I am trying to decide whether to buy a new laptop for college or stick with my desktop setup. The laptop would let me work from the library and study groups, but my desktop is vastly more powerful and already paid for. My budget is tight and I would have to take out a small student loan to afford the laptop.",
    "Yesterday I went to a local coffee shop and paid for a $5 latte with a $20 bill. The cashier accidentally gave me change for a $50 bill instead. I realized the mistake only after I walked back to my car. Part of me wants to go back and return the extra cash, but another part feels like it was their operational error.",
    "Our landlord just announced a 15% rent increase starting next month, citing market adjustments. The apartment has several maintenance issues that have been completely ignored for six months, like a leaking kitchen pipe and a broken window latch. Is it worth fighting this legally or should I look for a new place?"
]

export default function Summarizer() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [metrics, setMetrics] = useState({ tokensPerSec: 0, time: 0, length: 0 })
  const [topTokens, setTopTokens] = useState([])
  const [inferenceSettings, setInferenceSettings] = useState({ temp: 0.5, topK: 40, numProbs: 5 })
  
  const [selectedModel, setSelectedModel] = useState('PPO Aligned')

  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setIsGenerating(true)
    setOutputText('Generating summary...')
    setMetrics({ tokensPerSec: 0, time: 0 })
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || '/api/generate';
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          model: selectedModel,
          temperature: inferenceSettings.temp,
          top_k: inferenceSettings.topK,
          num_probs: inferenceSettings.numProbs
        })
      })
      
      if (!response.ok) {
        setOutputText('Error generating summary.')
        setIsGenerating(false)
        return
      }

      setOutputText('')
      let fullText = ''
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const data = JSON.parse(dataStr);
              fullText += data.text;
              setOutputText(fullText);
              setMetrics({ tokensPerSec: data.tps, time: data.time, length: data.length });
              if (data.top_tokens) setTopTokens(data.top_tokens);
            } catch (err) {
              console.error("Error parsing JSON", err);
            }
          }
        }
      }
    } catch (e) {
      setOutputText('Failed to connect to backend.', e)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex-grow flex flex-col md:flex-row w-full gap-gutter h-full min-h-[600px] pt-8 pb-4"
      >
        {/* Left Panel: Sample Buckets */}
        <aside className="w-full md:w-64 flex flex-col gap-md border-r border-outline-variant pr-md flex-shrink-0 h-full overflow-y-auto">
          <div className="font-headline-md text-headline-md text-on-background flex justify-between items-center mb-sm">
            Sample Scenarios
          </div>
          
          <div className="flex flex-col gap-unit">
            {preset_prompts.map((prompt, i) => (
              <div 
                key={i}
                onClick={() => setInputText(prompt)}
                className="p-sm bg-surface-container-lowest border border-outline-variant rounded cursor-pointer hover:bg-surface-container-low transition-colors shadow-stroke text-sm mb-2"
              >
                <p className="font-body-sm text-body-sm text-on-surface line-clamp-3">
                  {prompt}
                </p>
                <div className="mt-unit flex justify-between items-center text-secondary">
                  <span className="font-mono-label text-mono-label text-[10px]">Sample {i+1}</span>
                  <MessageSquare size={14} />
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Center & Right Panel Container (Flexible area) */}
        <div className="flex-grow flex flex-col h-full gap-lg overflow-hidden">
          {/* Source Input Section */}
          <section className="flex flex-col gap-sm h-2/5 flex-shrink-0">
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
            <div className="flex justify-between items-center flex-wrap gap-sm">
              <div className="flex items-center gap-md flex-wrap">
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
                  
                  <label className="text-body-md text-on-surface-variant whitespace-nowrap ml-sm">Probs: {inferenceSettings.numProbs}</label>
                  <input type="range" min="0" max="20" value={inferenceSettings.numProbs} onChange={(e) => setInferenceSettings({...inferenceSettings, numProbs: parseInt(e.target.value)})} className="w-24 accent-primary" />
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
          <section className="flex-grow flex gap-gutter h-3/5 overflow-hidden pb-4">
            <div className={`flex-1 flex flex-col border rounded bg-surface-container-lowest overflow-hidden shadow-stroke relative ${selectedModel === 'PPO Aligned' ? 'border-[#C27D38]' : 'border-outline-variant'}`}>
              <div className="bg-surface-container-low px-md py-sm border-b border-outline-variant flex justify-between items-center">
                <div className="flex items-center gap-xs">
                  <span className={`font-mono-label text-mono-label font-bold ${selectedModel === 'PPO Aligned' ? 'text-primary' : 'text-on-surface'}`}>
                    {selectedModel}
                  </span>
                  {selectedModel === 'PPO Aligned' && <CheckCircle className="text-primary" size={14} />}
                </div>
                <span className="font-mono-label text-mono-label text-secondary text-[11px]">
                  Temperature: {inferenceSettings.temp}
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
                  <span className="font-medium">{metrics.time ? `${metrics.time.toFixed(2)}s` : '0.00s'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="uppercase tracking-wider opacity-70">Length:</span>
                  <span className="font-medium">{metrics.length || 0}</span>
                </div>
              </div>
            </div>

            {/* Right: Live Probabilities */}
            {inferenceSettings.numProbs > 0 && (
              <div className="w-64 flex flex-col border rounded bg-surface-container-lowest overflow-hidden shadow-stroke border-outline-variant flex-shrink-0">
                <div className="bg-surface-container-low px-md py-sm border-b border-outline-variant">
                  <span className="font-mono-label text-mono-label font-bold text-on-surface">Top Tokens</span>
                </div>
                <div className="p-md flex-grow overflow-y-auto flex flex-col gap-3">
                  {topTokens.map((t, idx) => (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="bg-surface-container px-1 rounded truncate max-w-[120px] border border-outline-variant">
                          {t.token.replace(/ /g, '\u00A0')}
                        </span>
                        <span>{(t.prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-surface-container-high rounded-full h-1.5 overflow-hidden">
                        <div className="bg-primary h-full rounded-full transition-all duration-75" style={{ width: `${t.prob * 100}%` }}></div>
                      </div>
                    </div>
                  ))}
                  {topTokens.length === 0 && <span className="text-xs text-on-surface-variant italic">Waiting for tokens...</span>}
                </div>
              </div>
            )}
          </section>
        </div>
      </motion.div>
    </>
  )
}
