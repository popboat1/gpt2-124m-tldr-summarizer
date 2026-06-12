import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, CheckCircle, ChevronDown, Check, Plus, Mic, Volume2, Sparkles, Settings2 } from 'lucide-react'

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
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  
  const dropdownRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [dropdownRef])

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
                className="p-sm bg-surface-container-lowest border border-outline-variant rounded-none-none cursor-pointer hover:bg-surface-container-low transition-colors shadow-sm text-sm mb-2"
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

        {/* Center & Right Panel Container */}
        <div className="flex-grow flex gap-gutter h-full overflow-hidden">
          
          {/* Main Chat Interface */}
          <div className="flex-1 flex flex-col items-center pt-8 px-2 md:px-6 overflow-y-auto h-full pb-10">

            {/* Input Container */}
            <div className="w-full max-w-3xl bg-surface-container-low border-outline-variant rounded-none-none p-4 shadow-sm flex flex-col gap-2 relative transition-all focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary/30">
              <textarea 
                className="w-full h-32 bg-transparent text-body-lg text-on-surface resize-none focus:outline-none placeholder:text-on-surface-variant font-body-lg" 
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Start typing or paste in a reddit post to summarize"
              />
              
              <div className="flex justify-end items-center w-full mt-2">
                <div className="flex items-center gap-1.5">
                  {/* Custom Dropdown */}
                  <div className="relative" ref={dropdownRef}>
                    <button 
                      onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-none-none bg-surface-container-low text-on-surface hover:bg-surface-container-high transition-colors text-sm font-medium"
                    >
                      {selectedModel} <ChevronDown size={14} className="text-on-surface-variant" />
                    </button>
                    
                    {isDropdownOpen && (
                      <div className="absolute bottom-full right-0 mb-2 w-64 bg-surface-container-lowest border border-outline-variant rounded-none-none shadow-lg overflow-hidden z-50">
                        <button 
                          onClick={() => { setSelectedModel('PPO Aligned'); setIsDropdownOpen(false); }}
                          className="w-full flex items-center justify-between p-3 hover:bg-surface-container-low transition-colors text-left"
                        >
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-on-surface">PPO Aligned</span>
                            <span className="text-[11px] text-on-surface-variant">RLHF model for condensed output</span>
                          </div>
                          {selectedModel === 'PPO Aligned' && <Check size={16} className="text-primary" />}
                        </button>
                        <button 
                          onClick={() => { setSelectedModel('SFT Baseline'); setIsDropdownOpen(false); }}
                          className="w-full flex items-center justify-between p-3 hover:bg-surface-container-low transition-colors text-left border-t border-outline-variant"
                        >
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-on-surface">SFT Baseline</span>
                            <span className="text-[11px] text-on-surface-variant">Standard fine-tuned model</span>
                          </div>
                          {selectedModel === 'SFT Baseline' && <Check size={16} className="text-primary" />}
                        </button>
                      </div>
                    )}
                  </div>
                  
                  <button 
                    onClick={handleGenerate}
                    disabled={isGenerating || !inputText.trim()}
                    className="ml-2 px-4 py-1.5 bg-primary text-on-primary rounded-none-none font-medium text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
                  >
                    {isGenerating ? 'Running...' : 'Summarize'}
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            {!outputText && !isGenerating && (
               <div className="flex items-center gap-3 mt-6">
                 <button className="flex items-center gap-2 px-4 py-2 rounded-none-none border border-outline-variant text-sm text-on-surface hover:bg-surface-container transition-colors">
                   <Settings2 size={16} /> Inference Configured
                 </button>
               </div>
            )}

            {/* Output Section */}
            {(outputText || isGenerating) && (
              <div className="w-full max-w-3xl mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className={`w-full flex flex-col border rounded-none-none bg-surface-container-lowest overflow-hidden shadow-sm relative ${selectedModel === 'PPO Aligned' ? 'border-[#C27D38]' : 'border-outline-variant'}`}>
                  <div className="bg-surface-container-low px-md py-sm border-b border-outline-variant flex justify-between items-center">
                    <div className="flex items-center gap-xs">
                      <span className={`font-mono-label text-mono-label font-bold ${selectedModel === 'PPO Aligned' ? 'text-primary' : 'text-on-surface'}`}>
                        {selectedModel} Summary
                      </span>
                      {selectedModel === 'PPO Aligned' && <CheckCircle className="text-[#C27D38]" size={14} />}
                    </div>
                  </div>
                  <div className="p-md flex-grow font-body-lg text-body-lg text-on-surface leading-relaxed whitespace-pre-wrap min-h-[100px]">
                    {outputText || <span className="text-on-surface-variant italic">Generating summary...</span>}
                  </div>
                  {/* Inline Metrics Dashboard */}
                  <div className="bg-surface-container-low border-t border-outline-variant py-sm px-md flex flex-wrap justify-between items-center text-[12px] font-mono-metric">
                    <div className="flex items-center gap-2">
                      <span className="uppercase tracking-wider opacity-70">Velocity:</span>
                      <span className="text-[#C27D38] font-medium">{metrics.tokensPerSec ? `${metrics.tokensPerSec.toFixed(1)} t/s` : '0.0 t/s'}</span>
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
              </div>
            )}
          </div>

          {/* Right Panel */}
          <aside className="w-64 flex flex-col gap-gutter flex-shrink-0 h-full overflow-hidden">
            {/* Settings Card */}
            <div className="flex flex-col border rounded-none-none bg-surface-container-lowest shadow-sm border-outline-variant p-4 gap-4 flex-shrink-0">
              <h3 className="font-mono-label font-bold text-on-surface text-sm">Inference Settings</h3>
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="text-on-surface-variant">Temp</label>
                  <span className="font-mono">{inferenceSettings.temp}</span>
                </div>
                <input type="range" min="0" max="2" step="0.1" value={inferenceSettings.temp} onChange={(e) => setInferenceSettings({...inferenceSettings, temp: parseFloat(e.target.value)})} className="w-full accent-primary h-1 bg-surface-container-high rounded-none-none appearance-none cursor-pointer" />
                
                <div className="flex justify-between items-center text-xs mt-3">
                  <label className="text-on-surface-variant">Top K</label>
                  <span className="font-mono">{inferenceSettings.topK}</span>
                </div>
                <input type="range" min="1" max="100" value={inferenceSettings.topK} onChange={(e) => setInferenceSettings({...inferenceSettings, topK: parseInt(e.target.value)})} className="w-full accent-primary h-1 bg-surface-container-high rounded-none-none appearance-none cursor-pointer" />
                
                <div className="flex justify-between items-center text-xs mt-3">
                  <label className="text-on-surface-variant">Probs</label>
                  <span className="font-mono">{inferenceSettings.numProbs}</span>
                </div>
                <input type="range" min="0" max="20" value={inferenceSettings.numProbs} onChange={(e) => setInferenceSettings({...inferenceSettings, numProbs: parseInt(e.target.value)})} className="w-full accent-primary h-1 bg-surface-container-high rounded-none-none appearance-none cursor-pointer" />
              </div>
            </div>

            {/* Live Probabilities */}
            {inferenceSettings.numProbs > 0 && (
              <div className="flex-1 flex flex-col border rounded-none-none bg-surface-container-lowest overflow-hidden shadow-sm border-outline-variant">
                <div className="bg-surface-container-low px-md py-sm border-b border-outline-variant">
                  <span className="font-mono-label text-mono-label font-bold text-on-surface">Top Tokens</span>
                </div>
                <div className="p-md flex-grow overflow-y-auto flex flex-col gap-3">
                  {topTokens.map((t, idx) => (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="flex justify-between text-[11px] font-mono text-center">
                        <span className="text-center bg-surface-container-low px-1 rounded-none truncate max-w-[120px] border border-outline-variant text-on-surface">
                          {t.token.replace(/ /g, '\u00A0')}
                        </span>
                        <span className="text-on-surface-variant">{(t.prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-surface-container-high dark:bg-outline-variant rounded-none-none h-1.5 overflow-hidden">
                        <div className="bg-primary h-full rounded-none-none transition-all duration-75" style={{ width: `${t.prob * 100}%` }}></div>
                      </div>
                    </div>
                  ))}
                  {topTokens.length === 0 && <span className="text-xs text-on-surface-variant italic">Waiting for tokens...</span>}
                </div>
              </div>
            )}
          </aside>
        </div>
      </motion.div>
    </>
  )
}
