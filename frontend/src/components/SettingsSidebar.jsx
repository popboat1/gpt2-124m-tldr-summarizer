import React from 'react'
import { motion } from 'framer-motion'
import { X } from 'lucide-react'

export default function SettingsSidebar({ isOpen, onClose, settings, onUpdate }) {
  const handleTempChange = (e) => {
    onUpdate({ ...settings, temp: parseFloat(e.target.value) })
  }

  const handleTopKChange = (e) => {
    onUpdate({ ...settings, topK: parseInt(e.target.value, 10) })
  }

  return (
    <>
      {/* Optional: Overlay to close when clicking outside */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40" 
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      
      <motion.div
        className="fixed top-0 right-0 h-full w-80 bg-gray-900 border-l border-gray-800 z-50 p-6 flex flex-col shadow-xl"
        initial={{ x: '100%' }}
        animate={{ x: isOpen ? 0 : '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
      >
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-xl font-semibold text-gray-100">Inference Settings</h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close settings"
          >
            <X size={24} />
          </button>
        </div>

        <div className="space-y-6 flex-1 overflow-y-auto">
          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label htmlFor="temp" className="text-sm font-medium text-gray-300">
                Temperature
              </label>
              <span className="text-xs text-gray-400">{settings.temp.toFixed(1)}</span>
            </div>
            <input
              id="temp"
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={settings.temp}
              onChange={handleTempChange}
              className="w-full accent-blue-500"
            />
            <p className="text-xs text-gray-500">
              Controls randomness. Lower values are more deterministic, higher values are more creative.
            </p>
          </div>

          {/* Top-K */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label htmlFor="topK" className="text-sm font-medium text-gray-300">
                Top-K
              </label>
              <span className="text-xs text-gray-400">{settings.topK}</span>
            </div>
            <input
              id="topK"
              type="range"
              min="1"
              max="100"
              step="1"
              value={settings.topK}
              onChange={handleTopKChange}
              className="w-full accent-blue-500"
            />
            <p className="text-xs text-gray-500">
              Limits the next token selection to the K most likely tokens.
            </p>
          </div>
        </div>
      </motion.div>
    </>
  )
}
