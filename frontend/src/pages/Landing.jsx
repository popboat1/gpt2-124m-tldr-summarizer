import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="text-center"
      >
        <h1 className="text-5xl font-light tracking-tight text-white mb-4">
          GPT-2 <span className="text-gray-400">Summarizer</span>
        </h1>
        <p className="text-lg text-gray-500 mb-12 max-w-lg mx-auto leading-relaxed">
          A 124M parameter model fine-tuned on the openai_summarizer_tldr dataset.
          Minimalist, fast, and local.
        </p>
        <Link
          to="/summarizer"
          className="inline-flex items-center px-6 py-3 bg-white text-black rounded-full font-medium transition-transform hover:scale-105 active:scale-95"
        >
          Start Summarizing
        </Link>
      </motion.div>
    </div>
  )
}
