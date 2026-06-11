import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <main className="max-w-container-max mx-auto px-md md:px-gutter pb-xl">
      {/* Hero Section */}
      <section className="py-24 md:py-32 flex flex-col items-center text-center gap-lg border-b border-outline-variant">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full flex justify-between items-start mb-12"
        >
          <div className="font-mono-metric text-mono-metric text-on-surface-variant">TL;DR</div>
          <div className="font-mono-metric text-mono-metric text-primary border border-primary px-sm py-xs rounded-DEFAULT bg-primary-container/10">v1.0.0</div>
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg max-w-[800px]"
        >
          GPT-2 124M TLDR Summarizer
        </motion.h1>

        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="font-body-lg text-body-lg text-on-surface-variant max-w-[680px]"
        >
          A 124M parameter model fine-tuned on the openai_summarizer_tldr dataset. Minimalist, fast, and local. Pushing small parameter models beyond supervised fine-tuning limits using Proximal Policy Optimization.
        </motion.p>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-8 flex gap-md"
        >
          <Link to="/summarizer" className="bg-[#191919] text-[#FBF9F6] px-xl py-md font-body-md text-body-md rounded-DEFAULT hover:bg-surface-tint transition-colors">
            Start Summarizing
          </Link>
          <a href="https://github.com/popboat1/gpt2-124m-tldr-summarizer" target="_blank" rel="noreferrer" className="bg-transparent text-[#191919] border border-outline px-xl py-md font-body-md text-body-md rounded-DEFAULT hover:bg-surface-container-lowest transition-colors dark:text-[#FBF9F6] dark:hover:text-[#191919]">
            Explore GitHub
          </a>
        </motion.div>
      </section>

      {/* Feature A: The Hallucination Trap */}
      <section className="py-24 grid md:grid-cols-2 gap-xl items-center border-b border-outline-variant overflow-hidden">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="flex flex-col gap-md"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest">Section 01</div>
          <h2 className="font-headline-md text-headline-md">The Hallucination Trap</h2>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[400px]">
              Supervised Fine-Tuning (SFT) teaches models how to talk, but not what to say. When tasked with summarization, SFT models frequently fabricate details to match the linguistic distribution of the training data, prioritizing fluency over factual adherence.
          </p>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[400px]">
              Our baseline GPT-2 exhibited a 42% hallucination rate on the CNN/DailyMail dataset post-SFT.
          </p>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="bg-surface-container-lowest border border-outline-variant rounded-DEFAULT p-lg shadow-stroke"
        >
          <div className="flex items-center gap-sm mb-md border-b border-outline-variant pb-sm">
            <span className="material-symbols-outlined text-error">warning</span>
            <span className="font-mono-label text-mono-label">SFT Baseline Output Error</span>
          </div>
          <div className="font-body-md text-body-md bg-error-container/20 p-md rounded-DEFAULT border border-error-container text-on-surface">
              "The local council voted unanimously <span className="bg-error/20 text-on-error-container px-1">to increase taxes by 15%</span> on Tuesday."
          </div>
          <div className="mt-sm font-mono-label text-mono-label text-on-surface-variant flex items-center gap-xs">
            <span className="material-symbols-outlined text-xs">info</span> Ground truth indicates a 5% increase.
          </div>
        </motion.div>
      </section>

      {/* Feature B: The Alignment Solution: PPO */}
      <section className="py-24 border-b border-outline-variant">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="text-center mb-xl"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest mb-xs">Section 02</div>
          <h2 className="font-headline-md text-headline-md">The Alignment Solution: PPO</h2>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="relative max-w-[800px] mx-auto bg-surface-container-lowest border border-outline-variant rounded-DEFAULT p-xl shadow-stroke"
        >
          {/* Abstract Diagram */}
          <div className="flex justify-between items-center relative">
            {/* Actor */}
            <div className="flex flex-col items-center gap-sm z-10 bg-surface-container-lowest p-md border border-outline-variant rounded-DEFAULT">
              <div className="w-16 h-16 border-2 border-primary rounded-full flex items-center justify-center bg-primary-container/10">
                <span className="material-symbols-outlined text-primary">psychology</span>
              </div>
              <span className="font-mono-label text-mono-label">Actor (Policy)</span>
            </div>
            {/* Flow lines */}
            <div className="absolute top-1/2 left-0 w-full h-px bg-outline-variant -z-0 border-dashed border-t"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-surface-container-lowest px-sm z-10">
              <span className="material-symbols-outlined text-secondary text-sm">arrow_forward</span>
            </div>
            {/* Critic */}
            <div className="flex flex-col items-center gap-sm z-10 bg-surface-container-lowest p-md border border-outline-variant rounded-DEFAULT">
              <div className="w-16 h-16 border-2 border-on-surface rounded-DEFAULT flex items-center justify-center bg-surface-variant">
                <span className="material-symbols-outlined text-on-surface">gavel</span>
              </div>
              <span className="font-mono-label text-mono-label">Critic (Value)</span>
            </div>
          </div>
          <div className="mt-lg text-center">
            <p className="font-body-md text-body-md text-on-surface-variant max-w-[600px] mx-auto">
                We deploy Proximal Policy Optimization. The Critic model evaluates the Actor's summaries, dispensing scalar rewards based on human preference data. The Actor updates its policy weights, constrained by KL-divergence to prevent catastrophic forgetting of language fundamentals.
            </p>
          </div>
        </motion.div>
      </section>

      {/* Feature C: The Evaluation Paradigm */}
      <section className="py-24">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="mb-xl text-center"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest mb-xs">Section 03</div>
          <h2 className="font-headline-md text-headline-md">The Evaluation Paradigm</h2>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[600px] mx-auto mt-sm">
              Relying solely on ROUGE scores is insufficient for semantic alignment. We utilize an "LLM-as-a-Judge" framework (Llama-3-70B) to score outputs across three critical dimensions.
          </p>
        </motion.div>
        
        <div className="max-w-[800px] mx-auto flex flex-col gap-lg">
          {/* Metric 1 */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.8 }}
            className="bg-surface-container-lowest border border-outline-variant rounded-DEFAULT p-md shadow-stroke"
          >
            <div className="flex justify-between items-end mb-sm">
              <div>
                <div className="font-mono-label text-mono-label text-on-surface-variant">Metric</div>
                <div className="font-body-md text-body-md font-medium">Faithfulness Score</div>
              </div>
              <div className="font-mono-metric text-mono-metric text-primary">8.4<span className="text-sm text-outline">/10</span></div>
            </div>
            <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: "84%" }}
                viewport={{ once: true }}
                transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
                className="h-full bg-[#191919]"
              />
            </div>
          </motion.div>

          {/* Metric 2 */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="bg-surface-container-lowest border border-outline-variant rounded-DEFAULT p-md shadow-stroke"
          >
            <div className="flex justify-between items-end mb-sm">
              <div>
                <div className="font-mono-label text-mono-label text-on-surface-variant">Metric</div>
                <div className="font-body-md text-body-md font-medium">Conciseness Ratio</div>
              </div>
              <div className="font-mono-metric text-mono-metric text-primary">0.32<span className="text-sm text-outline">tgt/src</span></div>
            </div>
            <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: "65%" }}
                viewport={{ once: true }}
                transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1], delay: 0.4 }}
                className="h-full bg-primary"
              />
            </div>
          </motion.div>

          {/* Metric 3 */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-surface-container-lowest border border-outline-variant rounded-DEFAULT p-md shadow-stroke"
          >
            <div className="flex justify-between items-end mb-sm">
              <div>
                <div className="font-mono-label text-mono-label text-on-surface-variant">Metric</div>
                <div className="font-body-md text-body-md font-medium">Aggregate Reward</div>
              </div>
              <div className="font-mono-metric text-mono-metric text-primary">+2.14<span className="text-sm text-outline">nats</span></div>
            </div>
            <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: "75%" }}
                viewport={{ once: true }}
                transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1], delay: 0.5 }}
                className="h-full bg-tertiary"
              />
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  )
}
