import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import LossChart from '../components/LossChart'
import HellaChart from '../components/HellaChart'
import SFTLossChart from '../components/SFTLossChart'
import RougeChart from '../components/RougeChart'
import MermaidDiagram from '../components/MermaidDiagram'

export default function Landing() {
  const ppoDiagram = `
graph TD;
    A[Pretrained Model] --> B(SFT Model);
    B --> C{PPO Optimization};
    C --> D[Final Model];
    E[Reward Model] -.-> C;
`;

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
          <Link to="/summarizer" className="bg-inverse-surface text-inverse-on-surface px-xl py-md font-body-md text-body-md rounded-DEFAULT hover:opacity-80 transition-opacity cursor-pointer border border-inverse-surface">
            Start Summarizing
          </Link>
          <a href="https://github.com/popboat1/gpt2-124m-tldr-summarizer" target="_blank" rel="noreferrer" className="bg-transparent text-on-background border border-outline px-xl py-md font-body-md text-body-md rounded-DEFAULT hover:bg-on-background hover:text-background transition-colors cursor-pointer">
            Explore GitHub
          </a>
        </motion.div>
      </section>

      {/* Feature A: Pretraining */}
      <section className="py-24 grid md:grid-cols-2 gap-xl items-center border-b border-outline-variant overflow-hidden">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="flex flex-col gap-md"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest">Section 01</div>
          <h2 className="font-headline-md text-headline-md">Pretraining</h2>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[400px]">
              Trained on Kaggle's dual T4 GPU environment, the base model learned language representations from the FineWeb-Edu dataset (10 billion tokens). It reached a validation loss of 3.0048 and a HellaSwag accuracy of 28.95%.
          </p>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex flex-col gap-4"
        >
          <div className="w-full h-full min-h-[250px]">
            <LossChart dataUrl="/logs/pretraining_log.txt" />
          </div>
          <div className="w-full h-full min-h-[250px]">
            <HellaChart dataUrl="/logs/pretraining_log.txt" />
          </div>
        </motion.div>
      </section>

      {/* Feature B: Supervised Fine Tuning */}
      <section className="py-24 grid md:grid-cols-2 gap-xl items-center border-b border-outline-variant overflow-hidden">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="flex flex-col gap-md"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest">Section 02</div>
          <h2 className="font-headline-md text-headline-md">Supervised Fine Tuning</h2>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[400px]">
              We then fine-tuned the model on the OpenAI Summarize TL;DR dataset, bringing the validation loss down to 2.5321 and drastically improving ROUGE scores.
          </p>
        </motion.div>
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex flex-col gap-4 w-full"
        >
          <div className="w-full h-full min-h-[250px]">
            <SFTLossChart />
          </div>
          <div className="w-full h-full min-h-[250px]">
            <RougeChart />
          </div>
        </motion.div>
      </section>

      {/* Feature C: The Alignment Solution: PPO */}
      <section className="py-24 border-b border-outline-variant">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="text-center mb-xl"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest mb-xs">Section 03</div>
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
          <div className="w-full">
            <MermaidDiagram chart={ppoDiagram} />
          </div>
          <div className="mt-lg text-center">
            <p className="font-body-md text-body-md text-on-surface-variant max-w-[600px] mx-auto">
                We deploy Proximal Policy Optimization. The Critic model evaluates the Actor's summaries, dispensing scalar rewards based on human preference data. The Actor updates its policy weights, constrained by KL-divergence to prevent catastrophic forgetting of language fundamentals.
            </p>
          </div>
        </motion.div>
      </section>

      {/* Feature D: The Evaluation Paradigm */}
      <section className="py-24">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 0.8 }}
          className="mb-xl text-center"
        >
          <div className="font-mono-label text-mono-label text-primary uppercase tracking-widest mb-xs">Section 04</div>
          <h2 className="font-headline-md text-headline-md">The Evaluation Paradigm</h2>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-[600px] mx-auto mt-sm">
              We utilize an "LLM-as-a-Judge" framework via openrouter using gpt-oss-120B. Our PPO aligned model achieved a significant winrate when compared against the baseline SFT model.
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
                <div className="font-body-md text-body-md font-medium">Winrate vs SFT Baseline</div>
              </div>
              <div className="font-mono-metric text-mono-metric text-primary">52<span className="text-sm text-outline">%</span></div>
            </div>
            <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: "52%" }}
                viewport={{ once: true }}
                transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
                className="h-full bg-[#191919]"
              />
            </div>
          </motion.div>

        </div>
      </section>
    </main>
  )
}
