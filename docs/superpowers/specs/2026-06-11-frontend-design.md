# Frontend Design: GPT-2 (124M) Summarizer

## Overview
A hyper-minimalist, Claude-inspired web application for comparing text summarization from an SFT Baseline model and a PPO Aligned model. The app provides a quiet, intellectual aesthetic with scroll animations, data buckets for testing, and adjustable inference settings.

## Architecture
- **Stack:** React + Vite
- **Styling:** Tailwind CSS (v4.3.0) + Vanilla CSS for scroll animations and specific custom properties.
- **Routing:** Single Page Application (SPA). The landing page and the summarizer tool live on the same page with a smooth scrolling experience.

## Visual Language
- **Theme:** "Intellectual Minimalist"
- **Colors:**
  - Background: Warm cream (`#fff8f5`)
  - Text: Dark charcoal (`#211a15`)
  - Accent: Rich amber (`#8b500b` / `#C27D38`)
  - Structure: Muted slate borders
- **Typography:**
  - Inter for body and UI prose.
  - JetBrains Mono for metrics, tags, and data.

## Components & Layout

### 1. Landing / Hero Section
- A distraction-free, centered area featuring the project title and a brief introduction.
- Smooth scroll arrow/button guiding the user down to the Playground.
- Scroll Animations: Elements fade in and slide up slightly as they intersect the viewport.

### 2. The Playground (Summarizer Workspace)
- **Top Bar:** Simple navigation and a "Launch Playground" anchor.
- **Left Panel (Data Buckets):**
  - Buttons for predefined subreddits (`r/relationships`, `r/tifu`, etc.) to load sample text.
  - A list of sample prompts that populate the text input on click.
- **Center Top (Input & Settings):**
  - A text area for the source input.
  - A settings panel to configure inference options: `max_new_tokens`, `temperature`, `top_k`, `repetition_penalty`.
  - A primary "Generate Aligned Summary" button.
- **Center Bottom (Comparison Grid):**
  - Two side-by-side output cards: SFT Baseline vs PPO Aligned.
  - Read-only text areas that display the returned text.
- **Bottom Metrics Dashboard:**
  - Sticky footer displaying Generation Metrics: Tokens/sec, Total Inference Time, and Settings Used.

## Data Flow
1. **User Action:** The user selects a sample or types their own text and hits "Generate".
2. **API Request:** The frontend sends a `POST /api/generate` request containing:
   - `text` (string)
   - `inference_settings` (object: `max_new_tokens`, `temperature`, etc.)
3. **API Response:** The backend returns the SFT and PPO texts, inference time, and TPS.
4. **State Update:** The frontend populates the comparison grid and updates the metrics dashboard.

## Error Handling
- The UI will display a toast or inline error message if the API call fails or times out.
- Fallbacks for empty inputs (disable generate button).

## Required Backend Changes
- Modify `backend/main.py` and `promptrequest` model to accept optional `inference_settings`.
- Pass these settings into the `generate_text()` calls.

## Testing Strategy
- Use **Test-Driven Development (TDD)** for React components (using testing-library/react) and state management.
- Test the API integration utility.
- Use Parallel Agents to handle backend modifications and frontend scaffolding simultaneously.
