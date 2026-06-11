import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import MermaidDiagram from './MermaidDiagram'

// Mock mermaid to prevent actual rendering
vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    run: vi.fn()
  }
}))

describe('MermaidDiagram', () => {
  it('renders the chart definition inside a mermaid div', () => {
    render(<MermaidDiagram chart="graph TD; A-->B;" />)
    expect(screen.getByText('graph TD; A-->B;')).toBeInTheDocument()
  })
})
