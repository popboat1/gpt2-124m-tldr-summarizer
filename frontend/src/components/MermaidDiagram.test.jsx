import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import MermaidDiagram from './MermaidDiagram'

// Mock mermaid to prevent actual rendering
vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg data-testid="mock-svg">Mocked Diagram</svg>' })
  }
}))

describe('MermaidDiagram', () => {
  it('renders the mocked svg diagram', async () => {
    render(<MermaidDiagram chart="graph TD; A-->B;" />)
    
    await waitFor(() => {
      expect(screen.getByTestId('mock-svg')).toBeInTheDocument()
    })
  })
})
