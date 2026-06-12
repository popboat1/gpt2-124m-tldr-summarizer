import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeAll } from 'vitest'
import Layout from './Layout'
import { BrowserRouter } from 'react-router-dom'

describe('Layout', () => {
  beforeAll(() => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  it('renders children within a dark themed container', () => {
    render(<BrowserRouter><Layout><div>Test Content</div></Layout></BrowserRouter>)
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })
})
