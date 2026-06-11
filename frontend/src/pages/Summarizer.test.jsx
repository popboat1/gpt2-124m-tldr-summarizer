import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Summarizer from './Summarizer'

describe('Summarizer Page', () => {
  it('renders input area, output area, and metrics', () => {
    render(<Summarizer />)
    expect(screen.getByPlaceholderText(/Enter text to summarize/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Summarize/i })).toBeInTheDocument()
    expect(screen.getByText(/tokens\/sec/i)).toBeInTheDocument()
  })
})
