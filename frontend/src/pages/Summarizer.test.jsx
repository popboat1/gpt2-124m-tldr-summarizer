import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import Summarizer from './Summarizer'

describe('Summarizer Page', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders input area, output area, and metrics', () => {
    render(<Summarizer />)
    expect(screen.getByRole('textbox', { name: /input document/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Summarize/i })).toBeInTheDocument()
    expect(screen.getByText(/t\/s/i)).toBeInTheDocument()
    expect(screen.getByText(/Summary will appear here.../i)).toBeInTheDocument()
    expect(screen.getByText(/Temp: 0.7/i)).toBeInTheDocument()
  })

  it('updates input text when typed into', async () => {
    const user = userEvent.setup()
    render(<Summarizer />)
    
    const inputArea = screen.getByRole('textbox', { name: /input document/i })
    await user.type(inputArea, 'Hello world')
    
    expect(inputArea).toHaveValue('Hello world')
  })

  it('fetches a reddit post when clicking a data bucket', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ text: 'Fetched Reddit Post' }),
      })
    )
    
    const user = userEvent.setup()
    render(<Summarizer />)
    
    const tifuButton = screen.getByText('r/tifu')
    await user.click(tifuButton)
    
    expect(global.fetch).toHaveBeenCalledWith('/api/reddit/tifu')
    
    const inputArea = screen.getByRole('textbox', { name: /input document/i })
    expect(inputArea).toHaveValue('Fetched Reddit Post')
  })

  it('calls generate API and updates output', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ text: 'Generated Summary', time: 1.5, tps: 20.0 }),
      })
    )
    
    const user = userEvent.setup()
    render(<Summarizer />)

    const inputArea = screen.getByRole('textbox', { name: /input document/i })
    await user.type(inputArea, 'Some text')
    
    const summarizeButton = screen.getByRole('button', { name: /Summarize/i })
    await user.click(summarizeButton)
    
    expect(global.fetch).toHaveBeenCalledWith('/api/generate', expect.any(Object))
    
    expect(await screen.findByText('Generated Summary')).toBeInTheDocument()
    expect(await screen.findByText(/20.0 t\/s/i)).toBeInTheDocument()
  })
})
