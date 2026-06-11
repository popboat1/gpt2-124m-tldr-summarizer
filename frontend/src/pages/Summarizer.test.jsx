import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach } from 'vitest'
import Summarizer from './Summarizer'

describe('Summarizer Page', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders input area, output area, and metrics', () => {
    render(<Summarizer />)
    expect(screen.getByRole('textbox', { name: /input document/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Summarize/i })).toBeInTheDocument()
    expect(screen.getByText(/tokens\/sec/i)).toBeInTheDocument()
    expect(screen.getByText(/Summary will appear here.../i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Settings' })).toBeInTheDocument()
  })

  it('updates input text when typed into', async () => {
    const user = userEvent.setup()
    render(<Summarizer />)
    
    const inputArea = screen.getByRole('textbox', { name: /input document/i })
    await user.type(inputArea, 'Hello world')
    
    expect(inputArea).toHaveValue('Hello world')
  })
})
