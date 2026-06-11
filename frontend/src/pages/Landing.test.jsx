import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeAll } from 'vitest'
import Landing from './Landing'
import { BrowserRouter } from 'react-router-dom'


describe('Landing Page', () => {
  it('renders title and start button', () => {
    render(<BrowserRouter><Landing /></BrowserRouter>)
    expect(screen.getByRole('heading', { name: /GPT-2/i, level: 1 })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /start summarizing/i })).toBeInTheDocument()
  })
})
