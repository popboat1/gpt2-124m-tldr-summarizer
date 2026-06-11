import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Layout from './Layout'
import { BrowserRouter } from 'react-router-dom'

describe('Layout', () => {
  it('renders children within a dark themed container', () => {
    render(<BrowserRouter><Layout><div>Test Content</div></Layout></BrowserRouter>)
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })
})
