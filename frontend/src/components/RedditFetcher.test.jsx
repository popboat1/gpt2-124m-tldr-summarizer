import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import RedditFetcher from './RedditFetcher'

describe('RedditFetcher Component', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn()
  })

  it('renders input and button', () => {
    render(<RedditFetcher onFetch={() => {}} />)
    expect(screen.getByPlaceholderText(/Paste Reddit post URL.../i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Fetch from Reddit/i })).toBeInTheDocument()
  })

  it('shows error on invalid URL', async () => {
    const user = userEvent.setup()
    render(<RedditFetcher onFetch={() => {}} />)
    
    const input = screen.getByPlaceholderText(/Paste Reddit post URL.../i)
    await user.type(input, 'not-a-url')
    
    const button = screen.getByRole('button', { name: /Fetch from Reddit/i })
    await user.click(button)
    
    expect(await screen.findByText(/Please enter a valid URL/i)).toBeInTheDocument()
  })

  it('calls onFetch with combined text on successful fetch', async () => {
    const mockOnFetch = vi.fn()
    const mockResponse = [
      {
        data: {
          children: [
            {
              data: {
                title: 'Test Title',
                selftext: 'Test body content'
              }
            }
          ]
        }
      }
    ]
    
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse
    })

    const user = userEvent.setup()
    render(<RedditFetcher onFetch={mockOnFetch} />)
    
    const input = screen.getByPlaceholderText(/Paste Reddit post URL.../i)
    await user.type(input, 'https://reddit.com/r/test/comments/123/test/')
    
    const button = screen.getByRole('button', { name: /Fetch from Reddit/i })
    await user.click(button)
    
    await waitFor(() => {
      expect(mockOnFetch).toHaveBeenCalledWith('Title: Test Title\n\nTest body content')
    })
  })
})
