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

  it('shows error on non-Reddit URL', async () => {
    const user = userEvent.setup()
    render(<RedditFetcher onFetch={() => {}} />)
    
    const input = screen.getByPlaceholderText(/Paste Reddit post URL.../i)
    await user.type(input, 'https://google.com')
    
    const button = screen.getByRole('button', { name: /Fetch from Reddit/i })
    await user.click(button)
    
    expect(await screen.findByText(/Please enter a valid Reddit URL/i)).toBeInTheDocument()
  })

  it('correctly appends .json when URL has query parameters', async () => {
    const mockOnFetch = vi.fn()
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{ data: { children: [{ data: { title: 'T', selftext: 'B' } }] } }]
    })

    const user = userEvent.setup()
    render(<RedditFetcher onFetch={mockOnFetch} />)
    
    const input = screen.getByPlaceholderText(/Paste Reddit post URL.../i)
    await user.type(input, 'https://reddit.com/r/test/comments/123/test/?utm_source=share')
    
    const button = screen.getByRole('button', { name: /Fetch from Reddit/i })
    await user.click(button)
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('https://reddit.com/r/test/comments/123/test.json?utm_source=share')
    })
  })

  it('correctly resolves short links via allorigins proxy', async () => {
    const mockOnFetch = vi.fn()
    
    // First fetch: the proxy returns HTML with a canonical link
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        contents: '<html><head><link rel="canonical" href="https://www.reddit.com/r/test/comments/12345/test_post/"></head><body></body></html>'
      })
    })

    // Second fetch: actual post JSON
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{ data: { children: [{ data: { title: 'Short Link Title', selftext: 'Short Link Body' } }] } }]
    })

    const user = userEvent.setup()
    render(<RedditFetcher onFetch={mockOnFetch} />)
    
    const input = screen.getByPlaceholderText(/Paste Reddit post URL.../i)
    await user.type(input, 'https://www.reddit.com/r/ask/s/bkGdsmLFPL/')
    
    const button = screen.getByRole('button', { name: /Fetch from Reddit/i })
    await user.click(button)
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenNthCalledWith(1, `https://api.allorigins.win/get?url=${encodeURIComponent('https://www.reddit.com/r/ask/s/bkGdsmLFPL/')}`)
      expect(global.fetch).toHaveBeenNthCalledWith(2, 'https://www.reddit.com/r/test/comments/12345/test_post.json')
      expect(mockOnFetch).toHaveBeenCalledWith('Title: Short Link Title\n\nShort Link Body')
    })
  })
})
