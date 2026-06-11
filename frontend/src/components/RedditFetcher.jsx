import React, { useState } from 'react'
import { Link } from 'lucide-react'

export default function RedditFetcher({ onFetch }) {
  const [url, setUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFetch = async () => {
    setError(null)
    setIsLoading(true)
    try {
      let formattedUrl = url.trim()
      if (!formattedUrl) {
        throw new Error('Please enter a Reddit URL')
      }
      
      try {
        const urlObj = new URL(formattedUrl)
        if (!urlObj.hostname.includes('reddit.com') && !urlObj.hostname.includes('redd.it')) {
          throw new Error('Please enter a valid Reddit URL')
        }
      } catch (err) {
        throw new Error('Please enter a valid URL')
      }

      // If the URL ends with a slash, remove it before appending .json
      if (formattedUrl.endsWith('/')) {
        formattedUrl = formattedUrl.slice(0, -1)
      }
      const fetchUrl = formattedUrl.endsWith('.json') ? formattedUrl : `${formattedUrl}.json`
      
      const response = await fetch(fetchUrl)
      if (!response.ok) {
        throw new Error(`Failed to fetch from Reddit: ${response.status}`)
      }
      
      const json = await response.json()
      
      if (!Array.isArray(json) || !json[0]?.data?.children?.[0]?.data) {
        throw new Error('Invalid Reddit post data format')
      }
      
      const postData = json[0].data.children[0].data
      const title = postData.title || ''
      const selftext = postData.selftext || ''
      
      const combinedText = `Title: ${title}\n\n${selftext}`
      onFetch(combinedText)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-2 w-full mb-2">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
            <Link size={16} />
          </div>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleFetch()}
            placeholder="Paste Reddit post URL..."
            className="w-full bg-gray-900 border border-gray-800 rounded-lg py-2 pl-10 pr-4 text-gray-200 focus:outline-none focus:border-gray-600 focus:ring-1 focus:ring-gray-600 transition-all text-sm"
          />
        </div>
        <button
          onClick={handleFetch}
          disabled={isLoading || !url.trim()}
          className="px-4 py-2 bg-gray-800 text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
        >
          {isLoading ? 'Fetching...' : 'Fetch from Reddit'}
        </button>
      </div>
      {error && (
        <p className="text-red-400 text-xs pl-1">{error}</p>
      )}
    </div>
  )
}
