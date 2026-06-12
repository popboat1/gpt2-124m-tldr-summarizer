import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'

export default function Layout({ children }) {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('darkMode')
      if (saved !== null) return JSON.parse(saved)
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    return false
  })

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  return (
    <div className="min-h-screen flex flex-col font-body-md text-body-md selection:bg-primary-fixed selection:text-on-primary-fixed">
      <header className="sticky top-0 bg-background docked full-width border-b border-outline-variant flat no-shadows z-50">
        <div className="flex justify-between items-center w-full px-lg py-md max-w-container-max mx-auto relative">
          <div className="font-headline-md text-headline-md font-bold text-on-background tracking-tight z-10">
            GPT-2 Summarizer
          </div>
          <nav className="hidden md:flex gap-lg items-center absolute left-1/2 -translate-x-1/2">
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded cursor-pointer" to="/">Home</Link>
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded cursor-pointer" to="/summarizer">Playground</Link>
          </nav>
          <div className="flex items-center gap-4 z-10">
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="bg-surface-container-high dark:bg-background text-on-surface hover:text-primary transition-colors hover:bg-surface-container-highest rounded-full w-10 h-10 flex items-center justify-center cursor-pointer border border-outline-variant"
              aria-label="Toggle dark mode"
            >
              <span className="material-symbols-outlined text-[20px]">
                {darkMode ? 'light_mode' : 'dark_mode'}
              </span>
            </button>
          </div>
        </div>
      </header>
      <main className="flex-grow flex flex-col w-full h-full max-w-container-max mx-auto relative">
        {children}
      </main>
    </div>
  )
}
