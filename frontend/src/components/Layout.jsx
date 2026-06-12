import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'

export default function Layout({ children }) {
  const [darkMode, setDarkMode] = useState(false)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  return (
    <div className="min-h-screen flex flex-col font-body-md text-body-md selection:bg-primary-fixed selection:text-on-primary-fixed">
      <header className="bg-background docked full-width top-0 border-b border-outline-variant flat no-shadows z-50">
        <div className="flex justify-between items-center w-full px-lg py-md max-w-container-max mx-auto">
          <div className="font-headline-md text-headline-md font-bold text-on-background tracking-tight">
            GPT-2 Summarizer
          </div>
          <nav className="hidden md:flex gap-lg items-center">
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded cursor-pointer" to="/">Home</Link>
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded cursor-pointer" to="/summarizer">Playground</Link>
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded cursor-pointer flex items-center justify-center"
            >
              <span className="material-symbols-outlined text-[20px]">
                {darkMode ? 'light_mode' : 'dark_mode'}
              </span>
            </button>
          </nav>
          <div className="flex items-center gap-4">
            <Link to="/summarizer">
              <button className="bg-inverse-surface text-inverse-on-surface px-md py-sm rounded hover:opacity-80 transition-opacity font-body-md text-body-md border border-inverse-surface cursor-pointer">
                Launch Playground
              </button>
            </Link>
          </div>
        </div>
      </header>
      <main className="flex-grow flex flex-col w-full h-full max-w-container-max mx-auto relative">
        {children}
      </main>
    </div>
  )
}
