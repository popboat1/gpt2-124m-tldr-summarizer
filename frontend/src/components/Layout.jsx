import { Link } from 'react-router-dom'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col font-body-md text-body-md selection:bg-primary-fixed selection:text-on-primary-fixed">
      <header className="bg-background docked full-width top-0 border-b border-outline-variant flat no-shadows z-50">
        <div className="flex justify-between items-center w-full px-lg py-md max-w-container-max mx-auto">
          <div className="font-headline-md text-headline-md font-bold text-on-background tracking-tight">
            GPT-2 Summarizer
          </div>
          <nav className="hidden md:flex gap-lg items-center">
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded" to="/">Home</Link>
            <Link className="text-on-surface-variant hover:text-primary transition-colors hover:bg-surface-container-low transition-all px-xs py-xs rounded" to="/summarizer">Playground</Link>
          </nav>
          <Link to="/summarizer">
            <button className="bg-[#191919] text-[#FBF9F6] px-md py-sm rounded hover:opacity-90 transition-opacity font-body-md text-body-md border border-[#191919]">
              Launch Playground
            </button>
          </Link>
        </div>
      </header>
      <main className="flex-grow flex flex-col w-full h-full max-w-container-max mx-auto relative">
        {children}
      </main>
    </div>
  )
}
