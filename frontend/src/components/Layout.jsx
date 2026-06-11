export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans selection:bg-gray-700">
      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {children}
      </main>
    </div>
  )
}
