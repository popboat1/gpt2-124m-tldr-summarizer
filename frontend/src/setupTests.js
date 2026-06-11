import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

class IntersectionObserverMock {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() { return [] }
  unobserve() {}
}

vi.stubGlobal('IntersectionObserver', IntersectionObserverMock)
