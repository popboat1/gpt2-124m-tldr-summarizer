import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import RougeChart from './RougeChart'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="xaxis" />,
  YAxis: () => <div data-testid="yaxis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />
}))

describe('RougeChart', () => {
  it('renders a loading state initially', () => {
    render(<RougeChart />)
    expect(screen.getByText('Loading chart data...')).toBeInTheDocument()
  })
})
