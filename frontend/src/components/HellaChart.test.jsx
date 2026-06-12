import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import HellaChart from './HellaChart'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="xaxis" />,
  YAxis: () => <div data-testid="yaxis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ReferenceLine: () => <div data-testid="reference-line" />
}))

describe('HellaChart', () => {
  it('renders a loading state initially', () => {
    render(<HellaChart dataUrl="/logs/pretraining_log.txt" />)
    expect(screen.getByText('Loading chart data...')).toBeInTheDocument()
  })
})
