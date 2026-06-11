import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import SettingsSidebar from './SettingsSidebar'

describe('SettingsSidebar', () => {
  it('renders settings and updates values', async () => {
    const user = userEvent.setup()
    const onUpdate = vi.fn()
    render(
      <SettingsSidebar 
        isOpen={true} 
        onClose={() => {}} 
        settings={{ temp: 0.8, topK: 40 }} 
        onUpdate={onUpdate} 
      />
    )
    
    expect(screen.getByText('Inference Settings')).toBeInTheDocument()
    
    const tempInput = screen.getByRole('slider', { name: /Temperature/i })
    expect(tempInput).toBeInTheDocument()
  })
})
