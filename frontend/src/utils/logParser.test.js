import { parseLogData } from './logParser'
import { describe, it, expect } from 'vitest'

describe('parseLogData', () => {
  it('groups train, val, and hella metrics by step', () => {
    const rawLog = `0 train 10.954966\n0 val 10.902752\n250 hella 0.2428\n2 train 10.804342`
    const result = parseLogData(rawLog)
    
    expect(result).toEqual([
      { step: 0, loss: 10.954966, val_loss: 10.902752 },
      { step: 2, loss: 10.804342 },
      { step: 250, hella_acc: 0.2428 }
    ])
  })
})
