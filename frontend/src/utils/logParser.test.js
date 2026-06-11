import { parseLogData } from './logParser'
import { describe, it, expect } from 'vitest'

describe('parseLogData', () => {
  it('parses valid training log lines and ignores hella lines', () => {
    const rawLog = `0 train 10.954966\n1 train 10.902752\n250 hella 0.2428\n2 train 10.804342`
    const result = parseLogData(rawLog)
    
    expect(result).toEqual([
      { step: 0, loss: 10.954966 },
      { step: 1, loss: 10.902752 },
      { step: 2, loss: 10.804342 }
    ])
  })
})
