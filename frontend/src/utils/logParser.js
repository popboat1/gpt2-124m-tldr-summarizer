export function parseLogData(text) {
  if (!text) return [];
  return text.split('\n')
    .filter(line => line.includes('train'))
    .map(line => {
      const parts = line.trim().split(' ');
      if (parts.length >= 3) {
        return { step: parseInt(parts[0], 10), loss: parseFloat(parts[2]) };
      }
      return null;
    })
    .filter(item => item !== null && !isNaN(item.step) && !isNaN(item.loss));
}
