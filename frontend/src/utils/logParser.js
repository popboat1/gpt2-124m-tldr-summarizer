export function parseLogData(text) {
  if (!text) return [];
  const map = new Map();
  text.split('\n').forEach(line => {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 3) {
      const step = parseInt(parts[0], 10);
      const type = parts[1];
      const value = parseFloat(parts[2]);
      
      if (!isNaN(step) && !isNaN(value)) {
        if (!map.has(step)) map.set(step, { step });
        
        if (type === 'train') map.get(step).loss = value;
        else if (type === 'val') map.get(step).val_loss = value;
        else if (type === 'hella') map.get(step).hella_acc = value;
      }
    }
  });
  return Array.from(map.values()).sort((a, b) => a.step - b.step);
}
