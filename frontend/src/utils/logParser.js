export function parseLogData(text) {
  if (!text) return [];
  
  const items = text.split('\n')
    .map(line => line.trim().split(/\s+/))
    .filter(parts => parts.length >= 3)
    .map(parts => ({
      step: parseInt(parts[0], 10),
      type: parts[1],
      value: parseFloat(parts[2])
    }))
    .filter(item => !isNaN(item.step) && !isNaN(item.value));

  const itemsByType = {};
  items.forEach(item => {
    if (!itemsByType[item.type]) itemsByType[item.type] = [];
    itemsByType[item.type].push(item);
  });

  const validItems = [];
  
  for (const type in itemsByType) {
    const list = itemsByType[type];
    let minStep = Infinity;
    // Iterate backwards to drop rollbacks (restarted runs)
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].step < minStep) {
        validItems.push(list[i]);
        minStep = list[i].step;
      }
    }
  }

  // Group by step
  const map = new Map();
  validItems.forEach(item => {
    if (!map.has(item.step)) map.set(item.step, { step: item.step });
    if (item.type === 'train') map.get(item.step).loss = item.value;
    else if (item.type === 'val') map.get(item.step).val_loss = item.value;
    else if (item.type === 'hella') map.get(item.step).hella_acc = item.value;
  });

  return Array.from(map.values()).sort((a, b) => a.step - b.step);
}
