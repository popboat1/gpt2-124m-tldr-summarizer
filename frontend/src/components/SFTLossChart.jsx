import { useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts';
import { parseLogData } from '../utils/logParser';

export default function SFTLossChart({ dataUrl = "/logs/sft_log.txt" }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl)
      .then(res => res.text())
      .then(text => {
        // Sample data to prevent rendering too many points
        const parsed = parseLogData(text).filter((_, i) => i % 10 === 0);
        setData(parsed);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [dataUrl]);

  if (loading) {
    return <div className="w-full h-64 flex items-center justify-center text-on-surface-variant font-body-md" data-testid="loading-indicator">Loading chart data...</div>;
  }

  return (
    <div className="w-full h-64 bg-surface-container-lowest p-sm rounded-DEFAULT border border-outline-variant">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tick={{fontSize: 12}} stroke="#857467" />
          <YAxis domain={['auto', 'auto']} tick={{fontSize: 12}} stroke="#857467" />
          <Tooltip contentStyle={{backgroundColor: '#fff8f5', borderColor: '#d7c3b3', borderRadius: '4px'}} />
          <Line type="monotone" dataKey="loss" stroke="#884e08" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="val_loss" stroke="#ffb876" strokeWidth={2} dot={false} />
          <ReferenceLine y={2.5321} label="Best Val Loss (2.5321)" stroke="#4ade80" strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
