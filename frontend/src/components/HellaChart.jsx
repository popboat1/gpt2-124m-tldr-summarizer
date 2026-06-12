import { useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';
import { parseLogData } from '../utils/logParser';

export default function HellaChart({ dataUrl }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(dataUrl)
      .then(res => res.text())
      .then(text => {
        const parsed = parseLogData(text).filter(item => item.hella_acc !== undefined);
        setData(parsed);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [dataUrl]);

  if (loading) {
    return <div className="w-full h-64 flex items-center justify-center text-on-surface-variant font-body-md">Loading chart data...</div>;
  }

  return (
    <div className="w-full h-64 bg-surface-container-lowest p-sm rounded-DEFAULT border border-outline-variant">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tick={{fontSize: 12}} stroke="#857467" />
          <YAxis domain={['auto', 'auto']} tick={{fontSize: 12}} stroke="#857467" />
          <Tooltip contentStyle={{backgroundColor: 'var(--color-surface-container-lowest)', borderColor: 'var(--color-outline-variant)', borderRadius: '4px', color: 'var(--color-on-surface)'}} />
          <Line type="monotone" dataKey="hella_acc" stroke="#ffb876" strokeWidth={2} dot={true} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
