import { useState, useEffect } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export default function RougeChart() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/logs/sft_rouge.json')
      .then(res => res.json())
      .then(json => {
        if (json.pretrained_model) {
          const chartData = [
            { 
              name: 'Rouge-1', 
              pretrained: json.pretrained_model['rouge-1'],
              best: json.best_model['rouge-1'],
              latest: json.latest_model['rouge-1']
            },
            { 
              name: 'Rouge-2', 
              pretrained: json.pretrained_model['rouge-2'],
              best: json.best_model['rouge-2'],
              latest: json.latest_model['rouge-2']
            },
            { 
              name: 'Rouge-L', 
              pretrained: json.pretrained_model['rouge-l'],
              best: json.best_model['rouge-l'],
              latest: json.latest_model['rouge-l']
            }
          ];
          setData(chartData);
        } else if (Array.isArray(json)) {
            setData(json);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="w-full h-64 flex items-center justify-center text-on-surface-variant font-body-md" data-testid="loading-indicator">Loading chart data...</div>;
  }

  return (
    <div className="w-full h-64 bg-surface-container-lowest p-sm rounded-DEFAULT border border-outline-variant">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#555555" vertical={false} opacity={0.3} />
          <XAxis dataKey="name" tick={{fontSize: 12}} stroke="#857467" />
          <YAxis tick={{fontSize: 12}} stroke="#857467" />
          <Tooltip contentStyle={{backgroundColor: 'var(--color-surface-container-lowest)', borderColor: 'var(--color-outline-variant)', borderRadius: '4px', color: 'var(--color-on-surface)'}} />
          <Legend wrapperStyle={{fontSize: '12px', color: '#b0b0b0'}} />
          <Bar dataKey="pretrained" name="Pretrained" fill="#555555" radius={[4, 4, 0, 0]} />
          <Bar dataKey="best" name="Best Model" fill="#ffb876" radius={[4, 4, 0, 0]} />
          <Bar dataKey="latest" name="Latest Model" fill="#4ade80" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
