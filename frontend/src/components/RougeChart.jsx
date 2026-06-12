import { useState, useEffect } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function RougeChart() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/logs/sft_rouge.json')
      .then(res => res.json())
      .then(json => {
        // Transform the JSON object into an array for Recharts
        // We will plot the best_model's scores
        if (json.best_model) {
          const chartData = [
            { name: 'Rouge-1', score: json.best_model['rouge-1'] },
            { name: 'Rouge-2', score: json.best_model['rouge-2'] },
            { name: 'Rouge-L', score: json.best_model['rouge-l'] }
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
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
          <XAxis dataKey="name" tick={{fontSize: 12}} stroke="#857467" />
          <YAxis tick={{fontSize: 12}} stroke="#857467" />
          <Tooltip contentStyle={{backgroundColor: '#fff8f5', borderColor: '#d7c3b3', borderRadius: '4px'}} />
          <Bar dataKey="score" fill="#ffb876" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
