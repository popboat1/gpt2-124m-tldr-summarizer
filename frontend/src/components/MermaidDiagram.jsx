import React, { useEffect, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    primaryColor: '#3d2e24',
    primaryTextColor: '#d7c3b3',
    primaryBorderColor: '#857467',
    lineColor: '#857467',
    secondaryColor: '#2b211a',
    tertiaryColor: '#1a140f',
  }
});

export default function MermaidDiagram({ chart }) {
  const [svgContent, setSvgContent] = useState('');
  const [error, setError] = useState(false);
  
  useEffect(() => {
    let isMounted = true;
    const renderDiagram = async () => {
      try {
        setError(false);
        const { svg } = await mermaid.render(`mermaid-${Date.now()}`, chart);
        if (isMounted) setSvgContent(svg);
      } catch (err) {
        console.error(err);
        if (isMounted) setError(true);
      }
    };
    renderDiagram();
    return () => { isMounted = false; };
  }, [chart]);

  if (error) {
    return <div className="p-md bg-surface-container-lowest">Failed to render diagram</div>;
  }

  return (
    <div className="flex justify-center p-md bg-surface-container-lowest rounded-DEFAULT border border-outline-variant">
      <div 
        className="mermaid text-on-surface-variant font-body-md" 
        dangerouslySetInnerHTML={{ __html: svgContent }} 
      />
    </div>
  );
}
