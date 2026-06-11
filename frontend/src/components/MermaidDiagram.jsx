import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

export default function MermaidDiagram({ chart }) {
  const ref = useRef(null);

  useEffect(() => {
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
    
    if (ref.current) {
      mermaid.run({ nodes: [ref.current] });
    }
  }, [chart]);

  return (
    <div className="flex justify-center p-md bg-surface-container-lowest rounded-DEFAULT border border-outline-variant">
      <div ref={ref} className="mermaid text-on-surface-variant font-body-md">
        {chart}
      </div>
    </div>
  );
}
