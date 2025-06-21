import React, { useState, useEffect } from 'react';
import axios from 'axios';

const tabs = [
  { name: 'Status', endpoint: '/status' },
  { name: 'Config', endpoint: '/config' },
  { name: 'DNA', endpoint: '/dna' },
  { name: 'Nodes', endpoint: '/nodes' },
  { name: 'Self-Repair', endpoint: '/self_repair' },
  { name: 'Context', endpoint: '/context' },
  { name: 'Hardware', endpoint: '/hardware' },
  { name: 'Security', endpoint: '/security' },
];

export default function AdminPanel() {
  const [active, setActive] = useState(0);
  const [data, setData] = useState(null);
  useEffect(() => {
    axios.get(tabs[active].endpoint).then(res => setData(res.data));
    const interval = setInterval(() => {
      axios.get(tabs[active].endpoint).then(res => setData(res.data));
    }, 5000);
    return () => clearInterval(interval);
  }, [active]);
  return (
    <div className="bg-gray-900 text-white rounded-lg shadow-lg p-4 mt-4">
      <div className="flex space-x-4 mb-4">
        {tabs.map((tab, i) => (
          <button key={tab.name} className={`px-4 py-2 rounded ${i === active ? 'bg-blue-600' : 'bg-gray-700'}`} onClick={() => setActive(i)}>{tab.name}</button>
        ))}
      </div>
      <div className="bg-gray-800 rounded p-4 min-h-[200px]">
        <pre className="whitespace-pre-wrap">{data ? JSON.stringify(data, null, 2) : 'Loading...'}</pre>
      </div>
    </div>
  );
} 