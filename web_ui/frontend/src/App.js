import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ChatInterface from './ChatInterface';
import AdminPanel from './AdminPanel';

function Status() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    axios.get('/status').then(res => setStatus(res.data));
  }, []);
  return (
    <div className="bg-gray-800 text-white rounded-lg p-4 shadow mb-4">
      <h2 className="text-lg font-bold mb-2">System Status</h2>
      {status ? (
        <ul>
          <li>Status: {status.status}</li>
          <li>CPU: {status.cpu}%</li>
          <li>RAM: {status.ram} MB</li>
        </ul>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );
}

function LogsPanel() {
  const [logs, setLogs] = useState([]);
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('/logs').then(res => setLogs(res.data.logs || []));
    }, 2000);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="bg-gray-800 text-green-400 rounded-lg p-4 shadow mb-4 h-40 overflow-y-auto">
      <h2 className="text-lg font-bold mb-2 text-white">Real-Time Logs</h2>
      <pre className="whitespace-pre-wrap">{logs.join('\n')}</pre>
    </div>
  );
}

function MetricsChart() {
  const [metrics, setMetrics] = useState([]);
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('/metrics').then(res => setMetrics(res.data.metrics || []));
    }, 5000);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="bg-gray-800 text-white rounded-lg p-4 shadow mb-4">
      <h2 className="text-lg font-bold mb-2">System Metrics</h2>
      <div className="flex space-x-2">
        {metrics.map((m, i) => (
          <div key={i} className="flex flex-col items-center">
            <div className="w-8 h-24 bg-blue-600 mb-1" style={{ height: `${m.value}px` }}></div>
            <span className="text-xs">{m.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [currentView, setCurrentView] = useState('chat'); // 'chat' or 'admin'

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="bg-gray-800 text-white p-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">Atulya AI</h1>
          <div className="flex space-x-4">
            <button
              onClick={() => setCurrentView('chat')}
              className={`px-4 py-2 rounded ${
                currentView === 'chat' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setCurrentView('admin')}
              className={`px-4 py-2 rounded ${
                currentView === 'admin' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              Admin Panel
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {currentView === 'chat' ? (
          <ChatInterface />
        ) : (
          <AdminPanel />
        )}
      </main>
    </div>
  );
}

export default App; 