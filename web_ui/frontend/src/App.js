import React, { useState, useEffect, useRef } from 'react';
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
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('Loading...');
  const [uploadedFiles, setUploadedFiles] = useState({ audio: null, video: null });
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    checkModelStatus();
  }, [messages]);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/status');
      setModelStatus(response.data.status);
    } catch (error) {
      setModelStatus('Offline');
    }
  };

  const handleFileUpload = (type, file) => {
    setUploadedFiles(prev => ({ ...prev, [type]: file }));
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = { text: inputText, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('text', inputText);
      if (uploadedFiles.audio) formData.append('audio', uploadedFiles.audio);
      if (uploadedFiles.video) formData.append('video', uploadedFiles.video);

      const response = await axios.post('http://localhost:5000/api/chat', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const aiMessage = { 
        text: response.data.response, 
        sender: 'ai', 
        timestamp: new Date(),
        confidence: response.data.confidence
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = { 
        text: 'Sorry, I encountered an error. Please try again.', 
        sender: 'ai', 
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

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
          <div className="app">
            {/* Header */}
            <header className="header">
              <div className="header-content">
                <h1>🤖 AtulyaAI</h1>
                <div className="status-indicator">
                  <span className={`status-dot ${modelStatus === 'Online' ? 'online' : 'offline'}`}></span>
                  <span className="status-text">{modelStatus}</span>
                </div>
              </div>
            </header>

            {/* Main Chat Area */}
            <div className="chat-container">
              <div className="messages">
                {messages.length === 0 && (
                  <div className="welcome-message">
                    <h2>Welcome to AtulyaAI! 🚀</h2>
                    <p>I'm your multimodal AI assistant. I can process text, audio, and video inputs.</p>
                    <p>Try asking me something or upload a file!</p>
                  </div>
                )}
                
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.sender} ${message.isError ? 'error' : ''}`}>
                    <div className="message-content">
                      <div className="message-text">{message.text}</div>
                      {message.confidence && (
                        <div className="confidence">Confidence: {Math.round(message.confidence * 100)}%</div>
                      )}
                      <div className="timestamp">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="message ai">
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* File Upload Area */}
              <div className="file-upload-area">
                <div className="upload-section">
                  <label className="upload-label">
                    🎵 Audio File
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => handleFileUpload('audio', e.target.files[0])}
                      className="file-input"
                    />
                  </label>
                  {uploadedFiles.audio && (
                    <span className="file-name">{uploadedFiles.audio.name}</span>
                  )}
                </div>
                
                <div className="upload-section">
                  <label className="upload-label">
                    🎬 Video File
                    <input
                      type="file"
                      accept="video/*"
                      onChange={(e) => handleFileUpload('video', e.target.files[0])}
                      className="file-input"
                    />
                  </label>
                  {uploadedFiles.video && (
                    <span className="file-name">{uploadedFiles.video.name}</span>
                  )}
                </div>
              </div>

              {/* Input Area */}
              <div className="input-area">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here... (Press Enter to send)"
                  className="message-input"
                  rows="3"
                />
                <button 
                  onClick={sendMessage} 
                  disabled={isLoading || !inputText.trim()}
                  className="send-button"
                >
                  {isLoading ? '⏳' : '🚀'}
                </button>
              </div>
            </div>
          </div>
        ) : (
          <AdminPanel />
        )}
      </main>
    </div>
  );
}

export default App; 