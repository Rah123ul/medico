import React, { useState, useEffect } from 'react';

export default function Header({ isSessionRunning, onBack, sessionStatusText }) {
  const [userName, setUserName] = useState('');
  const [time, setTime] = useState('00:00:00');

  useEffect(() => {
    setUserName(localStorage.getItem('sns_user_name') || 'Meditator');
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      const n = new Date();
      setTime(
        String(n.getHours()).padStart(2, '0') + ':' +
        String(n.getMinutes()).padStart(2, '0') + ':' +
        String(n.getSeconds()).padStart(2, '0')
      );
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="dash-header" id="dashHeader">
      <div className="dash-header-left">
        <svg className="dash-brand-icon" width="32" height="32" viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="22" stroke="url(#brandGrad2)" strokeWidth="2"/>
          <circle cx="24" cy="24" r="5" fill="url(#brandGrad2)"/>
          <defs><linearGradient id="brandGrad2" x1="0" y1="0" x2="48" y2="48"><stop offset="0%" stopColor="#7c3aed"/><stop offset="100%" stopColor="#06b6d4"/></linearGradient></defs>
        </svg>
        <div>
          <span className="dash-brand-name">MindFlow</span>
          <span className="dash-user-greeting">Welcome, {userName}</span>
        </div>
      </div>
      <div className="dash-header-center">
        <div className="dash-clock">{time}</div>
        <div className="dash-status-pills">
          <div className="status-pill status-pill--active"><span className="pill-dot"></span>Neural Link</div>
          <div className="status-pill"><span className={`pill-dot ${isSessionRunning ? '' : 'pill-dot--off'}`}></span>Camera</div>
          <div className="status-pill"><span className={`pill-dot ${isSessionRunning ? '' : 'pill-dot--off'}`}></span>Audio</div>
        </div>
      </div>
      <div className="dash-header-right">
        <button className="icon-btn" title="Export CSV">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        </button>
        <button className="icon-btn" title="Clear History">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
        <div className={`session-badge ${isSessionRunning ? 'active' : ''}`}>
          <span className="session-dot"></span>
          <span>{sessionStatusText || 'Ready'}</span>
        </div>
        <button className="icon-btn" title="Back to Setup" onClick={onBack}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5"/><polyline points="12 19 5 12 12 5"/></svg>
        </button>
      </div>
    </header>
  );
}
