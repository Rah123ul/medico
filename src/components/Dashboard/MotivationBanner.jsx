import React from 'react';

export default function MotivationBanner({ onStart, onStop, onMute, isSessionRunning }) {
  return (
    <div className="motivation-banner" id="motivationBanner">
      <img src="assets/images/hands-meditation.png" alt="" className="banner-img" />
      <div className="banner-content">
        <span className="banner-tag">Today's Intention</span>
        <h2 className="banner-title">Find Your Center</h2>
        <p className="banner-desc">Every breath is a fresh start. Let the AI guide you to deep calm.</p>
      </div>
      <div className="banner-actions">
        {!isSessionRunning && (
          <button className="action-btn action-btn--engage" onClick={onStart}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Engage Session
          </button>
        )}
        {isSessionRunning && (
          <button className="action-btn action-btn--stop" onClick={onStop}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
            Stop
          </button>
        )}
        <button className="icon-btn" title="Toggle Voice" onClick={onMute}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
        </button>
      </div>
    </div>
  );
}
