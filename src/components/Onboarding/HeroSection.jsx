import React from 'react';

export default function HeroSection({ onScrollToSetup }) {
  return (
    <div className="hero-section" id="heroSection">
      <div className="hero-image-wrap">
        <img src={`${import.meta.env.BASE_URL}assets/images/hero-meditation.png`} alt="Meditation at sunset" className="hero-image" />
        <div className="hero-gradient-overlay"></div>
      </div>
      <div className="hero-content">
        <div className="hero-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2m10-10h-2M4 12H2m15.07-7.07l-1.41 1.41M8.34 15.66l-1.41 1.41m12.14 0l-1.41-1.41M8.34 8.34L6.93 6.93" />
          </svg>
          AI-Powered Wellness
        </div>
        <h1 className="hero-title">MindFlow</h1>
        <p className="hero-subtitle">
          Take a deep breath. Let technology guide your inner peace with real-time biometric meditation analysis.
        </p>
        <button id="heroScrollBtn" className="hero-cta" type="button" onClick={onScrollToSetup}>
          Get Started
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <polyline points="19 12 12 19 5 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
