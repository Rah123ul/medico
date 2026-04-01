import React, { useEffect, useState } from 'react';

export default function BreathingModule({ isSessionRunning, bpm, hrv, rppgConfidence, onSessionProgress }) {
  const [sessionMeterPct, setSessionMeterPct] = useState(0);

  useEffect(() => {
    let interval;
    if (isSessionRunning) {
      const breathCount = parseInt(localStorage.getItem('sns_session_breath') || '15', 10);
      const sessionDurationSeconds = breathCount * (4 + 7 + 8); // Example duration based on 4-7-8 breathing
      let elapsed = 0;
      
      interval = setInterval(() => {
        elapsed += 1;
        const pct = Math.min(100, Math.round((elapsed / sessionDurationSeconds) * 100));
        setSessionMeterPct(pct);
        onSessionProgress(pct);
        
        if (pct >= 100) clearInterval(interval);
      }, 1000);
      
      const circle = document.getElementById('breathingCircle');
      if (circle) {
        circle.style.animation = `breathCycle 19s infinite ease-in-out`;
        circle.style.transform = 'scale(1.5)';
      }
    } else {
      setSessionMeterPct(0);
      const circle = document.getElementById('breathingCircle');
      if (circle) {
        circle.style.animation = 'none';
        circle.style.transform = 'scale(1)';
      }
    }
    
    return () => clearInterval(interval);
  }, [isSessionRunning]);

  return (
    <article className="dash-card dash-card--breath" id="breathCard">
      <div className="dc-header">
        <div className="dc-icon dc-icon--teal">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
        </div>
        <div>
          <h3 className="dc-title">Breath Module</h3>
          <span className="dc-sub">Neural sync guidance</span>
        </div>
      </div>

      <div className="breath-stage">
        <div className="breath-rings">
          <div className="breath-ring br-1"></div>
          <div className="breath-ring br-2"></div>
          <div className="breath-ring br-3"></div>
        </div>
        <div id="breathingCircle" className="breath-heart"></div>
        <div id="breathStatus" className="breath-readout" role="status">
          {isSessionRunning ? 'Inhale deeply...' : 'Ready to engage'}
        </div>
      </div>

      <div className="progress-block">
        <div className="pb-header"><span>Session Progress</span><span id="progressPct" className="mono">{sessionMeterPct}%</span></div>
        <div className="pb-track"><div className="pb-fill" id="sessionMeter" style={{width: `${sessionMeterPct}%`}}></div></div>
      </div>

      {/* Heart Rate Panel */}
      <div className="hrv-panel">
        <div className="hrv-header">
          <div className="hrv-title-row"><span className="hrv-heart-icon">♥</span><span>Cardiac Monitor</span></div>
          <div className="bpm-readout" id="bpmDisplay">
            <span className="bpm-num">{bpm > 0 ? bpm : '--'}</span>
            <span className="bpm-unit">BPM</span>
          </div>
        </div>
        <div className="metrics-row">
          <div className="metric-card metric-card--rose">
            <span className="metric-label">Heart Rate</span>
            <span className="metric-value mono" id="heartrateTxt">{bpm > 0 ? bpm : 0} bpm</span>
            <div className="metric-bar">
              <div className="metric-fill metric-fill--rose" id="heartrateBar" style={{width: `${Math.min(100, Math.max(0, ((bpm - 40) / 80) * 100))}%`}}></div>
            </div>
          </div>
          <div className="metric-card metric-card--teal">
            <span className="metric-label">SDNN</span>
            <span className="metric-value mono" id="sdnnTxt">{hrv.sdnn} ms</span>
            <div className="metric-bar"><div className="metric-fill metric-fill--teal" id="sdnnBar" style={{width: `${Math.min(100, (hrv.sdnn / 100) * 100)}%`}}></div></div>
          </div>
          <div className="metric-card metric-card--violet">
            <span className="metric-label">RMSSD</span>
            <span className="metric-value mono" id="rmssdTxt">{hrv.rmssd} ms</span>
            <div className="metric-bar"><div className="metric-fill" id="rmssdBar" style={{width: `${Math.min(100, (hrv.rmssd / 100) * 100)}%`}}></div></div>
          </div>
        </div>
        <div className="signal-row">
          <span className="signal-label">Signal Quality</span>
          <span id="signalQualityTxt" className="mono signal-val">
            {rppgConfidence === 0 ? 'Waiting' : rppgConfidence < 1 ? 'Poor' : 'Excellent'}
          </span>
          <div className="signal-bar">
            <div className="signal-fill" id="signalQualityBar" style={{width: `${Math.min(100, Math.max(0, (rppgConfidence + 5) * 5))}%`}}></div>
          </div>
        </div>
      </div>
    </article>
  );
}
