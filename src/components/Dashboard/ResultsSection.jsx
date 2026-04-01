import React, { useEffect, useState } from 'react';
import { getSessions, calculateWeeklyForecast } from '../../utils/engine';

export default function ResultsSection({ faceCalmness, breathCons, meditationIndex, overallScore }) {
  const [sessions, setSessions] = useState([]);
  
  useEffect(() => {
    setSessions(getSessions().reverse());
  }, [overallScore]); // refresh history when a new score comes in

  return (
    <>
      <section className="results-section" id="resultsSection">
        <h2 className="section-title">Session Debrief</h2>
        <div className="results-grid">
          <div className="result-card result-card--violet">
            <span className="rc-label">Face Calmness</span>
            <span className="rc-value mono" id="faceCalmTxt">{faceCalmness > 0 ? faceCalmness : '--'}</span>
            <div className="metric-bar"><div className="metric-fill" id="faceCalmBar" style={{width: `${faceCalmness}%`}}></div></div>
          </div>
          <div className="result-card result-card--amber">
            <span className="rc-label">Breath Consistency</span>
            <span className="rc-value mono" id="breathConsTxt2">{breathCons > 0 ? `${breathCons}%` : '--'}</span>
            <div className="metric-bar"><div className="metric-fill metric-fill--amber" id="breathConsBar2" style={{width: `${breathCons}%`}}></div></div>
          </div>
          <div className="result-card result-card--teal">
            <span className="rc-label">Meditation Index</span>
            <span className="rc-value mono" id="meditationIndexTxt">{meditationIndex > 0 ? meditationIndex : '--'}</span>
            <div className="metric-bar"><div className="metric-fill metric-fill--teal" id="meditationIndexBar" style={{width: `${meditationIndex}%`}}></div></div>
          </div>
          <div className="result-card result-card--primary">
            <span className="rc-label">Overall Score</span>
            <span className="rc-value mono" id="overallTxt">{overallScore > 0 ? overallScore : '--'}</span>
            <div className="metric-bar"><div className="metric-fill metric-fill--primary" id="overallBar" style={{width: `${overallScore}%`}}></div></div>
          </div>
        </div>
        <p className="debrief-note" id="overallNote" role="status" aria-live="polite">
          {overallScore > 0 ? 'Session Complete. Check your wellness forecast above.' : ''}
        </p>
      </section>

      <section className="history-section">
        <h2 className="section-title">Session History</h2>
        <div className="table-container">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Name</th>
                <th>Profile</th>
                <th>Duration</th>
                <th>BPM</th>
                <th>Face</th>
                <th>Breath</th>
                <th>Meditation</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody id="historyBody">
              {sessions.map((s, i) => {
                const d = new Date(s.date);
                return (
                  <tr key={i}>
                    <td>{d.toLocaleDateString()} {d.toLocaleTimeString()}</td>
                    <td>{localStorage.getItem('sns_user_name') || 'User'}</td>
                    <td>{localStorage.getItem('sns_user_tone') || 'Type 3'}</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td><strong>{s.score}</strong></td>
                  </tr>
                );
              })}
              {sessions.length === 0 && (
                <tr><td colSpan="9" style={{textAlign: 'center', padding: '1rem', color: 'var(--text-muted)'}}>No sessions recorded yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="dash-footer">
        <div className="footer-row">
          <span className="footer-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            All processing local · Zero cloud transmission
          </span>
        </div>
        <p className="footer-disclaimer">MindFlow is a wellness instrument — not a medical device. HRV & BPM metrics are optical estimates. Not for clinical use.</p>
      </footer>
    </>
  );
}
