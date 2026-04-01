import React from 'react';

export default function SessionsShowcase() {
  return (
    <div className="sessions-showcase" id="sessionsShowcase">
      <div className="sessions-header">
        <span className="tag-pill tag-pill--violet">Today's Sessions</span>
        <h2 className="section-heading">Ready When You Are</h2>
        <p className="section-desc">Choose a session that fits your mood</p>
      </div>

      <div className="sessions-scroll">
        <div className="session-card" id="sCard1">
          <img src={`${import.meta.env.BASE_URL}assets/images/breathing-exercise.png`} alt="Breathing" className="session-img" loading="lazy" />
          <div className="session-card-body">
            <span className="session-duration">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              15 MIN
            </span>
            <h3>Deep Breathing</h3>
            <p>Focus on breath patterns with AI guidance</p>
            <span className="session-tag">Beginner Friendly</span>
          </div>
        </div>

        <div className="session-card" id="sCard2">
          <img src={`${import.meta.env.BASE_URL}assets/images/nature-calm.png`} alt="Nature" className="session-img" loading="lazy" />
          <div className="session-card-body">
            <span className="session-duration">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              20 MIN
            </span>
            <h3>Nature Sync</h3>
            <p>Ambient soundscapes for deep relaxation</p>
            <span className="session-tag session-tag--teal">Popular</span>
          </div>
        </div>

        <div className="session-card" id="sCard3">
          <img src={`${import.meta.env.BASE_URL}assets/images/forest-meditation.png`} alt="Forest" className="session-img" loading="lazy" />
          <div className="session-card-body">
            <span className="session-duration">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              25 MIN
            </span>
            <h3>Forest Walk</h3>
            <p>Guided meditation through enchanted woods</p>
            <span className="session-tag session-tag--emerald">New</span>
          </div>
        </div>

        <div className="session-card" id="sCard4">
          <img src={`${import.meta.env.BASE_URL}assets/images/aurora-sleep.png`} alt="Aurora" className="session-img" loading="lazy" />
          <div className="session-card-body">
            <span className="session-duration">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              30 MIN
            </span>
            <h3>Northern Lights</h3>
            <p>Deep sleep meditation under the aurora</p>
            <span className="session-tag session-tag--purple">Sleep</span>
          </div>
        </div>
      </div>
    </div>
  );
}
