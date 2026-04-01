import React from 'react';

export default function FeaturesSection() {
  return (
    <div className="features-section" id="featuresSection">
      <div className="features-header">
        <span className="tag-pill">Why MindFlow</span>
        <h2 className="section-heading">Elevate Your Practice</h2>
        <p className="section-desc">Cutting-edge AI technology meets ancient meditation wisdom</p>
      </div>

      <div className="features-grid">
        <div className="feature-card feature-card--tall" id="feat1">
          <img src="/assets/images/breathing-exercise.png" alt="Deep Breathing" className="feature-img" loading="lazy" />
          <div className="feature-overlay">
            <div className="feature-badge">Breath Tracking</div>
            <h3 className="feature-title">Deep Breathing Analysis</h3>
            <p className="feature-desc">Real-time audio analysis decodes your breathing patterns for precise guidance.</p>
          </div>
        </div>

        <div className="feature-card" id="feat2">
          <img src="/assets/images/nature-calm.png" alt="Nature calm" className="feature-img" loading="lazy" />
          <div className="feature-overlay">
            <div className="feature-badge">Ambient Sound</div>
            <h3 className="feature-title">Nature Soundscapes</h3>
            <p className="feature-desc">Immersive audio environments for deeper sessions.</p>
          </div>
        </div>

        <div className="feature-card" id="feat3">
          <img src="/assets/images/hands-meditation.png" alt="Meditation mudra" className="feature-img" loading="lazy" />
          <div className="feature-overlay">
            <div className="feature-badge">Heart Rate</div>
            <h3 className="feature-title">rPPG Cardiac Monitor</h3>
            <p className="feature-desc">Contactless heart rate via facial blood flow analysis.</p>
          </div>
        </div>

        <div className="feature-card feature-card--wide" id="feat4">
          <img src="/assets/images/forest-meditation.png" alt="Forest meditation" className="feature-img" loading="lazy" />
          <div className="feature-overlay">
            <div className="feature-badge feature-badge--gold">AI Powered</div>
            <h3 className="feature-title">Face & Gaze Tracking</h3>
            <p className="feature-desc">478 facial landmarks track your eye relaxation, head stability, and gaze focus in real-time using MediaPipe neural networks.</p>
          </div>
        </div>

        <div className="feature-card" id="feat5">
          <img src="/assets/images/aurora-sleep.png" alt="Aurora northern lights" className="feature-img" loading="lazy" />
          <div className="feature-overlay">
            <div className="feature-badge">CIELAB</div>
            <h3 className="feature-title">Skin-Tone Calibration</h3>
            <p className="feature-desc">Fitzpatrick-aware for all skin tones.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
