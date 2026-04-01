import React, { useState, useEffect } from 'react';

export default function SetupSection({ onStartSession, multiplayer }) {
  const [userName, setUserName] = useState(localStorage.getItem('sns_user_name') || '');
  const [userAge, setUserAge] = useState(localStorage.getItem('sns_user_age') || '');
  const [userGender, setUserGender] = useState('male');
  const [userSkinTone, setUserSkinTone] = useState(localStorage.getItem('sns_user_tone') || 'Type 3');
  const [cielabEnabled, setCielabEnabled] = useState(localStorage.getItem('sns_cielab_enabled') !== 'false');

  const [micDevices, setMicDevices] = useState([]);
  const [selectedMic, setSelectedMic] = useState('');
  
  const [breathCount, setBreathCount] = useState('15');
  const [music, setMusic] = useState(`${import.meta.env.BASE_URL}assets/ohhm.mp3`);
  const [volume, setVolume] = useState('30');

  // Multiplayer State
  const [isMultiplayer, setIsMultiplayer] = useState(false);
  const [joinCode, setJoinCode] = useState('');

  useEffect(() => {
    localStorage.setItem('sns_user_name', userName);
    localStorage.setItem('sns_user_age', userAge);
    localStorage.setItem('sns_user_tone', userSkinTone);
    localStorage.setItem('sns_cielab_enabled', cielabEnabled);
  }, [userName, userAge, userSkinTone, cielabEnabled]);

  const loadDevices = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter(d => d.kind === 'audioinput');
      setMicDevices(mics);
      if (mics.length > 0) setSelectedMic(mics[0].deviceId);
    } catch (e) {
      console.error(e);
    }
  };

  const handleMicPerm = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());
      await loadDevices();
      alert('Microphone permission granted.');
    } catch (e) {
      alert('Mic permission denied.');
    }
  };

  const handleCamPerm = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(t => t.stop());
      alert('Camera permission granted.');
    } catch (e) {
      alert('Camera permission denied.');
    }
  };

  const handleBegin = () => {
    if (isMultiplayer) {
      if (!joinCode || joinCode.length < 4) {
        alert('Please enter a valid 4-digit Room Code.');
        return;
      }
      if (!userName) {
        alert('Please enter your name for the Arena.');
        return;
      }
      
      // Connect and Join
      multiplayer.connect();
      multiplayer.joinRoom(joinCode, userName);
      localStorage.setItem('sns_room_code', joinCode);
    }

    localStorage.setItem('sns_session_mic', selectedMic);
    localStorage.setItem('sns_session_breath', breathCount);
    localStorage.setItem('sns_session_music', music);
    localStorage.setItem('sns_session_volume', volume);
    localStorage.setItem('sns_is_multiplayer', isMultiplayer);
    
    onStartSession();
  };

  return (
    <div className="setup-section" id="setupSection">
      <div className="setup-left">
        <div className="setup-image-stack">
          <img src={`${import.meta.env.BASE_URL}assets/images/nature-calm.png`} alt="Nature" className="setup-img setup-img--back" loading="lazy" />
          <img src={`${import.meta.env.BASE_URL}assets/images/hero-meditation.png`} alt="Meditation" className="setup-img setup-img--front" loading="lazy" />
        </div>
        <div className="setup-quote">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" opacity="0.3"><path d="M6 17h3l2-4V7H5v6h3zm8 0h3l2-4V7h-6v6h3z"/></svg>
          <p>"The mind is everything. What you think, you become."</p>
          <span>— Buddha</span>
        </div>

        {/* Multiplayer Lobby Card */}
        <div className="onboard-card" style={{ marginTop: '2rem' }}>
          <div className="arena-badge">MULTIPLAYER ARENA</div>
          <h2 className="card-title">Meditation Lobby</h2>
          <p className="card-desc">Meditate together and sync your flow states.</p>
          
          <div className="cielab-row" style={{ marginTop: '1rem' }}>
            <div className="cielab-info">
              <span className="cielab-text">Enable Multiplayer</span>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={isMultiplayer} onChange={e => setIsMultiplayer(e.target.checked)} />
              <span className="toggle-track"><span className="toggle-thumb"></span></span>
            </label>
          </div>

          {isMultiplayer && (
            <div className="room-code-group">
              <label className="field-label">Room Join Code</label>
              <input 
                type="text" 
                className="room-input" 
                placeholder="0000" 
                maxLength="4"
                value={joinCode}
                onChange={e => setJoinCode(e.target.value.toUpperCase())}
              />
              <p className="field-help" style={{ fontSize: '0.7rem', opacity: 0.6 }}>
                Share this code with friends to synchronize your session.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="setup-right">
        {/* Profile Card */}
        <div className="onboard-card" id="profileCard">
          <div className="card-icon-badge">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>
          </div>
          <h2 className="card-title">Your Profile</h2>
          <p className="card-desc">Personalize your meditation experience</p>

          <div className="field-grid">
            <div className="field-group field-group--full">
              <label className="field-label" htmlFor="userName">Your Name</label>
              <input type="text" id="userName" className="field-input" placeholder="Enter your name" value={userName} onChange={e => setUserName(e.target.value)} />
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="userAge">Age</label>
              <input type="number" id="userAge" className="field-input" placeholder="Age" min="1" max="120" value={userAge} onChange={e => setUserAge(e.target.value)} />
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="userGender">Gender</label>
              <select id="userGender" className="field-select" value={userGender} onChange={e => setUserGender(e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div className="field-group field-group--full">
              <label className="field-label" htmlFor="userSkinTone">Skin Profile (Fitzpatrick Scale)</label>
              <select id="userSkinTone" className="field-select" value={userSkinTone} onChange={e => setUserSkinTone(e.target.value)}>
                <option value="Type 1">Type I — Very Light</option>
                <option value="Type 2">Type II — Light</option>
                <option value="Type 3">Type III — Medium</option>
                <option value="Type 4">Type IV — Olive</option>
                <option value="Type 5">Type V — Dark Brown</option>
                <option value="Type 6">Type VI — Black</option>
              </select>
            </div>
          </div>

          <div className="cielab-row">
            <div className="cielab-info">
              <span className="cielab-badge">CIELAB</span>
              <span className="cielab-text">Skin-tone bias reduction</span>
            </div>
            <label className="toggle-switch" aria-label="Toggle CIELAB">
              <input type="checkbox" checked={cielabEnabled} onChange={e => setCielabEnabled(e.target.checked)} />
              <span className="toggle-track"><span className="toggle-thumb"></span></span>
            </label>
            <span className="cielab-status">{cielabEnabled ? 'Active' : 'Off'}</span>
          </div>
        </div>

        {/* Session Setup Card */}
        <div className="onboard-card" id="sessionCard">
          <div className="card-icon-badge card-icon-badge--teal">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
          </div>
          <h2 className="card-title">Session Setup</h2>
          <p className="card-desc">Configure your meditation settings</p>

          <div className="field-grid">
            <div className="field-group field-group--full">
              <label className="field-label" htmlFor="micSelect">Microphone</label>
              <select id="micSelect" className="field-select" value={selectedMic} onChange={e => setSelectedMic(e.target.value)}>
                {micDevices.length === 0 && <option value="">No microphone found</option>}
                {micDevices.map((m, i) => (
                  <option key={m.deviceId} value={m.deviceId}>{m.label || `Microphone ${i + 1}`}</option>
                ))}
              </select>
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="breathCount">Breathing Cycles</label>
              <select id="breathCount" className="field-select" value={breathCount} onChange={e => setBreathCount(e.target.value)}>
                <option value="10">10 cycles</option>
                <option value="15">15 cycles</option>
                <option value="20">20 cycles</option>
                <option value="25">25 cycles</option>
              </select>
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="musicSelect">Ambient Music</label>
              <select id="musicSelect" className="field-select" value={music} onChange={e => setMusic(e.target.value)}>
                <option value="none">No music</option>
                <option value={`${import.meta.env.BASE_URL}assets/ohhm.mp3`}>Calm Meditation</option>
                <option value={`${import.meta.env.BASE_URL}assets/deep_relaxation.mp3`}>Deep Relaxation</option>
                <option value="https://cdn.pixabay.com/audio/2022/08/23/audio_2e6d339bcc.mp3">Peaceful Zen</option>
                <option value="https://cdn.pixabay.com/audio/2023/02/28/audio_4dfed6d6c1.mp3">Nature Sync</option>
              </select>
            </div>

            <div className="field-group field-group--full">
              <label className="field-label" htmlFor="musicVolume">Volume — <span>{volume}%</span></label>
              <div className="range-container">
                <input type="range" id="musicVolume" className="range-input" min="0" max="100" value={volume} onChange={e => setVolume(e.target.value)} />
              </div>
            </div>
          </div>

          <div className="permission-row">
            <button className="perm-btn" type="button" onClick={handleMicPerm}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/></svg>
              Enable Microphone
            </button>
            <button className="perm-btn perm-btn--accent" type="button" onClick={handleCamPerm}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
              Enable Camera
            </button>
            <button className="perm-btn perm-btn--ghost" type="button" title="Refresh" onClick={loadDevices}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
            </button>
          </div>
        </div>

        {/* Begin Button */}
        <button id="beginSessionBtn" className="begin-btn" type="button" onClick={handleBegin}>
          <span className="begin-btn-glow"></span>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Begin {isMultiplayer ? 'Arena' : 'Meditation'}
        </button>

        <p className="privacy-note">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          {isMultiplayer ? 'Signals synced via secure WebSocket.' : 'All processing happens locally. Zero cloud transmission.'}
        </p>
      </div>
    </div>
  );
}
