import React, { useState, useRef, useEffect } from 'react';
import Header from './Header';
import MotivationBanner from './MotivationBanner';
import FaceAnalytics from './FaceAnalytics';
import AudioAnalytics from './AudioAnalytics';
import BreathingModule from './BreathingModule';
import ResultsSection from './ResultsSection';

import MultiplayerLeaderboard from './MultiplayerLeaderboard';

export default function DashboardScreen({ onBack, multiplayer }) {
  const [isSessionRunning, setIsSessionRunning] = useState(false);
  const [sessionStatusText, setSessionStatusText] = useState('Ready');
  const isMultiplayer = localStorage.getItem('sns_is_multiplayer') === 'true';
  const joinCode = localStorage.getItem('sns_room_code');
  
  // Real-time metrics
  const [bpm, setBpm] = useState(0);
  const [hrv, setHrv] = useState({ sdnn: 0, rmssd: 0 });
  
  const [eyeScore, setEyeScore] = useState(0);
  const [headScore, setHeadScore] = useState(0);
  const [gazeScore, setGazeScore] = useState(0);
  const [faceCalmness, setFaceCalmness] = useState(0);
  const [breathCons, setBreathCons] = useState(0);
  const [meditationIndex, setMeditationIndex] = useState(0);
  const [overallScore, setOverallScore] = useState(0);

  const [fps, setFps] = useState(0);
  const [facesDetected, setFacesDetected] = useState(0);
  const [rppgConfidence, setRppgConfidence] = useState(0);

  const bgAudioRef = useRef(null);

  const handleStart = () => {
    setIsSessionRunning(true);
    setSessionStatusText('running');
    if (bgAudioRef.current) {
      bgAudioRef.current.src = localStorage.getItem('sns_session_music') || '/assets/ohhm.mp3';
      const vol = localStorage.getItem('sns_session_volume') || '30';
      bgAudioRef.current.volume = Number(vol) / 100;
      if (bgAudioRef.current.src !== 'none') {
        bgAudioRef.current.play().catch(console.error);
      }
    }
  };

  const handleStop = () => {
    setIsSessionRunning(false);
    setSessionStatusText('Ready');
    if (bgAudioRef.current) bgAudioRef.current.pause();
  };

  const handleMute = () => {
    if (bgAudioRef.current) {
      bgAudioRef.current.muted = !bgAudioRef.current.muted;
    }
  };

  return (
    <section id="dashboardScreen" className="screen screen--dashboard screen--entering">
      <div className="dash-container">
        <Header isSessionRunning={isSessionRunning} onBack={onBack} sessionStatusText={sessionStatusText} />
        
        {isMultiplayer && (
          <MultiplayerLeaderboard 
            leaderboard={multiplayer.leaderboard} 
            currentUserId={null} // We'll handle self detection in the component
          />
        )}

        <MotivationBanner 
          isSessionRunning={isSessionRunning} 
          onStart={handleStart} 
          onStop={handleStop} 
          onMute={handleMute} 
        />

        <audio ref={bgAudioRef} loop crossOrigin="anonymous" />

        <div className="analytics-grid">
          <FaceAnalytics 
            isSessionRunning={isSessionRunning}
            onMetricsUpdate={({ fps, faces, eye, head, gaze, confidence, faceCalm }) => {
              setFps(fps); setFacesDetected(faces); setEyeScore(eye); 
              setHeadScore(head); setGazeScore(gaze); setRppgConfidence(confidence);
              if (faceCalm !== undefined) setFaceCalmness(faceCalm);
            }}
            onRawSync={(signals) => {
              if (isMultiplayer && isSessionRunning) {
                multiplayer.pushMetrics(joinCode, signals);
              }
            }}
            onRppgData={({ bpm, hrv, meditationIndex }) => {
              setBpm(bpm); setHrv(hrv); setMeditationIndex(meditationIndex);
            }}
          />

          <AudioAnalytics 
            isSessionRunning={isSessionRunning} 
            onMetricsUpdate={({ breathConsistency }) => setBreathCons(breathConsistency)}
          />

          <BreathingModule 
            isSessionRunning={isSessionRunning}
            bpm={bpm}
            hrv={hrv}
            rppgConfidence={rppgConfidence}
            onSessionProgress={(pct) => {
              if (pct >= 100 && isSessionRunning) {
                // Auto stop logic
              }
            }}
          />
        </div>

        <ResultsSection 
          faceCalmness={faceCalmness}
          breathCons={breathCons}
          meditationIndex={meditationIndex}
          overallScore={overallScore}
        />
      </div>
    </section>
  );
}
