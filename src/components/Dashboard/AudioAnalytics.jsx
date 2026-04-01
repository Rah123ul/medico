import React, { useRef, useEffect, useState } from 'react';

export default function AudioAnalytics({ isSessionRunning, onMetricsUpdate }) {
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const animFrameRef = useRef(null);
  
  const [breathConsTxt, setBreathConsTxt] = useState('--');
  const [breathConsBar, setBreathConsBar] = useState(0);
  const [micLvl, setMicLvl] = useState('--');
  const [micLvlBar, setMicLvlBar] = useState(0);

  useEffect(() => {
    if (isSessionRunning) {
      startAudio();
    } else {
      stopAudio();
    }
    return () => stopAudio();
  }, [isSessionRunning]);

  const startAudio = async () => {
    try {
      const selectedMic = localStorage.getItem('sns_session_mic');
      const constraints = { audio: selectedMic ? { deviceId: { exact: selectedMic } } : true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      sourceRef.current = stream;

      drawWaveform();
    } catch (e) {
      console.error('Microphone access denied or error:', e);
    }
  };

  const stopAudio = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (sourceRef.current) {
      sourceRef.current.getTracks().forEach(t => t.stop());
      sourceRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas || !analyserRef.current) return;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;

    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let frameCount = 0;

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#0f1629';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#f59e0b';
      ctx.beginPath();

      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;
      let sum = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        sum += Math.abs(dataArray[i] - 128);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);

        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      // Basic Mic Level Calculation
      const avgAmp = sum / bufferLength;
      const micVolPct = Math.min(100, Math.round((avgAmp / 64) * 100));

      frameCount++;
      if (frameCount % 30 === 0) {
        setMicLvl(`${micVolPct}%`);
        setMicLvlBar(micVolPct);
        
        // Simulating breath consistency since backend does real complex FFT
        // (Assuming backend will use the amplitude history via API in later additions or this simple metric for UI)
        const simBreathCons = Math.max(0, 100 - Math.abs(50 - micVolPct));
        setBreathConsTxt(`${simBreathCons}%`);
        setBreathConsBar(simBreathCons);
        onMetricsUpdate({ breathConsistency: simBreathCons });
      }
    };
    draw();
  };

  return (
    <article className="dash-card" id="audioCard">
      <div className="dc-header">
        <div className="dc-icon dc-icon--amber">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
        </div>
        <div>
          <h3 className="dc-title">Audio Analytics</h3>
          <span className="dc-sub">Breath pattern decoder</span>
        </div>
      </div>

      <div className="wave-viewport">
        <div className="wave-overlay">
          <span className="wo-label">Breath Waveform</span>
          <span className="wo-status">{isSessionRunning ? 'Monitoring' : 'Standby'}</span>
        </div>
        <canvas id="waveCanvas" ref={canvasRef}></canvas>
      </div>

      <div className="metrics-row metrics-row--2col">
        <div className="metric-card metric-card--amber">
          <span className="metric-label">Breath Consistency</span>
          <span className="metric-value mono" id="breathConsTxt">{breathConsTxt}</span>
          <div className="metric-bar"><div className="metric-fill metric-fill--amber" id="breathConsBar" style={{width: `${breathConsBar}%`}}></div></div>
        </div>
        <div className="metric-card metric-card--teal">
          <span className="metric-label">Mic Amplitude</span>
          <span className="metric-value mono" id="micLvl">{micLvl}</span>
          <div className="metric-bar"><div className="metric-fill metric-fill--teal" id="micLvlBar" style={{width: `${micLvlBar}%`}}></div></div>
        </div>
      </div>
    </article>
  );
}
