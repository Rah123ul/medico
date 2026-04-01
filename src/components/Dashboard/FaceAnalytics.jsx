import React, { useRef, useEffect, useState } from 'react';
import { useMediaPipe } from '../../hooks/useMediaPipe';
import { RPPG_CONFIG, cielabSkinTransfer } from '../../utils/engine';

export default function FaceAnalytics({ isSessionRunning, onMetricsUpdate, onRppgData, onRawSync }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [fps, setFps] = useState('--');
  const [facesDetected, setFacesDetected] = useState('0');
  const [eyeScoreTxt, setEyeScoreTxt] = useState('0%');
  const [headScoreTxt, setHeadScoreTxt] = useState('0%');
  const [gazeScoreTxt, setGazeScoreTxt] = useState('0%');
  const [faceConfidence, setFaceConfidence] = useState('--');
  
  const [eyeBar, setEyeBar] = useState(0);
  const [headBar, setHeadBar] = useState(0);
  const [gazeBar, setGazeBar] = useState(0);

  const lastRppgTimeRef = useRef(0);
  const rppgSignalRef = useRef({ red: [], green: [], blue: [], timestamps: [] });
  const bpmUpdateCounterRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const cielabEnabled = localStorage.getItem('sns_cielab_enabled') !== 'false';

  const { isReady, startCamera, stopCamera } = useMediaPipe(videoRef, canvasRef, (results) => {
    const now = performance.now();
    if (now - lastRppgTimeRef.current < 1000 / RPPG_CONFIG.FPS) return;
    lastRppgTimeRef.current = now;
    
    const deltaTime = now - lastFrameTimeRef.current;
    const estimatedFPS = deltaTime > 0 ? 1000 / deltaTime : 30;
    lastFrameTimeRef.current = now;
    
    setFps(Math.round(estimatedFPS).toString());

    if (!results.multiFaceLandmarks?.[0]) {
      setFacesDetected('0');
      onMetricsUpdate({ fps: Math.round(estimatedFPS), faces: 0, eye: 0, head: 0, gaze: 0, confidence: 0 });
      return;
    }
    
    setFacesDetected('1');
    const lm = results.multiFaceLandmarks[0];
    
    const faceCtx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    faceCtx.save();
    faceCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    faceCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);

    if (isSessionRunning) {
      const roi = getRoiFromLandmarks(lm, canvasRef.current.width, canvasRef.current.height);
      if (roi) {
        const avgRGB = getAvgRGBFromRoi(faceCtx, roi, cielabEnabled);
        if (avgRGB) {
          const sig = rppgSignalRef.current;
          sig.red.push(avgRGB.red);
          sig.green.push(avgRGB.green);
          sig.blue.push(avgRGB.blue);
          sig.timestamps.push(now);
          while (sig.green.length > RPPG_CONFIG.BUFFER_SIZE) {
            sig.red.shift(); sig.green.shift(); sig.blue.shift(); sig.timestamps.shift();
          }
        }
      }
      
      bpmUpdateCounterRef.current++;
      const minBuf = Math.floor(RPPG_CONFIG.BUFFER_SIZE * RPPG_CONFIG.MIN_BUFFER_FILL);
      if (rppgSignalRef.current.green.length >= minBuf && bpmUpdateCounterRef.current % RPPG_CONFIG.UPDATE_INTERVAL === 0) {
        const signals = {
          redSignal: rppgSignalRef.current.red.slice(-minBuf),
          greenSignal: rppgSignalRef.current.green.slice(-minBuf),
          blueSignal: rppgSignalRef.current.blue.slice(-minBuf),
          fps: estimatedFPS
        };
        
        // Multiplayer Sync (Socket.io)
        if (onRawSync) onRawSync(signals);

        // Legacy/Solo API Analysis
        fetch('http://localhost:3000/api/analyze-session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-api-key': 'sk_test_demo456' },
          body: JSON.stringify(signals)
        })
        .then(res => res.json())
        .then(response => {
          if (!response.success) return;
          const { bpm, finalConfidence, hrv, meditationIndex } = response.data;
          setFaceConfidence(finalConfidence.toFixed(2));
          onRppgData({ bpm, hrv, meditationIndex });
          onMetricsUpdate({ fps: Math.round(estimatedFPS), faces: 1, eye: Math.random()*100, head: Math.random()*100, gaze: Math.random()*100, confidence: finalConfidence });
        }).catch(err => console.log('API Error:', err));
      }
    }
    faceCtx.restore();
  });

  useEffect(() => {
    if (isReady && isSessionRunning) {
      startCamera();
    } else {
      stopCamera();
    }
  }, [isReady, isSessionRunning]);

  return (
    <article className="dash-card" id="faceCard">
      <div className="dc-header">
        <div className="dc-icon dc-icon--violet">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>
        </div>
        <div>
          <h3 className="dc-title">Facial Analytics</h3>
          <span className="dc-sub">Real-time state detection</span>
        </div>
        <div className="dc-badges">
          <span className="live-tag"><span className="live-dot"></span>LIVE</span>
          <span className="data-tag">FPS <span id="fps" className="mono">{fps}</span></span>
          <span className="data-tag">FC <span id="facesDetected" className="mono">{facesDetected}</span></span>
        </div>
      </div>

      <div id="faceWrap" className="face-viewport">
        <div className="face-overlay-ui">
          <span className="fo-label">OPTICS</span>
          <span className="fo-status" id="detectionState">{isSessionRunning ? 'Scanning' : 'Standby'}</span>
        </div>
        <video ref={videoRef} className="input_video" autoPlay playsInline muted style={{display: 'none'}}></video>
        <canvas ref={canvasRef} className="output_canvas" width="420" height="300"></canvas>
        <div className="fo-bottom">
          <span id="faceConfidence">Confidence: {faceConfidence}</span>
        </div>
      </div>

      <div className="metrics-row">
        <div className="metric-card metric-card--violet">
          <span className="metric-label">Eye Relax</span>
          <span className="metric-value mono" id="eyeScoreTxt">{eyeScoreTxt}</span>
          <div className="metric-bar"><div className="metric-fill" id="eyeBar" style={{width: `${eyeBar}%`}}></div></div>
        </div>
        <div className="metric-card metric-card--amber">
          <span className="metric-label">Head Lock</span>
          <span className="metric-value mono" id="headScoreTxt">{headScoreTxt}</span>
          <div className="metric-bar"><div className="metric-fill metric-fill--amber" id="headBar" style={{width: `${headBar}%`}}></div></div>
        </div>
        <div className="metric-card metric-card--teal">
          <span className="metric-label">Gaze Stable</span>
          <span className="metric-value mono" id="gazeScoreTxt">{gazeScoreTxt}</span>
          <div className="metric-bar"><div className="metric-fill metric-fill--teal" id="gazeBar" style={{width: `${gazeBar}%`}}></div></div>
        </div>
      </div>
    </article>
  );
}

function getRoiFromLandmarks(lm, width, height) {
  const foreheadIds = [10, 67, 69, 104, 108, 151, 337, 333, 297, 338];
  const points = foreheadIds.map(idx => lm[idx]).filter(Boolean);
  if (!points.length) return null;

  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const w = (maxX - minX) * width;
  const h = (maxY - minY) * height;
  const x = minX * width;
  const y = minY * height;

  return {
    x: Math.floor(x + (w - Math.max(10, Math.floor(w * RPPG_CONFIG.ROI_WIDTH_FACTOR))) / 2),
    y: Math.floor(y + h * RPPG_CONFIG.ROI_Y_OFFSET),
    w: Math.max(10, Math.floor(w * RPPG_CONFIG.ROI_WIDTH_FACTOR)),
    h: Math.max(8, Math.floor(h * RPPG_CONFIG.ROI_HEIGHT_FACTOR))
  };
}

function getAvgRGBFromRoi(ctx, roi, cielabEnabled) {
  try {
    const { x, y, w, h } = roi;
    if (w <= 0 || h <= 0) return null;

    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    if (!data || !data.length) return null;

    if (cielabEnabled) {
      cielabSkinTransfer(data, w, h);
      ctx.putImageData(imageData, x, y);
    }

    let totalR = 0, totalG = 0, totalB = 0;
    const pxCount = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      totalR += data[i];
      totalG += data[i + 1];
      totalB += data[i + 2];
    }

    ctx.strokeStyle = cielabEnabled ? 'rgba(245,158,11,0.95)' : 'rgba(0,255,0,0.9)';
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    return { red: totalR / pxCount, green: totalG / pxCount, blue: totalB / pxCount };
  } catch (_) {
    return null;
  }
}
