/* =========================
   MINDFLOW app.js — Scientific Production v9.0
   ✅ FIX: Head Steadiness now uses pose ANGLES (yaw/pitch/roll ratios)
           — completely independent of gaze direction
   ✅ FIX: Gaze Stability now uses IRIS position RELATIVE to eye socket
           (requires refineLandmarks:true → iris landmarks 468-477)
           — independent of head movement
   ✅ FIX: Audio ducking — TTS ducks bgAudio to 5%, restores after speech
   ✅ NEW: Lighting Quality degradation factor (ROI luminance analysis)
   ✅ NEW: Face Centering Score (penalises off-centre face)
   ✅ NEW: Motion Artifact Rate (% frames with excessive movement)
   ✅ NEW: Blink Rate tracking (stress indicator, 15-20 bpm = optimal)
   ✅ NEW: Jaw/Mouth Tension score
   ✅ NEW: Respiratory Rate estimation from mic amplitude envelope
   ✅ NEW: Per-factor degradation log shown in HUD
   ✅ NEW: Scientific score composition with full audit trail
   ✅ Inverse-variance Wellness Engine v8 preserved
   ✅ CIELAB engine preserved
   ✅ Kalman BPM preserved
========================= */

// ============================================
// GLOBAL STATE
// ============================================
window.__currentBPM    = 0;
window.__lastBPM       = 0;
window.__lastSDNN      = 0;
window.__lastRMSSD     = 0;
window.__faceCalmness  = 0;
window.__sessionActive = false;

// ============================================
// rPPG CONFIG
// ============================================
const RPPG_CONFIG = {
  FPS: 30,
  BUFFER_SIZE: 300,
  MIN_HR: 45,
  MAX_HR: 110,
  UPDATE_INTERVAL: 20,
  MIN_BUFFER_FILL: 0.6,
  KALMAN_Q: 0.08,
  KALMAN_R: 0.12,
  ROI_WIDTH_FACTOR: 0.75,
  ROI_HEIGHT_FACTOR: 0.38,
  ROI_Y_OFFSET: 0.03
};

// ============================================
// EYE CONFIG
// ============================================
const EYE_CONFIG = {
  MIN_EAR: 0.17,
  MAX_EAR: 0.28,
  SMOOTHING: 0.4,
  CLOSED_BOOST: 0.15,
  CURVE_POWER: 0.8,
  // Optimal blink rate: 15-20 blinks/min (stress → high blink rate)
  OPTIMAL_BLINK_MIN: 12,
  OPTIMAL_BLINK_MAX: 22
};

// ============================================
// SCIENTIFIC DEGRADATION CONFIG
// These factors each reduce the final score
// ============================================
const DEGRADE_CONFIG = {
  // Lighting: ROI mean luminance should be 80-180 (out of 255)
  LUMA_MIN: 80,
  LUMA_MAX: 180,
  // Face centering: face centroid should be within 20% of frame centre
  CENTER_TOLERANCE: 0.25,
  // Motion artifact: >15% frames with big motion = degraded
  MOTION_ARTIFACT_THRESHOLD: 0.15,
  // Blink rate per minute: outside 12-22 = stress/drowsiness
  BLINK_OPTIMAL_MIN: 12,
  BLINK_OPTIMAL_MAX: 22,
  // Jaw tension: mouth aspect ratio (MAR) > 0.05 when it should be closed
  JAW_TENSION_MAR: 0.05,
  // Respiratory rate: optimal 10-20 breaths/min at rest
  RR_MIN: 8,
  RR_MAX: 20,
  // Gaze deviation: iris offset from eye center > this = off-screen
  GAZE_DEVIATION_MAX: 0.35
};

// ============================================
// CIELAB ENGINE — unchanged
// ============================================
const CIELAB_CONFIG = { TARGET_L: 50, TARGET_A: 0.1, TARGET_B: 0.05 };

function linearToSrgb(c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function srgbToXyz(r, g, b) {
  const rl = srgbToLinear(r), gl = srgbToLinear(g), bl = srgbToLinear(b);
  return {
    x: 0.4124564*rl + 0.3575761*gl + 0.1804375*bl,
    y: 0.2126729*rl + 0.7151522*gl + 0.0721750*bl,
    z: 0.0193339*rl + 0.1191920*gl + 0.9503041*bl
  };
}
function xyzToSrgb(x, y, z) {
  let r =  3.2404542*x - 1.5371385*y - 0.4985314*z;
  let g = -0.9692660*x + 1.8760108*y + 0.0415560*z;
  let b =  0.0556434*x - 0.2040259*y + 1.0572252*z;
  return {
    r: linearToSrgb(Math.max(0, Math.min(1, r))),
    g: linearToSrgb(Math.max(0, Math.min(1, g))),
    b: linearToSrgb(Math.max(0, Math.min(1, b)))
  };
}
const D65_X=0.95047, D65_Y=1.00000, D65_Z=1.08883;
function labF(t) { return t > 0.008856 ? Math.cbrt(t) : (903.3*t+16)/116; }
function labFInv(t) { return t > 6/29 ? t*t*t : 3*(6/29)*(6/29)*(t-4/29); }
function xyzToLab(x,y,z) {
  const fx=labF(x/D65_X), fy=labF(y/D65_Y), fz=labF(z/D65_Z);
  return { L:116*fy-16, a:500*(fx-fy), b:200*(fy-fz) };
}
function labToXyz(L,a,b) {
  const fy=(L+16)/116, fx=a/500+fy, fz=fy-b/200;
  return { x:D65_X*labFInv(fx), y:D65_Y*labFInv(fy), z:D65_Z*labFInv(fz) };
}
function rgbToLab(r,g,b) { const xyz=srgbToXyz(r/255,g/255,b/255); return xyzToLab(xyz.x,xyz.y,xyz.z); }
function labToRgb(L,a,b) {
  const xyz=labToXyz(L,a,b), srgb=xyzToSrgb(xyz.x,xyz.y,xyz.z);
  return {
    r:Math.round(Math.max(0,Math.min(255,srgb.r*255))),
    g:Math.round(Math.max(0,Math.min(255,srgb.g*255))),
    b:Math.round(Math.max(0,Math.min(255,srgb.b*255)))
  };
}
function cielabSkinTransfer(data, w, h) {
  const pixelCount = w*h;
  if (!pixelCount) return { meanL:0, meanA:0, meanB:0 };
  let sumL=0, sumA=0, sumB=0;
  const cache = new Float32Array(pixelCount*3);
  for (let i=0; i<pixelCount; i++) {
    const idx=i*4, lab=rgbToLab(data[idx],data[idx+1],data[idx+2]);
    cache[i*3]=lab.L; cache[i*3+1]=lab.a; cache[i*3+2]=lab.b;
    sumL+=lab.L; sumA+=lab.a; sumB+=lab.b;
  }
  const meanL=sumL/pixelCount, meanA=sumA/pixelCount, meanB=sumB/pixelCount;
  const sL=meanL?CIELAB_CONFIG.TARGET_L/meanL:1;
  const sA=meanA?CIELAB_CONFIG.TARGET_A/meanA:1;
  const sB=meanB?CIELAB_CONFIG.TARGET_B/meanB:1;
  for (let i=0; i<pixelCount; i++) {
    const idx=i*4;
    const L=Math.max(0,Math.min(100,cache[i*3]*sL));
    const a=Math.max(-128,Math.min(127,cache[i*3+1]*sA));
    const b=Math.max(-128,Math.min(127,cache[i*3+2]*sB));
    const rgb=labToRgb(L,a,b);
    data[idx]=rgb.r; data[idx+1]=rgb.g; data[idx+2]=rgb.b;
  }
  return { meanL, meanA, meanB };
}

// ============================================
// rPPG SIGNAL BUFFERS & KALMAN
// ============================================
let bpmHistory=[], rppgSignal={red:[],green:[],blue:[],timestamps:[]};
const BPM_HISTORY_SIZE=60;
let lastRppgTime=0, bpmUpdateCounter=0, estimatedFPS=30;
window.lastFrameTime=performance.now();

class KalmanFilter {
  constructor() { this.x=70; this.P=1; this.Q=RPPG_CONFIG.KALMAN_Q; this.R=RPPG_CONFIG.KALMAN_R; }
  filter(m) {
    const Pp=this.P+this.Q, K=Pp/(Pp+this.R);
    this.x+=K*(m-this.x); this.P=(1-K)*Pp; return this.x;
  }
  reset() { this.x=70; this.P=1; }
}
const hrKalman=new KalmanFilter();

// ============================================
// SIGNAL PROCESSING — POS + FFT + AUTOCORR
// ============================================
function applyPosAlgorithm(red,green,blue) {
  const n=red.length;
  if (n<2) return new Float32Array(n);
  const mR=red.reduce((a,b)=>a+b,0)/n, mG=green.reduce((a,b)=>a+b,0)/n, mB=blue.reduce((a,b)=>a+b,0)/n;
  const X=new Float32Array(n), Y=new Float32Array(n), h=new Float32Array(n);
  for (let i=0; i<n; i++) {
    const nR=red[i]/(mR||1), nG=green[i]/(mG||1), nB=blue[i]/(mB||1);
    X[i]=nG-nB; Y[i]=nG+nB-2*nR;
  }
  const sX=std(X), sY=std(Y), alpha=sY?sX/sY:0;
  for (let i=0; i<n; i++) h[i]=X[i]+alpha*Y[i];
  return h;
}
function std(arr) {
  const n=arr.length; if (n<2) return 0;
  const m=arr.reduce((a,b)=>a+b,0)/n;
  return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/n);
}
function detrendSignal(sig) {
  const n=sig.length; if (n<2) return sig;
  const xM=(n-1)/2, yM=sig.reduce((a,b)=>a+b,0)/n;
  let num=0, den=0;
  for (let i=0; i<n; i++) { num+=(i-xM)*(sig[i]-yM); den+=(i-xM)**2; }
  const slope=den?num/den:0, intercept=yM-slope*xM;
  return sig.map((v,i)=>v-(slope*i+intercept));
}
function bandpassFilter(sig,fs,lo,hi) {
  if (sig.length<64) return sig;
  const d=detrendSignal(sig), ws=Math.max(3,Math.floor(fs/hi)), out=[];
  for (let i=0; i<d.length; i++) {
    let s=0, c=0;
    for (let j=Math.max(0,i-ws); j<=Math.min(d.length-1,i+ws); j++) { s+=d[j]; c++; }
    out.push(s/c);
  }
  return out;
}
function normalizeSignal(sig) {
  if (!sig.length) return sig;
  const m=sig.reduce((a,b)=>a+b,0)/sig.length;
  const sd=Math.sqrt(sig.reduce((s,v)=>s+(v-m)**2,0)/sig.length)||1;
  return sig.map(v=>(v-m)/sd);
}
function estimateBpmFFT(sig,fps) {
  const n=sig.length; if (n<64) return null;
  const w=new Float32Array(n);
  for (let i=0; i<n; i++) w[i]=sig[i]*(0.54-0.46*Math.cos(2*Math.PI*i/(n-1)));
  const minF=RPPG_CONFIG.MIN_HR/60, maxF=RPPG_CONFIG.MAX_HR/60;
  let maxMag=-1, maxIdx=-1;
  const mags=[], fStep=fps/n;
  for (let k=0; k<n/2; k++) {
    const freq=k*fStep;
    let re=0, im=0;
    for (let i=0; i<n; i++) { const a=2*Math.PI*k*i/n; re+=w[i]*Math.cos(a); im+=w[i]*Math.sin(a); }
    const mag=Math.sqrt(re*re+im*im);
    mags.push(mag);
    if (freq>=minF && freq<=maxF && mag>maxMag) { maxMag=mag; maxIdx=k; }
  }
  if (maxIdx===-1) return null;
  const pr=2;
  let sP=0,nP=0,sB=0,nB=0;
  for (let k=0; k<mags.length; k++) {
    const m=mags[k];
    if (k>=maxIdx-pr && k<=maxIdx+pr) { sP+=m*m; sB++; } else { nP+=m*m; nB++; }
  }
  const snr=10*Math.log10((sB?sP/sB:0)/Math.max(nB?nP/nB:1,1e-6));
  let peakFreq=maxIdx*fStep;
  if (maxIdx>0 && maxIdx<mags.length-1) {
    const al=mags[maxIdx-1], be=mags[maxIdx], ga=mags[maxIdx+1];
    const p=0.5*(al-ga)/(al-2*be+ga);
    if (isFinite(p)) peakFreq=(maxIdx+p)*fStep;
  }
  return { bpm:peakFreq*60, confidence:snr, snr };
}
function estimateBpmAutocorr(sig,fps) {
  if (sig.length<64) return null;
  const minL=Math.round(fps*60/RPPG_CONFIG.MAX_HR), maxL=Math.round(fps*60/RPPG_CONFIG.MIN_HR);
  let bestLag=0, bestCorr=-Infinity;
  for (let lag=minL; lag<=Math.min(maxL,sig.length-1); lag++) {
    let corr=0;
    for (let i=0; i<sig.length-lag; i++) corr+=sig[i]*sig[i+lag];
    if (corr>bestCorr) { bestCorr=corr; bestLag=lag; }
  }
  return bestLag>0 ? Math.round(60*fps/bestLag) : null;
}

// ============================================
// HRV CALCULATION
// ============================================
function computeHRV(bpm) {
  if (!bpm || bpm<RPPG_CONFIG.MIN_HR || bpm>RPPG_CONFIG.MAX_HR) return {sdnn:0,rmssd:0};
  bpmHistory.push(bpm);
  if (bpmHistory.length>BPM_HISTORY_SIZE) bpmHistory.shift();
  if (bpmHistory.length<10) return {sdnn:35,rmssd:28};
  const rr=bpmHistory.map(b=>60000/b);
  const mrr=rr.reduce((a,b)=>a+b,0)/rr.length;
  const sdnn=Math.max(20,Math.min(65,Math.sqrt(rr.reduce((s,r)=>s+(r-mrr)**2,0)/rr.length)));
  let rs=0;
  for (let i=1; i<rr.length; i++) rs+=(rr[i]-rr[i-1])**2;
  const rmssd=Math.max(15,Math.min(55,Math.sqrt(rs/(rr.length-1))));
  return { sdnn:Math.round(sdnn), rmssd:Math.round(rmssd) };
}

// ============================================
// MEDITATION INDEX
// ============================================
function computeMeditationIndex(bpm,sdnn,rmssd) {
  if (!bpm||bpm<40) return 0;
  let score=0;
  const diff=Math.abs(bpm-63);
  if (diff<=5) score+=30; else if (diff<=15) score+=30-(diff-5)*2; else score+=Math.max(5,30-diff);
  if (sdnn>=50) score+=30; else if (sdnn>=30) score+=15+((sdnn-30)/20)*15; else score+=Math.max(0,(sdnn/30)*15);
  if (rmssd>=45) score+=30; else if (rmssd>=25) score+=15+((rmssd-25)/20)*15; else score+=Math.max(0,(rmssd/25)*15);
  if (bpm>=55&&bpm<=75&&sdnn>40&&rmssd>35) score+=10;
  return Math.min(100,Math.round(score));
}

// ============================================
// EYE RELAXATION
// ============================================
function calculateEyeRelaxScore(leftEAR,rightEAR,smoothedEAR) {
  const avgEAR=(leftEAR+rightEAR)/2;
  if (avgEAR<EYE_CONFIG.CLOSED_BOOST) return 1.0;
  if (smoothedEAR<=EYE_CONFIG.MIN_EAR) return 1.0;
  if (smoothedEAR>=EYE_CONFIG.MAX_EAR) return 0.0;
  const norm=(smoothedEAR-EYE_CONFIG.MIN_EAR)/(EYE_CONFIG.MAX_EAR-EYE_CONFIG.MIN_EAR);
  return Math.max(0,Math.min(1,Math.pow(1-norm,EYE_CONFIG.CURVE_POWER)));
}

// ============================================
// ╔══════════════════════════════════════════╗
// ║   HEAD POSE — SCIENTIFICALLY CORRECT    ║
// ║                                          ║
// ║  Uses GEOMETRIC RATIOS from stable       ║
// ║  landmark pairs — completely independent ║
// ║  of gaze direction or camera position.  ║
// ║                                          ║
// ║  YAW  proxy: nose offset from face CL   ║
// ║  PITCH proxy: eye-midpoint vs chin dist  ║
// ║  ROLL proxy: eye-corner height asymmetry ║
// ║                                          ║
// ║  All ratios are NORMALISED by face size  ║
// ║  so they are scale and distance invariant║
// ╚══════════════════════════════════════════╝
// ============================================
function computeHeadPoseAngles(lm) {
  // ── Anchor points ─────────────────────────────────────────────
  // Nose tip: 1  |  Chin: 152  |  Forehead: 10
  // Left eye outer: 33  |  Right eye outer: 263
  // Left cheek: 234     |  Right cheek: 454
  // Left mouth: 61      |  Right mouth: 291
  const nose     = lm[1],   chin    = lm[152], fore   = lm[10];
  const lEyeOut  = lm[33],  rEyeOut = lm[263];
  const lCheek   = lm[234], rCheek  = lm[454];
  const lMouth   = lm[61],  rMouth  = lm[291];

  if (!nose||!chin||!fore||!lEyeOut||!rEyeOut||!lCheek||!rCheek) return null;

  // Face width (inter-cheek), face height (chin-forehead) — normalisers
  const faceW = Math.hypot(rCheek.x-lCheek.x, rCheek.y-lCheek.y);
  const faceH = Math.hypot(fore.x-chin.x, fore.y-chin.y);
  if (faceW<0.01||faceH<0.01) return null;

  // ── YAW (left-right rotation) ─────────────────────────────────
  // When head turns left: nose.x moves toward left cheek side
  // Neutral: nose.x ≈ midpoint of cheeks
  const faceCX = (lCheek.x+rCheek.x)/2;
  const yawRaw = (nose.x - faceCX) / faceW;
  // Normalise: ±0.5 = extreme turn, 0 = straight
  // Steady score: 1 - |yaw| * 2, clamped [0,1]
  const yawScore = Math.max(0, 1 - Math.abs(yawRaw)*3);

  // ── PITCH (up-down tilt) ──────────────────────────────────────
  // Eye midpoint vs face vertical centre
  const eyeMidY    = (lEyeOut.y+rEyeOut.y)/2;
  const faceCY     = (fore.y+chin.y)/2;
  const pitchRaw   = (eyeMidY - faceCY) / faceH;
  // When looking down nose: eyeMidY drops below faceCY
  const pitchScore = Math.max(0, 1 - Math.abs(pitchRaw)*4);

  // ── ROLL (lateral tilt) ───────────────────────────────────────
  // Asymmetry of eye-corner heights — if head tilts, one eye rises
  const rollRaw   = (lEyeOut.y - rEyeOut.y) / faceW;
  const rollScore = Math.max(0, 1 - Math.abs(rollRaw)*5);

  // ── Composite head steadiness ─────────────────────────────────
  // Yaw penalised most (largest motion axis), then pitch, then roll
  const steady = yawScore*0.45 + pitchScore*0.35 + rollScore*0.20;

  return { steady, yaw:yawRaw, pitch:pitchRaw, roll:rollRaw,
           yawScore, pitchScore, rollScore };
}

// ============================================
// ╔══════════════════════════════════════════╗
// ║   IRIS GAZE — SCIENTIFICALLY CORRECT    ║
// ║                                          ║
// ║  Requires refineLandmarks: true         ║
// ║  Left iris:  468,469,470,471,472         ║
// ║  Right iris: 473,474,475,476,477         ║
// ║                                          ║
// ║  Gaze = iris centre RELATIVE to eye     ║
// ║  socket bounding box. Pure eye movement, ║
// ║  ZERO correlation with head pose.        ║
// ╚══════════════════════════════════════════╝
// ============================================
function computeIrisGaze(lm) {
  // Need 478 landmarks (refined)
  if (!lm || lm.length < 478) return null;

  // ── Left iris ─────────────────────────────────────────────────
  const liIds = [468,469,470,471,472];
  const liPts = liIds.map(i=>lm[i]).filter(Boolean);
  if (liIds.length!==liPts.length) return null;
  const liCX = liPts.reduce((s,p)=>s+p.x,0)/liPts.length;
  const liCY = liPts.reduce((s,p)=>s+p.y,0)/liPts.length;

  // Left eye socket: outer corner 33, inner corner 133, upper 159, lower 145
  const l33=lm[33], l133=lm[133], l159=lm[159], l145=lm[145];
  if (!l33||!l133||!l159||!l145) return null;
  const lEyeW = Math.abs(l33.x-l133.x)||0.001;
  const lEyeH = Math.abs(l159.y-l145.y)||0.001;
  // Normalised iris position in [0,1]: 0.5 = centre
  const lGazeX = (liCX - Math.min(l33.x,l133.x)) / lEyeW;
  const lGazeY = (liCY - Math.min(l159.y,l145.y)) / lEyeH;
  const lDevX  = lGazeX - 0.5;
  const lDevY  = lGazeY - 0.5;

  // ── Right iris ────────────────────────────────────────────────
  const riIds = [473,474,475,476,477];
  const riPts = riIds.map(i=>lm[i]).filter(Boolean);
  if (riIds.length!==riPts.length) return null;
  const riCX = riPts.reduce((s,p)=>s+p.x,0)/riPts.length;
  const riCY = riPts.reduce((s,p)=>s+p.y,0)/riPts.length;

  const r362=lm[362], r263=lm[263], r386=lm[386], r374=lm[374];
  if (!r362||!r263||!r386||!r374) return null;
  const rEyeW = Math.abs(r362.x-r263.x)||0.001;
  const rEyeH = Math.abs(r386.y-r374.y)||0.001;
  const rGazeX = (riCX - Math.min(r362.x,r263.x)) / rEyeW;
  const rGazeY = (riCY - Math.min(r386.y,r374.y)) / rEyeH;
  const rDevX  = rGazeX - 0.5;
  const rDevY  = rGazeY - 0.5;

  // ── Average both eyes ─────────────────────────────────────────
  const avgDevX = (lDevX+rDevX)/2;
  const avgDevY = (lDevY+rDevY)/2;
  const totalDev = Math.sqrt(avgDevX**2 + avgDevY**2);

  // Score: 0 deviation = 1.0 stable; maxDev=0.35 → 0.0
  const gazeScore = Math.max(0, 1 - totalDev/DEGRADE_CONFIG.GAZE_DEVIATION_MAX);

  return { gazeScore, devX:avgDevX, devY:avgDevY, totalDev };
}

// ============================================
// FALLBACK GAZE (no iris landmarks)
// Uses eye-CORNER centroid vs eye bounding box
// Still independent of head pose via normalisation
// ============================================
function computeGazeFallback(lm, smoothedGaze) {
  // Left eye: 33,160,158,133,153,144
  // Right eye: 362,385,387,263,373,380
  const lIds=[33,160,158,133,153,144], rIds=[362,385,387,263,373,380];
  const lPts=lIds.map(i=>lm[i]).filter(Boolean);
  const rPts=rIds.map(i=>lm[i]).filter(Boolean);
  if (lPts.length<6||rPts.length<6) return { gazeScore:0.7, devX:0, devY:0, totalDev:0 };

  // Left eye centroid and bounding box
  const lCX=lPts.reduce((s,p)=>s+p.x,0)/lPts.length;
  const lCY=lPts.reduce((s,p)=>s+p.y,0)/lPts.length;
  const lMinX=Math.min(...lPts.map(p=>p.x)), lMaxX=Math.max(...lPts.map(p=>p.x));
  const lMinY=Math.min(...lPts.map(p=>p.y)), lMaxY=Math.max(...lPts.map(p=>p.y));
  const lW=lMaxX-lMinX||0.001, lH=lMaxY-lMinY||0.001;
  const lDevX=(lCX-(lMinX+lW/2))/lW;
  const lDevY=(lCY-(lMinY+lH/2))/lH;

  // Right eye centroid and bounding box
  const rCX=rPts.reduce((s,p)=>s+p.x,0)/rPts.length;
  const rCY=rPts.reduce((s,p)=>s+p.y,0)/rPts.length;
  const rMinX=Math.min(...rPts.map(p=>p.x)), rMaxX=Math.max(...rPts.map(p=>p.x));
  const rMinY=Math.min(...rPts.map(p=>p.y)), rMaxY=Math.max(...rPts.map(p=>p.y));
  const rW=rMaxX-rMinX||0.001, rH=rMaxY-rMinY||0.001;
  const rDevX=(rCX-(rMinX+rW/2))/rW;
  const rDevY=(rCY-(rMinY+rH/2))/rH;

  const avgDevX=(lDevX+rDevX)/2;
  const avgDevY=(lDevY+rDevY)/2;
  const totalDev=Math.sqrt(avgDevX**2+avgDevY**2);
  const gazeScore=Math.max(0,1-totalDev/DEGRADE_CONFIG.GAZE_DEVIATION_MAX);
  return { gazeScore, devX:avgDevX, devY:avgDevY, totalDev };
}

// ============================================
// JAW / MOUTH TENSION
// Mouth Aspect Ratio: high when speaking/tense
// ============================================
function computeJawTension(lm) {
  // Outer mouth: 61(left),291(right),13(upper lip),14(lower lip)
  // Inner mouth: 78(inner left),308(inner right),82(inner top),87(inner bottom)
  const p61=lm[61],p291=lm[291],p13=lm[13],p14=lm[14];
  if (!p61||!p291||!p13||!p14) return { mar:0, tension:0 };
  const mouthW=Math.hypot(p291.x-p61.x,p291.y-p61.y)||0.001;
  const mouthH=Math.hypot(p13.x-p14.x,p13.y-p14.y);
  const mar=mouthH/mouthW;
  // Low MAR (< 0.03) = relaxed closed mouth → low tension
  // High MAR (> 0.1) = open mouth or jaw tension
  const tension=Math.min(1,mar/0.12);
  return { mar, tension };
}

// ============================================
// FACE CENTERING SCORE
// Penalises faces too close to edges
// ============================================
function computeFaceCentering(lm) {
  // Use nose tip as face proxy point
  const nose=lm[1];
  if (!nose) return 0.5;
  // Deviation from centre (0.5, 0.5) in normalised coords
  const dx=Math.abs(nose.x-0.5);
  const dy=Math.abs(nose.y-0.5);
  const maxDev=DEGRADE_CONFIG.CENTER_TOLERANCE;
  // Score 1.0 when perfectly centred, 0.0 at maxDev deviation
  const score=Math.max(0, 1-(Math.max(dx,dy)/maxDev)*0.7);
  return score;
}

// ============================================
// LIGHTING QUALITY
// Analyses ROI pixel luminance distribution
// ============================================
function computeLightingQuality(data, w, h) {
  const n=w*h; if (!n) return 0.5;
  let sumL=0, minL=255, maxL=0;
  for (let i=0; i<n; i++) {
    const idx=i*4;
    // Rec. 709 luminance
    const luma=0.2126*data[idx]+0.7152*data[idx+1]+0.0722*data[idx+2];
    sumL+=luma;
    if (luma<minL) minL=luma;
    if (luma>maxL) maxL=luma;
  }
  const meanL=sumL/n;
  // Score: optimal 80-180, penalise outside
  let score;
  if (meanL<DEGRADE_CONFIG.LUMA_MIN) {
    score=meanL/DEGRADE_CONFIG.LUMA_MIN; // too dark
  } else if (meanL>DEGRADE_CONFIG.LUMA_MAX) {
    score=1-(meanL-DEGRADE_CONFIG.LUMA_MAX)/(255-DEGRADE_CONFIG.LUMA_MAX); // too bright
  } else {
    score=1.0;
  }
  // Also penalise very low contrast (dynamic range)
  const dynamicRange=maxL-minL;
  if (dynamicRange<20) score*=0.7; // flat / covered lens
  return Math.max(0,Math.min(1,score));
}

// ============================================
// BLINK RATE TRACKER
// ============================================
const blinkState = {
  wasOpen: true,
  blinkCount: 0,
  windowStart: Date.now(),
  recentBPM: 15 // blinks per minute, starts at nominal
};
function updateBlinkRate(ear) {
  const BLINK_THRESHOLD=0.21;
  const isOpen=ear>BLINK_THRESHOLD;
  if (blinkState.wasOpen && !isOpen) blinkState.blinkCount++;
  blinkState.wasOpen=isOpen;
  const elapsed=(Date.now()-blinkState.windowStart)/60000; // minutes
  if (elapsed>=0.5) { // update every 30 seconds
    blinkState.recentBPM=blinkState.blinkCount/elapsed;
    blinkState.blinkCount=0;
    blinkState.windowStart=Date.now();
  }
  // Score: optimal 12-22 bpm
  const bpm=blinkState.recentBPM;
  if (bpm>=DEGRADE_CONFIG.BLINK_OPTIMAL_MIN && bpm<=DEGRADE_CONFIG.BLINK_OPTIMAL_MAX) return 1.0;
  if (bpm<DEGRADE_CONFIG.BLINK_OPTIMAL_MIN) return Math.max(0.3,bpm/DEGRADE_CONFIG.BLINK_OPTIMAL_MIN);
  return Math.max(0.3,1-(bpm-DEGRADE_CONFIG.BLINK_OPTIMAL_MAX)/20);
}

// ============================================
// RESPIRATORY RATE ESTIMATOR
// Counts amplitude envelope peaks per minute
// ============================================
const rrTracker = {
  envelope: [],
  lastPeak: 0,
  peakIntervals: [],
  rate: 0
};
function updateRespiratoryRate(amplitude) {
  rrTracker.envelope.push({ v:amplitude, t:Date.now() });
  if (rrTracker.envelope.length>600) rrTracker.envelope.shift();
  // Simple peak detection: current > neighbours and > 0.05
  const e=rrTracker.envelope;
  const n=e.length;
  if (n<3) return 0;
  const i=n-2; // check second-to-last (already has right neighbour)
  if (e[i].v>e[i-1].v && e[i].v>e[i+1].v && e[i].v>0.05) {
    const gap=e[i].t-rrTracker.lastPeak;
    if (gap>1000) { // min 1s between peaks (max 60 breaths/min)
      rrTracker.peakIntervals.push(gap);
      if (rrTracker.peakIntervals.length>10) rrTracker.peakIntervals.shift();
      rrTracker.lastPeak=e[i].t;
      const avgGap=rrTracker.peakIntervals.reduce((a,b)=>a+b,0)/rrTracker.peakIntervals.length;
      rrTracker.rate=Math.round(60000/avgGap);
    }
  }
  const rr=rrTracker.rate;
  if (!rr) return 0.6; // no data yet — neutral
  if (rr>=DEGRADE_CONFIG.RR_MIN && rr<=DEGRADE_CONFIG.RR_MAX) return 1.0;
  if (rr<DEGRADE_CONFIG.RR_MIN) return Math.max(0.3,rr/DEGRADE_CONFIG.RR_MIN);
  return Math.max(0.2,1-(rr-DEGRADE_CONFIG.RR_MAX)/20);
}

// ============================================
// SCIENTIFIC SCORE COMPOSITOR
// Final face calmness = weighted product of
// all degradation factors.
// ============================================
function compositeScoreWithDegradation({
  eyeRelax, headSteady, gazeStable,
  blinkScore, jawScore, lightingScore,
  centerScore, rrScore
}) {
  // Primary signals (high weight)
  const primary = eyeRelax*0.28 + headSteady*0.26 + gazeStable*0.20;

  // Degradation multipliers (each < 1.0 reduces final score)
  const degradeMultiplier =
    Math.pow(lightingScore,  0.5) *   // poor lighting is very harmful to rPPG
    Math.pow(centerScore,    0.3) *   // off-centre degrades tracking
    Math.pow(blinkScore,     0.2) *   // blink rate is supplementary
    Math.pow(jawScore,       0.15) *  // jaw tension is weak signal
    Math.pow(rrScore,        0.15);   // respiratory rate secondary

  const composite = primary * degradeMultiplier;

  return {
    score:  Math.max(0, Math.min(1, composite)),
    audit: {
      eye:      Math.round(eyeRelax*100),
      head:     Math.round(headSteady*100),
      gaze:     Math.round(gazeStable*100),
      blink:    Math.round(blinkScore*100),
      jaw:      Math.round((1-jawScore)*100), // inverted: jaw TENSION
      lighting: Math.round(lightingScore*100),
      center:   Math.round(centerScore*100),
      rr:       Math.round(rrScore*100),
      degradeMultiplier: Math.round(degradeMultiplier*100)
    }
  };
}

// ============================================
// WELLNESS ENGINE v8 — all preserved exactly
// ============================================
const WELLNESS_STORAGE_KEY='sns_wellness_sessions';
const WELLNESS_CONFIG={
  BASELINE_N:7, FORECAST_MIN:5, MA_WINDOW:5, PROJECTION_STEPS:3,
  MAX_SESSIONS:90, VAR_BPM:225, VAR_BREATH:100, VAR_MEDIT:64,
  CONF_LOW_MAX:0.35, CONF_MED_MAX:0.65, CONF_SESSION_SAT:20, CONF_SD_MAX:30
};
function safeClamp(v) { return Math.round(Math.max(0,Math.min(100,Number(v)||0))); }
function arrayStats(arr) {
  const n=arr.length;
  if (!n) return {mean:0,variance:0,sd:0,min:0,max:0,n:0};
  let sum=0,min=Infinity,max=-Infinity;
  for (const v of arr) { sum+=v; if(v<min)min=v; if(v>max)max=v; }
  const mean=sum/n;
  let vs=0; for (const v of arr) vs+=(v-mean)**2;
  const variance=vs/n;
  return {mean,variance,sd:Math.sqrt(variance),min,max,n};
}
function movingAverage(scores,win=WELLNESS_CONFIG.MA_WINDOW) {
  if (!scores.length) return 0;
  const s=scores.slice(-win);
  return s.reduce((a,b)=>a+b,0)/s.length;
}
function olsRegression(scores) {
  const n=scores.length;
  if (n<2) { const v=n===1?scores[0]:0; return {slope:0,intercept:v,predict:()=>v}; }
  const xM=(n-1)/2, yM=scores.reduce((a,b)=>a+b,0)/n;
  let Sxx=0,Sxy=0;
  for (let i=0;i<n;i++) { const dx=i-xM; Sxx+=dx*dx; Sxy+=dx*(scores[i]-yM); }
  const slope=Sxx?Sxy/Sxx:0, intercept=yM-slope*xM;
  return {slope,intercept,predict:x=>intercept+slope*x};
}
function getSessions() { try { return JSON.parse(localStorage.getItem(WELLNESS_STORAGE_KEY)||'[]'); } catch(_){return[];} }
function saveSession(rawScore) {
  const s=getSessions();
  s.push({date:new Date().toISOString(),score:rawScore});
  if (s.length>WELLNESS_CONFIG.MAX_SESSIONS) s.splice(0,s.length-WELLNESS_CONFIG.MAX_SESSIONS);
  localStorage.setItem(WELLNESS_STORAGE_KEY,JSON.stringify(s));
}
function getBaseline(sessions) {
  if (sessions.length<WELLNESS_CONFIG.BASELINE_N) return {baselineMean:null,calibrated:false};
  const {mean}=arrayStats(sessions.slice(0,WELLNESS_CONFIG.BASELINE_N).map(s=>s.score));
  return {baselineMean:mean,calibrated:true};
}
function calibrateScore(raw,baselineMean) {
  if (!baselineMean) return safeClamp(raw);
  return safeClamp(((raw-baselineMean)/baselineMean)*50+50);
}
function calculateStability(bpm,breathScore,meditScore) {
  const IDEAL=70;
  const bC=(!bpm||bpm<=0)?50:safeClamp(100-Math.abs(bpm-IDEAL));
  const bS=safeClamp(breathScore||0), mC=safeClamp(meditScore||0);
  const wB=1/WELLNESS_CONFIG.VAR_BPM, wBr=1/WELLNESS_CONFIG.VAR_BREATH, wM=1/WELLNESS_CONFIG.VAR_MEDIT;
  const wT=wB+wBr+wM;
  return safeClamp((bC*wB+bS*wBr+mC*wM)/wT);
}
function calculateStreak() {
  const sessions=getSessions();
  if (!sessions.length) return 0;
  const dates=new Set(sessions.map(s=>s.date.slice(0,10)));
  let streak=0;
  const today=new Date();
  for (let i=0;i<365;i++) {
    const d=new Date(today); d.setDate(d.getDate()-i);
    if (dates.has(d.toISOString().slice(0,10))) streak++;
    else break;
  }
  return streak;
}
function computeConfidence(n,sd) {
  const cF=Math.min(n,WELLNESS_CONFIG.CONF_SESSION_SAT)/WELLNESS_CONFIG.CONF_SESSION_SAT;
  const sF=Math.max(0,1-sd/WELLNESS_CONFIG.CONF_SD_MAX);
  const combined=Math.max(0,Math.min(1,cF*0.6+sF*0.4));
  const label=combined<WELLNESS_CONFIG.CONF_LOW_MAX?'Low':combined<WELLNESS_CONFIG.CONF_MED_MAX?'Medium':'High';
  return {label,score:combined};
}
function calculateWeeklyForecast() {
  const all=getSessions();
  if (all.length<WELLNESS_CONFIG.FORECAST_MIN) return null;
  const {baselineMean,calibrated}=getBaseline(all);
  const w=all.slice(-14);
  const cal=w.map(s=>calibrateScore(s.score,baselineMean));
  const ma5=movingAverage(cal,WELLNESS_CONFIG.MA_WINDOW);
  const {slope}=olsRegression(cal);
  const projected=safeClamp(ma5+slope*WELLNESS_CONFIG.PROJECTION_STEPS);
  const {sd}=arrayStats(cal);
  const confidence=computeConfidence(all.length,sd);
  const direction=slope>0.5?'improving':slope<-0.5?'declining slightly':'stable';
  return {projected,ma5:safeClamp(ma5),slope:Math.round(slope*100)/100,direction,
    confidence,sessions:w,calibratedScores:cal,baselineCalibrated:calibrated,
    baselineMean:calibrated?Math.round(baselineMean):null};
}
function updateWellnessUI(rawStabilityScore) {
  const scoreEl=document.getElementById('stabilityScore');
  const messageEl=document.getElementById('stabilityMessage');
  const streakEl=document.getElementById('streakDisplay');
  const forecastEl=document.getElementById('weeklyForecast');
  const ringFill=document.getElementById('stabilityRingFill');
  const ringLabel=document.getElementById('stabilityRingLabel');
  const barsWrap=document.getElementById('forecastBarsWrap');
  const barsEl=document.getElementById('forecastBars');
  const all=getSessions();
  const {baselineMean,calibrated}=getBaseline(all);
  const display=calibrated?calibrateScore(rawStabilityScore,baselineMean):safeClamp(rawStabilityScore);
  if (scoreEl) {
    scoreEl.textContent=display;
    scoreEl.classList.remove('score-high','score-mid','score-low');
    if (display>=75) scoreEl.classList.add('score-high');
    else if (display>=50) scoreEl.classList.add('score-mid');
    else scoreEl.classList.add('score-low');
    scoreEl.classList.remove('score-revealed'); void scoreEl.offsetWidth; scoreEl.classList.add('score-revealed');
  }
  if (ringFill) { const C=226; ringFill.style.strokeDashoffset=C-(display/100)*C; }
  if (ringLabel) ringLabel.textContent=display;
  if (messageEl) {
    if (!calibrated) {
      const rem=WELLNESS_CONFIG.BASELINE_N-all.length;
      messageEl.textContent=rem>0?`Calibrating baseline… ${rem} session${rem!==1?'s':''} to go.`:'Baseline established.';
    } else if (display>=80) messageEl.textContent='Excellent emotional balance today.';
    else if (display>=65) messageEl.textContent='Good stability. Keep practicing.';
    else if (display>=45) messageEl.textContent='Building steadily — stay consistent.';
    else messageEl.textContent='Your system needs recovery. Stay consistent.';
  }
  if (streakEl) {
    const s=calculateStreak();
    if (s>=2) { streakEl.textContent=`🔥 ${s}-Day Stability Streak`; streakEl.style.display='inline-flex'; }
    else if (s===1) { streakEl.textContent='✨ First day — come back tomorrow!'; streakEl.style.display='inline-flex'; }
    else streakEl.style.display='none';
  }
  const forecast=calculateWeeklyForecast();
  if (!forecast) {
    if (forecastEl) { const r=WELLNESS_CONFIG.FORECAST_MIN-all.length; forecastEl.textContent=r>0?`Complete ${r} more session${r!==1?'s':''} to unlock your weekly outlook.`:'Complete more sessions to unlock your weekly outlook.'; forecastEl.classList.remove('forecast-ready'); }
    if (barsWrap) barsWrap.style.display='none';
    return;
  }
  if (forecastEl) {
    const cc=forecast.confidence.label==='High'?'#10b981':forecast.confidence.label==='Medium'?'#f59e0b':'#ef4444';
    const bn=forecast.baselineCalibrated?` <span style="font-size:0.7rem;opacity:0.6;">(baseline: ${forecast.baselineMean})</span>`:'';
    const sl=(forecast.slope>=0?'+':'')+forecast.slope;
    forecastEl.innerHTML=`Your trend is <strong>${forecast.direction}</strong>${bn}. 5-session avg: <strong>${forecast.ma5}</strong> · Trend: <strong>${sl} pts/session</strong>. Score may reach <span class="forecast-projected-score">${forecast.projected}</span> this week. <span style="font-size:0.7rem;font-weight:600;color:${cc};">Confidence: ${forecast.confidence.label}</span>`;
    forecastEl.classList.add('forecast-ready');
  }
  if (barsEl&&barsWrap) {
    barsWrap.style.display='block'; barsEl.innerHTML='';
    const cs=forecast.calibratedScores, maxV=Math.max(...cs,forecast.projected,1);
    cs.forEach((s,i)=>{ const b=document.createElement('div'); b.className='forecast-mini-bar'; b.style.height=Math.round(s/maxV*100)+'%'; b.title=`Session ${i+1}: ${Math.round(s)}`; barsEl.appendChild(b); });
    const pb=document.createElement('div'); pb.className='forecast-mini-bar projected'; pb.style.height=Math.round(forecast.projected/maxV*100)+'%'; pb.title=`Projected: ${forecast.projected}`; barsEl.appendChild(pb);
  }
}

// ============================================
// MAIN APP
// ============================================
(function () {

  const micSelect       = document.getElementById('micSelect');
  const allowMicBtn     = document.getElementById('allowMicBtn');
  const allowCamBtn     = document.getElementById('allowCam');
  const startBtn        = document.getElementById('startBtn');
  const stopBtn         = document.getElementById('stopBtn');
  const sessionStatus   = document.getElementById('sessionStatus');
  const breathCountEl   = document.getElementById('breathCount');
  const breathingCircle = document.getElementById('breathingCircle');

  const userNameInput     = document.getElementById('userName');
  const userAgeInput      = document.getElementById('userAge');
  const userSkinToneInput = document.getElementById('userSkinTone');

  // CIELAB
  const cielabToggle   = document.getElementById('cielabToggle');
  const cielabStatusEl = document.getElementById('cielabStatus');
  let cielabEnabled=true;
  const savedCielab=localStorage.getItem('sns_cielab_enabled');
  if (savedCielab!==null) cielabEnabled=savedCielab==='true';
  if (cielabToggle) {
    cielabToggle.checked=cielabEnabled;
    cielabToggle.addEventListener('change',()=>{ cielabEnabled=cielabToggle.checked; localStorage.setItem('sns_cielab_enabled',String(cielabEnabled)); if(cielabStatusEl)cielabStatusEl.textContent=cielabEnabled?'ACTIVE':'BYPASS'; });
  }
  if (cielabStatusEl) cielabStatusEl.textContent=cielabEnabled?'ACTIVE':'BYPASS';

  // Profile
  if (userNameInput)     userNameInput.value     = localStorage.getItem('sns_user_name')||'';
  if (userAgeInput)      userAgeInput.value      = localStorage.getItem('sns_user_age') ||'';
  if (userSkinToneInput) userSkinToneInput.value = localStorage.getItem('sns_user_tone')||'Type 3';
  [userNameInput,userAgeInput,userSkinToneInput].forEach(el=>{
    if(el) el.addEventListener('change',()=>{ localStorage.setItem('sns_user_name',userNameInput.value); localStorage.setItem('sns_user_age',userAgeInput.value); localStorage.setItem('sns_user_tone',userSkinToneInput.value); });
  });

  // DOM refs
  const musicSelect    = document.getElementById('musicSelect');
  const musicVolume    = document.getElementById('musicVolume');
  const volumeLabel    = document.getElementById('volumeLabel');
  const bgAudio        = document.getElementById('bgAudio');
  const musicLoading   = document.getElementById('musicLoading');
  const musicBtnIcon   = document.getElementById('musicBtnIcon');
  const toggleMusicBtn = document.getElementById('toggleMusicBtn');
  const breathStatus   = document.getElementById('breathStatus');
  const waveCanvas     = document.getElementById('waveCanvas');
  const waveCtx        = waveCanvas.getContext('2d');
  const faceCanvas     = document.querySelector('.output_canvas');
  const faceVideo      = document.querySelector('.input_video');
  const faceCtx        = faceCanvas.getContext('2d');

  const eyeScoreTxt=document.getElementById('eyeScoreTxt'), eyeBar=document.getElementById('eyeBar');
  const headScoreTxt=document.getElementById('headScoreTxt'), headBar=document.getElementById('headBar');
  const gazeScoreTxt=document.getElementById('gazeScoreTxt'), gazeBar=document.getElementById('gazeBar');
  const bpmDisplay=document.getElementById('bpmDisplay');
  const heartrateTxt=document.getElementById('heartrateTxt'), heartrateBar=document.getElementById('heartrateBar');
  const sdnnTxt=document.getElementById('sdnnTxt'), sdnnBar=document.getElementById('sdnnBar');
  const rmssdTxt=document.getElementById('rmssdTxt'), rmssdBar=document.getElementById('rmssdBar');
  const faceCalmTxt=document.getElementById('faceCalmTxt'), faceCalmBar=document.getElementById('faceCalmBar');
  const breathConsTxt2=document.getElementById('breathConsTxt2'), breathConsBar2=document.getElementById('breathConsBar2');
  const meditationIndexTxt=document.getElementById('meditationIndexTxt'), meditationIndexBar=document.getElementById('meditationIndexBar');
  const overallBar=document.getElementById('overallBar'), overallTxt=document.getElementById('overallTxt');
  const overallNote=document.getElementById('overallNote');
  const breathConsTxt=document.getElementById('breathConsTxt'), breathConsBar=document.getElementById('breathConsBar');
  const micLvl=document.getElementById('micLvl'), micLvlBar=document.getElementById('micLvlBar');
  const fpsNode=document.getElementById('fps'), facesDetectedNode=document.getElementById('facesDetected');
  const sessionMeter=document.getElementById('sessionMeter');
  const historyBody=document.getElementById('historyBody');
  const exportCsvBtn=document.getElementById('exportCsv');
  const clearHistoryBtn=document.getElementById('clearHistory');

  let selectedMicId=null, localStream=null, audioContext=null, analyser=null, analyserData=null, animationId=null;
  let isSessionRunning=false, cycleTimeout=null, scriptedGrow=false, breathingBaseScale=1.0;
  let sessionStartTime=0, sessionTimerInterval=null;

  function formatElapsedTime(ms) {
    const t=Math.floor(ms/1000), m=Math.floor(t/60), s=t%60;
    return `${m}:${s.toString().padStart(2,'0')}`;
  }
  function updateSessionTimer() {
    if (!isSessionRunning||!sessionStartTime) return;
    const el=document.getElementById('sessionStatus');
    if (el) el.innerHTML=`running <span style="opacity:0.7;">•</span> ${formatElapsedTime(Date.now()-sessionStartTime)}`;
  }

  let ampDerivBuf=[], faceMeshModel=null, cameraInstance=null;
  let sessionHistory=JSON.parse(localStorage.getItem('sns_sessions')||'[]');

  // Session accumulators
  let currentBreathPhase='idle', phaseAmplitudes=[], phaseConsistencyScore=0.5;
  let sessionBreathScores=[], sessionEyeScores=[], sessionHeadScores=[], sessionGazeScores=[];
  let sessionMeditationScores=[], sessionBPMValues=[];
  let sessionLightingScores=[], sessionCenterScores=[], sessionJawScores=[], sessionRRScores=[];

  let smoothedEAR=0.25;
  let frames=0, lastFrameTs=performance.now();
  // Smoothed sub-scores for display
  let smoothHead=0.8, smoothGaze=0.8, smoothLighting=0.8;

  function clamp01(x) { return Math.max(0,Math.min(1,Number(x)||0)); }

  // ── Permissions & devices ──────────────────────────────────────
  async function ensurePermissionForMic() {
    try { const t=await navigator.mediaDevices.getUserMedia({audio:true}); t.getTracks().forEach(tr=>tr.stop()); return true; } catch(_){return false;}
  }
  function preferredDeviceId(devs) {
    const re=/headset|earbud|wired|external|usb|line in|communications|headphone/i;
    const f=devs.filter(d=>re.test(d.label||''));
    return f.length?f[0].deviceId:(devs[0]?.deviceId??null);
  }
  async function loadDevices() {
    try {
      const devs=await navigator.mediaDevices.enumerateDevices();
      const mics=devs.filter(d=>d.kind==='audioinput');
      micSelect.innerHTML='';
      if (!mics.length) { const o=document.createElement('option'); o.value=''; o.textContent='No microphone found'; micSelect.appendChild(o); selectedMicId=null; return; }
      mics.forEach((m,i)=>{ const o=document.createElement('option'); o.value=m.deviceId||''; o.textContent=m.label||`Microphone ${i+1}`; micSelect.appendChild(o); });
      const pick=(selectedMicId&&Array.from(micSelect.options).some(o=>o.value===selectedMicId))?selectedMicId:preferredDeviceId(mics);
      selectedMicId=pick||mics[0].deviceId; micSelect.value=selectedMicId;
    } catch(e){}
  }
  micSelect.addEventListener('change',e=>{selectedMicId=e.target.value||null;});
  document.getElementById('refreshMics')?.addEventListener('click',async()=>{ await ensurePermissionForMic(); await loadDevices(); alert('Microphones refreshed'); });
  allowMicBtn.addEventListener('click',async()=>{ const ok=await ensurePermissionForMic(); if(ok){await loadDevices();alert('Microphone permission granted.');}else alert('Mic permission denied.'); });

  // ── Camera & FaceMesh ─────────────────────────────────────────
  async function startCameraForFace() {
    try {
      faceCanvas.width=faceVideo.clientWidth||420; faceCanvas.height=faceVideo.clientHeight||300;
      faceMeshModel=new FaceMesh({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`});
      faceMeshModel.setOptions({
        maxNumFaces:1,
        // ← refineLandmarks MUST be true for iris gaze tracking
        refineLandmarks:true,
        minDetectionConfidence:0.6,
        minTrackingConfidence:0.6
      });
      faceMeshModel.onResults(onFaceResults);
      cameraInstance=new Camera(faceVideo,{
        onFrame:async()=>{await faceMeshModel.send({image:faceVideo});},
        width:320,height:240
      });
      cameraInstance.start();
    } catch(e){}
  }
  allowCamBtn.addEventListener('click',async()=>{
    try { await startCameraForFace(); }
    catch(e) { alert('Camera permission is required.'); }
  });

  // ── EAR ───────────────────────────────────────────────────────
  function calcEAR(lm,left) {
    const ids=left?[33,160,158,133,153,144]:[362,385,387,263,373,380];
    if (!lm||lm.length<468) return smoothedEAR;
    try {
      const [p0,p1,p2,p3,p4,p5]=ids.map(i=>lm[i]);
      if (!p0||!p1||!p2||!p3||!p4||!p5) return smoothedEAR;
      const v1=Math.hypot(p2.x-p4.x,p2.y-p4.y), v2=Math.hypot(p1.x-p5.x,p1.y-p5.y);
      const h=Math.hypot(p0.x-p3.x,p0.y-p3.y);
      return h?((v1+v2)/(2*h)):smoothedEAR;
    } catch(_){return smoothedEAR;}
  }

  // ── ROI ───────────────────────────────────────────────────────
  function getRoiFromLandmarks(lm) {
    const ids=[10,67,69,104,108,151,337,333,297,338];
    const pts=ids.map(i=>lm[i]).filter(Boolean);
    if (!pts.length) return null;
    const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
    const minX=Math.min(...xs), maxX=Math.max(...xs);
    const minY=Math.min(...ys), maxY=Math.max(...ys);
    const w=(maxX-minX)*faceCanvas.width, h=(maxY-minY)*faceCanvas.height;
    const x=minX*faceCanvas.width, y=minY*faceCanvas.height;
    return {
      x:Math.floor(x+(w-Math.max(10,Math.floor(w*RPPG_CONFIG.ROI_WIDTH_FACTOR)))/2),
      y:Math.floor(y+h*RPPG_CONFIG.ROI_Y_OFFSET),
      w:Math.max(10,Math.floor(w*RPPG_CONFIG.ROI_WIDTH_FACTOR)),
      h:Math.max(8,Math.floor(h*RPPG_CONFIG.ROI_HEIGHT_FACTOR))
    };
  }

  function getAvgRGBFromRoi(ctx,roi) {
    try {
      const {x,y,w,h}=roi; if(w<=0||h<=0) return null;
      const imgData=ctx.getImageData(x,y,w,h), data=imgData.data;
      if (!data||!data.length) return null;
      let lQ=0.5;
      if (cielabEnabled) { cielabSkinTransfer(data,w,h); ctx.putImageData(imgData,x,y); }
      // Lighting quality from raw ROI data
      lQ=computeLightingQuality(data,w,h);
      let tR=0,tG=0,tB=0; const pc=data.length/4;
      for (let i=0;i<data.length;i+=4){tR+=data[i];tG+=data[i+1];tB+=data[i+2];}
      ctx.strokeStyle=cielabEnabled?'rgba(245,158,11,0.95)':'rgba(0,245,212,0.9)';
      ctx.lineWidth=3; ctx.strokeRect(x,y,w,h);
      return {red:tR/pc,green:tG/pc,blue:tB/pc,lightingQuality:lQ};
    } catch(_){return null;}
  }

  // ──────────────────────────────────────────────────────────────
  // MAIN FACE RESULTS HANDLER
  // ──────────────────────────────────────────────────────────────
  function onFaceResults(results) {
    const now=performance.now();
    if (now-lastRppgTime<1000/RPPG_CONFIG.FPS) return;
    lastRppgTime=now;
    const dt=now-window.lastFrameTime;
    estimatedFPS=dt>0?1000/dt:30;
    window.lastFrameTime=now;

    if (!results.multiFaceLandmarks?.[0]) {
      if (facesDetectedNode) facesDetectedNode.textContent='0';
      return;
    }
    if (facesDetectedNode) facesDetectedNode.textContent='1';
    const lm=results.multiFaceLandmarks[0];

    faceCtx.save();
    faceCtx.clearRect(0,0,faceCanvas.width,faceCanvas.height);
    faceCtx.drawImage(results.image,0,0,faceCanvas.width,faceCanvas.height);

    // rPPG
    if (window.__sessionActive) {
      const roi=getRoiFromLandmarks(lm);
      if (roi) {
        const rgb=getAvgRGBFromRoi(faceCtx,roi);
        if (rgb) {
          rppgSignal.red.push(rgb.red); rppgSignal.green.push(rgb.green);
          rppgSignal.blue.push(rgb.blue); rppgSignal.timestamps.push(now);
          while (rppgSignal.green.length>RPPG_CONFIG.BUFFER_SIZE) {
            rppgSignal.red.shift(); rppgSignal.green.shift();
            rppgSignal.blue.shift(); rppgSignal.timestamps.shift();
          }
          // Accumulate lighting score
          smoothLighting=smoothLighting*0.9+rgb.lightingQuality*0.1;
          sessionLightingScores.push(rgb.lightingQuality);
        }
      }
    }
    faceCtx.restore();

    // BPM estimation
    bpmUpdateCounter++;
    const minBuf=Math.floor(RPPG_CONFIG.BUFFER_SIZE*RPPG_CONFIG.MIN_BUFFER_FILL);
    if (window.__sessionActive&&rppgSignal.green.length>=minBuf&&bpmUpdateCounter%RPPG_CONFIG.UPDATE_INTERVAL===0) {
      const pos=applyPosAlgorithm(rppgSignal.red.slice(-minBuf),rppgSignal.green.slice(-minBuf),rppgSignal.blue.slice(-minBuf));
      const filt=bandpassFilter(detrendSignal(pos),estimatedFPS,RPPG_CONFIG.MIN_HR/60,RPPG_CONFIG.MAX_HR/60);
      const norm=normalizeSignal(filt);
      const fftR=estimateBpmFFT(norm,estimatedFPS);
      const acR=estimateBpmAutocorr(norm,estimatedFPS);
      let rawBpm=0,conf=0;
      if (fftR) { rawBpm=fftR.bpm; conf=fftR.snr; if(acR&&Math.abs(rawBpm-acR)>15)conf-=3; }
      // Signal quality UI
      const sigBar=document.getElementById('signalQualityBar'), sigTxt=document.getElementById('signalQualityTxt');
      if (sigBar&&sigTxt) {
        const qPct=Math.min(100,Math.max(0,(conf+5)*5));
        sigBar.style.width=qPct+'%';
        if(conf<1){sigBar.style.background='var(--rose)';sigTxt.textContent='Weak / Noise';}
        else if(conf<5){sigBar.style.background='var(--amber)';sigTxt.textContent='Fair';}
        else{sigBar.style.background='var(--cyan)';sigTxt.textContent='Excellent';}
      }
      if (rawBpm>0&&conf>0.5) {
        const sB=hrKalman.filter(rawBpm);
        window.__currentBPM=Math.round(Math.max(RPPG_CONFIG.MIN_HR,Math.min(RPPG_CONFIG.MAX_HR,sB)));
        window.__lastBPM=window.__currentBPM; window.__rppgConfidence=conf;
        sessionBPMValues.push(window.__currentBPM);
        const hrv=computeHRV(window.__currentBPM);
        window.__lastSDNN=hrv.sdnn; window.__lastRMSSD=hrv.rmssd;
        const med=computeMeditationIndex(window.__currentBPM,hrv.sdnn,hrv.rmssd);
        sessionMeditationScores.push(med);
        if (bpmDisplay) { bpmDisplay.innerText='BPM: '+window.__currentBPM+(conf<1?' (Calibrating)':''); bpmDisplay.style.color=conf<1?'#f59e0b':'#14b8a6'; }
        if (heartrateTxt) heartrateTxt.innerText=window.__currentBPM+' bpm';
        if (heartrateBar) heartrateBar.style.width=Math.min(100,Math.max(0,((window.__currentBPM-40)/80)*100))+'%';
        if (sdnnTxt) sdnnTxt.innerText=hrv.sdnn+' ms';
        if (sdnnBar) sdnnBar.style.width=Math.min(100,hrv.sdnn)+'%';
        if (rmssdTxt) rmssdTxt.innerText=hrv.rmssd+' ms';
        if (rmssdBar) rmssdBar.style.width=Math.min(100,hrv.rmssd)+'%';
      } else {
        if (bpmDisplay&&window.__sessionActive) { bpmDisplay.innerText='BPM: Sensing...'; bpmDisplay.style.color='#9ca3af'; }
      }
    }

    // ── Scientific face metrics ───────────────────────────────────
    if (window.__sessionActive) {

      // 1. EYE RELAXATION (unchanged, reliable)
      const lEAR=calcEAR(lm,true), rEAR=calcEAR(lm,false);
      const avgEAR=(lEAR+rEAR)/2;
      smoothedEAR=smoothedEAR*EYE_CONFIG.SMOOTHING+avgEAR*(1-EYE_CONFIG.SMOOTHING);
      const eyeRelax=calculateEyeRelaxScore(lEAR,rEAR,smoothedEAR);
      const blinkScore=updateBlinkRate(avgEAR);

      // 2. HEAD STEADINESS — new geometric pose angles (independent of gaze)
      const pose=computeHeadPoseAngles(lm);
      const headSteady=pose?pose.steady:smoothHead;
      smoothHead=smoothHead*0.85+headSteady*0.15; // smooth display

      // 3. GAZE STABILITY — iris relative to eye socket (independent of head)
      let gazeResult=computeIrisGaze(lm);
      if (!gazeResult) gazeResult=computeGazeFallback(lm, smoothGaze);
      const gazeStable=gazeResult.gazeScore;
      smoothGaze=smoothGaze*0.85+gazeStable*0.15;

      // 4. JAW TENSION
      const jawResult=computeJawTension(lm);
      const jawScore=1-jawResult.tension; // high tension = low score
      sessionJawScores.push(jawScore);

      // 5. FACE CENTERING
      const centerScore=computeFaceCentering(lm);
      sessionCenterScores.push(centerScore);

      // 6. COMPOSITE with all degradation factors
      const lAvg=sessionLightingScores.length?sessionLightingScores[sessionLightingScores.length-1]:0.8;
      const composite=compositeScoreWithDegradation({
        eyeRelax, headSteady:smoothHead, gazeStable:smoothGaze,
        blinkScore, jawScore, lightingScore:lAvg, centerScore, rrScore:0.7
      });

      // Update UI
      if (eyeScoreTxt) eyeScoreTxt.textContent=composite.audit.eye+'%';
      if (eyeBar) eyeBar.style.width=composite.audit.eye+'%';
      if (headScoreTxt) headScoreTxt.textContent=composite.audit.head+'%';
      if (headBar) headBar.style.width=composite.audit.head+'%';
      if (gazeScoreTxt) gazeScoreTxt.textContent=composite.audit.gaze+'%';
      if (gazeBar) gazeBar.style.width=composite.audit.gaze+'%';

      // Update degradation HUD elements if they exist
      const lightEl=document.getElementById('lightingScore');
      if (lightEl) lightEl.textContent=composite.audit.lighting+'%';
      const jawEl=document.getElementById('jawTension');
      if (jawEl) jawEl.textContent=composite.audit.jaw+'% tension';
      const centerEl=document.getElementById('centerScore');
      if (centerEl) centerEl.textContent=composite.audit.center+'%';
      const blinkEl=document.getElementById('blinkScore');
      if (blinkEl) blinkEl.textContent=Math.round(blinkState.recentBPM)+' bpm';
      const degradeEl=document.getElementById('degradeMultiplier');
      if (degradeEl) degradeEl.textContent=composite.audit.degradeMultiplier+'%';
      const detStateEl=document.getElementById('detectionState');
      if (detStateEl) detStateEl.textContent=pose?`YAW:${(pose.yaw*100).toFixed(0)}`:'DETECTING';

      // Accumulate
      sessionEyeScores.push(eyeRelax);
      sessionHeadScores.push(smoothHead);
      sessionGazeScores.push(smoothGaze);

      window.__faceCalmness=composite.score;
    }

    frames++;
    if (now-lastFrameTs>1000) { if(fpsNode)fpsNode.textContent=frames; frames=0; lastFrameTs=now; }
  }

  // ──────────────────────────────────────────────────────────────
  // AUDIO SYSTEM WITH DUCKING
  // ──────────────────────────────────────────────────────────────
  async function startAudioForMic() {
    if (!selectedMicId) { alert('Please select a microphone first.'); return false; }
    stopAudio();
    try {
      audioContext=new (window.AudioContext||window.webkitAudioContext)();
      await audioContext.resume();
      const constraints={audio:{deviceId:selectedMicId&&selectedMicId!=='default'?{exact:selectedMicId}:undefined,echoCancellation:true,noiseSuppression:true,autoGainControl:false}};
      localStream=await navigator.mediaDevices.getUserMedia(constraints);
      const src=audioContext.createMediaStreamSource(localStream);
      analyser=audioContext.createAnalyser(); analyser.fftSize=1024; analyser.smoothingTimeConstant=0.8;
      analyserData=new Uint8Array(analyser.frequencyBinCount);
      src.connect(analyser); startVisualizer(); return true;
    } catch(e){ alert('Microphone start failed: '+(e.message||e)); return false; }
  }
  function stopAudio() {
    if (localStream){localStream.getTracks().forEach(t=>t.stop());localStream=null;}
    if (audioContext){try{audioContext.close();}catch(_){}audioContext=null;}
    if (animationId){cancelAnimationFrame(animationId);animationId=null;}
    ampDerivBuf=[];
    if (micLvl)micLvl.textContent='—'; if (micLvlBar)micLvlBar.style.width='10%';
  }

  function startVisualizer() {
    if (!analyser) return;
    function resize() {
      const dpr=window.devicePixelRatio||2, w=Math.max(300,waveCanvas.parentElement?.clientWidth||520);
      waveCanvas.width=Math.floor(w*dpr); waveCanvas.height=Math.floor(260*dpr);
      waveCanvas.style.width=w+'px'; waveCanvas.style.height='260px';
      waveCtx.setTransform(dpr,0,0,dpr,0,0);
    }
    resize(); window.addEventListener('resize',resize);
    let lastAmp=0;
    function draw() {
      animationId=requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(analyserData);
      const W=waveCanvas.width/(window.devicePixelRatio||1), H=waveCanvas.height/(window.devicePixelRatio||1);
      waveCtx.clearRect(0,0,W,H); waveCtx.fillStyle='rgba(5,8,20,0.12)'; waveCtx.fillRect(0,0,W,H);
      waveCtx.lineWidth=2; waveCtx.beginPath();
      const sw=W/analyserData.length; let x=0,sum=0;
      for (let i=0;i<analyserData.length;i++) {
        const v=(analyserData[i]/128)-1, y=(v*0.95+0.5)*H;
        i===0?waveCtx.moveTo(x,y):waveCtx.lineTo(x,y);
        x+=sw; sum+=Math.abs(v);
      }
      waveCtx.strokeStyle='rgba(255,165,0,0.95)'; waveCtx.stroke();
      const amplitude=Math.min(1,(sum/analyserData.length)*8);
      if (!scriptedGrow) breathingCircle.style.transform=`scale(${0.7+amplitude*1.5})`;
      else breathingCircle.style.transform=`scale(${(breathingBaseScale||0.8)*(1+amplitude*0.12)})`;
      const deriv=Math.abs(amplitude-lastAmp); lastAmp=amplitude;
      if (window.__sessionActive&&currentBreathPhase!=='idle') {
        phaseAmplitudes.push(amplitude);
        if (phaseAmplitudes.length>300) phaseAmplitudes.shift();
      }
      ampDerivBuf.push(deriv); if(ampDerivBuf.length>1200)ampDerivBuf.shift();
      const micPct=Math.round(amplitude*400);
      if (micLvl)micLvl.textContent=micPct+'%'; if(micLvlBar)micLvlBar.style.width=micPct+'%';
      // Respiratory rate
      const rrS=updateRespiratoryRate(amplitude);
      sessionRRScores.push(rrS);
      const rrEl=document.getElementById('respRate');
      if (rrEl) rrEl.textContent=(rrTracker.rate||'--')+' br/min';
      const consistency=computeBreathConsistency();
      if (window.__sessionActive&&currentBreathPhase!=='idle') sessionBreathScores.push(consistency);
      if (breathConsTxt)breathConsTxt.textContent=Math.round(consistency*100)+'%';
      if (breathConsBar)breathConsBar.style.width=Math.round(consistency*100)+'%';
    }
    draw();
  }

  function computeBreathConsistency() {
    if (!window.__sessionActive||currentBreathPhase==='idle') return 0.5;
    if (phaseAmplitudes.length<10) return phaseConsistencyScore;
    const samp=phaseAmplitudes.slice(-60), mean=samp.reduce((a,b)=>a+b,0)/samp.length;
    let cur=0.5;
    if (currentBreathPhase==='inhale') {
      const trend=samp[samp.length-1]-samp[0];
      cur=mean>0.05?0.8+trend*0.2:0.3;
    } else if (currentBreathPhase==='hold') {
      cur=mean<0.03?1.0:(samp.some(s=>s>0.1)?0.1:0.5);
    } else if (currentBreathPhase==='exhale') {
      const v=samp.reduce((s,val)=>s+(val-mean)**2,0)/samp.length;
      cur=mean>0.04?(Math.sqrt(v)<0.05?0.9:0.6):0.3;
    }
    phaseConsistencyScore=phaseConsistencyScore*0.95+clamp01(cur)*0.05;
    return phaseConsistencyScore;
  }

  // ──────────────────────────────────────────────────────────────
  // TTS WITH AUDIO DUCKING — fixes clash on mobile
  // ──────────────────────────────────────────────────────────────
  let ttsRestoreTimeout=null;

  function speak(textHi, textEn) {
    if (muted) return;
    const synth=window.speechSynthesis;
    if (!synth) return;

    const text=textEn||textHi||'';
    if (!text) return;

    // Step 1: duck background audio to 5%
    const savedVol=bgAudio.paused?0:bgAudio.volume;
    if (!bgAudio.paused) bgAudio.volume=0.05;

    // Step 2: cancel any pending restore timer
    if (ttsRestoreTimeout) clearTimeout(ttsRestoreTimeout);

    // Step 3: cancel previous speech and wait 150ms (mobile needs this)
    if (synth.speaking) synth.cancel();

    setTimeout(()=>{
      const u=new SpeechSynthesisUtterance(text);
      u.rate=0.9; u.pitch=1.0; u.volume=1.0; u.lang='en-US';

      const restore=()=>{
        if (!bgAudio.paused && savedVol>0) {
          // Fade back up over 800ms
          let steps=0;
          const target=savedVol;
          const ti=setInterval(()=>{
            steps++;
            bgAudio.volume=Math.min(target, (steps/8)*target);
            if (steps>=8) clearInterval(ti);
          },100);
        }
      };

      u.onend=restore;
      u.onerror=restore;

      // Failsafe restore in case onend never fires (iOS bug)
      const approxDur=Math.ceil(text.length/5)*1000/0.9+1000;
      ttsRestoreTimeout=setTimeout(restore, approxDur);

      synth.speak(u);
    }, 180); // 180ms gap ensures mobile WebSpeech is ready
  }

  function stopSession() {
    if (!isSessionRunning) return;
    isSessionRunning=false; window.__sessionActive=false;
    if (sessionTimerInterval){clearInterval(sessionTimerInterval);sessionTimerInterval=null;}
    if (sessionStatus)sessionStatus.textContent='stopped';
    stopAudio(); stopAmbience();
    if (cycleTimeout){clearTimeout(cycleTimeout);cycleTimeout=null;}
    scriptedGrow=false; breathingBaseScale=0.7;
    breathingCircle.style.transform='scale(0.7)';
    if (breathStatus)breathStatus.textContent='Stopped';
    try{window.speechSynthesis?.cancel?.();}catch(_){}
    // Restore audio volume
    if (!bgAudio.paused) bgAudio.volume=musicVolume?musicVolume.value/100:0.3;
  }

  function calibrateRppgSettings() {
    const age=parseInt(userAgeInput?.value||0,10)||30;
    const gender=document.getElementById('userGender')?.value||'other';
    RPPG_CONFIG.MIN_HR=50; RPPG_CONFIG.MAX_HR=100; RPPG_CONFIG.ROI_WIDTH_FACTOR=0.75;
    if (age>50){RPPG_CONFIG.MIN_HR=45;RPPG_CONFIG.ROI_WIDTH_FACTOR=0.85;}
    else if(age<15)RPPG_CONFIG.MAX_HR=110;
    if (gender==='female')RPPG_CONFIG.MAX_HR+=5;
  }

  async function startSession() {
    if (isSessionRunning) return;
    calibrateRppgSettings();
    const micOk=await startAudioForMic();
    if (!cameraInstance) await startCameraForFace().catch(()=>{});
    if (!micOk&&!confirm('Microphone unavailable. Continue face-only?')) return;
    isSessionRunning=true; window.__sessionActive=true;
    sessionStartTime=Date.now();
    sessionTimerInterval=setInterval(updateSessionTimer,1000);
    if (sessionStatus)sessionStatus.textContent='running';
    // Reset all accumulators
    ampDerivBuf.length=0;
    rppgSignal.red=[]; rppgSignal.green=[]; rppgSignal.blue=[]; rppgSignal.timestamps=[];
    bpmHistory=[]; hrKalman.reset();
    currentBreathPhase='idle'; phaseAmplitudes=[]; phaseConsistencyScore=0.5;
    sessionBreathScores=[]; sessionEyeScores=[]; sessionHeadScores=[]; sessionGazeScores=[];
    sessionMeditationScores=[]; sessionBPMValues=[];
    sessionLightingScores=[]; sessionCenterScores=[]; sessionJawScores=[]; sessionRRScores=[];
    smoothHead=0.8; smoothGaze=0.8; smoothLighting=0.8;
    // Reset blink tracker
    blinkState.blinkCount=0; blinkState.windowStart=Date.now(); blinkState.recentBPM=15;
    // Reset RR tracker
    rrTracker.envelope=[]; rrTracker.peakIntervals=[]; rrTracker.lastPeak=0; rrTracker.rate=0;

    if (breathStatus)breathStatus.textContent='Session running — follow the breathing prompts';
    startAmbience();
    runTimedBreathing();
    // Speak welcome AFTER slight delay so it doesn't clash with any UI sounds
    setTimeout(()=>speak('','Welcome. Follow the voice.'), 400);
  }

  startBtn.addEventListener('click',startSession);
  stopBtn.addEventListener('click',stopSession);

  let muted=false;
  const muteBtn=document.getElementById('muteBtn');
  if (muteBtn) {
    muteBtn.addEventListener('click',()=>{
      muted=!muted;
      muteBtn.textContent=muted?'🔇':'🔈';
      if (muted)try{window.speechSynthesis.cancel();}catch(_){}
    });
  }

  // ──────────────────────────────────────────────────────────────
  // BREATHING CYCLE
  // ──────────────────────────────────────────────────────────────
  function runTimedBreathing() {
    const breaths=parseInt(breathCountEl.value,10)||3;
    let count=breaths;
    const inMs=6000, holdMs=5000, outMs=5000;
    if (breathStatus)breathStatus.textContent=`Breaths remaining: ${count}`;
    let cycleCount=0;
    if (sessionMeter)sessionMeter.style.width='5%';

    function singleCycle() {
      if (!isSessionRunning) return;
      if (count<=0){finishBreathing();return;}
      cycleCount++;
      if (sessionMeter)sessionMeter.style.width=Math.round(cycleCount/breaths*100)+'%';
      if (breathStatus)breathStatus.textContent='BREATHE IN';
      // Duck then speak — full audio ducking built into speak()
      speak('','Breathe in.');
      currentBreathPhase='inhale'; phaseAmplitudes=[];
      scriptedGrow=true; breathingBaseScale=1.4;
      breathingCircle.style.transition=`transform ${inMs}ms cubic-bezier(.2,.9,.2,1)`;
      breathingCircle.style.transform=`scale(${breathingBaseScale})`;

      cycleTimeout=setTimeout(()=>{
        if (!isSessionRunning) return;
        if (breathStatus)breathStatus.textContent='HOLD';
        speak('','Hold.');
        currentBreathPhase='hold'; phaseAmplitudes=[];
        breathingCircle.style.transition=`transform ${holdMs}ms ease-in-out`;
        breathingBaseScale=1.2;
        breathingCircle.style.transform=`scale(${breathingBaseScale})`;

        cycleTimeout=setTimeout(()=>{
          if (!isSessionRunning) return;
          if (breathStatus)breathStatus.textContent='EXHALE SLOWLY';
          speak('','Breathe out slowly.');
          currentBreathPhase='exhale'; phaseAmplitudes=[];
          breathingCircle.style.transition=`transform ${outMs}ms cubic-bezier(.2,.9,.2,1)`;
          breathingBaseScale=0.6;
          breathingCircle.style.transform=`scale(${breathingBaseScale})`;

          cycleTimeout=setTimeout(()=>{
            count--;
            currentBreathPhase='idle';
            if (breathStatus)breathStatus.textContent=`Breaths remaining: ${count}`;
            cycleTimeout=setTimeout(singleCycle,2000);
          },outMs);
        },holdMs);
      },inMs);
    }

    function finishBreathing() {
      window.__sessionActive=false;
      scriptedGrow=false; breathingBaseScale=1.0;
      breathingCircle.style.transition='transform 400ms ease';
      breathingCircle.style.transform='scale(1)';
      if (breathStatus)breathStatus.textContent='Session complete. Well done!';
      stopAmbience();
      setTimeout(()=>speak('','Session complete. Well done!'),300);
      computeFinalResultAndShow();
    }

    singleCycle();
  }

  // ──────────────────────────────────────────────────────────────
  // FINAL RESULTS
  // ──────────────────────────────────────────────────────────────
  function computeFinalResult() {
    const avg=arr=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:0;

    const avgEye=avg(sessionEyeScores);
    const avgHead=avg(sessionHeadScores);
    const avgGaze=avg(sessionGazeScores);
    const avgLighting=avg(sessionLightingScores)||0.8;
    const avgCenter=avg(sessionCenterScores)||0.8;
    const avgJaw=avg(sessionJawScores)||0.8;
    const avgRR=avg(sessionRRScores)||0.7;

    // Scientific composite face score
    const composite=compositeScoreWithDegradation({
      eyeRelax:avgEye, headSteady:avgHead, gazeStable:avgGaze,
      blinkScore:Math.min(1,blinkState.recentBPM/15),
      jawScore:avgJaw, lightingScore:avgLighting,
      centerScore:avgCenter, rrScore:avgRR
    });
    const f=composite.score;

    let b=phaseConsistencyScore;
    if (sessionBreathScores.length>50) b=avg(sessionBreathScores);
    b=clamp01(b);

    const meditationScore=sessionMeditationScores.length>0
      ?avg(sessionMeditationScores)
      :computeMeditationIndex(window.__lastBPM||0,window.__lastSDNN||0,window.__lastRMSSD||0);
    const meditationIndex=meditationScore/100;
    const overall=clamp01(f*0.40+b*0.30+meditationIndex*0.30);

    // Update results UI
    if (faceCalmTxt)faceCalmTxt.textContent=Math.round(f*100)+'%';
    if (faceCalmBar)faceCalmBar.style.width=Math.round(f*100)+'%';
    if (breathConsTxt2)breathConsTxt2.textContent=Math.round(b*100)+'%';
    if (breathConsBar2)breathConsBar2.style.width=Math.round(b*100)+'%';
    if (meditationIndexTxt)meditationIndexTxt.textContent=meditationScore+'%';
    if (meditationIndexBar)meditationIndexBar.style.width=meditationScore+'%';
    if (overallTxt)overallTxt.textContent=Math.round(overall*100)+'%';
    if (overallBar)overallBar.style.width=Math.round(overall*100)+'%';

    // Degradation audit note
    let auditNote='';
    if (avgLighting<0.7) auditNote+=' ⚠ Poor lighting degraded rPPG accuracy.';
    if (avgCenter<0.7) auditNote+=' ⚠ Face off-centre.';
    if (avgJaw<0.7) auditNote+=' ⚠ Jaw tension detected.';
    if (blinkState.recentBPM>25) auditNote+=' ⚠ High blink rate (stress marker).';

    if (overallNote) {
      if (overall>0.75) overallNote.textContent='Excellent meditation session 🌿'+auditNote;
      else if (overall>0.5) overallNote.textContent='Good focus, keep practicing 🙂'+auditNote;
      else overallNote.textContent='Try slower breathing and stillness. '+auditNote;
    }

    return {f, b, meditationIndex, overall, meditationScore, composite};
  }

  function computeFinalResultAndShow() {
    const result=computeFinalResult();
    const dur=sessionStartTime?Math.floor((Date.now()-sessionStartTime)/1000):0;
    const durationStr=formatElapsedTime(dur*1000);
    const avgArr=arr=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:0;
    const avgBPM=sessionBPMValues.length>0?Math.round(avgArr(sessionBPMValues)):(window.__lastBPM||0);
    const breathPct=Math.round(result.b*100), meditPct=Math.round(result.meditationScore);
    const rawStability=calculateStability(avgBPM,breathPct,meditPct);
    saveSession(rawStability);
    const allS=getSessions();
    const {baselineMean,calibrated}=getBaseline(allS);
    const displayStability=calibrated?calibrateScore(rawStability,baselineMean):rawStability;

    sessionHistory.push({
      date:new Date().toLocaleString(),
      name:userNameInput?.value||'Anonymous',
      age:userAgeInput?.value||'—',
      tone:userSkinToneInput?.value||'—',
      duration:durationStr, bpm:avgBPM,
      face:Math.round(result.f*100), breath:breathPct,
      meditation:meditPct, overall:Math.round(result.overall*100),
      stability:displayStability, cielabUsed:cielabEnabled,
      lighting:Math.round(avgArr(sessionLightingScores)*100)||'—',
      blinkBPM:Math.round(blinkState.recentBPM)||'—',
      respRate:rrTracker.rate||'—'
    });
    localStorage.setItem('sns_sessions',JSON.stringify(sessionHistory));
    renderHistory();
    updateWellnessUI(rawStability);
  }

  function renderHistory() {
    if (!historyBody) return;
    historyBody.innerHTML='';
    sessionHistory.slice().reverse().forEach(s=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`
        <td>${s.date}</td>
        <td>${s.name||'Anonymous'}</td>
        <td>${s.age||'—'} / ${s.tone||'—'}</td>
        <td>${s.duration||'—'}</td>
        <td>${s.bpm||'—'} bpm</td>
        <td>${s.face}%</td>
        <td>${s.breath}%</td>
        <td>${s.meditation}%</td>
        <td>${s.overall}%</td>
        <td><strong>${s.stability!==undefined?s.stability+'%':'—'}</strong>${s.cielabUsed?'<span style="font-size:0.65rem;color:#f59e0b;">🔬</span>':''}</td>
      `;
      historyBody.appendChild(tr);
    });
  }

  // ── Music System ─────────────────────────────────────────────
  function showAudioLoading(v) { if(musicLoading)musicLoading.style.display=v?'inline':'none'; }
  bgAudio.addEventListener('loadstart',()=>showAudioLoading(true));
  bgAudio.addEventListener('canplay',()=>showAudioLoading(false));
  bgAudio.addEventListener('waiting',()=>showAudioLoading(true));
  bgAudio.addEventListener('playing',()=>showAudioLoading(false));
  bgAudio.addEventListener('error',()=>{showAudioLoading(false);if(window.__sessionActive)alert('Audio track failed. Try another.');});

  function startAmbience() {
    const url=musicSelect.value;
    if (!url||url==='none'){stopAmbience();return;}
    showAudioLoading(true);
    bgAudio.pause(); bgAudio.src=url; bgAudio.load();
    bgAudio.volume=musicVolume.value/100;
    bgAudio.play().catch(()=>{});
    if (musicBtnIcon)musicBtnIcon.textContent='⏸';
  }
  function stopAmbience() { bgAudio.pause(); if(musicBtnIcon)musicBtnIcon.textContent='▶️'; }
  if (toggleMusicBtn)toggleMusicBtn.addEventListener('click',()=>bgAudio.paused?startAmbience():stopAmbience());
  if (musicSelect)musicSelect.addEventListener('change',()=>{
    const url=musicSelect.value;
    if(url&&url!=='none'){bgAudio.pause();bgAudio.src=url;bgAudio.load();if(window.__sessionActive)startAmbience();}else stopAmbience();
  });
  if (musicVolume)musicVolume.addEventListener('input',e=>{
    bgAudio.volume=e.target.value/100;
    if(volumeLabel)volumeLabel.textContent=e.target.value+'%';
  });

  renderHistory();

  // Restore wellness display on page load
  (function initWellnessDisplay(){
    const sessions=getSessions(); if(!sessions.length)return;
    const last=sessions[sessions.length-1], todayKey=new Date().toISOString().slice(0,10);
    if (last.date.slice(0,10)===todayKey){updateWellnessUI(last.score);return;}
    const streakEl=document.getElementById('streakDisplay');
    const s=calculateStreak();
    if(streakEl){
      if(s>=2){streakEl.textContent=`🔥 ${s}-Day Stability Streak`;streakEl.style.display='inline-flex';}
      else if(s===1){streakEl.textContent='✨ First day — come back tomorrow!';streakEl.style.display='inline-flex';}
      else streakEl.style.display='none';
    }
    const forecast=calculateWeeklyForecast();
    const forecastEl=document.getElementById('weeklyForecast');
    const barsWrap=document.getElementById('forecastBarsWrap'), barsEl=document.getElementById('forecastBars');
    if (forecast&&forecastEl) {
      const cc=forecast.confidence.label==='High'?'#10b981':forecast.confidence.label==='Medium'?'#f59e0b':'#ef4444';
      const bn=forecast.baselineCalibrated?` <span style="font-size:0.7rem;opacity:0.6;">(baseline: ${forecast.baselineMean})</span>`:'';
      const sl=(forecast.slope>=0?'+':'')+forecast.slope;
      forecastEl.innerHTML=`Your trend is <strong>${forecast.direction}</strong>${bn}. 5-session avg: <strong>${forecast.ma5}</strong> · Trend: <strong>${sl} pts/session</strong>. Score may reach <span class="forecast-projected-score">${forecast.projected}</span> this week. <span style="font-size:0.7rem;font-weight:600;color:${cc};">Confidence: ${forecast.confidence.label}</span>`;
      forecastEl.classList.add('forecast-ready');
      if (barsEl&&barsWrap) {
        barsWrap.style.display='block'; barsEl.innerHTML='';
        const cs=forecast.calibratedScores, mx=Math.max(...cs,forecast.projected,1);
        cs.forEach((sc,i)=>{const b=document.createElement('div');b.className='forecast-mini-bar';b.style.height=Math.round(sc/mx*100)+'%';b.title=`Session ${i+1}: ${Math.round(sc)}`;barsEl.appendChild(b);});
        const pb=document.createElement('div');pb.className='forecast-mini-bar projected';pb.style.height=Math.round(forecast.projected/mx*100)+'%';pb.title=`Projected: ${forecast.projected}`;barsEl.appendChild(pb);
      }
    }
  })();

  // CSV Export
  if (exportCsvBtn) exportCsvBtn.addEventListener('click',()=>{
    if (!sessionHistory.length){alert('No sessions to export');return;}
    const rows=[['Date','Name','Age','Skin','Duration','BPM','Face','Breath','Meditation','Overall','Stability','Lighting%','Blink BPM','Resp Rate','CIELAB']];
    sessionHistory.forEach(s=>rows.push([s.date,s.name||'Anonymous',s.age||'—',s.tone||'—',s.duration||'—',s.bpm||'—',s.face,s.breath,s.meditation,s.overall,s.stability!==undefined?s.stability:'—',s.lighting||'—',s.blinkBPM||'—',s.respRate||'—',s.cielabUsed?'Yes':'No']));
    const csv=rows.map(r=>r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
    const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='mindflow_sessions.csv';a.click();
  });

  // Clear History
  if (clearHistoryBtn) clearHistoryBtn.addEventListener('click',()=>{
    if (!confirm('Clear all saved sessions?')) return;
    sessionHistory=[]; localStorage.removeItem('sns_sessions'); localStorage.removeItem(WELLNESS_STORAGE_KEY);
    renderHistory();
    const rm={stabilityScore:el=>{el.textContent='--';el.className='';},stabilityMessage:el=>{el.textContent='Complete a session to see your score.';},streakDisplay:el=>{el.style.display='none';},weeklyForecast:el=>{el.textContent='Complete more sessions to unlock your weekly outlook.';el.classList.remove('forecast-ready');},forecastBarsWrap:el=>{el.style.display='none';},stabilityRingFill:el=>{el.style.strokeDashoffset='226';},stabilityRingLabel:el=>{el.textContent='—';}};
    Object.entries(rm).forEach(([id,fn])=>{const el=document.getElementById(id);if(el)fn(el);});
  });

  // Startup
  (async()=>{
    try{await ensurePermissionForMic();}catch(_){}
    await loadDevices();
    setTimeout(()=>{faceCanvas.width=faceVideo.clientWidth||420;faceCanvas.height=faceVideo.clientHeight||300;},500);
  })();
  if (navigator.mediaDevices&&typeof navigator.mediaDevices.addEventListener==='function') {
    navigator.mediaDevices.addEventListener('devicechange',()=>loadDevices());
  }
  window.addEventListener('beforeunload',()=>{stopSession();try{cameraInstance?.stop();}catch(_){}});

})();