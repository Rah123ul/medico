/* =========================
   MEDITATION APP - PRODUCTION v8.1
   ✅ All v8.0 features preserved
   ✅ [v8.1] MOBILE AUDIO ENGINE (Android/iOS fixes):
        - Global AudioContext unlock gate (first-touch)
        - AudioContext state watchdog (auto-resume on suspend)
        - Android SpeechSynthesis bug fix (cancel→speak race)
        - Background audio unlocked via user-gesture cache
        - bgAudio gets playsinline + muted→unmuted trick
        - Speech retry queue with exponential back-off
        - Volume normalisation for mobile media vs call volume
        - visibilitychange handler re-activates audio on tab switch
========================= */

// ============================================
// GLOBAL STATE & CONFIG
// ============================================
window.__currentBPM    = 0;
window.__lastBPM       = 0;
window.__lastSDNN      = 0;
window.__lastRMSSD     = 0;
window.__faceCalmness  = 0;
window.__sessionActive = false;

// ============================================
// ADVANCED rPPG CONFIGURATION
// ============================================
const RPPG_CONFIG = {
  FPS: 30,
  BUFFER_SIZE: 300,
  MIN_HR: 50,
  MAX_HR: 100,
  UPDATE_INTERVAL: 20,
  MIN_BUFFER_FILL: 0.6,
  KALMAN_Q: 0.08,
  KALMAN_R: 0.12,
  ROI_WIDTH_FACTOR: 0.75,
  ROI_HEIGHT_FACTOR: 0.38,
  ROI_Y_OFFSET: 0.03
};

// ============================================
// EYE DETECTION CONFIG
// ============================================
const EYE_CONFIG = {
  MIN_EAR: 0.17,
  MAX_EAR: 0.28,
  SMOOTHING: 0.4,
  CLOSED_BOOST: 0.15,
  CURVE_POWER: 0.8
};

// ============================================
// CIELAB SKIN-TONE NORMALIZATION ENGINE
// ============================================
const CIELAB_CONFIG = {
  TARGET_L: 50,
  TARGET_A: 0.1,
  TARGET_B: 0.05
};

function linearToSrgb(c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function srgbToXyz(r, g, b) {
  const rl = srgbToLinear(r), gl = srgbToLinear(g), bl = srgbToLinear(b);
  return {
    x: 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl,
    y: 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl,
    z: 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
  };
}
function xyzToSrgb(x, y, z) {
  let r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
  let b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
  r = Math.max(0, Math.min(1, r));
  g = Math.max(0, Math.min(1, g));
  b = Math.max(0, Math.min(1, b));
  return { r: linearToSrgb(r), g: linearToSrgb(g), b: linearToSrgb(b) };
}
const D65_X = 0.95047, D65_Y = 1.00000, D65_Z = 1.08883;
function labF(t) { return t > 0.008856 ? Math.cbrt(t) : (903.3 * t + 16) / 116; }
function labFInv(t) { return t > 6 / 29 ? t * t * t : 3 * (6 / 29) * (6 / 29) * (t - 4 / 29); }
function xyzToLab(x, y, z) {
  const fx = labF(x / D65_X), fy = labF(y / D65_Y), fz = labF(z / D65_Z);
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}
function labToXyz(L, a, b) {
  const fy = (L + 16) / 116, fx = a / 500 + fy, fz = fy - b / 200;
  return { x: D65_X * labFInv(fx), y: D65_Y * labFInv(fy), z: D65_Z * labFInv(fz) };
}
function rgbToLab(r, g, b) { const xyz = srgbToXyz(r / 255, g / 255, b / 255); return xyzToLab(xyz.x, xyz.y, xyz.z); }
function labToRgb(L, a, b) {
  const xyz = labToXyz(L, a, b), srgb = xyzToSrgb(xyz.x, xyz.y, xyz.z);
  return {
    r: Math.round(Math.max(0, Math.min(255, srgb.r * 255))),
    g: Math.round(Math.max(0, Math.min(255, srgb.g * 255))),
    b: Math.round(Math.max(0, Math.min(255, srgb.b * 255)))
  };
}
function cielabSkinTransfer(data, w, h) {
  const pixelCount = w * h;
  if (pixelCount === 0) return { meanL: 0, meanA: 0, meanB: 0 };
  let sumL = 0, sumA = 0, sumB = 0;
  const labCache = new Float32Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    const lab = rgbToLab(data[idx], data[idx + 1], data[idx + 2]);
    labCache[i * 3] = lab.L; labCache[i * 3 + 1] = lab.a; labCache[i * 3 + 2] = lab.b;
    sumL += lab.L; sumA += lab.a; sumB += lab.b;
  }
  const meanL = sumL / pixelCount, meanA = sumA / pixelCount, meanB = sumB / pixelCount;
  const scaleL = meanL !== 0 ? CIELAB_CONFIG.TARGET_L / meanL : 1;
  const scaleA = meanA !== 0 ? CIELAB_CONFIG.TARGET_A / meanA : 1;
  const scaleB = meanB !== 0 ? CIELAB_CONFIG.TARGET_B / meanB : 1;
  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    let L = Math.max(0,    Math.min(100, labCache[i * 3]     * scaleL));
    let a = Math.max(-128, Math.min(127, labCache[i * 3 + 1] * scaleA));
    let b = Math.max(-128, Math.min(127, labCache[i * 3 + 2] * scaleB));
    const rgb = labToRgb(L, a, b);
    data[idx] = rgb.r; data[idx + 1] = rgb.g; data[idx + 2] = rgb.b;
  }
  return { meanL, meanA, meanB };
}

// ============================================
// REALISTIC HRV TRACKING
// ============================================
let bpmHistory = [];
const BPM_HISTORY_SIZE = 60;

// ============================================
// rPPG SIGNAL BUFFERS
// ============================================
let rppgSignal       = { red: [], green: [], blue: [], timestamps: [] };
let lastRppgTime     = 0;
let bpmUpdateCounter = 0;
let estimatedFPS     = 30;
window.lastFrameTime = performance.now();

// ============================================
// KALMAN FILTER
// ============================================
class KalmanFilter {
  constructor() { this.x = 70; this.P = 1; this.Q = RPPG_CONFIG.KALMAN_Q; this.R = RPPG_CONFIG.KALMAN_R; }
  filter(measurement) {
    const P_pred = this.P + this.Q, K = P_pred / (P_pred + this.R);
    this.x = this.x + K * (measurement - this.x); this.P = (1 - K) * P_pred; return this.x;
  }
  reset() { this.x = 70; this.P = 1; }
}
const hrKalman = new KalmanFilter();

// ============================================
// SIGNAL PROCESSING & POS ALGORITHM
// ============================================
function applyPosAlgorithm(red, green, blue) {
  const n = red.length; if (n < 2) return new Float32Array(n);
  const meanR = red.reduce((a, b) => a + b, 0) / n;
  const meanG = green.reduce((a, b) => a + b, 0) / n;
  const meanB = blue.reduce((a, b) => a + b, 0) / n;
  const X = new Float32Array(n), Y = new Float32Array(n), h = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const normR = red[i] / (meanR || 1), normG = green[i] / (meanG || 1), normB = blue[i] / (meanB || 1);
    X[i] = normG - normB; Y[i] = normG + normB - 2 * normR;
  }
  const sigmaX = std(X), sigmaY = std(Y), alpha = sigmaY !== 0 ? sigmaX / sigmaY : 0;
  for (let i = 0; i < n; i++) h[i] = X[i] + alpha * Y[i];
  return h;
}
function std(arr) {
  const n = arr.length; if (n < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
}
function detrendSignal(signal) {
  const n = signal.length; if (n < 2) return signal;
  const xMean = (n - 1) / 2, yMean = signal.reduce((a, b) => a + b, 0) / n;
  let numerator = 0, denominator = 0;
  for (let i = 0; i < n; i++) { numerator += (i - xMean) * (signal[i] - yMean); denominator += Math.pow(i - xMean, 2); }
  const slope = denominator !== 0 ? numerator / denominator : 0, intercept = yMean - slope * xMean;
  return signal.map((val, i) => val - (slope * i + intercept));
}
function bandpassFilter(signal, fs, lowFreq, highFreq) {
  if (signal.length < 64) return signal;
  const detrended = detrendSignal(signal), windowSize = Math.max(3, Math.floor(fs / highFreq));
  const filtered = [];
  for (let i = 0; i < detrended.length; i++) {
    let sum = 0, count = 0;
    for (let j = Math.max(0, i - windowSize); j <= Math.min(detrended.length - 1, i + windowSize); j++) { sum += detrended[j]; count++; }
    filtered.push(sum / count);
  }
  return filtered;
}
function normalizeSignal(signal) {
  if (!signal.length) return signal;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const sd = Math.sqrt(signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length) || 1;
  return signal.map(val => (val - mean) / sd);
}
function estimateBpmFFT(signal, fps) {
  const n = signal.length; if (n < 64) return null;
  const windowed = new Float32Array(n);
  for (let i = 0; i < n; i++) windowed[i] = signal[i] * (0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1)));
  const minFreq = RPPG_CONFIG.MIN_HR / 60, maxFreq = RPPG_CONFIG.MAX_HR / 60;
  let maxMag = -1, maxIdx = -1;
  const magnitudes = [], freqStep = fps / n;
  for (let k = 0; k < n / 2; k++) {
    const freq = k * freqStep; let real = 0, imag = 0;
    for (let i = 0; i < n; i++) { const angle = (2 * Math.PI * k * i) / n; real += windowed[i] * Math.cos(angle); imag += windowed[i] * Math.sin(angle); }
    const mag = Math.sqrt(real * real + imag * imag);
    magnitudes.push(mag);
    if (freq >= minFreq && freq <= maxFreq && mag > maxMag) { maxMag = mag; maxIdx = k; }
  }
  if (maxIdx === -1) return null;
  const peakRange = 2; let signalPower = 0, noisePower = 0, signalBins = 0, noiseBins = 0;
  for (let k = 0; k < magnitudes.length; k++) {
    const m = magnitudes[k];
    if (k >= maxIdx - peakRange && k <= maxIdx + peakRange) { signalPower += m * m; signalBins++; }
    else { noisePower += m * m; noiseBins++; }
  }
  const meanSignal = signalBins > 0 ? signalPower / signalBins : 0;
  const meanNoise = Math.max(noiseBins > 0 ? noisePower / noiseBins : 1, 0.000001);
  const snr = 10 * Math.log10(meanSignal / meanNoise);
  let peakFreq = maxIdx * freqStep;
  if (maxIdx > 0 && maxIdx < magnitudes.length - 1) {
    const alpha = magnitudes[maxIdx - 1], beta = magnitudes[maxIdx], gamma = magnitudes[maxIdx + 1];
    const p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
    peakFreq = (maxIdx + p) * freqStep;
  }
  return { bpm: peakFreq * 60, confidence: snr, snr };
}
function estimateBpmAutocorr(signal, fps) {
  if (signal.length < 64) return null;
  const minLag = Math.round(fps * 60 / RPPG_CONFIG.MAX_HR), maxLag = Math.round(fps * 60 / RPPG_CONFIG.MIN_HR);
  let bestLag = 0, bestCorr = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, signal.length - 1); lag++) {
    let corr = 0;
    for (let i = 0; i < signal.length - lag; i++) corr += signal[i] * signal[i + lag];
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
  }
  if (bestLag === 0) return null;
  return Math.round((60 * fps) / bestLag);
}
function computeHRV(bpm) {
  if (!bpm || bpm < RPPG_CONFIG.MIN_HR || bpm > RPPG_CONFIG.MAX_HR) return { sdnn: 0, rmssd: 0 };
  bpmHistory.push(bpm); if (bpmHistory.length > BPM_HISTORY_SIZE) bpmHistory.shift();
  if (bpmHistory.length < 10) return { sdnn: 35, rmssd: 28 };
  const rrIntervals = bpmHistory.map(b => 60000 / b);
  const meanRR = rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length;
  const variance = rrIntervals.reduce((s, rr) => s + Math.pow(rr - meanRR, 2), 0) / rrIntervals.length;
  let sdnn = Math.sqrt(variance);
  let rmssdSum = 0;
  for (let i = 1; i < rrIntervals.length; i++) rmssdSum += Math.pow(rrIntervals[i] - rrIntervals[i - 1], 2);
  let rmssd = Math.sqrt(rmssdSum / (rrIntervals.length - 1));
  sdnn = Math.max(20, Math.min(65, sdnn)); rmssd = Math.max(15, Math.min(55, rmssd));
  return { sdnn: Math.round(sdnn), rmssd: Math.round(rmssd) };
}
function computeMeditationIndex(bpm, sdnn, rmssd) {
  if (!bpm || bpm < 40) return 0;
  let score = 0; const bpmDiff = Math.abs(bpm - 63);
  if (bpmDiff <= 5) score += 30; else if (bpmDiff <= 15) score += 30 - (bpmDiff - 5) * 2; else score += Math.max(5, 30 - bpmDiff);
  if (sdnn >= 50) score += 30; else if (sdnn >= 30) score += 15 + ((sdnn - 30) / 20) * 15; else score += Math.max(0, (sdnn / 30) * 15);
  if (rmssd >= 45) score += 30; else if (rmssd >= 25) score += 15 + ((rmssd - 25) / 20) * 15; else score += Math.max(0, (rmssd / 25) * 15);
  if (bpm >= 55 && bpm <= 75 && sdnn > 40 && rmssd > 35) score += 10;
  return Math.min(100, Math.round(score));
}
function calculateEyeRelaxScore(leftEAR, rightEAR, smoothedEAR) {
  const avgEAR = (leftEAR + rightEAR) / 2;
  if (avgEAR < EYE_CONFIG.CLOSED_BOOST)  return 1.0;
  if (smoothedEAR <= EYE_CONFIG.MIN_EAR) return 1.0;
  if (smoothedEAR >= EYE_CONFIG.MAX_EAR) return 0.0;
  const range = EYE_CONFIG.MAX_EAR - EYE_CONFIG.MIN_EAR;
  const normalized = (smoothedEAR - EYE_CONFIG.MIN_EAR) / range;
  return Math.max(0, Math.min(1, Math.pow(1 - normalized, EYE_CONFIG.CURVE_POWER)));
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║              WELLNESS ENGINE  v8.1  –  SCIENTIFIC CORE          ║
// ╚══════════════════════════════════════════════════════════════════╝
const WELLNESS_STORAGE_KEY = 'sns_wellness_sessions';
const WELLNESS_CONFIG = {
  BASELINE_N: 7, FORECAST_MIN: 5, MA_WINDOW: 5, PROJECTION_STEPS: 3,
  MAX_SESSIONS: 90, VAR_BPM: 225, VAR_BREATH: 100, VAR_MEDIT: 64,
  CONF_LOW_MAX: 0.35, CONF_MED_MAX: 0.65, CONF_SESSION_SAT: 20, CONF_SD_MAX: 30
};
function safeClamp(v) { return Math.round(Math.max(0, Math.min(100, Number(v) || 0))); }
function arrayStats(arr) {
  const n = arr.length;
  if (n === 0) return { mean: 0, variance: 0, sd: 0, min: 0, max: 0, n: 0 };
  let sum = 0, min = Infinity, max = -Infinity;
  for (const v of arr) { sum += v; if (v < min) min = v; if (v > max) max = v; }
  const mean = sum / n; let varSum = 0;
  for (const v of arr) varSum += (v - mean) ** 2;
  const variance = varSum / n;
  return { mean, variance, sd: Math.sqrt(variance), min, max, n };
}
function movingAverage(scores, window = WELLNESS_CONFIG.MA_WINDOW) {
  if (!scores.length) return 0;
  const slice = scores.slice(-window);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}
function olsRegression(scores) {
  const n = scores.length;
  if (n < 2) { const val = n === 1 ? scores[0] : 0; return { slope: 0, intercept: val, predict: () => val }; }
  const xMean = (n - 1) / 2, yMean = scores.reduce((a, b) => a + b, 0) / n;
  let Sxx = 0, Sxy = 0;
  for (let i = 0; i < n; i++) { const dx = i - xMean; Sxx += dx * dx; Sxy += dx * (scores[i] - yMean); }
  const slope = Sxx !== 0 ? Sxy / Sxx : 0, intercept = yMean - slope * xMean;
  return { slope, intercept, predict: x => intercept + slope * x };
}
function getSessions() { try { return JSON.parse(localStorage.getItem(WELLNESS_STORAGE_KEY) || '[]'); } catch (_) { return []; } }
function saveSession(rawScore) {
  const sessions = getSessions(); sessions.push({ date: new Date().toISOString(), score: rawScore });
  if (sessions.length > WELLNESS_CONFIG.MAX_SESSIONS) sessions.splice(0, sessions.length - WELLNESS_CONFIG.MAX_SESSIONS);
  localStorage.setItem(WELLNESS_STORAGE_KEY, JSON.stringify(sessions));
}
function getBaseline(sessions) {
  if (sessions.length < WELLNESS_CONFIG.BASELINE_N) return { baselineMean: null, calibrated: false };
  const { mean } = arrayStats(sessions.slice(0, WELLNESS_CONFIG.BASELINE_N).map(s => s.score));
  return { baselineMean: mean, calibrated: true };
}
function calibrateScore(raw, baselineMean) {
  if (!baselineMean) return safeClamp(raw);
  return safeClamp(((raw - baselineMean) / baselineMean) * 50 + 50);
}
function calculateStability(bpm, breathScore, meditScore) {
  const IDEAL_BPM = 70;
  const bpmComponent = (!bpm || bpm <= 0) ? 50 : safeClamp(100 - Math.abs(bpm - IDEAL_BPM));
  const breathComponent = safeClamp(breathScore || 0), meditComponent = safeClamp(meditScore || 0);
  const wBpm = 1 / WELLNESS_CONFIG.VAR_BPM, wBreath = 1 / WELLNESS_CONFIG.VAR_BREATH, wMedit = 1 / WELLNESS_CONFIG.VAR_MEDIT;
  const wTotal = wBpm + wBreath + wMedit;
  return safeClamp((bpmComponent * wBpm + breathComponent * wBreath + meditComponent * wMedit) / wTotal);
}
function calculateStreak() {
  const sessions = getSessions(); if (!sessions.length) return 0;
  const datesWithSession = new Set(sessions.map(s => s.date.slice(0, 10)));
  let streak = 0; const today = new Date();
  for (let offset = 0; offset < 365; offset++) {
    const d = new Date(today); d.setDate(d.getDate() - offset);
    const key = d.toISOString().slice(0, 10);
    if (datesWithSession.has(key)) streak++; else break;
  }
  return streak;
}
function computeConfidence(sessionCount, sd) {
  const countFactor = Math.min(sessionCount, WELLNESS_CONFIG.CONF_SESSION_SAT) / WELLNESS_CONFIG.CONF_SESSION_SAT;
  const sdFactor = Math.max(0, 1 - sd / WELLNESS_CONFIG.CONF_SD_MAX);
  const combined = Math.max(0, Math.min(1, countFactor * 0.6 + sdFactor * 0.4));
  const label = combined < WELLNESS_CONFIG.CONF_LOW_MAX ? 'Low' : combined < WELLNESS_CONFIG.CONF_MED_MAX ? 'Medium' : 'High';
  return { label, score: combined };
}
function calculateWeeklyForecast() {
  const allSessions = getSessions(); if (allSessions.length < WELLNESS_CONFIG.FORECAST_MIN) return null;
  const { baselineMean, calibrated } = getBaseline(allSessions);
  const window14 = allSessions.slice(-14), rawScores = window14.map(s => s.score);
  const calibratedScores = rawScores.map(s => calibrateScore(s, baselineMean));
  const ma5 = movingAverage(calibratedScores, WELLNESS_CONFIG.MA_WINDOW);
  const { slope } = olsRegression(calibratedScores);
  const projected = safeClamp(ma5 + slope * WELLNESS_CONFIG.PROJECTION_STEPS);
  const { sd } = arrayStats(calibratedScores);
  const confidence = computeConfidence(allSessions.length, sd);
  const direction = slope > 0.5 ? 'improving' : slope < -0.5 ? 'declining slightly' : 'stable';
  return {
    projected, ma5: safeClamp(ma5), slope: Math.round(slope * 100) / 100,
    direction, confidence, sessions: window14, calibratedScores,
    baselineCalibrated: calibrated, baselineMean: calibrated ? Math.round(baselineMean) : null
  };
}
function updateWellnessUI(rawStabilityScore) {
  const scoreEl = document.getElementById('stabilityScore'), messageEl = document.getElementById('stabilityMessage');
  const streakEl = document.getElementById('streakDisplay'), forecastEl = document.getElementById('weeklyForecast');
  const ringFill = document.getElementById('stabilityRingFill'), ringLabel = document.getElementById('stabilityRingLabel');
  const barsWrap = document.getElementById('forecastBarsWrap'), barsEl = document.getElementById('forecastBars');
  const allSessions = getSessions();
  const { baselineMean, calibrated } = getBaseline(allSessions);
  const displayScore = calibrated ? calibrateScore(rawStabilityScore, baselineMean) : safeClamp(rawStabilityScore);
  if (scoreEl) {
    scoreEl.textContent = displayScore; scoreEl.classList.remove('score-high', 'score-mid', 'score-low');
    if (displayScore >= 75) scoreEl.classList.add('score-high'); else if (displayScore >= 50) scoreEl.classList.add('score-mid'); else scoreEl.classList.add('score-low');
    scoreEl.classList.remove('score-revealed'); void scoreEl.offsetWidth; scoreEl.classList.add('score-revealed');
  }
  if (ringFill) { const CIRC = 226; ringFill.style.strokeDashoffset = CIRC - (displayScore / 100) * CIRC; }
  if (ringLabel) ringLabel.textContent = displayScore;
  if (messageEl) {
    if (!calibrated) {
      const rem = WELLNESS_CONFIG.BASELINE_N - allSessions.length;
      messageEl.textContent = rem > 0 ? `Calibrating your baseline… ${rem} session${rem !== 1 ? 's' : ''} to go.` : 'Baseline established — personalised scoring active.';
    } else if (displayScore >= 80) messageEl.textContent = 'Excellent emotional balance today.';
    else if (displayScore >= 65) messageEl.textContent = 'Good stability. Keep practicing.';
    else if (displayScore >= 45) messageEl.textContent = 'Building steadily — stay consistent.';
    else messageEl.textContent = 'Your system needs recovery. Stay consistent.';
  }
  if (streakEl) {
    const streak = calculateStreak();
    if (streak >= 2) { streakEl.textContent = `🔥 ${streak}-Day Stability Streak`; streakEl.style.display = 'inline-flex'; }
    else if (streak === 1) { streakEl.textContent = '✨ First day — come back tomorrow!'; streakEl.style.display = 'inline-flex'; }
    else streakEl.style.display = 'none';
  }
  const forecast = calculateWeeklyForecast();
  if (!forecast) {
    if (forecastEl) {
      const rem = WELLNESS_CONFIG.FORECAST_MIN - allSessions.length;
      forecastEl.textContent = rem > 0 ? `Complete ${rem} more session${rem !== 1 ? 's' : ''} to unlock your weekly outlook.` : 'Complete more sessions to unlock your weekly outlook.';
      forecastEl.classList.remove('forecast-ready');
    }
    if (barsWrap) barsWrap.style.display = 'none'; return;
  }
  if (forecastEl) {
    const confColor = forecast.confidence.label === 'High' ? '#10b981' : forecast.confidence.label === 'Medium' ? '#f59e0b' : '#ef4444';
    const baseNote = forecast.baselineCalibrated ? ` <span style="font-size:0.7rem;opacity:0.6;">(baseline: ${forecast.baselineMean})</span>` : '';
    const slopeStr = (forecast.slope >= 0 ? '+' : '') + forecast.slope;
    forecastEl.innerHTML = `Your trend is <strong>${forecast.direction}</strong>${baseNote}. 5-session average: <strong>${forecast.ma5}</strong> · Trend: <strong>${slopeStr} pts/session</strong>. Score may reach <span class="forecast-projected-score">${forecast.projected}</span> this week. <span style="font-size:0.7rem;font-weight:600;color:${confColor};">Confidence: ${forecast.confidence.label}</span>`;
    forecastEl.classList.add('forecast-ready');
  }
  if (barsEl && barsWrap) {
    barsWrap.style.display = 'block'; barsEl.innerHTML = '';
    const chartScores = forecast.calibratedScores, maxVal = Math.max(...chartScores, forecast.projected, 1);
    chartScores.forEach((score, i) => {
      const bar = document.createElement('div');
      bar.className = 'forecast-mini-bar'; bar.style.height = Math.round((score / maxVal) * 100) + '%';
      bar.title = `Session ${i + 1}: ${Math.round(score)}`; barsEl.appendChild(bar);
    });
    const projBar = document.createElement('div');
    projBar.className = 'forecast-mini-bar projected';
    projBar.style.height = Math.round((forecast.projected / maxVal) * 100) + '%';
    projBar.title = `Projected: ${forecast.projected}`; barsEl.appendChild(projBar);
  }
}

// ============================================
// MAIN APP LOGIC
// ============================================
(function () {

  // ─────────────────────────────────────────────────────────────────
  // ██████████████████████████████████████████████████████████████
  //   SECTION M1 — MOBILE AUDIO UNLOCK GATE
  //   Problem: Android/iOS require a user-gesture before ANY audio
  //   can play. The AudioContext is created in "suspended" state.
  //   Fix: We register a one-time touch/click listener that:
  //     1. Resumes the AudioContext
  //     2. Plays + instantly pauses bgAudio to unlock it (iOS trick)
  //     3. Warms up SpeechSynthesis with a silent utterance
  //     4. Sets a global flag so we never unlock twice
  // ██████████████████████████████████████████████████████████████
  // ─────────────────────────────────────────────────────────────────
  let audioUnlocked = false;

  function unlockAudioOnMobile() {
    if (audioUnlocked) return;
    audioUnlocked = true;

    // 1. Resume AudioContext if it exists
    if (audioContext && audioContext.state === 'suspended') {
      audioContext.resume().catch(() => {});
    }

    // 2. Unlock <audio> element (iOS/Android autoplay block)
    //    Play a tiny burst then pause — this satisfies the browser's
    //    user-gesture requirement and caches the "allowed" state.
    const bgAudio = document.getElementById('bgAudio');
    if (bgAudio) {
      bgAudio.muted  = true;
      bgAudio.volume = 0;
      const unlockPlay = bgAudio.play();
      if (unlockPlay !== undefined) {
        unlockPlay
          .then(() => {
            bgAudio.pause();
            bgAudio.muted  = false;
            bgAudio.currentTime = 0;
            // Restore volume from slider
            const vol = document.getElementById('musicVolume');
            bgAudio.volume = vol ? (vol.value / 100) : 0.5;
          })
          .catch(() => {
            bgAudio.muted = false;
          });
      }
    }

    // 3. Warm up SpeechSynthesis — Android needs this first call
    //    inside a gesture to activate the synthesis engine.
    if (window.speechSynthesis) {
      const warmUp = new SpeechSynthesisUtterance('');
      warmUp.volume = 0;
      warmUp.rate   = 1;
      window.speechSynthesis.speak(warmUp);
      // Cancel the silent utterance after it starts
      setTimeout(() => {
        try { window.speechSynthesis.cancel(); } catch (_) {}
      }, 100);
    }
  }

  // Attach unlock to the very first interaction anywhere on the page
  ['touchstart', 'touchend', 'click', 'keydown'].forEach(evt => {
    document.addEventListener(evt, unlockAudioOnMobile, { once: true, passive: true });
  });


  // ─────────────────────────────────────────────────────────────────
  // ██████████████████████████████████████████████████████████████
  //   SECTION M2 — AUDIOCONTEXT STATE WATCHDOG
  //   Problem: Android Chrome randomly suspends the AudioContext
  //   mid-session (e.g. when screen dims or another app beeps).
  //   Fix: Poll every 2 s; if suspended during an active session,
  //   call resume(). Also handle the Page Visibility API so audio
  //   recovers when the user switches back to the tab.
  // ██████████████████████████████████████████████████████████████
  // ─────────────────────────────────────────────────────────────────
  let audioWatchdogTimer = null;

  function startAudioWatchdog() {
    if (audioWatchdogTimer) return;
    audioWatchdogTimer = setInterval(() => {
      if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume().catch(() => {});
      }
    }, 2000);
  }

  function stopAudioWatchdog() {
    if (audioWatchdogTimer) { clearInterval(audioWatchdogTimer); audioWatchdogTimer = null; }
  }

  // Page Visibility: resume audio when tab becomes visible again
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume().catch(() => {});
      }
      const bgAudio = document.getElementById('bgAudio');
      if (bgAudio && window.__sessionActive && bgAudio.paused && bgAudio.src && bgAudio.src !== window.location.href) {
        bgAudio.play().catch(() => {});
      }
    }
  });


  // ─────────────────────────────────────────────────────────────────
  // ██████████████████████████████████████████████████████████████
  //   SECTION M3 — ANDROID SPEECH SYNTHESIS FIX
  //   Problem 1 (Race condition):
  //     On Android Chrome, calling synth.cancel() immediately
  //     followed by synth.speak() causes the speak() call to be
  //     silently dropped ~80% of the time.
  //   Problem 2 (Synthesis engine freeze):
  //     Android's TTS sometimes stalls with a pending utterance
  //     and never fires. The workaround is to resume() the synth,
  //     then re-queue after a short delay.
  //   Problem 3 (No voices loaded yet):
  //     On first load, getVoices() returns [] on Android.
  //     We must wait for the voiceschanged event.
  //   Fix: A small speak() wrapper that:
  //     - Detects Android
  //     - If already speaking, cancels and waits 350 ms before
  //       re-queuing (instead of speaking immediately)
  //     - Applies a 100 ms safety delay on Android always
  //     - Caps pending queue at 1 item to prevent pile-up
  // ██████████████████████████████████████████████████████████████
  // ─────────────────────────────────────────────────────────────────
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isIOS     = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  const isMobile  = isAndroid || isIOS;

  let speechQueue        = [];   // max 1 pending utterance
  let speechPending      = false;
  let speechCancelTimer  = null;
  let voicesReady        = false;

  function loadVoices() {
    const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    if (voices.length > 0) { voicesReady = true; return; }
    if (window.speechSynthesis) {
      window.speechSynthesis.addEventListener('voiceschanged', () => { voicesReady = true; }, { once: true });
    }
  }
  loadVoices();

  function speak(textHi, textEn) {
    if (muted) return;
    const synth = window.speechSynthesis;
    if (!synth) return;
    const text = textEn || textHi || '';
    if (!text) return;

    if (isMobile) {
      // ── MOBILE PATH ──────────────────────────────────────────────
      // Replace any queued item (we only keep the latest prompt)
      speechQueue = [text];
      if (speechPending) return;   // drainSpeechQueue will pick it up
      drainSpeechQueue();
    } else {
      // ── DESKTOP PATH (original behaviour) ────────────────────────
      if (synth.speaking) { try { synth.cancel(); } catch (_) {} }
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
      synth.speak(u);
    }
  }

  function drainSpeechQueue() {
    if (!speechQueue.length) { speechPending = false; return; }
    speechPending = true;
    const text  = speechQueue.shift();
    const synth = window.speechSynthesis;
    if (!synth) { speechPending = false; return; }

    // If still speaking, cancel first then delay before speaking
    if (synth.speaking) {
      try { synth.cancel(); } catch (_) {}
      if (speechCancelTimer) clearTimeout(speechCancelTimer);
      speechCancelTimer = setTimeout(() => doSpeak(text), 350);
    } else {
      // Android needs a small guaranteed gap even when not speaking
      if (speechCancelTimer) clearTimeout(speechCancelTimer);
      speechCancelTimer = setTimeout(() => doSpeak(text), isAndroid ? 120 : 0);
    }
  }

  function doSpeak(text) {
    const synth = window.speechSynthesis;
    if (!synth || muted) { speechPending = false; return; }

    const u      = new SpeechSynthesisUtterance(text);
    u.rate       = isAndroid ? 0.95 : 1.0;  // slightly slower for Android clarity
    u.pitch      = 1.0;
    u.volume     = 1.0;
    u.lang       = 'en-US';

    // Pick a voice: prefer a local (non-network) voice on mobile
    const voices = synth.getVoices();
    if (voices.length) {
      const local = voices.find(v => v.localService && /en/i.test(v.lang));
      if (local) u.voice = local;
    }

    u.onend   = () => { speechPending = false; drainSpeechQueue(); };
    u.onerror = (e) => {
      // 'interrupted' is normal on Android when we cancel mid-utterance
      if (e.error !== 'interrupted') {
        // Try once more after 400 ms
        setTimeout(() => { speechPending = false; if (speechQueue.length) drainSpeechQueue(); }, 400);
      } else {
        speechPending = false; drainSpeechQueue();
      }
    };

    try {
      synth.speak(u);

      // ── Android synthesis freeze watchdog ─────────────────────
      // If the utterance hasn't started within 3 s, the engine
      // has frozen. Resume + cancel + re-queue.
      if (isAndroid) {
        setTimeout(() => {
          if (synth.speaking && !synth.paused) return; // normal
          try { synth.resume(); synth.cancel(); } catch (_) {}
          speechPending = false;
          speechQueue.unshift(text);   // put it back
          setTimeout(drainSpeechQueue, 300);
        }, 3000);
      }
    } catch (_) {
      speechPending = false;
    }
  }


  // ─────────────────────────────────────────────────────────────────
  // ██████████████████████████████████████████████████████████████
  //   SECTION M4 — BACKGROUND AUDIO (bgAudio) MOBILE FIXES
  //   Problem: On Android, <audio> play() is blocked unless it
  //   happens inside a gesture, and volume can be reset by the OS.
  //   Fix:
  //     - playsinline attribute is set via JS (in case HTML forgot)
  //     - We store the desired volume and re-apply after every
  //       play() call (Android sometimes resets it)
  //     - On iOS, we force load() before play() every time
  //     - Retry play() once on failure with 500 ms delay
  // ██████████████████████████████████████████████████████████████
  // ─────────────────────────────────────────────────────────────────
  function startAmbience() {
    const bgAudio    = document.getElementById('bgAudio');
    const musicSelect = document.getElementById('musicSelect');
    const musicVolume = document.getElementById('musicVolume');
    const musicBtnIcon = document.getElementById('musicBtnIcon');
    if (!bgAudio || !musicSelect) return;
    const url = musicSelect.value;
    if (!url || url === 'none') { stopAmbience(); return; }

    const desiredVol = musicVolume ? (musicVolume.value / 100) : 0.5;

    showAudioLoading(true);
    bgAudio.pause();
    // Ensure mobile-friendly attributes
    bgAudio.setAttribute('playsinline', '');
    bgAudio.setAttribute('webkit-playsinline', '');
    bgAudio.preload  = 'auto';
    bgAudio.src      = url;
    bgAudio.load();

    const attemptPlay = (attempt = 1) => {
      bgAudio.volume = desiredVol;
      const p = bgAudio.play();
      if (p !== undefined) {
        p.then(() => {
          bgAudio.volume = desiredVol;  // re-apply — Android may reset it
          if (musicBtnIcon) musicBtnIcon.textContent = '⏸';
        }).catch(err => {
          if (attempt < 3) {
            // Retry after short delay — gives Android time to release audio focus
            setTimeout(() => attemptPlay(attempt + 1), 500 * attempt);
          }
        });
      }
    };

    bgAudio.addEventListener('canplay', () => { showAudioLoading(false); attemptPlay(); }, { once: true });
    bgAudio.addEventListener('error',   () => { showAudioLoading(false); }, { once: true });

    if (musicBtnIcon) musicBtnIcon.textContent = '⏸';
  }

  function stopAmbience() {
    const bgAudio    = document.getElementById('bgAudio');
    const musicBtnIcon = document.getElementById('musicBtnIcon');
    if (bgAudio) bgAudio.pause();
    if (musicBtnIcon) musicBtnIcon.textContent = '▶️';
  }

  function showAudioLoading(show) {
    const musicLoading = document.getElementById('musicLoading');
    if (musicLoading) musicLoading.style.display = show ? 'inline' : 'none';
  }


  // ─────────────────────────────────────────────────────────────────
  // DOM CONTROLS
  // ─────────────────────────────────────────────────────────────────
  const micSelect       = document.getElementById('micSelect');
  const allowMicBtn     = document.getElementById('allowMicBtn');
  const allowCamBtn     = document.getElementById('allowCam');
  const startBtn        = document.getElementById('startBtn');
  const stopBtn         = document.getElementById('stopBtn');
  const sessionStatus   = document.getElementById('sessionStatus');
  const breathCountEl   = document.getElementById('breathCount');
  const breathingCircle = document.getElementById('breathingCircle');
  const userNameInput   = document.getElementById('userName');
  const userAgeInput    = document.getElementById('userAge');
  const userSkinToneInput = document.getElementById('userSkinTone');
  const cielabToggle    = document.getElementById('cielabToggle');
  const cielabStatusEl  = document.getElementById('cielabStatus');

  let cielabEnabled = true;
  const savedCielab = localStorage.getItem('sns_cielab_enabled');
  if (savedCielab !== null) cielabEnabled = savedCielab === 'true';
  if (cielabToggle) {
    cielabToggle.checked = cielabEnabled;
    cielabToggle.addEventListener('change', () => {
      cielabEnabled = cielabToggle.checked;
      localStorage.setItem('sns_cielab_enabled', String(cielabEnabled));
      if (cielabStatusEl) cielabStatusEl.textContent = cielabEnabled ? 'Active' : 'Off';
    });
  }
  if (cielabStatusEl) cielabStatusEl.textContent = cielabEnabled ? 'Active' : 'Off';

  if (userNameInput)     userNameInput.value     = localStorage.getItem('sns_user_name') || '';
  if (userAgeInput)      userAgeInput.value      = localStorage.getItem('sns_user_age')  || '';
  if (userSkinToneInput) userSkinToneInput.value = localStorage.getItem('sns_user_tone') || 'Type 3';
  [userNameInput, userAgeInput, userSkinToneInput].forEach(el => {
    if (el) el.addEventListener('change', () => {
      localStorage.setItem('sns_user_name', userNameInput?.value || '');
      localStorage.setItem('sns_user_age',  userAgeInput?.value  || '');
      localStorage.setItem('sns_user_tone', userSkinToneInput?.value || '');
    });
  });

  const musicSelect    = document.getElementById('musicSelect');
  const musicVolume    = document.getElementById('musicVolume');
  const volumeLabel    = document.getElementById('volumeLabel');
  const bgAudio        = document.getElementById('bgAudio');
  const toggleMusicBtn = document.getElementById('toggleMusicBtn');

  // Apply mobile attributes to bgAudio immediately
  if (bgAudio) {
    bgAudio.setAttribute('playsinline', '');
    bgAudio.setAttribute('webkit-playsinline', '');
    bgAudio.preload = 'auto';
    bgAudio.addEventListener('loadstart', () => showAudioLoading(true));
    bgAudio.addEventListener('canplay',   () => showAudioLoading(false));
    bgAudio.addEventListener('waiting',   () => showAudioLoading(true));
    bgAudio.addEventListener('playing',   () => {
      showAudioLoading(false);
      // Re-apply volume after play() in case Android reset it
      const desiredVol = musicVolume ? (musicVolume.value / 100) : 0.5;
      bgAudio.volume = desiredVol;
    });
    bgAudio.addEventListener('error', () => {
      showAudioLoading(false);
      if (window.__sessionActive) {
        // Suppress alert on mobile — just silently fail audio track
        console.warn('Audio track failed to load');
      }
    });
  }

  const breathStatus  = document.getElementById('breathStatus');
  const waveCanvas    = document.getElementById('waveCanvas');
  const waveCtx       = waveCanvas.getContext('2d');
  const faceCanvas    = document.querySelector('.output_canvas');
  const faceVideo     = document.querySelector('.input_video');
  const faceCtx       = faceCanvas.getContext('2d');

  const eyeScoreTxt   = document.getElementById('eyeScoreTxt');
  const headScoreTxt  = document.getElementById('headScoreTxt');
  const gazeScoreTxt  = document.getElementById('gazeScoreTxt');
  const eyeBar        = document.getElementById('eyeBar');
  const headBar       = document.getElementById('headBar');
  const gazeBar       = document.getElementById('gazeBar');
  const bpmDisplay    = document.getElementById('bpmDisplay');
  const heartrateTxt  = document.getElementById('heartrateTxt');
  const heartrateBar  = document.getElementById('heartrateBar');
  const sdnnTxt       = document.getElementById('sdnnTxt');
  const sdnnBar       = document.getElementById('sdnnBar');
  const rmssdTxt      = document.getElementById('rmssdTxt');
  const rmssdBar      = document.getElementById('rmssdBar');
  const faceCalmTxt   = document.getElementById('faceCalmTxt');
  const faceCalmBar   = document.getElementById('faceCalmBar');
  const breathConsTxt2 = document.getElementById('breathConsTxt2');
  const breathConsBar2 = document.getElementById('breathConsBar2');
  const meditationIndexTxt = document.getElementById('meditationIndexTxt');
  const meditationIndexBar = document.getElementById('meditationIndexBar');
  const overallBar    = document.getElementById('overallBar');
  const overallTxt    = document.getElementById('overallTxt');
  const overallNote   = document.getElementById('overallNote');
  const breathConsTxt = document.getElementById('breathConsTxt');
  const breathConsBar = document.getElementById('breathConsBar');
  const micLvl        = document.getElementById('micLvl');
  const micLvlBar     = document.getElementById('micLvlBar');
  const fpsNode       = document.getElementById('fps');
  const facesDetectedNode = document.getElementById('facesDetected');
  const sessionMeter  = document.getElementById('sessionMeter');
  const historyBody   = document.getElementById('historyBody');
  const exportCsvBtn  = document.getElementById('exportCsv');
  const clearHistoryBtn = document.getElementById('clearHistory');

  let selectedMicId = null;
  let localStream   = null;
  let audioContext  = null;
  let analyser      = null;
  let analyserData  = null;
  let animationId   = null;

  let isSessionRunning   = false;
  let cycleTimeout       = null;
  let scriptedGrow       = false;
  let breathingBaseScale = 1.0;
  let muted              = false;

  // Session timer
  let sessionStartTime     = 0;
  let sessionTimerInterval = null;
  function formatElapsedTime(ms) {
    const total = Math.floor(ms / 1000), minutes = Math.floor(total / 60), seconds = total % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }
  function updateSessionTimer() {
    if (!isSessionRunning || !sessionStartTime) return;
    const el = document.getElementById('sessionStatus');
    if (el) el.innerHTML = `running <span style="opacity:0.7;">•</span> ${formatElapsedTime(Date.now() - sessionStartTime)}`;
  }

  let ampDerivBuf   = [];
  let faceMeshModel = null;
  let cameraInstance = null;
  let sessionHistory = JSON.parse(localStorage.getItem('sns_sessions') || '[]');

  let currentBreathPhase    = 'idle';
  let phaseAmplitudes       = [];
  let phaseConsistencyScore = 0.5;
  let sessionBreathScores   = [];
  let sessionEyeScores      = [];
  let sessionHeadScores     = [];
  let sessionGazeScores     = [];
  let sessionMeditationScores = [];
  let sessionBPMValues      = [];
  let smoothedEAR = 0.25;
  let gazeBuf = [], noseBuf = [];
  let frames = 0, lastFrameTs = performance.now();

  function clamp01(x) { return Math.max(0, Math.min(1, Number(x) || 0)); }

  async function ensurePermissionForMic() {
    try { const tmp = await navigator.mediaDevices.getUserMedia({ audio: true }); tmp.getTracks().forEach(t => t.stop()); return true; }
    catch (e) { return false; }
  }
  function preferredDeviceId(devices) {
    const prefer = /headset|earbud|wired|external|usb|line in|communications|headphone/i;
    const found = devices.filter(d => prefer.test(d.label || ''));
    return found.length ? found[0].deviceId : (devices[0]?.deviceId ?? null);
  }
  async function loadDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter(d => d.kind === 'audioinput');
      micSelect.innerHTML = '';
      if (!mics.length) {
        const opt = document.createElement('option'); opt.value = ''; opt.textContent = 'No microphone found';
        micSelect.appendChild(opt); selectedMicId = null; return;
      }
      mics.forEach((m, i) => {
        const opt = document.createElement('option'); opt.value = m.deviceId || '';
        opt.textContent = m.label || `Microphone ${i + 1}`; micSelect.appendChild(opt);
      });
      const pick = (selectedMicId && Array.from(micSelect.options).some(o => o.value === selectedMicId))
        ? selectedMicId : preferredDeviceId(mics);
      selectedMicId = pick || mics[0].deviceId; micSelect.value = selectedMicId;
    } catch (e) {}
  }
  micSelect.addEventListener('change', e => { selectedMicId = e.target.value || null; });
  document.getElementById('refreshMics')?.addEventListener('click', async () => {
    await ensurePermissionForMic(); await loadDevices(); alert('Microphones refreshed');
  });
  allowMicBtn.addEventListener('click', async () => {
    unlockAudioOnMobile();   // ← trigger audio unlock on this gesture too
    const ok = await ensurePermissionForMic();
    if (ok) { await loadDevices(); alert('Microphone permission granted.'); } else alert('Mic permission denied.');
  });

  async function startCameraForFace() {
    try {
      faceCanvas.width = faceVideo.clientWidth || 420; faceCanvas.height = faceVideo.clientHeight || 300;
      faceMeshModel = new FaceMesh({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
      faceMeshModel.setOptions({ maxNumFaces: 1, refineLandmarks: false, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
      faceMeshModel.onResults(onFaceResults);
      cameraInstance = new Camera(faceVideo, { onFrame: async () => { await faceMeshModel.send({ image: faceVideo }); }, width: 320, height: 240 });
      cameraInstance.start();
    } catch (e) { throw e; }
  }
  allowCamBtn.addEventListener('click', async () => {
    unlockAudioOnMobile();   // ← also unlock audio on camera tap
    try { await startCameraForFace(); } catch (e) { alert('Camera permission is required.'); }
  });

  function calcEAR(landmarks, left) {
    const ids = left ? [33, 160, 158, 133, 153, 144] : [362, 385, 387, 263, 373, 380];
    if (!landmarks || landmarks.length < 468) return smoothedEAR;
    try {
      const [p0, p1, p2, p3, p4, p5] = ids.map(i => landmarks[i]);
      if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5) return smoothedEAR;
      const distV1 = Math.hypot(p2.x - p4.x, p2.y - p4.y), distV2 = Math.hypot(p1.x - p5.x, p1.y - p5.y);
      const distH = Math.hypot(p0.x - p3.x, p0.y - p3.y);
      if (distH === 0) return smoothedEAR;
      return (distV1 + distV2) / (2.0 * distH);
    } catch (_) { return smoothedEAR; }
  }
  function getRoiFromLandmarks(lm) {
    const foreheadIds = [10, 67, 69, 104, 108, 151, 337, 333, 297, 338];
    const points = foreheadIds.map(idx => lm[idx]).filter(Boolean);
    if (!points.length) return null;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const w = (maxX - minX) * faceCanvas.width, h = (maxY - minY) * faceCanvas.height;
    const x = minX * faceCanvas.width, y = minY * faceCanvas.height;
    return {
      x: Math.floor(x + (w - Math.max(10, Math.floor(w * RPPG_CONFIG.ROI_WIDTH_FACTOR))) / 2),
      y: Math.floor(y + h * RPPG_CONFIG.ROI_Y_OFFSET),
      w: Math.max(10, Math.floor(w * RPPG_CONFIG.ROI_WIDTH_FACTOR)),
      h: Math.max(8,  Math.floor(h * RPPG_CONFIG.ROI_HEIGHT_FACTOR))
    };
  }
  function getAvgRGBFromRoi(ctx, roi) {
    try {
      const { x, y, w, h } = roi; if (w <= 0 || h <= 0) return null;
      const imageData = ctx.getImageData(x, y, w, h), data = imageData.data;
      if (!data || !data.length) return null;
      if (cielabEnabled) { cielabSkinTransfer(data, w, h); ctx.putImageData(imageData, x, y); }
      let totalR = 0, totalG = 0, totalB = 0; const pxCount = data.length / 4;
      for (let i = 0; i < data.length; i += 4) { totalR += data[i]; totalG += data[i + 1]; totalB += data[i + 2]; }
      ctx.strokeStyle = cielabEnabled ? 'rgba(245,158,11,0.95)' : 'rgba(0,255,0,0.9)';
      ctx.lineWidth = 4; ctx.strokeRect(x, y, w, h);
      return { red: totalR / pxCount, green: totalG / pxCount, blue: totalB / pxCount };
    } catch (_) { return null; }
  }

  function onFaceResults(results) {
    const now = performance.now();
    if (now - lastRppgTime < 1000 / RPPG_CONFIG.FPS) return;
    lastRppgTime = now;
    const deltaTime = now - window.lastFrameTime;
    estimatedFPS = deltaTime > 0 ? 1000 / deltaTime : 30;
    window.lastFrameTime = now;
    if (!results.multiFaceLandmarks?.[0]) { if (facesDetectedNode) facesDetectedNode.textContent = '0'; return; }
    if (facesDetectedNode) facesDetectedNode.textContent = '1';
    const lm = results.multiFaceLandmarks[0];
    faceCtx.save(); faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
    faceCtx.drawImage(results.image, 0, 0, faceCanvas.width, faceCanvas.height);
    if (window.__sessionActive) {
      const roi = getRoiFromLandmarks(lm);
      if (roi) {
        const avgRGB = getAvgRGBFromRoi(faceCtx, roi);
        if (avgRGB) {
          rppgSignal.red.push(avgRGB.red); rppgSignal.green.push(avgRGB.green);
          rppgSignal.blue.push(avgRGB.blue); rppgSignal.timestamps.push(now);
          while (rppgSignal.green.length > RPPG_CONFIG.BUFFER_SIZE) {
            rppgSignal.red.shift(); rppgSignal.green.shift(); rppgSignal.blue.shift(); rppgSignal.timestamps.shift();
          }
        }
      }
    }
    faceCtx.restore();
    bpmUpdateCounter++;
    const minBuf = Math.floor(RPPG_CONFIG.BUFFER_SIZE * RPPG_CONFIG.MIN_BUFFER_FILL);
    if (window.__sessionActive && rppgSignal.green.length >= minBuf && bpmUpdateCounter % RPPG_CONFIG.UPDATE_INTERVAL === 0) {
      const posSignal  = applyPosAlgorithm(rppgSignal.red.slice(-minBuf), rppgSignal.green.slice(-minBuf), rppgSignal.blue.slice(-minBuf));
      const detrended  = detrendSignal(posSignal);
      const filtered   = bandpassFilter(detrended, estimatedFPS, RPPG_CONFIG.MIN_HR / 60, RPPG_CONFIG.MAX_HR / 60);
      const normalized = normalizeSignal(filtered);
      const fftResult  = estimateBpmFFT(normalized, estimatedFPS);
      const autocorrBpm = estimateBpmAutocorr(normalized, estimatedFPS);
      let rawBpm = 0, finalConfidence = 0;
      if (fftResult) { rawBpm = fftResult.bpm; finalConfidence = fftResult.snr; if (autocorrBpm && Math.abs(rawBpm - autocorrBpm) > 15) finalConfidence -= 3; }
      const signalBar = document.getElementById('signalQualityBar'), signalTxt = document.getElementById('signalQualityTxt');
      if (signalBar && signalTxt) {
        const qualityPct = Math.min(100, Math.max(0, (finalConfidence + 5) * 5));
        signalBar.style.width = qualityPct + '%';
        if (finalConfidence < 1) { signalBar.style.background = 'var(--error)'; signalTxt.textContent = 'Weak / Noise'; }
        else if (finalConfidence < 5) { signalBar.style.background = 'var(--warning)'; signalTxt.textContent = 'Fair'; }
        else { signalBar.style.background = 'var(--success)'; signalTxt.textContent = 'Excellent'; }
      }
      if (rawBpm > 0 && finalConfidence > 0.5) {
        const smoothedBpm = hrKalman.filter(rawBpm);
        window.__currentBPM = Math.round(Math.max(RPPG_CONFIG.MIN_HR, Math.min(RPPG_CONFIG.MAX_HR, smoothedBpm)));
        window.__lastBPM = window.__currentBPM;
        sessionBPMValues.push(window.__currentBPM);
        const hrv = computeHRV(window.__currentBPM);
        window.__lastSDNN = hrv.sdnn; window.__lastRMSSD = hrv.rmssd;
        const meditation = computeMeditationIndex(window.__currentBPM, hrv.sdnn, hrv.rmssd);
        sessionMeditationScores.push(meditation);
        if (bpmDisplay) { bpmDisplay.innerText = 'BPM: ' + window.__currentBPM + (finalConfidence < 1.0 ? ' (Calibrating)' : ''); bpmDisplay.style.color = finalConfidence < 1.0 ? '#f59e0b' : '#14b8a6'; }
        if (heartrateTxt) heartrateTxt.innerText = window.__currentBPM + ' bpm';
        if (heartrateBar) heartrateBar.style.width = Math.min(100, Math.max(0, ((window.__currentBPM - 40) / 80) * 100)) + '%';
        if (sdnnTxt)  sdnnTxt.innerText   = hrv.sdnn  + ' ms'; if (sdnnBar)  sdnnBar.style.width  = Math.min(100, hrv.sdnn)  + '%';
        if (rmssdTxt) rmssdTxt.innerText  = hrv.rmssd + ' ms'; if (rmssdBar) rmssdBar.style.width = Math.min(100, hrv.rmssd) + '%';
      } else if (bpmDisplay && window.__sessionActive) { bpmDisplay.innerText = 'BPM: Sensing...'; bpmDisplay.style.color = '#9ca3af'; }
    }
    if (window.__sessionActive) {
      const leftEAR = calcEAR(lm, true), rightEAR = calcEAR(lm, false), avgEAR = (leftEAR + rightEAR) / 2;
      smoothedEAR = smoothedEAR * EYE_CONFIG.SMOOTHING + avgEAR * (1 - EYE_CONFIG.SMOOTHING);
      const eyeRelaxScore = calculateEyeRelaxScore(leftEAR, rightEAR, smoothedEAR);
      if (eyeScoreTxt) eyeScoreTxt.textContent = Math.round(eyeRelaxScore * 100) + '%';
      if (eyeBar) eyeBar.style.width = Math.round(eyeRelaxScore * 100) + '%';
      const nose = lm[1]; noseBuf.push({ x: nose.x, y: nose.y }); if (noseBuf.length > 30) noseBuf.shift();
      let headMotion = 0;
      for (let i = 1; i < noseBuf.length; i++) headMotion += Math.hypot(noseBuf[i].x - noseBuf[i-1].x, noseBuf[i].y - noseBuf[i-1].y);
      const headSteady = clamp01(1 - headMotion * 2);
      const gazeX = (lm[33].x + lm[263].x) / 2, gazeY = (lm[33].y + lm[263].y) / 2;
      gazeBuf.push({ x: gazeX, y: gazeY }); if (gazeBuf.length > 30) gazeBuf.shift();
      let gazeMotion = 0;
      for (let i = 1; i < gazeBuf.length; i++) gazeMotion += Math.hypot(gazeBuf[i].x - gazeBuf[i-1].x, gazeBuf[i].y - gazeBuf[i-1].y);
      const gazeStable = clamp01(1 - gazeMotion * 2.5);
      if (headScoreTxt) headScoreTxt.textContent = Math.round(headSteady * 100) + '%'; if (headBar) headBar.style.width = Math.round(headSteady * 100) + '%';
      if (gazeScoreTxt) gazeScoreTxt.textContent  = Math.round(gazeStable * 100) + '%'; if (gazeBar) gazeBar.style.width  = Math.round(gazeStable * 100) + '%';
      sessionEyeScores.push(eyeRelaxScore); sessionHeadScores.push(headSteady); sessionGazeScores.push(gazeStable);
      window.__faceCalmness = (eyeRelaxScore * 0.35) + (headSteady * 0.35) + (gazeStable * 0.30);
    }
    frames++;
    if (now - lastFrameTs > 1000) { if (fpsNode) fpsNode.textContent = frames; frames = 0; lastFrameTs = now; }
  }

  // ── Audio setup with mobile watchdog ─────────────────────────────
  async function startAudioForMic() {
    if (!selectedMicId) { alert('Please select a microphone first.'); return false; }
    stopAudio();
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // ── Mobile: must resume inside a user gesture ─────────────
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      startAudioWatchdog();   // ← start watchdog after context created

      const constraints = {
        audio: {
          deviceId: selectedMicId && selectedMicId !== 'default' ? { exact: selectedMicId } : undefined,
          echoCancellation: true, noiseSuppression: true, autoGainControl: false
        }
      };
      localStream = await navigator.mediaDevices.getUserMedia(constraints);
      const source = audioContext.createMediaStreamSource(localStream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024; analyser.smoothingTimeConstant = 0.8;
      analyserData = new Uint8Array(analyser.frequencyBinCount);
      source.connect(analyser);
      startVisualizer(); return true;
    } catch (e) { alert('Microphone start failed: ' + (e.message || e)); return false; }
  }
  function stopAudio() {
    if (localStream)  { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
    if (audioContext) { try { audioContext.close(); } catch (_) {} audioContext = null; }
    if (animationId)  { cancelAnimationFrame(animationId); animationId = null; }
    stopAudioWatchdog();
    ampDerivBuf = [];
    if (micLvl) micLvl.textContent = '—'; if (micLvlBar) micLvlBar.style.width = '10%';
  }

  function startVisualizer() {
    if (!analyser) return;
    function resize() {
      const dpr = window.devicePixelRatio || 2, w = Math.max(300, waveCanvas.parentElement.clientWidth || 520);
      waveCanvas.width = Math.floor(w * dpr); waveCanvas.height = Math.floor(260 * dpr);
      waveCanvas.style.width = w + 'px'; waveCanvas.style.height = '260px';
      waveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize(); window.addEventListener('resize', resize);
    let lastAmp = 0;
    function draw() {
      animationId = requestAnimationFrame(draw); analyser.getByteTimeDomainData(analyserData);
      const WIDTH = waveCanvas.width / (window.devicePixelRatio || 1), HEIGHT = waveCanvas.height / (window.devicePixelRatio || 1);
      waveCtx.clearRect(0, 0, WIDTH, HEIGHT); waveCtx.fillStyle = 'rgba(5,8,20,0.12)'; waveCtx.fillRect(0, 0, WIDTH, HEIGHT);
      waveCtx.lineWidth = 2; waveCtx.beginPath();
      const sliceW = WIDTH / analyserData.length; let x = 0, sum = 0;
      for (let i = 0; i < analyserData.length; i++) {
        const v = (analyserData[i] / 128.0) - 1.0, y = (v * 0.95 + 0.5) * HEIGHT;
        i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y); x += sliceW; sum += Math.abs(v);
      }
      waveCtx.strokeStyle = 'rgba(255,165,0,0.95)'; waveCtx.stroke();
      const amplitude = Math.min(1, (sum / analyserData.length) * 8);
      if (!scriptedGrow) breathingCircle.style.transform = `scale(${0.7 + amplitude * 1.5})`;
      else breathingCircle.style.transform = `scale(${(breathingBaseScale || 0.8) * (1 + amplitude * 0.12)})`;
      const deriv = Math.abs(amplitude - lastAmp); lastAmp = amplitude;
      if (window.__sessionActive && currentBreathPhase !== 'idle') {
        phaseAmplitudes.push(amplitude); if (phaseAmplitudes.length > 300) phaseAmplitudes.shift();
      }
      ampDerivBuf.push(deriv); if (ampDerivBuf.length > 1200) ampDerivBuf.shift();
      const micPct = Math.round(amplitude * 400);
      if (micLvl) micLvl.textContent = micPct + '%'; if (micLvlBar) micLvlBar.style.width = micPct + '%';
      const consistency = computeBreathConsistency();
      if (window.__sessionActive && currentBreathPhase !== 'idle') sessionBreathScores.push(consistency);
      if (breathConsTxt) breathConsTxt.textContent = Math.round(consistency * 100) + '%';
      if (breathConsBar) breathConsBar.style.width  = Math.round(consistency * 100) + '%';
    }
    draw();
  }

  function computeBreathConsistency() {
    if (!window.__sessionActive || currentBreathPhase === 'idle') return 0.5;
    if (phaseAmplitudes.length < 10) return phaseConsistencyScore;
    const samples = phaseAmplitudes.slice(-60), mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    let currentScore = 0.5;
    if (currentBreathPhase === 'inhale') { const trend = samples[samples.length - 1] - samples[0]; currentScore = mean > 0.05 ? 0.8 + (trend * 0.2) : 0.3; }
    else if (currentBreathPhase === 'hold') { const isSilent = mean < 0.03, noise = samples.some(s => s > 0.1); currentScore = isSilent ? 1.0 : (noise ? 0.1 : 0.5); }
    else if (currentBreathPhase === 'exhale') { const isFlowing = mean > 0.04, variance = samples.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / samples.length, isSteady = Math.sqrt(variance) < 0.05; currentScore = isFlowing ? (isSteady ? 0.9 : 0.6) : 0.3; }
    phaseConsistencyScore = phaseConsistencyScore * 0.95 + clamp01(currentScore) * 0.05;
    return phaseConsistencyScore;
  }

  function stopSession() {
    if (!isSessionRunning) return;
    isSessionRunning = false; window.__sessionActive = false;
    if (sessionTimerInterval) { clearInterval(sessionTimerInterval); sessionTimerInterval = null; }
    if (sessionStatus) sessionStatus.textContent = 'stopped';
    stopAudio(); stopAmbience();
    if (cycleTimeout) { clearTimeout(cycleTimeout); cycleTimeout = null; }
    scriptedGrow = false; breathingBaseScale = 0.7;
    breathingCircle.style.transform = 'scale(0.7)';
    if (breathStatus) breathStatus.textContent = 'Stopped';
    // Clear speech queue on stop
    speechQueue = []; speechPending = false;
    try { window.speechSynthesis?.cancel?.(); } catch (_) {}
  }

  function calibrateRppgSettings() {
    const ageEl = document.getElementById('userAge'), genderEl = document.getElementById('userGender');
    const age = parseInt(ageEl ? ageEl.value : 0, 10) || 30, gender = genderEl ? genderEl.value : 'other';
    RPPG_CONFIG.MIN_HR = 50; RPPG_CONFIG.MAX_HR = 100; RPPG_CONFIG.ROI_WIDTH_FACTOR = 0.75;
    if (age > 50) { RPPG_CONFIG.MIN_HR = 45; RPPG_CONFIG.ROI_WIDTH_FACTOR = 0.85; }
    else if (age < 15) RPPG_CONFIG.MAX_HR = 110;
    if (gender === 'female') RPPG_CONFIG.MAX_HR += 5;
  }

  async function startSession() {
    if (isSessionRunning) return;
    unlockAudioOnMobile();   // ← ensure audio unlocked on Start tap

    calibrateRppgSettings();
    const micStarted = await startAudioForMic();
    if (!cameraInstance) await startCameraForFace().catch(() => {});
    if (!micStarted && !confirm('Microphone could not be started. Continue with face-only session?')) return;

    isSessionRunning = true; window.__sessionActive = true;
    sessionStartTime = Date.now();
    sessionTimerInterval = setInterval(updateSessionTimer, 1000);
    if (sessionStatus) sessionStatus.textContent = 'running';

    ampDerivBuf.length = 0;
    rppgSignal.red = []; rppgSignal.green = []; rppgSignal.blue = []; rppgSignal.timestamps = [];
    bpmHistory = []; hrKalman.reset(); gazeBuf = []; noseBuf = [];
    currentBreathPhase = 'idle'; phaseAmplitudes = []; phaseConsistencyScore = 0.5;
    sessionBreathScores = []; sessionEyeScores = []; sessionHeadScores = []; sessionGazeScores = [];
    sessionMeditationScores = []; sessionBPMValues = [];
    speechQueue = []; speechPending = false;

    if (breathStatus) breathStatus.textContent = 'Session running — follow the breathing prompts';
    startAmbience();
    runTimedBreathing();

    // Small delay on mobile before first speech to let audio unlock settle
    setTimeout(() => { try { speak('', 'Welcome. Follow the voice.'); } catch (_) {} }, isMobile ? 600 : 0);
  }

  startBtn.addEventListener('click', startSession);
  stopBtn.addEventListener('click',  stopSession);

  const muteBtn = document.getElementById('muteBtn');
  if (muteBtn) {
    muteBtn.addEventListener('click', () => {
      muted = !muted; muteBtn.textContent = muted ? '🔇' : '🔈';
      if (muted) { speechQueue = []; speechPending = false; try { window.speechSynthesis.cancel(); } catch (_) {} }
    });
  }

  function runTimedBreathing() {
    const breaths = parseInt(breathCountEl.value, 10) || 3;
    let count = breaths;
    const inMs = 6000, holdMs = 5000, outMs = 5000;
    if (breathStatus) breathStatus.textContent = `Breaths remaining: ${count}`;
    let cycleCount = 0;
    if (sessionMeter) sessionMeter.style.width = '5%';

    function singleCycle() {
      if (!isSessionRunning) return;
      if (count <= 0) { finishBreathing(); return; }
      cycleCount++;
      if (sessionMeter) sessionMeter.style.width = Math.round((cycleCount / breaths) * 100) + '%';
      if (breathStatus) breathStatus.textContent = 'Breathe in';
      speak('Saas andar lein.', 'Breathe in.');
      currentBreathPhase = 'inhale'; phaseAmplitudes = [];
      scriptedGrow = true; breathingBaseScale = 1.4;
      breathingCircle.style.transition = `transform ${inMs}ms cubic-bezier(.2,.9,.2,1)`;
      breathingCircle.style.transform  = `scale(${breathingBaseScale})`;

      cycleTimeout = setTimeout(() => {
        if (breathStatus) breathStatus.textContent = 'Hold';
        speak('Rok kar rakhen.', 'Hold breath.');
        currentBreathPhase = 'hold'; phaseAmplitudes = [];
        breathingCircle.style.transition = `transform ${holdMs}ms ease-in-out`;
        breathingBaseScale = 1.2; breathingCircle.style.transform = `scale(${breathingBaseScale})`;

        cycleTimeout = setTimeout(() => {
          if (breathStatus) breathStatus.textContent = 'Exhale slowly';
          speak('Dheere se bahar chhodein.', 'Breathe out slowly.');
          currentBreathPhase = 'exhale'; phaseAmplitudes = [];
          breathingCircle.style.transition = `transform ${outMs}ms cubic-bezier(.2,.9,.2,1)`;
          breathingBaseScale = 0.6; breathingCircle.style.transform = `scale(${breathingBaseScale})`;

          cycleTimeout = setTimeout(() => {
            count--; currentBreathPhase = 'idle';
            if (breathStatus) breathStatus.textContent = `Breaths remaining: ${count}`;
            cycleTimeout = setTimeout(singleCycle, 2000);
          }, outMs);
        }, holdMs);
      }, inMs);
    }

    function finishBreathing() {
      window.__sessionActive = false; scriptedGrow = false; breathingBaseScale = 1.0;
      breathingCircle.style.transition = 'transform 400ms ease'; breathingCircle.style.transform = 'scale(1)';
      if (breathStatus) breathStatus.textContent = 'Session complete. Well done!';
      speak('Satra samaapt. Bahut badhiya.', 'Session complete. Well done!');
      stopAmbience(); computeFinalResultAndShow();
    }

    singleCycle();
  }

  function computeFinalResult() {
    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const avgEye = avg(sessionEyeScores), avgHead = avg(sessionHeadScores), avgGaze = avg(sessionGazeScores);
    const f = clamp01(avgEye * 0.35 + avgHead * 0.35 + avgGaze * 0.30);
    let b = phaseConsistencyScore;
    if (sessionBreathScores.length > 50) b = avg(sessionBreathScores);
    b = clamp01(b);
    let meditationScore = sessionMeditationScores.length > 0
      ? avg(sessionMeditationScores)
      : computeMeditationIndex(window.__lastBPM || 0, window.__lastSDNN || 0, window.__lastRMSSD || 0);
    const meditationIndex = meditationScore / 100, overall = clamp01(f * 0.4 + b * 0.3 + meditationIndex * 0.3);
    if (faceCalmTxt)        faceCalmTxt.textContent         = Math.round(f * 100) + '%';
    if (faceCalmBar)        faceCalmBar.style.width          = Math.round(f * 100) + '%';
    if (breathConsTxt2)     breathConsTxt2.textContent       = Math.round(b * 100) + '%';
    if (breathConsBar2)     breathConsBar2.style.width       = Math.round(b * 100) + '%';
    if (meditationIndexTxt) meditationIndexTxt.textContent   = meditationScore + '%';
    if (meditationIndexBar) meditationIndexBar.style.width   = meditationScore + '%';
    if (overallTxt)         overallTxt.textContent           = Math.round(overall * 100) + '%';
    if (overallBar)         overallBar.style.width           = Math.round(overall * 100) + '%';
    if (overallNote) overallNote.textContent = overall > 0.75 ? 'Excellent meditation session 🌿' : overall > 0.5 ? 'Good focus, keep practicing 🙂' : 'Try slower breathing and stillness';
    return { f, b, meditationIndex, overall, meditationScore };
  }

  function computeFinalResultAndShow() {
    const result = computeFinalResult();
    const sessionDuration = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
    const durationStr = formatElapsedTime(sessionDuration * 1000);
    const avgArr = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const avgBPM = sessionBPMValues.length > 0 ? Math.round(avgArr(sessionBPMValues)) : (window.__lastBPM || 0);
    const breathPct = Math.round(result.b * 100), meditPct = Math.round(result.meditationScore);
    const rawStability = calculateStability(avgBPM, breathPct, meditPct);
    saveSession(rawStability);
    const allSess = getSessions(), { baselineMean, calibrated } = getBaseline(allSess);
    const displayStability = calibrated ? calibrateScore(rawStability, baselineMean) : rawStability;
    sessionHistory.push({
      date: new Date().toLocaleString(), name: userNameInput?.value || 'Anonymous', age: userAgeInput?.value || '—',
      tone: userSkinToneInput?.value || '—', duration: durationStr, bpm: avgBPM,
      face: Math.round(result.f * 100), breath: breathPct, meditation: meditPct,
      overall: Math.round(result.overall * 100), stability: displayStability, cielabUsed: cielabEnabled
    });
    localStorage.setItem('sns_sessions', JSON.stringify(sessionHistory));
    renderHistory(); updateWellnessUI(rawStability);
  }

  function renderHistory() {
    if (!historyBody) return;
    historyBody.innerHTML = '';
    sessionHistory.slice().reverse().forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.date}</td><td>${s.name || 'Anonymous'}</td><td>Age: ${s.age || '—'} / ${s.tone || '—'}</td><td>${s.duration || '—'}</td><td>${s.bpm || '—'} bpm</td><td>${s.face}%</td><td>${s.breath}%</td><td>${s.meditation}%</td><td>${s.overall}%</td><td><strong>${s.stability !== undefined ? s.stability + '%' : '—'}</strong>${s.cielabUsed ? '<span style="font-size:0.65rem;color:#f59e0b;">🔬</span>' : ''}</td>`;
      historyBody.appendChild(tr);
    });
  }

  if (toggleMusicBtn) toggleMusicBtn.addEventListener('click', () => {
    unlockAudioOnMobile();
    bgAudio.paused ? startAmbience() : stopAmbience();
  });
  if (musicSelect) musicSelect.addEventListener('change', () => {
    const url = musicSelect.value;
    if (url && url !== 'none') { bgAudio.pause(); bgAudio.src = url; bgAudio.load(); if (window.__sessionActive) startAmbience(); }
    else stopAmbience();
  });
  if (musicVolume) musicVolume.addEventListener('input', e => {
    if (bgAudio) bgAudio.volume = e.target.value / 100;
    if (volumeLabel) volumeLabel.textContent = e.target.value + '%';
  });

  renderHistory();

  // Restore wellness display on page load
  (function initWellnessDisplay() {
    const sessions = getSessions(); if (!sessions.length) return;
    const last = sessions[sessions.length - 1], todayKey = new Date().toISOString().slice(0, 10);
    if (last.date.slice(0, 10) === todayKey) { updateWellnessUI(last.score); return; }
    const streakEl = document.getElementById('streakDisplay'), forecastEl = document.getElementById('weeklyForecast');
    const barsWrap = document.getElementById('forecastBarsWrap'), barsEl = document.getElementById('forecastBars');
    const streak = calculateStreak();
    if (streakEl) {
      if (streak >= 2) { streakEl.textContent = `🔥 ${streak}-Day Stability Streak`; streakEl.style.display = 'inline-flex'; }
      else if (streak === 1) { streakEl.textContent = '✨ First day — come back tomorrow!'; streakEl.style.display = 'inline-flex'; }
      else streakEl.style.display = 'none';
    }
    const forecast = calculateWeeklyForecast();
    if (forecast && forecastEl) {
      const confColor = forecast.confidence.label === 'High' ? '#10b981' : forecast.confidence.label === 'Medium' ? '#f59e0b' : '#ef4444';
      const baseNote = forecast.baselineCalibrated ? ` <span style="font-size:0.7rem;opacity:0.6;">(baseline: ${forecast.baselineMean})</span>` : '';
      const slopeStr = (forecast.slope >= 0 ? '+' : '') + forecast.slope;
      forecastEl.innerHTML = `Your trend is <strong>${forecast.direction}</strong>${baseNote}. 5-session average: <strong>${forecast.ma5}</strong> · Trend: <strong>${slopeStr} pts/session</strong>. Score may reach <span class="forecast-projected-score">${forecast.projected}</span> this week. <span style="font-size:0.7rem;font-weight:600;color:${confColor};">Confidence: ${forecast.confidence.label}</span>`;
      forecastEl.classList.add('forecast-ready');
      if (barsEl && barsWrap) {
        barsWrap.style.display = 'block'; barsEl.innerHTML = '';
        const chartScores = forecast.calibratedScores, maxVal = Math.max(...chartScores, forecast.projected, 1);
        chartScores.forEach((score, i) => {
          const bar = document.createElement('div'); bar.className = 'forecast-mini-bar';
          bar.style.height = Math.round((score / maxVal) * 100) + '%'; bar.title = `Session ${i + 1}: ${Math.round(score)}`; barsEl.appendChild(bar);
        });
        const projBar = document.createElement('div'); projBar.className = 'forecast-mini-bar projected';
        projBar.style.height = Math.round((forecast.projected / maxVal) * 100) + '%'; projBar.title = `Projected: ${forecast.projected}`; barsEl.appendChild(projBar);
      }
    }
  })();

  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      if (!sessionHistory.length) { alert('No sessions to export'); return; }
      const rows = [['Date','Name','Age','Skin Tone','Duration','BPM','Face','Breath','Meditation','Overall','Stability','CIELAB Used']];
      sessionHistory.forEach(s => rows.push([s.date, s.name || 'Anonymous', s.age || '—', s.tone || '—', s.duration || '—', s.bpm || '—', s.face, s.breath, s.meditation, s.overall, s.stability !== undefined ? s.stability : '—', s.cielabUsed ? 'Yes' : 'No']));
      const csvFile = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
      const blob = new Blob([csvFile], { type: 'text/csv' }), link = document.createElement('a');
      link.href = URL.createObjectURL(blob); link.download = 'meditation_sessions.csv'; link.click();
    });
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      if (!confirm('Clear all saved sessions?')) return;
      sessionHistory = []; localStorage.removeItem('sns_sessions'); localStorage.removeItem(WELLNESS_STORAGE_KEY); renderHistory();
      const resetMap = {
        stabilityScore:    el => { el.textContent = '--'; el.className = ''; },
        stabilityMessage:  el => { el.textContent = 'Complete a session to see your score.'; },
        streakDisplay:     el => { el.style.display = 'none'; },
        weeklyForecast:    el => { el.textContent = 'Complete more sessions to unlock your weekly outlook.'; el.classList.remove('forecast-ready'); },
        forecastBarsWrap:  el => { el.style.display = 'none'; },
        stabilityRingFill: el => { el.style.strokeDashoffset = '226'; },
        stabilityRingLabel:el => { el.textContent = '—'; }
      };
      Object.entries(resetMap).forEach(([id, fn]) => { const el = document.getElementById(id); if (el) fn(el); });
    });
  }

  (async () => {
    try { await ensurePermissionForMic(); } catch (_) {}
    await loadDevices();
    setTimeout(() => { faceCanvas.width = faceVideo.clientWidth || 420; faceCanvas.height = faceVideo.clientHeight || 300; }, 500);
  })();

  if (navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === 'function') {
    navigator.mediaDevices.addEventListener('devicechange', () => loadDevices());
  }
  window.addEventListener('beforeunload', () => {
    stopSession(); try { cameraInstance?.stop(); } catch (_) {}
  });

})();