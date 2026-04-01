import { useEffect, useRef, useState } from 'react';

// Make sure these are globally available by Vite/index.html
// window.FaceMesh and window.Camera
export function useMediaPipe(videoRef, canvasRef, onFaceResults) {
  const [isReady, setIsReady] = useState(false);
  const faceMeshRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    // Wait until global FaceMesh is loaded
    if (typeof window.FaceMesh === 'undefined') {
      console.warn('FaceMesh global not found. Ensure script is loaded in index.html.');
      return;
    }

    try {
      const faceMesh = new window.FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });
      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
      });
      faceMesh.onResults(onFaceResults);
      faceMeshRef.current = faceMesh;

      setIsReady(true);
    } catch (e) {
      console.error('Failed to initialize FaceMesh:', e);
    }

    return () => {
      if (faceMeshRef.current) {
        try {
          // faceMeshRef.current.close() is typically not cleanly synchronous in MP, but we try.
          faceMeshRef.current.close();
        } catch (e) {}
      }
    };
  }, [videoRef, canvasRef]); // Usually only run once when refs are attached

  const startCamera = async () => {
    if (!isReady || !videoRef.current || !faceMeshRef.current) return false;
    
    // Size normalization is handled by CSS, but internal constraints:
    const videoEl = videoRef.current;
    if (canvasRef.current) {
      canvasRef.current.width = videoEl.clientWidth || 420;
      canvasRef.current.height = videoEl.clientHeight || 300;
    }

    try {
      const camera = new window.Camera(videoEl, {
        onFrame: async () => {
          await faceMeshRef.current.send({ image: videoEl });
        },
        width: 320,
        height: 240
      });
      camera.start();
      cameraRef.current = camera;
      return true;
    } catch (e) {
      console.error('startCamera error', e);
      return false;
    }
  };

  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  return { isReady, startCamera, stopCamera };
}
