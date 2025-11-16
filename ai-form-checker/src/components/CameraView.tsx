import { useEffect, useRef, useState } from "react";
import type { Exercise, WorkoutResults, RepResult } from "../App";
import "./CameraView.css";
import { io, Socket } from "socket.io-client";

interface CameraViewProps {
  exercise: Exercise;
  onStop: (results: WorkoutResults) => void;
}

export default function CameraView({ exercise, onStop }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);

  const [repCount, setRepCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(100);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [useDemoMode, setUseDemoMode] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [mirrorVideo, setMirrorVideo] = useState(false);

  // NEW: which camera to use ("user" = front, "environment" = back)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");

  const repDataRef = useRef<RepResult[]>([]);

  // ---------------------- DEMO MODE ----------------------
  useEffect(() => {
    if (useDemoMode) startDemoMode();
  }, [useDemoMode]);

  // ---------------------- CAMERA INITIALIZATION ----------------------
  useEffect(() => {
    if (!cameraEnabled) return;

    let mounted = true;

    const initializeCamera = async () => {
      if (!videoRef.current) return;

      try {
        setIsProcessing(true);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        setIsProcessing(false);
        setCameraError(null);

        startCanvasTracking();
      } catch (error: any) {
        console.error("Camera error:", error);
        setCameraError(error.message || "Camera access error");
        setUseDemoMode(true);
        setIsProcessing(false);
      }
    };

    initializeCamera();

    return () => {
      mounted = false;
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
    };
  }, [cameraEnabled, facingMode]); // facingMode triggers re-initialization

  // ---------------------- SOCKET.IO SETUP ----------------------
  useEffect(() => {
    if (!cameraEnabled) return;

    const socket = io(
      "https://shameka-unbridgeable-noncausally.ngrok-free.dev/",
      {
        transports: ["websocket"],
      }
    );

    socketRef.current = socket;

    socket.on("connect", () => console.log("Socket.IO connected"));
    socket.on("disconnect", () => console.log("Socket.IO disconnected"));

    socket.on("analysis", (data: any) => {
      if (data.repCount !== undefined) setRepCount(data.repCount);
      if (data.score !== undefined) setCurrentScore(data.score);
    });

    return () => {
      socket.disconnect();
    };
  }, [cameraEnabled]);

  // ---------------------- SEND FRAMES ----------------------
  const startCanvasTracking = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sendFPS = 10;
    const interval = 1000 / sendFPS;
    let lastSent = 0;

    const loop = (timestamp: number) => {
      if (!ctx || !videoRef.current) return;

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      // Mirror ONLY front camera
      ctx.save();

      if (mirrorVideo) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }

      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      if (timestamp - lastSent > interval) {
        lastSent = timestamp;

        const base64 = canvas.toDataURL("image/jpeg", 0.7);

        if (socketRef.current) {
          socketRef.current.emit("frame", { image: base64 });
        }
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  };

  // ---------------------- DEMO MODE ----------------------
  const startDemoMode = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (canvas.width === 0) {
      canvas.width = 640;
      canvas.height = 480;
    }

    const draw = () => {
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      requestAnimationFrame(draw);
    };

    draw();
  };

  // ---------------------- STOP ----------------------
  const handleStop = () => {
    const reps = repDataRef.current;
    const avgScore =
      reps.length > 0
        ? Math.round(reps.reduce((sum, r) => sum + r.score, 0) / reps.length)
        : 0;

    onStop({
      exercise,
      reps,
      overallScore: avgScore,
      overallNotes: [],
    });
  };

  // ---------------------- FLIP CAMERA BUTTON ----------------------
  const flipCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  // mirrors back camera
  useEffect(() => {
    setMirrorVideo(facingMode === "user");
  }, [facingMode]);

  return (
    <div className="camera-view">
      <div className="camera-container">
        <video
          ref={videoRef}
          className="video-element"
          autoPlay
          playsInline
          muted
          style={{
            transform: mirrorVideo ? "scaleX(-1)" : "scaleX(1)",
          }}
        />
        <canvas ref={canvasRef} className="pose-canvas" />

        {/* Enable camera button */}
        {useDemoMode && !cameraError && !isProcessing && (
          <div className="camera-toggle">
            <button
              className="enable-camera-button"
              onClick={() => {
                setCameraEnabled(true);
                setUseDemoMode(false);
              }}
            >
              Enable Camera
            </button>
          </div>
        )}

        {/* 🔄 FLIP CAMERA BUTTON */}
        {cameraEnabled && (
          <button
            className="flip-button"
            onClick={flipCamera}
            style={{
              position: "absolute",
              top: 10,
              right: 10,
              padding: "10px 14px",
              borderRadius: 8,
              background: "rgba(0,0,0,0.6)",
              color: "white",
              fontSize: 14,
            }}
          >
            Flip Camera
          </button>
        )}
      </div>

      {/* Stats */}
      <div className="stats-panel">
        <div className="stat-item">
          <span className="stat-label">Reps Completed</span>
          <span className="stat-value">{repCount}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Current Score</span>
          <span className="stat-value">{currentScore}</span>
        </div>
      </div>

      <button className="stop-button" onClick={handleStop}>
        Stop Analysis
      </button>
    </div>
  );
}
