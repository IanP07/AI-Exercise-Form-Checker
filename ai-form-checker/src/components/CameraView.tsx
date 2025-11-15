import { useEffect, useRef, useState } from "react";
import type { Exercise, WorkoutResults, RepResult } from "../App";
import "./CameraView.css";

interface CameraViewProps {
  exercise: Exercise;
  onStop: (results: WorkoutResults) => void;
}

export default function CameraView({ exercise, onStop }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [repCount, setRepCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(100);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [useDemoMode, setUseDemoMode] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const repDataRef = useRef<RepResult[]>([]);
  const exerciseStateRef = useRef<any>({
    phase: "down",
    cycleCount: 0,
    startTime: Date.now(),
  });

  useEffect(() => {
    let mounted = true;

    const startDemoMode = () => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      canvas.width = 640;
      canvas.height = 480;

      const trackFrame = () => {
        if (!mounted || !canvasRef.current) return;

        const ctx = canvasRef.current.getContext("2d");
        if (ctx) {
          // Draw gradient background for demo mode
          const gradient = ctx.createLinearGradient(
            0,
            0,
            canvas.width,
            canvas.height
          );
          gradient.addColorStop(0, "#1e293b");
          gradient.addColorStop(1, "#0f172a");
          ctx.fillStyle = gradient;
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // Add "Demo Mode" text
          ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
          ctx.font = "bold 24px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("DEMO MODE", canvas.width / 2, 40);

          ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
          ctx.font = "14px sans-serif";
          ctx.fillText("Simulated Exercise Tracking", canvas.width / 2, 65);

          // Simulate pose detection and analysis
          simulatePoseTracking(ctx, canvas.width, canvas.height);
        }

        animationFrameRef.current = requestAnimationFrame(trackFrame);
      };

      trackFrame();
    };

    const startTracking = () => {
      const trackFrame = () => {
        if (!mounted || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (video.readyState === video.HAVE_ENOUGH_DATA && ctx) {
          // Set canvas dimensions to match video
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          // Draw mirrored video
          ctx.save();
          ctx.scale(-1, 1);
          ctx.translate(-canvas.width, 0);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          // Simulate pose detection and analysis
          simulatePoseTracking(ctx, canvas.width, canvas.height);
        }

        animationFrameRef.current = requestAnimationFrame(trackFrame);
      };

      trackFrame();
    };

    const initializeCamera = async () => {
      if (!canvasRef.current) return;

      // Always start demo mode first
      if (useDemoMode) {
        setIsProcessing(false);
        startDemoMode();
        return;
      }

      // Only try camera if explicitly enabled
      if (!videoRef.current) {
        return;
      }

      try {
        setIsProcessing(true);

        // Get camera stream
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (!mounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          // Wait for metadata to load
          await new Promise<void>((resolve, reject) => {
            if (!videoRef.current) {
              reject(new Error("Video ref lost"));
              return;
            }

            videoRef.current.onloadedmetadata = () => resolve();
            videoRef.current.onerror = () =>
              reject(new Error("Video load error"));

            // Timeout after 10 seconds
            setTimeout(() => reject(new Error("Video load timeout")), 10000);
          });

          if (videoRef.current && mounted) {
            await videoRef.current.play();
            setIsProcessing(false);
            setCameraError(null);
            startTracking();
          }
        }
      } catch (error: any) {
        console.error("Error accessing camera:", error);

        if (!mounted) return;

        setIsProcessing(false);

        // Provide helpful error messages
        if (error.name === "NotAllowedError") {
          setCameraError(
            "Camera access denied. Please allow camera access in your browser settings."
          );
        } else if (error.name === "NotFoundError") {
          setCameraError("No camera found on this device.");
        } else if (error.name === "NotReadableError") {
          setCameraError("Camera is already in use by another app.");
        } else {
          setCameraError(`Camera error: ${error.message || "Unknown error"}`);
        }
      }
    };

    initializeCamera();

    return () => {
      mounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [useDemoMode]);

  const simulatePoseTracking = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ) => {
    const state = exerciseStateRef.current;
    const elapsed = Date.now() - state.startTime;

    // Simulate exercise motion with a cycling pattern
    const cycleTime = getExerciseCycleTime(exercise);
    const progress = (elapsed % cycleTime) / cycleTime;

    // Don't draw skeleton - just track reps

    // Simulate rep counting
    const currentPhase = progress < 0.5 ? "down" : "up";

    if (currentPhase === "up" && state.phase === "down") {
      // Rep completed
      const score = generateRepScore();
      const notes = generateRepNotes(score, exercise);
      completeRep(score, notes);
    }

    state.phase = currentPhase;
  };

  const getExerciseCycleTime = (exercise: Exercise): number => {
    switch (exercise) {
      case "pushups":
      case "squats":
        return 3000; // 3 seconds per rep
      case "jumping-jacks":
        return 2000; // 2 seconds per rep
      case "pullups":
        return 4000; // 4 seconds per rep
      case "russian-twists":
        return 2500; // 2.5 seconds per rep
      default:
        return 3000;
    }
  };

  const generateRepScore = (): number => {
    // Generate realistic scores with occasional form issues
    const baseScore = 70 + Math.random() * 30;
    return Math.round(baseScore);
  };

  const generateRepNotes = (score: number, exercise: Exercise): string[] => {
    const notes: string[] = [];

    if (score >= 90) {
      notes.push("Great form!");
      notes.push("Perfect range of motion");
    } else if (score >= 80) {
      notes.push("Good form!");
      const formTips = getFormTips(exercise);
      if (Math.random() > 0.5 && formTips.length > 0) {
        notes.push(formTips[Math.floor(Math.random() * formTips.length)]);
      }
    } else if (score >= 70) {
      notes.push("Fair form");
      const formTips = getFormTips(exercise);
      notes.push(formTips[Math.floor(Math.random() * formTips.length)]);
    } else {
      notes.push("Needs improvement");
      const formTips = getFormTips(exercise);
      notes.push(formTips[Math.floor(Math.random() * formTips.length)]);
      if (formTips.length > 1) {
        notes.push(
          formTips[
            (Math.floor(Math.random() * formTips.length) + 1) % formTips.length
          ]
        );
      }
    }

    return notes;
  };

  const getFormTips = (exercise: Exercise): string[] => {
    const tips: Record<Exercise, string[]> = {
      pushups: [
        "Keep your back straight",
        "Lower chest to ground",
        "Elbows at 45 degrees",
        "Core engaged throughout",
        "Full range of motion",
      ],
      squats: [
        "Knees behind toes",
        "Chest up, back straight",
        "Go deeper - aim for 90°",
        "Weight on heels",
        "Keep knees aligned",
      ],
      "jumping-jacks": [
        "Arms fully extended overhead",
        "Land softly on toes",
        "Keep core tight",
        "Maintain rhythm",
      ],
      pullups: [
        "Pull chin over bar",
        "Full extension at bottom",
        "Control the descent",
        "Engage your lats",
      ],
      "russian-twists": [
        "Rotate from core",
        "Keep back straight",
        "Feet off ground",
        "Touch floor each side",
      ],
    };

    return tips[exercise] || ["Keep practicing!"];
  };

  const completeRep = (score: number, notes: string[]) => {
    setRepCount((prevCount) => {
      const newRepCount = prevCount + 1;

      const repResult: RepResult = {
        repNumber: newRepCount,
        score: Math.max(0, score),
        notes: notes,
      };

      repDataRef.current.push(repResult);

      return newRepCount;
    });
  };

  const handleStop = () => {
    const reps = repDataRef.current;
    const avgScore =
      reps.length > 0
        ? Math.round(
            reps.reduce((sum, rep) => sum + rep.score, 0) / reps.length
          )
        : 0;

    const overallNotes: string[] = [];
    if (reps.length === 0) {
      overallNotes.push("No reps detected - try performing the exercise");
    } else if (avgScore >= 90) {
      overallNotes.push("Excellent form throughout!");
      overallNotes.push(`Completed ${reps.length} reps with great technique`);
    } else if (avgScore >= 75) {
      overallNotes.push("Good form, minor improvements needed");
      overallNotes.push("Focus on consistency across all reps");
    } else if (avgScore >= 60) {
      overallNotes.push("Fair form, focus on consistency");
      overallNotes.push("Review the form tips for each rep");
    } else {
      overallNotes.push("Keep practicing to improve form");
      overallNotes.push("Consider reviewing proper technique");
    }

    const results: WorkoutResults = {
      exercise,
      reps,
      overallScore: avgScore,
      overallNotes,
    };

    onStop(results);
  };

  const handleEnableCamera = async () => {
    setIsProcessing(true);
    setCameraError(null);
    setCameraEnabled(true);
    setUseDemoMode(false);
  };

  return (
    <div className="camera-view">
      <div className="camera-container">
        <video
          ref={videoRef}
          className="video-element"
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="pose-canvas" />

        {isProcessing && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Initializing camera...</p>
          </div>
        )}

        {cameraError && (
          <div className="error-overlay">
            <div className="error-icon">!</div>
            <p className="error-message">{cameraError}</p>
            <button
              className="demo-button"
              onClick={() => {
                setCameraError(null);
                setUseDemoMode(true);
              }}
            >
              Use Demo Mode
            </button>
          </div>
        )}

        {useDemoMode && !cameraError && !isProcessing && (
          <div className="camera-toggle">
            <button
              className="enable-camera-button"
              onClick={handleEnableCamera}
            >
              Enable Camera
            </button>
          </div>
        )}
      </div>

      <div className="stats-panel">
        <div className="stat-item">
          <span className="stat-label">Reps Completed</span>
          <span className="stat-value">{repCount}</span>
        </div>
      </div>

      <button className="stop-button" onClick={handleStop}>
        Stop Analysis
      </button>
    </div>
  );
}
