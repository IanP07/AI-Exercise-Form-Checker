import { useState } from "react";
import ExerciseSelector from "./components/ExerciseSelector";
import CameraView from "./components/CameraView";
import ResultsView from "./components/ResultsView";
import Particles from "./components/Particles";
import "./App.css";

export type Exercise = "Push-ups" | "Jumping Jacks" | "Squats" | "Lunges";

export interface RepResult {
  repNumber: number;
  score: number;
  notes: string[];
}

export interface WorkoutResults {
  exercise: Exercise;
  reps: RepResult[];
  overallScore: number;
  overallNotes: string[];
}

type AppState = "selection" | "analyzing" | "results";

export default function App() {
  const [state, setState] = useState<AppState>("selection");
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
  const [results, setResults] = useState<WorkoutResults | null>(null);

  const handleStartAnalysis = (exercise: Exercise) => {
    setSelectedExercise(exercise);
    setState("analyzing");
  };

  const handleStopAnalysis = (workoutResults: WorkoutResults) => {
    setResults(workoutResults);
    setState("results");
  };

  const handleReset = () => {
    setState("selection");
    setSelectedExercise(null);
    setResults(null);
  };

  return (
    <div className="app">
      <div className="particles-wrapper">
        <Particles
          particleColors={["#ffffff", "#7df9ff", "#4cc9f0"]}
          particleCount={1000}
          particleSpread={10}
          speed={0.15}
          particleBaseSize={130}
          moveParticlesOnHover={true}
          alphaParticles={true}
          disableRotation={false}
        />
      </div>

      <header className="app-header">
        <h1>AI Workout Form Tracker</h1>
      </header>

      <main className="app-main">
        {state === "selection" && (
          <ExerciseSelector onStart={handleStartAnalysis} />
        )}

        {state === "analyzing" && selectedExercise && (
          <CameraView exercise={selectedExercise} onStop={handleStopAnalysis} />
        )}

        {state === "results" && results && (
          <ResultsView results={results} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
