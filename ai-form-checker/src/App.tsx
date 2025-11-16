import { useState } from "react";
import ExerciseSelector from "./components/ExerciseSelector";
import CameraView from "./components/CameraView";
import ResultsView from "./components/ResultsView";
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
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(
    null
  );
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
      <header className="app-header">
        <h1>RepRadar</h1>
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
