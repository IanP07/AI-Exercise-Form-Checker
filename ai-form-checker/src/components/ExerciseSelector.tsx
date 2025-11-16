import { useState } from "react";
import type { Exercise } from "../App";
import "./ExerciseSelector.css";

interface ExerciseSelectorProps {
  onStart: (exercise: Exercise) => void;
}

const exercises: { id: Exercise; name: string; icon: string }[] = [
  { id: "Push-ups", name: "Push-ups", icon: "" },
  { id: "Jumping Jacks", name: "Jumping Jacks", icon: "" },
  { id: "Squats", name: "Squats", icon: "" },
  { id: "Lunges", name: "Lunges", icon: "" },
];

export default function ExerciseSelector({ onStart }: ExerciseSelectorProps) {
  const [selected, setSelected] = useState<Exercise | null>(null);

  const handleStart = () => {
    if (selected) {
      onStart(selected);
    }
  };

  return (
    <div className="exercise-selector">
      <h2>Select Your Exercise</h2>
      <div className="exercise-grid">
        {exercises.map((exercise) => (
          <button
            key={exercise.id}
            className={`exercise-card ${
              selected === exercise.id ? "selected" : ""
            }`}
            onClick={() => setSelected(exercise.id)}
          >
            <span className="exercise-icon">{exercise.icon}</span>
            <span className="exercise-name">{exercise.name}</span>
          </button>
        ))}
      </div>
      <button
        className="start-button"
        onClick={handleStart}
        disabled={!selected}
      >
        Start Analysis
      </button>
    </div>
  );
}
