import ExerciseSelector from "../components/ExerciseSelector";
import type { Exercise } from "../types";
import { useNavigate } from "react-router-dom";

export default function ExerciseSelection() {
  const navigate = useNavigate();

  const goToAnalyze = (exercise: Exercise) => {
    navigate(`/analyze/${encodeURIComponent(exercise)}`);
  };

  return <ExerciseSelector onStart={goToAnalyze} />;
}
