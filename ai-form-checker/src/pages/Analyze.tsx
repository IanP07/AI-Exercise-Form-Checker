import { useParams, useNavigate } from "react-router-dom";
import CameraView from "../components/CameraView";
import type { Exercise, WorkoutResults } from "../types";

export default function Analyze() {
  const { exercise } = useParams();
  const navigate = useNavigate();

  const goToResults = (results: WorkoutResults) => {
    navigate("/results", { state: results });
  };

  return <CameraView exercise={exercise as Exercise} onStop={goToResults} />;
}
