import { useLocation, useNavigate } from "react-router-dom";
import ResultsView from "../components/ResultsView";
import type { WorkoutResults } from "../types";

export default function Results() {
  const navigate = useNavigate();
  const location = useLocation();

  const results = location.state as WorkoutResults;

  return <ResultsView results={results} onReset={() => navigate("/")} />;
}
