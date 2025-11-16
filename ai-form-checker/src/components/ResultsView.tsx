import type { WorkoutResults } from "../App";
import "./ResultsView.css";

interface ResultsViewProps {
  results: WorkoutResults;
  onReset: () => void;
}

export default function ResultsView({ results, onReset }: ResultsViewProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "good";
    if (score >= 60) return "fair";
    return "poor";
  };

  const exerciseNames: Record<string, string> = {
    pushups: "Push-ups",
    "jumping-jacks": "Jumping Jacks",
    pullups: "Pull-ups",
    squats: "Squats",
    "russian-twists": "Russian Twists",
  };

  return (
    <div className="results-view">
      <div className="results-header">
        <h2>Workout Complete!</h2>
        <p className="exercise-name">{exerciseNames[results.exercise]}</p>
      </div>

      <div className="overall-score">
        <div className="score-circle">
          <div className={`score-value ${getScoreColor(results.overallScore)}`}>
            {results.overallScore}%
          </div>
          <div className="score-label">Overall Score</div>
        </div>
        <div className="overall-notes">
          {results.overallNotes.map((note, index) => (
            <div key={index} className="note-item overall">
              {note}
            </div>
          ))}
        </div>
      </div>

      <div className="reps-summary">
        <h3>Rep Breakdown ({results.reps.length} reps)</h3>
        <div className="reps-list">
          {results.reps.map((rep) => (
            <div key={rep.repNumber} className="rep-card">
              <div className="rep-header">
                <span className="rep-number">Rep {rep.repNumber}</span>
                <span className={`rep-score ${getScoreColor(rep.score)}`}>
                  {rep.score}%
                </span>
              </div>
              <div className="rep-notes">
                {rep.notes.map((note, index) => (
                  <div key={index} className="note-item">
                    {note}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <button className="reset-button" onClick={onReset}>
        Start New Workout
      </button>
    </div>
  );
}
