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
