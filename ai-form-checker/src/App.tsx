import { Routes, Route } from "react-router-dom";
import ExerciseSelection from "./pages/ExerciseSelection";
import Analyze from "./pages/Analyze";
import Results from "./pages/Results";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>RepRadar</h1>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<ExerciseSelection />} />
          <Route path="/analyze/:exercise" element={<Analyze />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </main>
    </div>
  );
}
