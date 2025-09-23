import PersonaVisualizer from "../components/PersonaVisualizer";
import personasData from "./personas.json";

export default function Home() {
  return <PersonaVisualizer personasData={personasData} />;
}
