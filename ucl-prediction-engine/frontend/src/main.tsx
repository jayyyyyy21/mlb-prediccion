import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes, Link } from "react-router-dom";
import "./styles.css";
import Home from "./pages/Home";
import MatchPage from "./pages/Match";
import Standings from "./pages/Standings";
import Validation from "./pages/Validation";
import Status from "./pages/Status";

function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <div className="container">
          <Link to="/" className="brand">
            <span className="mark"><span>★</span></span>
            <b>UCL <em>PREDICTION</em> ENGINE</b>
          </Link>
          <nav className="nav">
            <NavLink to="/" end>Partidos</NavLink>
            <NavLink to="/clasificacion">Clasificación</NavLink>
            <NavLink to="/validacion">Validación del modelo</NavLink>
            <NavLink to="/estado">Fuentes y estado</NavLink>
          </nav>
        </div>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/partido/:id" element={<MatchPage />} />
          <Route path="/clasificacion" element={<Standings />} />
          <Route path="/validacion" element={<Validation />} />
          <Route path="/estado" element={<Status />} />
        </Routes>
      </main>
      <footer className="container">
        Pronósticos estadísticos con incertidumbre explícita. No son garantía de resultado ni consejo de apuesta.
        Datos: ESPN (resultados y estadísticas), Open-Meteo (clima). Sin afiliación con la UEFA.
      </footer>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
