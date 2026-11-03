import { api, type Any } from "../api";
import { ErrorBox, Loading, fmtDate, useAsync } from "../components/ui";

const NAMES: Record<string, string> = {
  espn: "ESPN (calendario, resultados, estadísticas, alineaciones, cuotas)",
  open_meteo: "Open-Meteo (clima y altitud)",
  api_football: "API-Football (lesiones y sanciones)",
  xg_provider: "Proveedor de xG real (Opta / StatsBomb / Understat)",
};

export default function Status() {
  const st = useAsync(() => api.status(), []);
  if (st.loading) return <Loading />;
  if (st.error) return <div className="mt"><ErrorBox error={st.error} /></div>;
  const s = st.data as Any;
  return (
    <div className="mt">
      <h1>Fuentes y estado</h1>
      <div className="muted mb">Qué datos están conectados, cuándo se actualizaron y qué falta.</div>
      <div className="grid g2">
        <div className="panel">
          <h3>Integraciones</h3>
          {Object.entries(s.integrations as Record<string, Any>).map(([k, v]) => (
            <div className="finding" key={k}>
              <b>{NAMES[k] ?? k} <span className={`badge ${v.connected ? "alta" : "baja"}`}>{v.connected ? "Conectado" : "No conectado"}</span></b>
              {v.missing_message && <span className="muted small">{v.missing_message}</span>}
            </div>
          ))}
        </div>
        <div className="panel">
          <h3>Datos</h3>
          <div className="kv"><span>Última consulta correcta</span><span>{s.data.last_successful_fetch ? fmtDate(s.data.last_successful_fetch + "Z") : "nunca"}</span></div>
          <div className="kv"><span>Errores de consulta (24 h)</span><span>{s.data.fetch_errors_24h}</span></div>
          <div className="kv"><span>Partidos completados en BD</span><span>{s.data.completed_matches.toLocaleString("es-ES")}</span></div>
          <div className="kv"><span>Partidos UCL en BD</span><span>{s.data.ucl_matches}</span></div>
          <div className="kv"><span>Actualización automática</span><span>{s.refresh.interval_min ? `cada ${s.refresh.interval_min} min` : "desactivada"}</span></div>
          <div className="kv"><span>Último backtest</span><span>{s.model.last_backtest ? fmtDate(s.model.last_backtest + "Z") : "no ejecutado"}</span></div>
          <h3 className="mt">Competiciones cargadas</h3>
          <div className="row wrap">{s.data.competitions.map((c: string) => <span key={c} className="tag">{c}</span>)}</div>
        </div>
      </div>
      <div className="panel mt small muted">
        La app nunca simula actualizaciones: si una fuente falla, el error queda registrado y el pronóstico conserva su fecha de cálculo.
        Los pronósticos se recalculan cuando cambian alineaciones, bajas, clima, cuotas o resultados.
      </div>
    </div>
  );
}
