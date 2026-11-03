import { api, type Any } from "../api";
import { Crest, ErrorBox, Loading, useAsync } from "../components/ui";

export default function Standings() {
  const st = useAsync(() => api.standings(), []);
  if (st.loading) return <Loading />;
  if (st.error) return <div className="mt"><ErrorBox error={st.error} /></div>;
  const rows = st.data ?? [];
  return (
    <div className="mt">
      <h1>Clasificación · Fase de liga</h1>
      <div className="muted mb">Calculada con los resultados observados de la temporada actual (fuente ESPN).</div>
      <div className="panel table-wrap">
        <table>
          <thead><tr><th>#</th><th className="left">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th><th>DG</th><th>Pts</th></tr></thead>
          <tbody>
            {rows.map((r: Any) => (
              <tr key={r.team_id} style={{ boxShadow: r.pos === 8 ? "inset 0 -2px 0 var(--electric)" : r.pos === 24 ? "inset 0 -2px 0 var(--warn)" : undefined }}>
                <td>{r.pos}</td>
                <td className="left"><span className="row"><Crest team={{ name: r.name, logo: r.logo }} size="sm" />{r.name}</span></td>
                <td>{r.p}</td><td>{r.w}</td><td>{r.d}</td><td>{r.l}</td><td>{r.gf}</td><td>{r.ga}</td>
                <td>{r.gf - r.ga > 0 ? "+" : ""}{r.gf - r.ga}</td><td><b>{r.pts}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="dim small mt-s">Línea azul: clasificación directa (top 8). Línea ámbar: límite de playoff (puesto 24).</div>
    </div>
  );
}
