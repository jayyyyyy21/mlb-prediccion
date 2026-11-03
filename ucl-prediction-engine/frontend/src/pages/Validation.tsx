import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api, type Any } from "../api";
import { ErrorBox, Loading, fmtDate, num, useAsync } from "../components/ui";

const tooltipStyle = { background: "#0b1428", border: "1px solid #233764", borderRadius: 10, color: "#e8eefc" };

function MetricRow({ label, m, b, lower = true, d = 4 }: { label: string; m?: number; b?: number; lower?: boolean; d?: number }) {
  if (m == null || b == null) return null;
  const better = lower ? m < b : m > b;
  return (
    <tr>
      <td>{label}</td>
      <td><b>{num(m, d)}</b></td>
      <td>{num(b, d)}</td>
      <td style={{ color: better ? "var(--good)" : "var(--bad)" }}>{better ? "Mejor" : "Peor"} ({((lower ? b - m : m - b) / b * 100).toFixed(1)}%)</td>
    </tr>
  );
}

export default function Validation() {
  const st = useAsync(() => api.backtest(), []);
  if (st.loading) return <Loading />;
  if (st.error) return <div className="mt"><ErrorBox error={st.error} /></div>;
  const bt = st.data as Any;
  const t = bt.test;
  const cal = (t.result_1x2.calibration as Any[]).map((c) => ({ pred: c.pred * 100, obs: c.obs * 100, n: c.n }));
  return (
    <div className="mt">
      <h1>Validación del modelo</h1>
      <div className="muted mb">
        Backtest cronológico (walk-forward): cada semana el modelo se reentrena solo con partidos anteriores y predice los partidos
        europeos de esa semana. Ejecutado {fmtDate(bt.created_at + "Z")} · {bt.model_version}
      </div>
      <div className="grid g3 mb">
        <div className="bigprob"><div className="v">{t.n}</div><div className="l">Partidos de prueba</div><div className="b">{t.period?.[0]?.slice(0, 10)} → {t.period?.[1]?.slice(0, 10)}</div></div>
        <div className="bigprob"><div className="v">{num(t.result_1x2.model.log_loss, 3)}</div><div className="l">Log loss 1X2</div><div className="b">referencia {num(t.result_1x2.baseline.log_loss, 3)}</div></div>
        <div className="bigprob"><div className="v">{num(t.result_1x2.ece * 100, 1)}%</div><div className="l">Error de calibración (ECE)</div><div className="b">acierto del favorito: {num(t.result_1x2.accuracy_argmax * 100, 1)}%</div></div>
      </div>

      <div className="grid g2">
        <div className="panel table-wrap">
          <h3>Métricas en el periodo de prueba</h3>
          <table>
            <thead><tr><th>Métrica</th><th>Modelo</th><th>Referencia</th><th>Resultado</th></tr></thead>
            <tbody>
              <MetricRow label="1X2 · log loss" m={t.result_1x2.model.log_loss} b={t.result_1x2.baseline.log_loss} />
              <MetricRow label="1X2 · Brier" m={t.result_1x2.model.brier} b={t.result_1x2.baseline.brier} />
              <MetricRow label="1X2 · RPS" m={t.result_1x2.model.rps} b={t.result_1x2.baseline.rps} />
              <MetricRow label="Más de 2.5 · log loss" m={t.over_2_5?.model.log_loss} b={t.over_2_5?.baseline.log_loss} />
              <MetricRow label="Ambos marcan · log loss" m={t.btts?.model.log_loss} b={t.btts?.baseline.log_loss} />
              <MetricRow label="Córners +9.5 · log loss" m={t.corners_over_9_5?.model.log_loss} b={t.corners_over_9_5?.baseline.log_loss} />
              <MetricRow label="Tiros a puerta +8.5 · log loss" m={t.sot_over_8_5?.model.log_loss} b={t.sot_over_8_5?.baseline.log_loss} />
              <MetricRow label="Córners totales · MAE" m={t.corners_total_mae?.model} b={t.corners_total_mae?.baseline} d={2} />
              <MetricRow label="Tiros a puerta totales · MAE" m={t.sot_total_mae?.model} b={t.sot_total_mae?.baseline} d={2} />
              <MetricRow label="Posesión · MAE (pp)" m={t.possession_mae?.model} b={t.possession_mae?.baseline} d={2} />
              {t.vs_market && <MetricRow label={`1X2 vs mercado (n=${t.vs_market.n}) · log loss`} m={t.vs_market.model.log_loss} b={t.vs_market.market.log_loss} />}
            </tbody>
          </table>
          <div className="dim small mt-s">
            Referencia ingenua = frecuencias observadas en competiciones europeas del último año (para posesión: 50%).
            {!t.vs_market && " No hay suficientes cuotas históricas verificadas para comparar con el mercado."}
          </div>
        </div>
        <div className="panel">
          <h3>Calibración 1X2</h3>
          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ left: 0, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="pred" name="Predicha" unit="%" domain={[0, 100]} tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="obs" name="Observada" unit="%" domain={[0, 100]} tick={{ fontSize: 11 }} width={40} />
                <ZAxis dataKey="n" range={[40, 400]} name="n" />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => (typeof v === "number" ? v.toFixed(1) : v)} />
                <Scatter data={cal} fill="#2bb3ff" />
                <Scatter data={[{ pred: 0, obs: 0, n: 1 }, { pred: 100, obs: 100, n: 1 }]} line={{ stroke: "#5d7099", strokeDasharray: "4 4" }} shape={() => <g />} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="dim small">Cada punto agrupa probabilidades similares; tamaño = nº de casos. Sobre la diagonal = bien calibrado.</div>
        </div>
      </div>

      {t.over_2_5?.calibration && (
        <div className="panel mt">
          <h3>Calibración por mercado (predicho vs observado)</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <LineChart>
                <CartesianGrid />
                <XAxis type="number" dataKey="pred" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} allowDuplicatedCategory={false} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} width={40} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Legend />
                {([["over_2_5", "Más de 2.5", "#2bb3ff"], ["btts", "Ambos marcan", "#8b7bff"], ["corners_over_9_5", "Córners +9.5", "#2fd38a"], ["sot_over_8_5", "Tiros a puerta +8.5", "#ffb547"]] as [string, string, string][]).map(([k, l, c]) =>
                  t[k] ? <Line key={k} name={l} stroke={c} dot data={(t[k].calibration as Any[]).map((x) => ({ pred: x.pred * 100, obs: x.obs * 100 }))} dataKey="obs" /> : null,
                )}
                <Line name="Perfecta" stroke="#5d7099" strokeDasharray="4 4" dot={false} data={[{ pred: 0, obs: 0 }, { pred: 100, obs: 100 }]} dataKey="obs" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="panel mt table-wrap">
        <h3>Selección de hiperparámetros (periodo de validación {bt.validation.period[0].slice(0, 10)} → {bt.validation.period[1].slice(0, 10)})</h3>
        <table>
          <thead><tr><th>Vida media (días)</th><th>Peso xG-proxy</th><th>λ equipo</th><th>Log loss 1X2</th><th>Log loss +2.5</th><th>Total</th></tr></thead>
          <tbody>
            {(bt.validation.grid as Any[]).map((g, i) => {
              const chosen = g.hyper.half_life_days === bt.chosen_hyper.half_life_days && g.hyper.xg_blend === bt.chosen_hyper.xg_blend && g.hyper.lam_team === bt.chosen_hyper.lam_team;
              return <tr key={i} style={{ background: chosen ? "rgba(43,179,255,.08)" : undefined }}><td>{g.hyper.half_life_days}{chosen ? " ✓" : ""}</td><td>{g.hyper.xg_blend}</td><td>{g.hyper.lam_team}</td><td>{num(g.log_loss_1x2, 4)}</td><td>{num(g.log_loss_o25, 4)}</td><td><b>{num(g.score, 4)}</b></td></tr>;
            })}
          </tbody>
        </table>
        {bt.validation.count_grid && (
          <>
            <h3 className="mt">Calibración de córners y tiros a puerta</h3>
            <table>
              <thead><tr><th>Encogimiento equipo/liga</th><th>Peso del modelo en el total</th><th>LL córners (ref.)</th><th>LL tiros a puerta (ref.)</th></tr></thead>
              <tbody>
                {(bt.validation.count_grid as Any[]).map((g, i) => {
                  const chosen = g.count_shrink === bt.chosen_hyper.count_shrink && g.count_blend === bt.chosen_hyper.count_blend;
                  return <tr key={i} style={{ background: chosen ? "rgba(43,179,255,.08)" : undefined }}><td>{g.count_shrink}{chosen ? " ✓" : ""}</td><td>{g.count_blend}</td><td>{num(g.corners_ll, 4)} ({num(g.corners_baseline_ll, 4)})</td><td>{num(g.sot_ll, 4)} ({num(g.sot_baseline_ll, 4)})</td></tr>;
                })}
              </tbody>
            </table>
          </>
        )}
        <div className="dim small mt-s">Ajuste táctico por estilo: {bt.validation.style_adjust_used ? "mejoró en validación y se usa" : "no mejoró en validación → desactivado (evita sobreajuste)"}.</div>
        <ul className="list small muted">{bt.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
      </div>
    </div>
  );
}
