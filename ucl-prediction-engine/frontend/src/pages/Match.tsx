import { Fragment, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar,
  RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type Any, type PredictionPayload } from "../api";
import {
  Compare, ConfidenceBadge, Crest, ErrorBox, LineProb, Loading, PHASES, ProbBar, fmtDate, num, pct, useAsync,
} from "../components/ui";

const TABS = [
  ["pred", "MATCH PREDICTION"],
  ["tac", "TACTICAL BREAKDOWN"],
  ["stats", "ESTADÍSTICAS"],
  ["avail", "BAJAS Y ALINEACIONES"],
  ["opp", "OPORTUNIDADES"],
  ["trans", "TRANSPARENCIA"],
] as const;
type Tab = (typeof TABS)[number][0];

const tooltipStyle = { background: "#0b1428", border: "1px solid #233764", borderRadius: 10, color: "#e8eefc" };

export default function MatchPage() {
  const { id = "" } = useParams();
  const [tab, setTab] = useState<Tab>("pred");
  const st = useAsync(() => api.match(id), [id]);
  if (st.loading) return <Loading text="Calculando pronóstico (modelo + 20.000 simulaciones)…" />;
  if (st.error || !st.data) return <div className="mt"><ErrorBox error={st.error ?? "Sin datos"} /></div>;
  const p = st.data;
  const m = p.match;
  return (
    <div className="mt">
      <Link to="/" className="small">← Volver al calendario</Link>
      <div className="hero mt-s">
        <div className="side"><Crest team={m.home} size="lg" /><div className="name">{m.home.name}</div><span className="dim small">Local</span></div>
        <div className="center">
          <div className="badge">{PHASES[m.phase ?? ""] ?? m.phase} · {m.season}</div>
          <div className="score">{m.completed ? `${m.home_goals} - ${m.away_goals}` : "VS"}</div>
          <div className="muted small">{fmtDate(m.kickoff)}</div>
          <div className="dim small">{m.venue}{m.city ? ` · ${m.city}` : ""}</div>
          <div className="mt-s"><ConfidenceBadge level={p.confidence.level} /></div>
        </div>
        <div className="side"><Crest team={m.away} size="lg" /><div className="name">{m.away.name}</div><span className="dim small">Visitante</span></div>
      </div>
      <div className="dim small mt-s">
        Pronóstico generado {fmtDate(p.transparency.generated_at + "Z", { weekday: undefined })}
        {p.transparency.update_reason ? ` · motivo: ${p.transparency.update_reason}` : ""} · modelo {p.transparency.model_version}
      </div>

      <div className="tabs" role="tablist">
        {TABS.map(([k, l]) => (
          <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
        ))}
      </div>

      {tab === "pred" && <PredictionTab p={p} />}
      {tab === "tac" && <TacticalTab p={p} />}
      {tab === "stats" && <StatsTab p={p} />}
      {tab === "avail" && <AvailabilityTab p={p} />}
      {tab === "opp" && <OpportunitiesTab p={p} />}
      {tab === "trans" && <TransparencyTab p={p} />}
    </div>
  );
}

/* ----------------------------------------------------------- MATCH PREDICTION */
function PredictionTab({ p }: { p: PredictionPayload }) {
  const g = p.prediction.goals;
  const c = p.prediction.corners;
  const s = p.prediction.shots_on_target;
  const pos = p.prediction.possession;
  const hn = p.match.home.name;
  const an = p.match.away.name;
  const b = g.bands_p10_p90;
  const history = useAsync(() => api.history(p.match.id), [p.match.id]);

  const cornerDist = (c.total_dist as number[]).map((v, i) => ({ k: i, p: v * 100 }));
  return (
    <>
      <div className="panel">
        <h3>A · Probabilidades del resultado</h3>
        <div className="grid g3 probs">
          <div className="bigprob"><div className="v" style={{ color: "var(--home)" }}>{pct(g.home_win)}</div><div className="l">Victoria {hn}</div><div className="b">banda p10–p90: {pct(b.home_win[0])}–{pct(b.home_win[1])}</div></div>
          <div className="bigprob"><div className="v" style={{ color: "var(--draw)" }}>{pct(g.draw)}</div><div className="l">Empate</div><div className="b">banda p10–p90: {pct(b.draw[0])}–{pct(b.draw[1])}</div></div>
          <div className="bigprob"><div className="v" style={{ color: "var(--away)" }}>{pct(g.away_win)}</div><div className="l">Victoria {an}</div><div className="b">banda p10–p90: {pct(b.away_win[0])}–{pct(b.away_win[1])}</div></div>
        </div>
        <ProbBar h={g.home_win} d={g.draw} a={g.away_win} labels={[hn, "Empate", an]} />
        <div className="dim small mt-s">
          Las bandas reflejan la incertidumbre de los parámetros del modelo (aproximación de Laplace), no la aleatoriedad del partido.
        </div>
      </div>

      <div className="grid g2 mt">
        <div className="panel">
          <h3>B · Goles</h3>
          <div className="kv"><span>Goles esperados {hn}</span><span>{num(g.home_xg)}</span></div>
          <div className="kv"><span>Goles esperados {an}</span><span>{num(g.away_xg)}</span></div>
          <div className="kv"><span>Marcador más probable</span><span>{g.most_likely_score} ({pct(g.most_likely_score_prob, 1)})</span></div>
          <div className="mt-s">
            <LineProb label="Más de 1.5" p={g.over_1_5} />
            <LineProb label="Más de 2.5" p={g.over_2_5} />
            <LineProb label="Más de 3.5" p={g.over_3_5} />
            <LineProb label="Ambos marcan" p={g.btts} />
          </div>
        </div>
        <div className="panel">
          <h3>Matriz de marcadores</h3>
          <ScoreMatrix m={g.score_matrix} hn={hn} an={an} />
        </div>
      </div>

      <div className="grid g2 mt">
        <div className="panel">
          <h3>C · Córners</h3>
          <Compare label="Esperados" h={c.home_exp} a={c.away_exp} d={1} />
          <div className="kv"><span>Total esperado</span><span>{num(c.total_exp, 1)}</span></div>
          <div className="mt-s">
            {Object.entries(c.total_over as Record<string, number>).map(([l, v]) => <LineProb key={l} label={`Más de ${l}`} p={v} />)}
          </div>
          <div className="kv mt-s"><span>{hn} más córners</span><span>{pct(c.home_more)}</span></div>
          <div className="kv"><span>Igualados</span><span>{pct(c.equal)}</span></div>
          <div className="kv"><span>{an} más córners</span><span>{pct(c.away_more)}</span></div>
          <div style={{ height: 160 }} className="mt-s">
            <ResponsiveContainer>
              <BarChart data={cornerDist}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="k" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" width={36} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v.toFixed(1)}%`} labelFormatter={(l) => `${l} córners`} />
                <Bar dataKey="p" fill="#2bb3ff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="dim small">Distribución simulada del total de córners · binomial negativa (α={num(c.dispersion, 3)}) ajustada por rival.</div>
        </div>
        <div className="panel">
          <h3>D · Tiros a puerta</h3>
          <Compare label="Esperados" h={s.home_exp} a={s.away_exp} d={1} />
          <Compare label="Tiros totales" h={s.home_shots_exp} a={s.away_shots_exp} d={1} />
          <div className="kv"><span>Total combinado esperado</span><span>{num(s.total_exp, 1)}</span></div>
          <div className="mt-s">
            {Object.entries(s.total_over as Record<string, number>).map(([l, v]) => <LineProb key={l} label={`Total +${l}`} p={v} />)}
          </div>
          <div className="grid g2 mt-s">
            <div>
              <div className="muted small">{hn}</div>
              {Object.entries(s.home_over as Record<string, number>).map(([l, v]) => <LineProb key={l} label={`+${l}`} p={v} />)}
            </div>
            <div>
              <div className="muted small">{an}</div>
              {Object.entries(s.away_over as Record<string, number>).map(([l, v]) => <LineProb key={l} label={`+${l}`} p={v} />)}
            </div>
          </div>
        </div>
      </div>

      <div className="grid g2 mt">
        <div className="panel">
          <h3>E · Posesión estimada</h3>
          <div className="pbar" style={{ height: 22 }}>
            <div className="h" style={{ width: `${pos.home}%` }} />
            <div className="a" style={{ width: `${pos.away}%` }} />
          </div>
          <div className="plabels"><span>{hn} <b>{num(pos.home, 0)}%</b></span><span>{an} <b>{num(pos.away, 0)}%</b></span></div>
          <div className="dim small mt-s">
            Controlar la posesión no implica generar más ocasiones: compárese con los goles y tiros a puerta esperados.
          </div>
        </div>
        <div className="panel">
          <h3>Simulación Monte Carlo</h3>
          <div className="kv"><span>Iteraciones</span><span>{p.prediction.monte_carlo.iterations.toLocaleString("es-ES")}</span></div>
          <div className="kv"><span>1X2 simulado</span><span>{pct(p.prediction.monte_carlo.goals_check.home_win, 1)} / {pct(p.prediction.monte_carlo.goals_check.draw, 1)} / {pct(p.prediction.monte_carlo.goals_check.away_win, 1)}</span></div>
          <div className="kv"><span>Error estándar máx.</span><span>±{pct(p.prediction.monte_carlo.std_error_max, 2)}</span></div>
          <div className="kv"><span>Ajuste ataque por bajas</span><span>{num(p.prediction.adjustments.home_attack_mult, 3)} / {num(p.prediction.adjustments.away_attack_mult, 3)}</span></div>
          <div className="kv"><span>Días de descanso</span><span>{num(p.prediction.adjustments.rest_days_home, 1)} / {num(p.prediction.adjustments.rest_days_away, 1)}</span></div>
          <div className="kv"><span>ρ Dixon-Coles</span><span>{num(g.rho, 3)}</span></div>
        </div>
      </div>

      <div className="panel mt">
        <h3>G · Nivel de confianza: {p.confidence.level}</h3>
        <ul className="list">
          {p.confidence.reasons.map((r, i) => (
            <li key={i}><span className={`badge ${r.level === "high" ? "alta" : r.level === "medium" ? "media" : "baja"}`}>{r.level === "high" ? "OK" : r.level === "medium" ? "Moderado" : "Limita"}</span> {r.text}</li>
          ))}
        </ul>
      </div>

      {history.data && history.data.length > 1 && (
        <div className="panel mt">
          <h3>Evolución del pronóstico</h3>
          <div style={{ height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={history.data.map((h: Any) => ({ t: fmtDate(h.created_at + "Z"), "1": h.home_win * 100, X: h.draw * 100, "2": h.away_win * 100 }))}>
                <CartesianGrid />
                <XAxis dataKey="t" tick={{ fontSize: 10 }} />
                <YAxis unit="%" tick={{ fontSize: 11 }} width={40} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Legend />
                <Line dataKey="1" stroke="#2bb3ff" dot />
                <Line dataKey="X" stroke="#7b8bb0" dot />
                <Line dataKey="2" stroke="#8b7bff" dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <ul className="list small muted">{history.data.map((h: Any, i: number) => <li key={i}>{fmtDate(h.created_at + "Z")}: {h.trigger}</li>)}</ul>
        </div>
      )}
    </>
  );
}

function ScoreMatrix({ m, hn, an }: { m: number[][]; hn: string; an: string }) {
  const max = Math.max(...m.flat());
  return (
    <>
      <div className="heat" style={{ gridTemplateColumns: `28px repeat(${m[0].length}, 1fr)` }}>
        <div className="hdr" />
        {m[0].map((_, j) => <div key={j} className="hdr">{j}</div>)}
        {m.map((row, i) => (
          <Fragment key={i}>
            <div className="hdr" style={{ alignSelf: "center" }}>{i}</div>
            {row.map((v, j) => (
              <div key={`${i}-${j}`} className="c" title={`${hn} ${i} - ${j} ${an}: ${(v * 100).toFixed(1)}%`}
                style={{ background: `rgba(43,179,255,${0.06 + 0.8 * (v / max)})`, color: v / max > 0.5 ? "#001222" : undefined }}>
                {(v * 100).toFixed(1)}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
      <div className="dim small mt-s">Filas: goles {hn} · Columnas: goles {an} · valores en %</div>
    </>
  );
}

/* ----------------------------------------------------------- TACTICAL */
function TacticalTab({ p }: { p: PredictionPayload }) {
  const t = p.tactical;
  const hn = p.match.home.name;
  const an = p.match.away.name;
  const radar = (t.home_style.metrics as Any[]).map((m: Any, i: number) => ({
    k: m.label.replace(" (menor = más presión)", ""),
    [hn]: m.percentile ?? 0,
    [an]: t.away_style.metrics[i].percentile ?? 0,
  }));
  return (
    <>
      <div className="panel">
        <h3>Conclusión táctica</h3>
        <p style={{ fontSize: 16, margin: 0 }}>{t.conclusion}</p>
      </div>
      <div className="grid g2 mt">
        {[["home", hn], ["away", an]].map(([side, name]) => (
          <div className="panel" key={side}>
            <h3>{name}</h3>
            <div className="row wrap">{t[`${side}_style`].tags.map((x: Any) => <span className="tag" key={x.tag} title={x.evidence}>{x.tag}</span>)}</div>
            <ul className="list small muted mt-s">{t[`${side}_style`].tags.map((x: Any) => <li key={x.tag}>{x.tag}: {x.evidence}</li>)}</ul>
            <div className="kv"><span>Formación habitual</span><span>{t[`${side}_style`].formation?.[0]?.[0] ?? "n/d"}</span></div>
            <div className="kv"><span>Alternativas usadas</span><span>{(t[`${side}_style`].formation ?? []).slice(1, 4).map((f: Any) => `${f[0]} (${f[1]})`).join(", ") || "—"}</span></div>
          </div>
        ))}
      </div>
      <div className="panel mt">
        <h3>Perfil de estilo (percentil frente a equipos UCL)</h3>
        <div style={{ height: 340 }}>
          <ResponsiveContainer>
            <RadarChart data={radar} outerRadius="72%">
              <PolarGrid stroke="#1a2a4d" />
              <PolarAngleAxis dataKey="k" tick={{ fontSize: 11, fill: "#8fa3c8" }} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
              <Radar name={hn} dataKey={hn} stroke="#2bb3ff" fill="#2bb3ff" fillOpacity={0.25} />
              <Radar name={an} dataKey={an} stroke="#8b7bff" fill="#8b7bff" fillOpacity={0.2} />
              <Legend />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `p${v.toFixed(0)}`} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Métrica</th><th>{hn}</th><th>pct</th><th>{an}</th><th>pct</th></tr></thead>
            <tbody>
              {(t.home_style.metrics as Any[]).map((m: Any, i: number) => {
                const a = t.away_style.metrics[i];
                const f = (v: number | null) => (v === null ? "n/d" : m.key.includes("acc") || m.key.includes("share") ? pct(v, 1) : num(v, 1));
                return <tr key={m.key}><td>{m.label}</td><td>{f(m.value)}</td><td className="dim">{m.percentile === null ? "" : `p${m.percentile.toFixed(0)}`}</td><td>{f(a.value)}</td><td className="dim">{a.percentile === null ? "" : `p${a.percentile.toFixed(0)}`}</td></tr>;
              })}
            </tbody>
          </table>
        </div>
      </div>
      <div className="grid g2 mt">
        <div className="panel">
          <h3>Cómo puede desarrollarse</h3>
          {(t.findings as Any[]).map((f: Any, i: number) => (
            <div className="finding" key={i}><b>{f.title}{f.favors ? <span className="dim small"> · favorece a {f.favors}</span> : null}</b><span className="muted">{f.text}</span></div>
          ))}
        </div>
        <div className="panel">
          <h3>Vulnerabilidades explotables</h3>
          {t.vulnerabilities.length ? <ul className="list">{t.vulnerabilities.map((v: string, i: number) => <li key={i}>{v}</li>)}</ul> : <div className="muted">No se detectan vulnerabilidades destacadas con los umbrales definidos.</div>}
          <h3 className="mt">Limitaciones</h3>
          <ul className="list small muted">{t.limitations.map((v: string, i: number) => <li key={i}>{v}</li>)}</ul>
        </div>
      </div>
    </>
  );
}

/* ----------------------------------------------------------- STATS */
const WINDOWS: [string, string][] = [
  ["last5", "Últimos 5"], ["last10", "Últimos 10"], ["last15", "Últimos 15"], ["season", "Temporada"],
  ["home10", "Local (10)"], ["away10", "Visitante (10)"], ["europe", "Europa"],
];
const ROWS: [string, string, number, boolean?][] = [
  ["goals_for", "Goles a favor", 2], ["xg_proxy_for", "xG-proxy a favor", 2], ["shots_for", "Tiros", 1],
  ["sot_for", "Tiros a puerta", 1], ["conversion", "Conversión (gol/tiro)", 3, true], ["sot_accuracy", "Precisión de tiro", 3, true],
  ["corners_for", "Córners a favor", 1], ["possession", "Posesión %", 1],
  ["goals_against", "Goles en contra", 2], ["xg_proxy_against", "xG-proxy en contra", 2], ["shots_against", "Tiros permitidos", 1],
  ["sot_against", "Tiros a puerta permitidos", 1], ["corners_against", "Córners concedidos", 1],
  ["clean_sheet_pct", "Porterías a cero", 0, true], ["failed_to_score_pct", "Sin marcar", 0, true],
  ["scored_1plus_pct", "Marca 1+", 0, true], ["scored_2plus_pct", "Marca 2+", 0, true], ["scored_3plus_pct", "Marca 3+", 0, true],
  ["ppg", "Puntos/partido", 2],
];

function StatsTab({ p }: { p: PredictionPayload }) {
  const [win, setWin] = useState("last10");
  const h = p.teams.home;
  const a = p.teams.away;
  const hw = h.windows[win];
  const aw = a.windows[win];
  const hn = p.match.home.name;
  const an = p.match.away.name;
  const freq = [1, 2, 3].map((k) => ({ k: `${k}+ goles`, [hn]: (hw[`scored_${k}plus_pct`] ?? 0) * 100, [an]: (aw[`scored_${k}plus_pct`] ?? 0) * 100 }));
  return (
    <>
      <div className="notice info mb">Estadísticas <b>observadas</b> (fuente ESPN). El «xG-proxy» se calcula con tiros a puerta; no es el xG de Opta/Understat.</div>
      <div className="filters mb">
        {WINDOWS.map(([k, l]) => <button key={k} className={`chip ${win === k ? "on" : ""}`} onClick={() => setWin(k)}>{l}</button>)}
      </div>
      <div className="grid g2">
        <div className="panel">
          <div className="row between"><h3>Comparativa ({WINDOWS.find((w) => w[0] === win)?.[1]})</h3><span className="dim small">n = {hw.n ?? 0} / {aw.n ?? 0}</span></div>
          <div className="row between small muted mb"><span>{hn}</span><span>{an}</span></div>
          {ROWS.map(([k, l, d, isPct]) => (
            <Compare key={k} label={l} h={isPct ? (hw[k] ?? null) !== null ? hw[k] * 100 : null : hw[k]} a={isPct ? (aw[k] ?? null) !== null ? aw[k] * 100 : null : aw[k]} d={isPct ? (d === 3 ? 1 : 0) : d}
              higherIsBetter={!["goals_against", "xg_proxy_against", "shots_against", "sot_against", "corners_against", "failed_to_score_pct"].includes(k)} />
          ))}
        </div>
        <div>
          <div className="panel">
            <h3>Frecuencia de goles marcados</h3>
            <div style={{ height: 200 }}>
              <ResponsiveContainer>
                <BarChart data={freq}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="k" tick={{ fontSize: 12 }} />
                  <YAxis unit="%" width={40} tick={{ fontSize: 11 }} domain={[0, 100]} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v.toFixed(0)}%`} />
                  <Legend />
                  <Bar dataKey={hn} fill="#2bb3ff" radius={[4, 4, 0, 0]} />
                  <Bar dataKey={an} fill="#8b7bff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="panel mt">
            <h3>Forma reciente</h3>
            {[[h, hn], [a, an]].map(([t, n]) => (
              <div key={n as string} className="mb">
                <div className="muted small">{n as string}</div>
                {(t as Any).form.map((f: Any, i: number) => (
                  <div key={i} className="row small" style={{ padding: "3px 0" }}>
                    <span className={`form-pill ${f.res}`}>{f.res === "W" ? "V" : f.res === "D" ? "E" : "D"}</span>
                    <span className="num">{f.gf}-{f.ga}</span>
                    <span>{f.home ? "vs" : "@"} {f.opponent_name}</span>
                    <span className="dim">{f.league} · {new Date(f.date).toLocaleDateString("es-ES")}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="panel mt">
        <h3>Rendimiento por tipo de rival</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Split</th><th>Equipo</th><th>n</th><th>PPG</th><th>GF</th><th>GC</th><th>xG-p F</th><th>xG-p C</th><th>TaP F</th><th>TaP C</th></tr></thead>
            <tbody>
              {([["vs_strong", "vs rivales fuertes (top 15%)"], ["vs_similar", "vs rivales de nivel similar"], ["vs_possession_teams", "vs equipos de posesión"], ["vs_low_possession_teams", "vs equipos de poca posesión"]] as [string, string][]).flatMap(([k, l]) =>
                [[h, hn], [a, an]].map(([t, n]) => {
                  const s = (t as Any).splits[k];
                  return <tr key={k + n}><td>{l}</td><td className="left">{n as string}</td><td>{s.n}</td><td>{num(s.ppg)}</td><td>{num(s.goals_for)}</td><td>{num(s.goals_against)}</td><td>{num(s.xg_proxy_for)}</td><td>{num(s.xg_proxy_against)}</td><td>{num(s.sot_for, 1)}</td><td>{num(s.sot_against, 1)}</td></tr>;
                }),
              )}
            </tbody>
          </table>
        </div>
        <div className="dim small mt-s">Fuerza del rival según el rating del modelo (ataque + defensa, ajustado por liga). Muestras pequeñas → interpretar con cautela.</div>
      </div>
      <div className="grid g2 mt">
        {[[h, hn], [a, an]].map(([t, n]) => (
          <div className="panel" key={n as string}>
            <h3>Calendario · {n as string}</h3>
            <div className="kv"><span>Días de descanso</span><span>{num((t as Any).schedule.rest_days, 1)}</span></div>
            <div className="kv"><span>Partidos últimos 14 días</span><span>{(t as Any).schedule.matches_last_14d}</span></div>
            <div className="kv"><span>Partidos últimos 30 días</span><span>{(t as Any).schedule.matches_last_30d}</span></div>
            <div className="kv"><span>Rating ataque / defensa</span><span>{num((t as Any).rating.attack)} / {num((t as Any).rating.defense)}</span></div>
            <div className="kv"><span>Percentil global (modelo)</span><span>p{num((t as Any).rating.overall_percentile, 0)}</span></div>
            <div className="kv"><span>Liga de referencia</span><span>{(t as Any).league_group}</span></div>
          </div>
        ))}
      </div>
    </>
  );
}

/* ----------------------------------------------------------- AVAILABILITY */
const STATUS_LABEL: Record<string, string> = {
  confirmed: "Baja confirmada", probable: "Duda / probable", rumor: "Rumor (no usado)", not_in_xi: "Fuera del XI confirmado",
};

function AvailabilityTab({ p }: { p: PredictionPayload }) {
  const av = p.availability;
  return (
    <>
      {av.injury_source_missing && (
        <div className="notice mb">
          <b>Bajas no verificadas.</b> No hay una fuente de lesiones/sanciones conectada. Para activarla, configura
          <code> API_FOOTBALL_KEY</code> en el servidor (API-Football / api-sports.io). Mientras tanto la confianza se reduce
          y solo se usan las alineaciones confirmadas que publica ESPN (~1 h antes del partido).
        </div>
      )}
      <div className="grid g2">
        {[["home", p.match.home.name], ["away", p.match.away.name]].map(([side, name]) => {
          const a = av[side];
          return (
            <div className="panel" key={side}>
              <div className="row between"><h3>{name}</h3>
                <span className={`badge ${a.lineup_confirmed ? "alta" : "media"}`}>{a.lineup_confirmed ? "XI confirmado" : "XI no confirmado"}</span></div>
              <div className="kv"><span>Multiplicador ofensivo por bajas</span><span>{num(a.attack_multiplier, 3)}</span></div>
              {a.goalkeeper_absent && <div className="notice bad mt-s">Portero titular ausente: impacto no cuantificado (reduce confianza).</div>}
              {a.key_defensive_absences.length > 0 && <div className="notice mt-s">Bajas defensivas clave: {a.key_defensive_absences.join(", ")}</div>}
              <h3 className="mt">Bajas</h3>
              {a.absences.length === 0 ? <div className="muted small">Ninguna baja registrada por fuentes verificadas.</div> : (
                <table><thead><tr><th>Jugador</th><th className="left">Estado</th><th>Titular %</th><th>Peso ofensivo</th></tr></thead>
                  <tbody>{a.absences.map((x: Any, i: number) => (
                    <tr key={i}><td>{x.player}<div className="dim small">{x.reason} · {x.source}</div></td><td className="left">{STATUS_LABEL[x.status] ?? x.status}</td><td>{pct(x.start_rate)}</td><td>{pct(x.share, 1)}</td></tr>
                  ))}</tbody></table>
              )}
              {a.lineup_confirmed && (<><h3 className="mt">XI confirmado</h3><div className="small">{a.lineup.map((x: Any) => `${x.player} (${x.position})`).join(" · ")}</div></>)}
              <h3 className="mt">Jugadores habituales (últimos {a.sample_matches})</h3>
              <div className="table-wrap">
                <table><thead><tr><th>Jugador</th><th>Pos</th><th>Titular</th><th>G+A</th><th>Peso ofensivo</th></tr></thead>
                  <tbody>{a.regulars.map((x: Any, i: number) => (
                    <tr key={i}><td>{x.player}</td><td>{x.position}</td><td>{pct(x.start_rate)}</td><td>{x.ga}</td><td>{pct(x.share, 1)}</td></tr>
                  ))}</tbody></table>
              </div>
            </div>
          );
        })}
      </div>
      <div className="panel mt small muted">
        <b>Cómo se calcula el impacto:</b> peso ofensivo = 50% cuota de goles+asistencias + 50% cuota de tiros del equipo en los últimos
        partidos. Se asume que el sustituto recupera la mitad de esa producción (supuesto explícito). Bajas probables cuentan al 50%;
        los rumores nunca alteran el modelo. Bajas defensivas y de portero se señalan pero no se cuantifican.
      </div>
    </>
  );
}

/* ----------------------------------------------------------- OPPORTUNITIES */
const FAMILY: Record<string, string> = { "1x2": "Ganador / doble oportunidad", goals_total: "Total de goles", btts: "Ambos marcan", corners: "Córners", sot: "Tiros a puerta" };

function OpportunitiesTab({ p }: { p: PredictionPayload }) {
  const o = p.opportunities;
  const [fam, setFam] = useState("all");
  const rows = (o.markets as Any[]).filter((m) => fam === "all" || m.family === fam);
  return (
    <>
      <div className="decision">
        <div className="big" style={{ color: o.recommendation.decision === "BET" ? "var(--cyan)" : "var(--muted)" }}>{o.recommendation.decision}</div>
        <div className="muted">{o.recommendation.explanation}</div>
      </div>
      {o.recommendation.picks.length > 0 && (
        <div className="panel mt">
          <h3>Mercados con ventaja respaldada</h3>
          {o.recommendation.picks.map((m: Any, i: number) => (
            <div className="kv" key={i}><span>{m.market} @ {num(m.odds)}</span><span>modelo {pct(m.prob, 1)} (p10 {pct(m.band[0], 1)}) vs implícita {pct(m.implied, 1)} → +{pct(m.edge_conservative, 1)}</span></div>
          ))}
        </div>
      )}
      <div className="notice info mt">
        Alta probabilidad ≠ valor. El valor solo se evalúa con cuotas verificadas ({o.odds_source ?? "no disponibles para este partido"}),
        usando la probabilidad conservadora (p10) frente a la implícita sin margen de la casa.
      </div>
      <div className="filters mt">
        <button className={`chip ${fam === "all" ? "on" : ""}`} onClick={() => setFam("all")}>Todos</button>
        {Object.entries(FAMILY).map(([k, l]) => <button key={k} className={`chip ${fam === k ? "on" : ""}`} onClick={() => setFam(k)}>{l}</button>)}
      </div>
      <div className="panel mt table-wrap">
        <table>
          <thead><tr><th>Mercado</th><th>Prob.</th><th>Banda p10–p90</th><th>Implícita</th><th>Ventaja (p10)</th><th>Estable backtest</th><th>Respaldo</th><th className="left">Veredicto</th></tr></thead>
          <tbody>
            {rows.map((m: Any, i: number) => (
              <tr key={i}>
                <td>{m.market}</td>
                <td><b>{pct(m.prob, 1)}</b></td>
                <td className="dim">{pct(m.band[0])}–{pct(m.band[1])}</td>
                <td>{m.implied != null ? pct(m.implied, 1) : "—"}</td>
                <td style={{ color: m.edge_conservative > 0 ? "var(--good)" : m.edge_conservative != null ? "var(--bad)" : undefined }}>{m.edge_conservative != null ? `${(m.edge_conservative * 100).toFixed(1)} pp` : "—"}</td>
                <td>{m.stable_in_backtest == null ? "s/d" : m.stable_in_backtest ? "Sí" : "No"}</td>
                <td>{num(m.support_score * 100, 0)}</td>
                <td className="left small">{m.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="dim small mt-s">Respaldo = probabilidad conservadora × calidad de datos × estabilidad histórica de la familia de mercado. Ordenado de mayor a menor.</div>
    </>
  );
}

/* ----------------------------------------------------------- TRANSPARENCY */
function TransparencyTab({ p }: { p: PredictionPayload }) {
  const t = p.transparency;
  const c = p.context;
  return (
    <div className="grid g2">
      <div className="panel">
        <h3>Contexto del partido</h3>
        <div className="kv"><span>Fase</span><span>{PHASES[c.phase] ?? c.phase}</span></div>
        {c.first_leg && <div className="kv"><span>Eliminatoria</span><span>{c.first_leg.aggregate_text}</span></div>}
        {c.standings?.home && <div className="kv"><span>Clasificación {p.match.home.name}</span><span>{c.standings.home.pos}º · {c.standings.home.pts} pts</span></div>}
        {c.standings?.away && <div className="kv"><span>Clasificación {p.match.away.name}</span><span>{c.standings.away.pos}º · {c.standings.away.pts} pts</span></div>}
        <div className="kv"><span>Estadio</span><span>{c.venue ?? "n/d"}</span></div>
        <div className="kv"><span>Viaje del visitante</span><span>{c.travel_km_away != null ? `${c.travel_km_away} km desde ${c.away_home_city}` : "n/d"}</span></div>
        {c.weather ? (
          <>
            <div className="kv"><span>Temperatura</span><span>{c.weather.temperature_c != null ? `${num(c.weather.temperature_c, 1)} °C` : "pronóstico aún no disponible"}</span></div>
            <div className="kv"><span>Precipitación</span><span>{c.weather.precipitation_mm != null ? `${num(c.weather.precipitation_mm, 1)} mm` : "n/d"}</span></div>
            <div className="kv"><span>Viento</span><span>{c.weather.wind_kmh != null ? `${num(c.weather.wind_kmh, 0)} km/h` : "n/d"}</span></div>
            <div className="kv"><span>Altitud</span><span>{c.weather.elevation_m != null ? `${num(c.weather.elevation_m, 0)} m` : "n/d"}</span></div>
            <div className="dim small">Clima: {c.weather.source}, consultado {fmtDate(c.weather.fetched_at + "Z")}</div>
          </>
        ) : <div className="muted small">Clima no disponible.</div>}
        <ul className="list small muted mt-s">{c.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
      </div>
      <div className="panel">
        <h3>Fuentes</h3>
        {t.sources.map((s: Any, i: number) => (
          <div className="finding" key={i}><b>{s.name}</b><span className="muted small">{s.used_for} · <i>{s.type}</i></span></div>
        ))}
        <h3 className="mt">Qué es cada dato</h3>
        <ul className="list small">
          <li><b>Observado:</b> {t.labels.observed}</li>
          <li><b>Estimado por el modelo:</b> {t.labels.model}</li>
          <li><b>No confirmado:</b> {t.labels.unconfirmed}</li>
        </ul>
        <h3 className="mt">Datos no disponibles</h3>
        <ul className="list small muted">{t.not_available.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
        <h3 className="mt">Modelo</h3>
        <div className="kv"><span>Versión</span><span>{t.model_version}</span></div>
        <div className="kv"><span>Partidos de entrenamiento</span><span>{t.training_matches.toLocaleString("es-ES")}</span></div>
        <div className="kv"><span>Datos hasta</span><span>{t.training_until ? fmtDate(t.training_until + "Z") : "n/d"}</span></div>
        <div className="kv"><span>Vida media (decaimiento)</span><span>{t.hyperparameters.half_life_days} días</span></div>
        <div className="kv"><span>Peso xG-proxy en goles</span><span>{t.hyperparameters.xg_blend}</span></div>
        <div className="kv"><span>Penalización por equipo (λ)</span><span>{t.hyperparameters.lam_team}</span></div>
        <div className="kv"><span>Ajuste por estilo</span><span>{t.hyperparameters.style_adjust ? "activo" : "no (no mejoró en validación)"}</span></div>
      </div>
    </div>
  );
}
