import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type MatchItem } from "../api";
import {
  ConfidenceBadge, Crest, DecisionBadge, ErrorBox, Loading, PHASES, ProbBar, fmtDay, fmtTime, num, pct, useAsync,
} from "../components/ui";

type Range = "upcoming" | "today" | "week" | "results" | "custom";
type Market = "1x2" | "goals" | "btts" | "corners" | "sot";

const MARKETS: { k: Market; label: string }[] = [
  { k: "1x2", label: "1X2" },
  { k: "goals", label: "Goles" },
  { k: "btts", label: "Ambos marcan" },
  { k: "corners", label: "Córners" },
  { k: "sot", label: "Tiros a puerta" },
];

const iso = (d: Date) => d.toISOString().slice(0, 19);
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const dayKey = (s: string) => new Date(s).toLocaleDateString("en-CA");

export default function Home() {
  const [range, setRange] = useState<Range>("upcoming");
  const [team, setTeam] = useState("");
  const [market, setMarket] = useState<Market>("1x2");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const teams = useAsync(() => api.teams(), []);

  const params = useMemo(() => {
    const now = new Date();
    const start = new Date(now); start.setHours(0, 0, 0, 0);
    const p: { from?: string; to?: string; team?: string } = { team: team || undefined };
    if (range === "today") { p.from = iso(start); p.to = iso(new Date(start.getTime() + 86400e3)); }
    if (range === "week") { p.from = iso(start); p.to = iso(new Date(start.getTime() + 7 * 86400e3)); }
    if (range === "upcoming") { p.from = iso(new Date(now.getTime() - 3 * 3600e3)); p.to = iso(new Date(now.getTime() + 45 * 86400e3)); }
    if (range === "results") { p.from = iso(new Date(now.getTime() - 60 * 86400e3)); p.to = iso(now); }
    if (range === "custom") { p.from = from || undefined; p.to = to ? `${to}T23:59:59` : undefined; }
    return p;
  }, [range, team, from, to]);

  const matches = useAsync(() => api.matches(params), [JSON.stringify(params)]);

  const list = useMemo(() => {
    const ms = matches.data ?? [];
    return range === "results" ? [...ms].reverse() : ms;
  }, [matches.data, range]);

  const today = dayKey(new Date().toISOString());
  const todays = list.filter((m) => dayKey(m.kickoff) === today && !m.completed);
  const showToday = !matches.loading && range !== "today" && range !== "results" && todays.length > 0;
  const groups = useMemo(() => {
    const g = new Map<string, MatchItem[]>();
    list.forEach((m) => {
      const k = dayKey(m.kickoff);
      if (!g.has(k)) g.set(k, []);
      g.get(k)!.push(m);
    });
    return [...g.entries()];
  }, [list]);

  return (
    <div className="mt">
      <div className="row between wrap mb">
        <div>
          <h1>Calendario Champions League</h1>
          <div className="muted">Probabilidades del modelo para cada enfrentamiento · selecciona un partido para el análisis completo</div>
        </div>
      </div>

      <div className="panel mb">
        <div className="filters">
          {([
            ["upcoming", "Próximos"], ["today", "Hoy"], ["week", "7 días"], ["results", "Resultados"], ["custom", "Fechas"],
          ] as [Range, string][]).map(([k, l]) => (
            <button key={k} className={`chip ${range === k ? "on" : ""}`} onClick={() => setRange(k)}>{l}</button>
          ))}
          {range === "custom" && (
            <>
              <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} aria-label="Desde" />
              <input type="date" value={to} onChange={(e) => setTo(e.target.value)} aria-label="Hasta" />
            </>
          )}
          <select value={team} onChange={(e) => setTeam(e.target.value)} aria-label="Equipo">
            <option value="">Todos los equipos</option>
            {(teams.data ?? []).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <select value={market} onChange={(e) => setMarket(e.target.value as Market)} aria-label="Mercado">
            {MARKETS.map((m) => <option key={m.k} value={m.k}>Mercado: {m.label}</option>)}
          </select>
        </div>
      </div>

      {matches.loading && <Loading text="Cargando calendario…" />}
      {matches.error && <ErrorBox error={matches.error} />}

      {showToday && (
        <section className="mb">
          <h2>Partidos de hoy</h2>
          <div className="grid g2">{todays.map((m) => <MatchCard key={m.id} m={m} market={market} />)}</div>
        </section>
      )}

      {!matches.loading && list.length === 0 && !matches.error && (
        <div className="notice info">No hay partidos en este rango.</div>
      )}

      {groups.filter(([k]) => !(showToday && k === today)).map(([k, ms]) => (
        <section key={k} className="mb">
          <h3 style={{ textTransform: "none" }}>{cap(fmtDay(ms[0].kickoff))}</h3>
          <div className="grid g2">{ms.map((m) => <MatchCard key={m.id} m={m} market={market} />)}</div>
        </section>
      ))}
    </div>
  );
}

function MarketLine({ m, market }: { m: MatchItem; market: Market }) {
  const s = m.summary!;
  if (market === "1x2")
    return <ProbBar h={s.home_win} d={s.draw} a={s.away_win} />;
  if (market === "goals")
    return <div className="row between mt-s small"><span className="muted">xG {num(s.home_xg)} – {num(s.away_xg)}</span><span>+2.5: <b>{pct(s.over_2_5)}</b></span><span className="muted">Marcador: {s.most_likely_score}</span></div>;
  if (market === "btts")
    return <div className="row between mt-s small"><span className="muted">Ambos marcan</span><b>Sí {pct(s.btts)} · No {pct(1 - s.btts)}</b></div>;
  if (market === "corners")
    return <div className="row between mt-s small"><span className="muted">Córners esperados</span><b>{num(s.corners_total, 1)}</b></div>;
  return <div className="row between mt-s small"><span className="muted">Tiros a puerta esperados</span><b>{num(s.sot_total, 1)}</b></div>;
}

function MatchCard({ m, market }: { m: MatchItem; market: Market }) {
  return (
    <Link to={`/partido/${m.id}`} className="match-card">
      <div className="teams">
        <div className="team"><Crest team={m.home} /><span>{m.home.name}</span></div>
        <div className="mid">
          {m.completed ? (
            <div className="t">{m.home_goals} - {m.away_goals}</div>
          ) : (
            <div className="t">{fmtTime(m.kickoff)}</div>
          )}
          <div className="dim small">{m.completed ? "Final" : m.status === "STATUS_SCHEDULED" ? "" : m.status.replace("STATUS_", "")}</div>
        </div>
        <div className="team away"><Crest team={m.away} /><span>{m.away.name}</span></div>
      </div>
      {m.summary ? (
        <MarketLine m={m} market={market} />
      ) : (
        <div className="dim small mt-s">{m.completed ? "Partido disputado" : "Pronóstico aún no calculado (se calcula al abrir el partido o en la próxima actualización)"}</div>
      )}
      <div className="meta">
        {m.phase && <span className="badge">{PHASES[m.phase] ?? m.phase}</span>}
        {m.summary && <ConfidenceBadge level={m.summary.confidence} />}
        {m.summary && !m.completed && <DecisionBadge d={m.summary.decision} />}
        {m.venue && <span className="dim small">{m.venue}</span>}
      </div>
    </Link>
  );
}
