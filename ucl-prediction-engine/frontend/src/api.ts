/* Cliente de la API. El navegador solo habla con nuestro backend; las claves viven en el servidor. */

export interface TeamRef {
  id: string;
  name: string;
  short_name?: string | null;
  abbreviation?: string | null;
  logo?: string | null;
  color?: string | null;
}

export interface MatchSummary {
  home_win: number;
  draw: number;
  away_win: number;
  home_xg: number;
  away_xg: number;
  over_2_5: number;
  btts: number;
  most_likely_score: string;
  corners_total: number;
  sot_total: number;
  confidence: "Alta" | "Media" | "Baja";
  decision: "BET" | "NO BET";
  generated_at: string;
}

export interface MatchItem {
  id: string;
  league: string;
  season: string | null;
  phase: string | null;
  kickoff: string;
  status: string;
  completed: boolean;
  home: TeamRef;
  away: TeamRef;
  home_goals: number | null;
  away_goals: number | null;
  venue: string | null;
  city: string | null;
  country: string | null;
  summary?: MatchSummary;
}

// El payload del pronóstico es amplio; se tipa lo esencial y el resto como Record.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Any = any;

export interface PredictionPayload {
  match: MatchItem;
  prediction: Any;
  confidence: { level: "Alta" | "Media" | "Baja"; reasons: { level: string; text: string }[] };
  tactical: Any;
  teams: { home: Any; away: Any };
  availability: Any;
  context: Any;
  opportunities: Any;
  odds: Any;
  transparency: Any;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = `Error ${r.status}`;
    try {
      const j = await r.json();
      msg = j.detail || msg;
    } catch {
      /* sin cuerpo JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

/* Modo instantánea (VITE_STATIC=1): lee JSON exportados junto a la página, sin backend. */
export const STATIC = import.meta.env.VITE_STATIC === "1";
const SNAP = "snapshot/";
const cache = new Map<string, Promise<unknown>>();
function snap<T>(file: string, missing = "Dato no incluido en la instantánea"): Promise<T> {
  // Instantánea incrustada en la propia página (un solo archivo, sin peticiones de red).
  const embedded = (window as unknown as { __SNAPSHOT__?: Record<string, unknown> }).__SNAPSHOT__;
  if (embedded) {
    return file in embedded ? Promise.resolve(embedded[file] as T) : Promise.reject(new Error(missing));
  }
  if (!cache.has(file)) {
    cache.set(file, fetch(SNAP + file).then((r) => {
      if (!r.ok) throw new Error(missing);
      return r.json();
    }));
  }
  return cache.get(file) as Promise<T>;
}

const staticApi = {
  matches: async (params: { from?: string; to?: string; team?: string }) => {
    const all = await snap<MatchItem[]>("matches.json");
    const parse = (x: string) =>
      new Date(x.length === 10 ? `${x}T00:00:00Z` : /(Z|[+-]\d\d:\d\d)$/.test(x) ? x : `${x}Z`).getTime();
    const f = params.from ? parse(params.from) : -Infinity;
    const t = params.to ? parse(params.to) : Infinity;
    return all.filter((m) => {
      const k = new Date(m.kickoff).getTime();
      return k >= f && k < t && (!params.team || m.home.id === params.team || m.away.id === params.team);
    });
  },
  match: (id: string) => snap<PredictionPayload>(`match-${id}.json`,
    "Este partido no tiene pronóstico en la instantánea (solo se incluyen los próximos partidos)."),
  history: async (id: string) => ((await snap<Any>(`match-${id}.json`))._history ?? []) as Any[],
  teams: () => snap<{ id: string; name: string; logo: string | null; domestic_league: string | null }[]>("teams.json"),
  standings: () => snap<Any[]>("standings.json"),
  backtest: () => snap<Any>("backtest.json"),
  status: () => snap<Any>("status.json"),
};

const liveApi = {
  matches: (params: { from?: string; to?: string; team?: string }) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && q.set(k, v));
    return get<MatchItem[]>(`/api/matches?${q.toString()}`);
  },
  match: (id: string) => get<PredictionPayload>(`/api/matches/${id}`),
  history: (id: string) => get<Any[]>(`/api/matches/${id}/history`),
  teams: () => get<{ id: string; name: string; logo: string | null; domestic_league: string | null }[]>("/api/teams"),
  standings: () => get<Any[]>("/api/standings"),
  backtest: () => get<Any>("/api/backtest"),
  status: () => get<Any>("/api/status"),
};

export const api = STATIC ? staticApi : liveApi;
