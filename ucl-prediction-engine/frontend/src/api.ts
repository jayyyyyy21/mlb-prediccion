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

export const api = {
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
