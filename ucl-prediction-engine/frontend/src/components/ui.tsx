import { useEffect, useState } from "react";
import type { TeamRef } from "../api";

export const pct = (x: number | null | undefined, d = 0) =>
  x === null || x === undefined || Number.isNaN(x) ? "n/d" : `${(x * 100).toFixed(d)}%`;
export const num = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "n/d" : x.toFixed(d);

export const fmtDate = (iso: string, opts: Intl.DateTimeFormatOptions = {}) =>
  new Date(iso).toLocaleString("es-ES", {
    weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", ...opts,
  });
export const fmtDay = (iso: string) =>
  new Date(iso).toLocaleDateString("es-ES", { weekday: "long", day: "numeric", month: "long" });
export const fmtTime = (iso: string) =>
  new Date(iso).toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });

export const PHASES: Record<string, string> = {
  "league-phase": "Fase de liga",
  "knockout-round-playoffs": "Playoff eliminatorio",
  "round-of-16": "Octavos de final",
  quarterfinals: "Cuartos de final",
  semifinals: "Semifinales",
  final: "Final",
};

export function Crest({ team, size = "" }: { team: Partial<TeamRef>; size?: "" | "lg" | "sm" }) {
  const [err, setErr] = useState(false);
  const px = size === "lg" ? 72 : size === "sm" ? 22 : 36;
  // En la instantánea de un solo archivo los escudos van incrustados una vez y se buscan por ruta.
  const crests = (window as unknown as { __CRESTS__?: Record<string, string> }).__CRESTS__;
  const src = team.logo ? crests?.[team.logo] ?? team.logo : null;
  if (!src || err)
    return (
      <span className={`crest crest-fallback ${size}`} style={{ width: px, height: px }}>
        {(team.abbreviation || team.name || "?").slice(0, 3)}
      </span>
    );
  return <img className={`crest ${size}`} src={src} alt={team.name} loading="lazy" onError={() => setErr(true)} />;
}

export function ConfidenceBadge({ level }: { level: string }) {
  return <span className={`badge ${level.toLowerCase()}`}>Confianza {level}</span>;
}

export function DecisionBadge({ d }: { d: string }) {
  return <span className={`badge ${d === "BET" ? "bet" : "nobet"}`}>{d}</span>;
}

export function ProbBar({ h, d, a, labels }: { h: number; d: number; a: number; labels?: [string, string, string] }) {
  return (
    <>
      <div className="pbar" role="img" aria-label={`Local ${pct(h)}, empate ${pct(d)}, visitante ${pct(a)}`}>
        <div className="h" style={{ width: `${h * 100}%` }} />
        <div className="d" style={{ width: `${d * 100}%` }} />
        <div className="a" style={{ width: `${a * 100}%` }} />
      </div>
      <div className="plabels">
        <span>{labels?.[0] ?? "1"} <b>{pct(h)}</b></span>
        <span>{labels?.[1] ?? "X"} <b>{pct(d)}</b></span>
        <span>{labels?.[2] ?? "2"} <b>{pct(a)}</b></span>
      </div>
    </>
  );
}

export function LineProb({ label, p }: { label: string; p: number }) {
  return (
    <div className="lineprob">
      <span className="lbl">{label}</span>
      <div className="track"><div className="fill" style={{ width: `${p * 100}%` }} /></div>
      <span className="val">{pct(p)}</span>
    </div>
  );
}

export function Compare({ label, h, a, d = 2, higherIsBetter = true }: {
  label: string; h: number | null | undefined; a: number | null | undefined; d?: number; higherIsBetter?: boolean;
}) {
  const hv = h ?? 0;
  const av = a ?? 0;
  const max = Math.max(hv, av, 1e-9);
  const better = (x: number, y: number) => (higherIsBetter ? x > y : x < y);
  return (
    <div className="cmp">
      <span className="v" style={{ color: better(hv, av) ? "var(--home)" : undefined }}>{num(h, d)}</span>
      <div className="bar l"><div style={{ width: `${(hv / max) * 100}%` }} /></div>
      <span className="lbl">{label}</span>
      <div className="bar r"><div style={{ width: `${(av / max) * 100}%` }} /></div>
      <span className="v r" style={{ color: better(av, hv) ? "var(--away)" : undefined }}>{num(a, d)}</span>
    </div>
  );
}

export function Loading({ text = "Cargando…" }: { text?: string }) {
  return <div className="loading"><div className="spinner" />{text}</div>;
}

export function ErrorBox({ error }: { error: string }) {
  return <div className="notice bad">{error}</div>;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, set] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    let alive = true;
    set((s) => ({ ...s, loading: true, error: undefined }));
    fn()
      .then((data) => alive && set({ data, loading: false }))
      .catch((e: Error) => alive && set({ error: e.message, loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
