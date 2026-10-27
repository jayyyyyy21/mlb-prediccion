"""Disponibilidad de jugadores e impacto estimado de las bajas.

Clasificación (no se considera baja confirmada sin información oficial):
* confirmada   – reporte oficial NFL "Out" para ese partido, o jugador en lista
                 de reserva (IR/PUP/NFI/suspensión) según el roster oficial o
                 la transacción publicada.
* probable     – "Doubtful" oficial, u "Out"/"Doubtful" reportado por medios
                 (ESPN) antes de que exista el reporte oficial de esa semana.
* cuestionable – "Questionable".
* disponible   – sin designación.

Impacto en puntos:
* QB: diferencia de EPA por dropback (con encogimiento bayesiano) entre el
  titular y su reemplazo, multiplicada por los dropbacks por partido del equipo.
* Otras posiciones: valor posicional a priori x porcentaje de snaps reciente.
  Estos valores a priori NO se calibran en el backtest (no hay reportes
  históricos fiables tal como se conocían antes de cada partido); se muestran
  explícitamente para que el usuario los evalúe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db

# Puntos por partido que pierde (ofensiva) o concede de más (defensa) un equipo
# cuando falta un titular con 100% de snaps y entra un reemplazo típico.
POSITION_VALUE = {
    "WR": 0.55, "TE": 0.40, "RB": 0.30, "HB": 0.30, "FB": 0.10,
    "T": 0.45, "OT": 0.45, "G": 0.30, "C": 0.35, "OL": 0.35,
    "DE": 0.45, "EDGE": 0.45, "OLB": 0.40, "DT": 0.30, "NT": 0.25, "DL": 0.35,
    "LB": 0.25, "ILB": 0.25, "MLB": 0.25,
    "CB": 0.45, "S": 0.30, "FS": 0.30, "SS": 0.30, "DB": 0.30,
}
OFFENSE = {"QB", "WR", "TE", "RB", "HB", "FB", "T", "OT", "G", "C", "OL"}
UNIT_CAP = 4.0          # tope de impacto acumulado por unidad (sin contar QB)
QB_PRIOR_DB = 250.0     # dropbacks de encogimiento para EPA del QB
QB_CAP = 10.0
RESERVE_STATUS = {"RES", "PUP", "NFI", "SUS", "IR"}
POSITION_ES = {
    "QB": "Quarterback", "WR": "Receptor", "TE": "Ala cerrada", "RB": "Corredor", "HB": "Corredor", "FB": "Fullback",
    "T": "Tackle ofensivo", "OT": "Tackle ofensivo", "G": "Guardia", "C": "Centro", "OL": "Línea ofensiva",
    "DE": "Ala defensiva", "DT": "Tackle defensivo", "NT": "Nose tackle", "DL": "Línea defensiva",
    "LB": "Linebacker", "OLB": "Linebacker externo", "ILB": "Linebacker interno", "MLB": "Linebacker",
    "CB": "Cornerback", "S": "Safety", "FS": "Safety", "SS": "Safety", "DB": "Back defensivo",
    "K": "Pateador", "PK": "Pateador", "P": "Despejador", "LS": "Long snapper",
}


class InjuryData:
    """Carga una vez las tablas necesarias para evaluar disponibilidad."""

    def __init__(self) -> None:
        self.official = db.read_table("injuries_official")
        self.espn = db.read_table("injuries_espn")
        self.rosters = db.read_table("rosters")
        self.snaps = db.read_table("snaps")
        self.qb = db.read_table("qb_games")
        self._qb_rates = self._qb_epa_rates()

    # ---------------- QB ----------------
    def _qb_epa_rates(self) -> pd.DataFrame:
        if self.qb.empty:
            return pd.DataFrame()
        q = self.qb.copy()
        seasons = sorted(q["season"].unique())
        recent = q[q["season"] >= seasons[-1] - 2]
        per = recent.groupby("gsis_id").agg(db=("dropbacks", "sum"), epa=("qb_epa_sum", "sum"),
                                           name=("passer", "last")).reset_index()
        # Nivel de reemplazo: EPA/dropback de QBs con poco volumen (suplentes).
        backups = per[per["db"] < 300]
        self.replacement_epa = float(backups["epa"].sum() / max(backups["db"].sum(), 1)) if len(backups) else -0.15
        per["epa_shrunk"] = (per["epa"] + self.replacement_epa * QB_PRIOR_DB) / (per["db"] + QB_PRIOR_DB)
        return per.set_index("gsis_id")

    def qb_depth(self, team: str, season: int, week: int) -> list[dict]:
        """QBs del equipo ordenados por titularidad reciente (dropbacks en sus últimos 3 partidos)."""
        q = self.qb[(self.qb["team"] == team) & (self.qb["season"] == season) & (self.qb["week"] < week)]
        last_games = sorted(q["game_id"].unique(), key=lambda g: q[q["game_id"] == g]["week"].iloc[0])[-3:]
        recent = q[q["game_id"].isin(last_games)].groupby("gsis_id")["dropbacks"].sum()
        roster_qbs = self._roster(team, season, week)
        roster_qbs = roster_qbs[roster_qbs["position"] == "QB"] if not roster_qbs.empty else roster_qbs
        ids = set(recent.index) | set(roster_qbs["gsis_id"].dropna()) if not roster_qbs.empty else set(recent.index)
        out = []
        for gid in ids:
            rate = self._qb_rates.loc[gid] if gid in self._qb_rates.index else None
            name = None
            if not roster_qbs.empty and gid in set(roster_qbs["gsis_id"]):
                name = roster_qbs[roster_qbs["gsis_id"] == gid]["full_name"].iloc[0]
            if name is None and rate is not None:
                name = rate["name"]
            out.append({
                "gsis_id": gid, "name": name or gid,
                "recent_db": int(recent.get(gid, 0)),
                "career_db": int(rate["db"]) if rate is not None else 0,
                "epa_db": float(rate["epa_shrunk"]) if rate is not None else self.replacement_epa,
                "on_roster": bool(not roster_qbs.empty and gid in set(roster_qbs["gsis_id"])),
            })
        out.sort(key=lambda r: (-r["recent_db"], -r["career_db"]))
        return out

    def team_dropbacks_pg(self, team: str, season: int) -> float:
        q = self.qb[(self.qb["team"] == team) & (self.qb["season"] == season)]
        if q.empty:
            return 36.0
        return float(q.groupby("game_id")["dropbacks"].sum().mean())

    # ---------------- Roster / snaps ----------------
    def _roster(self, team: str, season: int, week: int) -> pd.DataFrame:
        r = self.rosters
        if r.empty:
            return r
        r = r[(r["team"] == team) & (r["season"] == season)]
        if r.empty:
            return r
        wk = r[r["week"] <= week]["week"].max() if (r["week"] <= week).any() else r["week"].min()
        return r[r["week"] == wk]

    def snap_shares(self, team: str, season: int, week: int) -> pd.DataFrame:
        s = self.snaps
        if s.empty:
            return pd.DataFrame()
        s = s[(s["team"] == team) & (s["season"] == season) & (s["week"] < week)]
        games = sorted(s["week"].unique())[-3:]
        s = s[s["week"].isin(games)]
        if s.empty:
            return pd.DataFrame()
        n = len(games)
        g = s.groupby(["pfr_player_id", "player", "position"]).agg(off=("offense_pct", "sum"), deff=("defense_pct", "sum")).reset_index()
        g["off"] /= n
        g["deff"] /= n
        g["share"] = np.where(g["position"].isin(OFFENSE), g["off"], g["deff"])
        return g

    # ---------------- Estado ----------------
    def official_week_published(self, team: str, season: int, week: int) -> bool:
        o = self.official
        return bool(not o.empty and ((o["team"] == team) & (o["season"] == season) & (o["week"] == week)).any())

    def statuses(self, team: str, season: int, week: int) -> pd.DataFrame:
        """Estado de cada jugador con designación (oficial o reportada)."""
        rows = []
        official = self.official_week_published(team, season, week)
        roster = self._roster(team, season, week)
        id_map = roster[["gsis_id", "espn_id", "pfr_id", "full_name", "position"]].copy() if not roster.empty else pd.DataFrame()
        if official:
            o = self.official[(self.official["team"] == team) & (self.official["season"] == season) & (self.official["week"] == week)]
            for r in o.itertuples(index=False):
                st = r.report_status if isinstance(r.report_status, str) else None
                if not st:
                    continue
                cat = {"Out": "confirmada", "Doubtful": "probable", "Questionable": "cuestionable"}.get(st, "disponible")
                rows.append({"gsis_id": r.gsis_id, "name": r.full_name, "position": r.position, "status": st,
                             "category": cat, "source": "Reporte oficial NFL",
                             "detail": r.report_primary_injury if isinstance(r.report_primary_injury, str) else ""})
        # Listas de reserva (transacción oficial) según roster semanal.
        if not roster.empty:
            res = roster[roster["status"].isin(RESERVE_STATUS)]
            for r in res.itertuples(index=False):
                if any(x["gsis_id"] == r.gsis_id for x in rows):
                    continue
                rows.append({"gsis_id": r.gsis_id, "name": r.full_name, "position": r.position,
                             "status": f"Reserva ({r.status_description_abbr or r.status})", "category": "confirmada",
                             "source": "Roster oficial (lista de reserva)", "detail": ""})
        # ESPN: transacciones (IR/suspensión) y reportes de medios antes del reporte oficial.
        if not self.espn.empty:
            e = self.espn[self.espn["team"] == team]
            for r in e.itertuples(index=False):
                gid = None
                if not id_map.empty and r.espn_id is not None:
                    m = id_map[id_map["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True) == str(r.espn_id)]
                    if len(m):
                        gid = m["gsis_id"].iloc[0]
                if gid and any(x["gsis_id"] == gid for x in rows):
                    continue
                st = r.status
                if st in ("Injured Reserve", "Suspension"):
                    cat = "confirmada"
                elif official:
                    continue  # con reporte oficial publicado, manda el reporte oficial
                elif st in ("Out", "Doubtful"):
                    cat = "probable"
                elif st == "Questionable":
                    cat = "cuestionable"
                else:
                    continue
                rows.append({"gsis_id": gid, "name": r.full_name, "position": r.position, "status": st,
                             "category": cat, "source": "ESPN (no oficial)" if cat != "confirmada" else "ESPN (transacción)",
                             "detail": r.injury_type if isinstance(r.injury_type, str) else ""})
        return pd.DataFrame(rows)


def _backup_order(qbs: list[dict]) -> list[dict]:
    """Suplente estimado: experiencia acumulada (dropbacks) con más peso a los recientes.

    Sin depth chart oficial en los datos abiertos, es una aproximación; la
    interfaz muestra qué suplente se asumió.
    """
    return sorted(qbs, key=lambda q: -(q["career_db"] + 3 * q["recent_db"]))


def evaluate_team(data: InjuryData, team: str, season: int, week: int,
                  forced_out: list[str] | None = None, include_probable: bool = False) -> dict:
    """Impacto estimado de las bajas de `team`.

    Devuelve ajustes para la ofensiva propia (off_adj, negativo = menos puntos)
    y para la ofensiva rival (def_adj, positivo = el rival anota más), además
    de la lista de jugadores evaluados.
    """
    forced_out = forced_out or []
    st = data.statuses(team, season, week)
    shares = data.snap_shares(team, season, week)
    roster = data._roster(team, season, week)
    pfr_by_gsis = dict(zip(roster["gsis_id"], roster["pfr_id"])) if not roster.empty else {}
    name_by_gsis = dict(zip(roster["gsis_id"], roster["full_name"])) if not roster.empty else {}
    pos_by_gsis = dict(zip(roster["gsis_id"], roster["position"])) if not roster.empty else {}

    players = []
    if not st.empty:
        players = st.to_dict("records")
    for gid in forced_out:  # simulador: ausencia hipotética
        existing = next((p for p in players if p["gsis_id"] == gid), None)
        if existing:
            existing.update(category="simulada", source="Simulación del usuario")
        else:
            players.append({"gsis_id": gid, "name": name_by_gsis.get(gid, gid), "position": pos_by_gsis.get(gid, "?"),
                            "status": "Baja simulada", "category": "simulada", "source": "Simulación del usuario", "detail": ""})

    applied = {"confirmada", "simulada"} | ({"probable"} if include_probable else set())
    qbs = data.qb_depth(team, season, week)
    unavailable_ids = {p["gsis_id"] for p in players if p["category"] in applied}
    off_adj = def_adj = 0.0
    off_other = def_other = 0.0
    qb_info = None
    if qbs:
        starter = qbs[0]
        healthy = _backup_order([q for q in qbs[1:] if q["gsis_id"] not in unavailable_ids and q["on_roster"]])
        dbpg = data.team_dropbacks_pg(team, season)
        qb_info = {"starter": starter["name"], "starter_epa_db": starter["epa_db"], "dropbacks_pg": dbpg}
        if starter["gsis_id"] in unavailable_ids:
            repl = healthy[0] if healthy else None
            repl_epa = repl["epa_db"] if repl else data.replacement_epa
            delta = float(np.clip((repl_epa - starter["epa_db"]) * dbpg, -QB_CAP, QB_CAP))
            off_adj += delta
            qb_info.update(replacement=repl["name"] if repl else "Reemplazo genérico", replacement_epa_db=repl_epa,
                           impact=delta)

    for p in players:
        pos = p.get("position") or "?"
        gid = p.get("gsis_id")
        share = 0.0
        pfr = pfr_by_gsis.get(gid)
        if not shares.empty and pfr is not None:
            m = shares[shares["pfr_player_id"] == pfr]
            share = float(m["share"].iloc[0]) if len(m) else 0.0
        elif not shares.empty and p.get("name"):
            m = shares[shares["player"] == p["name"]]
            share = float(m["share"].iloc[0]) if len(m) else 0.0
        p["snap_share"] = round(share, 3)
        if pos == "QB":
            impact = 0.0
            if qbs and gid == qbs[0]["gsis_id"]:
                starter_epa = qbs[0]["epa_db"]
                others = _backup_order([q for q in qbs[1:] if q["on_roster"] and q["gsis_id"] not in unavailable_ids])
                repl_epa = others[0]["epa_db"] if others else data.replacement_epa
                impact = float(np.clip((repl_epa - starter_epa) * data.team_dropbacks_pg(team, season), -QB_CAP, QB_CAP))
            p["impact"] = round(impact, 2)
            p["side"] = "ofensiva"
        else:
            val = POSITION_VALUE.get(pos, 0.0)
            is_off = pos in OFFENSE
            impact = val * share
            p["impact"] = round(-impact if is_off else impact, 2)
            p["side"] = "ofensiva" if is_off else "defensa"
            if p["category"] in applied:
                if is_off:
                    off_other += impact
                else:
                    def_other += impact
        p["applied"] = p["category"] in applied
        p["position_es"] = POSITION_ES.get(pos, pos)
    off_adj -= min(off_other, UNIT_CAP)
    def_adj += min(def_other, UNIT_CAP)

    # Escenario: impacto adicional si los jugadores NO confirmados no juegan.
    scen_off = scen_def = 0.0
    for p in players:
        if p["category"] in ("probable", "cuestionable") and not p["applied"]:
            if p["side"] == "ofensiva":
                scen_off += p["impact"]
            else:
                scen_def += p["impact"]
    players.sort(key=lambda p: -abs(p.get("impact", 0)))
    return {
        "team": team,
        "official_report": data.official_week_published(team, season, week),
        "off_adj": round(off_adj, 2),
        "def_adj": round(def_adj, 2),
        "scenario_off": round(scen_off, 2),
        "scenario_def": round(scen_def, 2),
        "qb": qb_info,
        "qb_depth": qbs[:4],
        "players": players,
        "unconfirmed_qb": any(p["position"] == "QB" and p["category"] in ("probable", "cuestionable")
                              and qbs and p["gsis_id"] == qbs[0]["gsis_id"] for p in players),
    }


def key_players(data: InjuryData, team: str, season: int, week: int, n: int = 22) -> list[dict]:
    """Jugadores con más snaps recientes (para el simulador de ausencias)."""
    shares = data.snap_shares(team, season, week)
    roster = data._roster(team, season, week)
    out = []
    for q in data.qb_depth(team, season, week)[:3]:
        out.append({"gsis_id": q["gsis_id"], "name": q["name"], "position": "QB", "snap_share": None,
                    "note": f"{q['recent_db']} dropbacks recientes"})
    if shares.empty or roster.empty:
        return out
    m = shares.merge(roster[["pfr_id", "gsis_id"]], left_on="pfr_player_id", right_on="pfr_id", how="inner")
    m = m[(m["position"] != "QB") & m["position"].isin(POSITION_VALUE)].sort_values("share", ascending=False).head(n)
    for r in m.itertuples(index=False):
        out.append({"gsis_id": r.gsis_id, "name": r.player, "position": r.position,
                    "snap_share": round(float(r.share), 3), "note": ""})
    return out
