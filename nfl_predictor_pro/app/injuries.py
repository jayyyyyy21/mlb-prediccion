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
* QB: el titular esperado sale del depth chart de ESPN (o, si no está
  disponible, de los dropbacks recientes) y de las bajas aplicadas. Su efecto
  entra al modelo como variable calibrada en el backtest (ver qb.py), no como
  un ajuste fijo.
* Otras posiciones: valor posicional a priori x porcentaje de snaps reciente.
  Estos valores a priori NO se calibran en el backtest (no hay reportes
  históricos fiables tal como se conocían antes de cada partido); se muestran
  explícitamente para que el usuario los evalúe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db
from .qb import QBModel

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
GENERIC_QB = "__reemplazo__"   # QB desconocido: se asume nivel de reemplazo
SRC_DEPTH = "Depth chart de ESPN"
SRC_RECENT = "Estimado por dropbacks recientes (sin depth chart)"
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

    def __init__(self, qbm: QBModel | None = None) -> None:
        self.official = db.read_table("injuries_official")
        self.espn = db.read_table("injuries_espn")
        self.rosters = db.read_table("rosters")
        self.snaps = db.read_table("snaps")
        self.qb = db.read_table("qb_games")
        self.depth = db.read_table("depth_qb")
        self.qbm = qbm or QBModel(self.qb)

    # ---------------- QB ----------------
    def qb_depth(self, team: str, season: int, week: int) -> list[dict]:
        """QBs del equipo en orden de titularidad esperada.

        Con depth chart de ESPN: ese orden. Sin él: primero quien más dropbacks
        tuvo en los últimos 3 partidos y después los suplentes por experiencia.
        """
        qsnap = self.qbm.at(season, week)
        q = self.qb[(self.qb["team"] == team) & (self.qb["season"] == season) & (self.qb["week"] < week)]
        last_games = list(q.sort_values("week")["game_id"].drop_duplicates())[-3:]
        recent = q[q["game_id"].isin(last_games)].groupby("gsis_id")["dropbacks"].sum()
        roster = self._roster(team, season, week)
        roster_qbs = roster[roster["position"] == "QB"] if not roster.empty else roster
        names = dict(zip(roster_qbs["gsis_id"], roster_qbs["full_name"])) if not roster_qbs.empty else {}
        depth = self.depth[self.depth["team"] == team].sort_values("rank") if not self.depth.empty else pd.DataFrame()
        rank: dict[str, int] = {}
        for r in depth.itertuples(index=False):
            gid = r.gsis_id if isinstance(r.gsis_id, str) else f"espn:{r.espn_id}"
            rank.setdefault(gid, int(r.rank))
            names.setdefault(gid, r.full_name)
        ids = set(recent.index) | set(names)
        out = []
        for gid in ids:
            out.append({
                "gsis_id": gid, "name": names.get(gid) or qsnap.name(gid) or gid,
                "recent_db": int(recent.get(gid, 0)),
                "career_db": qsnap.volume(gid),
                "epa_db": qsnap.level(gid),
                "on_roster": gid in names,
                "depth_rank": rank.get(gid),
            })
        if rank:
            out.sort(key=lambda r: (r["depth_rank"] is None, r["depth_rank"] or 0, -r["recent_db"], -r["career_db"]))
            source = SRC_DEPTH
        else:
            out.sort(key=lambda r: (-r["recent_db"], -r["career_db"]))
            out = out[:1] + _backup_order(out[1:])
            source = SRC_RECENT
        for r in out:
            r["source"] = source
        return out

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

    Sólo se usa cuando no hay depth chart disponible; la interfaz muestra qué
    suplente se asumió.
    """
    return sorted(qbs, key=lambda q: -(q["career_db"] + 3 * q["recent_db"]))


def evaluate_team(data: InjuryData, team: str, season: int, week: int,
                  forced_out: list[str] | None = None, include_probable: bool = False,
                  qb_points=None) -> dict:
    """Impacto estimado de las bajas de `team`.

    Devuelve ajustes de posiciones distintas al QB para la ofensiva propia
    (off_adj, negativo = menos puntos) y para la ofensiva rival (def_adj,
    positivo = el rival anota más), el QB titular esperado y la lista de
    jugadores evaluados. `qb_points(de, a)` da los puntos que cambia la
    proyección al sustituir un QB por otro según el modelo calibrado.
    """
    forced_out = forced_out or []
    qsnap = data.qbm.at(season, week)
    if qb_points is None:  # sin modelo: estimación bruta EPA x dropbacks
        def qb_points(a, b):
            return (qsnap.level(b) - qsnap.level(a)) * qsnap.dropbacks(team)
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
    # Jugadores del reporte sin id vinculado: se emparejan por nombre con el depth chart.
    unavailable_names = {p["name"] for p in players if p["category"] in applied and not p.get("gsis_id")}

    def available(q):
        return q["on_roster"] and q["gsis_id"] not in unavailable_ids and q["name"] not in unavailable_names

    def next_after(skip_id):
        return next((q for q in qbs if q["gsis_id"] != skip_id and available(q)), None)

    off_adj = def_adj = 0.0
    off_other = def_other = 0.0
    qb_info = None
    if qbs:
        starter = qbs[0]
        expected = next((q for q in qbs if available(q)), None)
        exp_id = expected["gsis_id"] if expected else GENERIC_QB
        recent_qb = max(qbs, key=lambda q: q["recent_db"])
        qb_info = {
            "starter": starter["name"], "starter_id": starter["gsis_id"], "starter_epa_db": starter["epa_db"],
            "expected": expected["name"] if expected else "Reemplazo genérico", "expected_id": exp_id,
            "expected_epa_db": qsnap.level(exp_id), "source": starter["source"],
            "dropbacks_pg": qsnap.dropbacks(team), "ratings_epa_db": qsnap.base(team),
            "recent": recent_qb["name"] if recent_qb["recent_db"] > 0 else None,
            "recent_differs": bool(recent_qb["recent_db"] > 0 and recent_qb["gsis_id"] != exp_id),
        }
        if exp_id != starter["gsis_id"]:
            qb_info.update(replacement=qb_info["expected"], replacement_epa_db=qb_info["expected_epa_db"],
                           impact=float(qb_points(starter["gsis_id"], exp_id)))

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
            is_starter = bool(qbs) and (gid == qbs[0]["gsis_id"] or (not gid and p.get("name") == qbs[0]["name"]))
            if is_starter:
                repl = next_after(qbs[0]["gsis_id"])
                impact = float(qb_points(qbs[0]["gsis_id"], repl["gsis_id"] if repl else GENERIC_QB))
            p["impact"] = round(impact, 2)
            p["side"] = "ofensiva"
            p["is_starter_qb"] = is_starter
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
    doubt = next((p for p in players if p.get("is_starter_qb") and p["category"] in ("probable", "cuestionable")
                  and not p["applied"]), None)
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
        "unconfirmed_qb": doubt is not None,
        "unconfirmed_qb_detail": ({"name": doubt["name"], "status": doubt["status"], "impact": doubt["impact"]}
                                  if doubt else None),
    }


def key_players(data: InjuryData, team: str, season: int, week: int, n: int = 22) -> list[dict]:
    """Jugadores con más snaps recientes (para el simulador de ausencias)."""
    shares = data.snap_shares(team, season, week)
    roster = data._roster(team, season, week)
    out = []
    for q in data.qb_depth(team, season, week)[:3]:
        out.append({"gsis_id": q["gsis_id"], "name": q["name"], "position": "QB", "snap_share": None,
                    "note": ("titular según depth chart · " if q.get("depth_rank") == 1 else "") + f"{q['recent_db']} dropbacks recientes"})
    if shares.empty or roster.empty:
        return out
    m = shares.merge(roster[["pfr_id", "gsis_id"]], left_on="pfr_player_id", right_on="pfr_id", how="inner")
    m = m[(m["position"] != "QB") & m["position"].isin(POSITION_VALUE)].sort_values("share", ascending=False).head(n)
    for r in m.itertuples(index=False):
        out.append({"gsis_id": r.gsis_id, "name": r.player, "position": r.position,
                    "snap_share": round(float(r.share), 3), "note": ""})
    return out
