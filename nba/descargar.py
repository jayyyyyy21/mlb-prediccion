"""Descarga resultados y box scores de la NBA desde la API pública de ESPN.

Uso:  python3 descargar.py            # actualiza todo (solo baja lo que falta)
Guarda un JSON compacto por partido en data/box/ y los calendarios en data/sb/.
"""
import datetime, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
# temporada -> (primer día, último día de fase regular, ¿bajar box scores?)
TEMPORADAS = {
    2023: ("2022-10-18", "2023-04-09", True),
    2024: ("2023-10-24", "2024-04-14", True),
    2025: ("2024-10-22", "2025-04-13", True),
    2026: ("2025-10-21", "2026-04-12", True),
    2027: ("2026-10-20", "2027-04-11", True),
}


def curl(url):
    for _ in range(4):
        r = subprocess.run(["curl", "-sS", "-m", "40", url], capture_output=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except ValueError:
                pass
    return None


def partidos_del_dia(sb, temporada):
    out = []
    for e in sb.get("events", []):
        if e.get("season", {}).get("slug") != "regular-season":
            continue
        c = e["competitions"][0]
        t = {x["homeAway"]: x for x in c["competitors"]}
        out.append(dict(
            id=e["id"], fecha=e["date"], temporada=temporada,
            terminado=c["status"]["type"]["completed"],
            local=t["home"]["team"]["abbreviation"], visita=t["away"]["team"]["abbreviation"],
            pl=int(t["home"].get("score") or 0), pv=int(t["away"].get("score") or 0),
            neutral=bool(c.get("neutralSite")),
        ))
    return out


def par(s):
    a, b = s.split("-")
    return int(a), int(b)


def compactar(d):
    comp = d["header"]["competitions"][0]
    g = dict(fecha=comp["date"], equipos={}, mercado=None)
    for c in comp["competitors"]:
        g["equipos"][c["team"]["abbreviation"]] = dict(
            local=c["homeAway"] == "home", pts=int(c["score"]),
            q=[int(float(x["displayValue"])) for x in c.get("linescores", [])], jug=[])
    for b in d["boxscore"].get("teams", []):
        s = {x["name"]: x["displayValue"] for x in b["statistics"]}
        e = g["equipos"][b["team"]["abbreviation"]]
        e["fgm"], e["fga"] = par(s["fieldGoalsMade-fieldGoalsAttempted"])
        e["tpm"], e["tpa"] = par(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
        e["ftm"], e["fta"] = par(s["freeThrowsMade-freeThrowsAttempted"])
        e["orb"], e["drb"], e["tov"] = int(s["offensiveRebounds"]), int(s["defensiveRebounds"]), int(s["totalTurnovers"])
    for p in d["boxscore"].get("players", []):
        e = g["equipos"][p["team"]["abbreviation"]]
        st = p["statistics"][0]
        k = st["keys"]
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            v = dict(zip(k, a["stats"]))
            try:
                fgm, fga = par(v["fieldGoalsMade-fieldGoalsAttempted"])
                tpm, _ = par(v["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
                ftm, fta = par(v["freeThrowsMade-freeThrowsAttempted"])
                e["jug"].append(dict(
                    id=a["athlete"]["id"], n=a["athlete"]["displayName"], tit=bool(a.get("starter")),
                    min=int(v["minutes"]), pts=int(v["points"]), fgm=fgm, fga=fga, tpm=tpm, ftm=ftm, fta=fta,
                    orb=int(v["offensiveRebounds"]), drb=int(v["defensiveRebounds"]), ast=int(v["assists"]),
                    stl=int(v["steals"]), blk=int(v["blocks"]), tov=int(v["turnovers"]), pf=int(v["fouls"])))
            except (ValueError, KeyError):
                continue
    for o in d.get("pickcenter") or []:
        if o.get("spread") is None:
            continue
        # spread de ESPN: negativo = local favorito. Lo guardamos como margen esperado del local.
        g["mercado"] = dict(margen_local=-float(o["spread"]), total=o.get("overUnder"),
                            ml_local=(o.get("homeTeamOdds") or {}).get("moneyLine"),
                            ml_visita=(o.get("awayTeamOdds") or {}).get("moneyLine"))
        break
    return g


def bajar_box(gid):
    ruta = os.path.join(DIR, "box", gid + ".json")
    if os.path.exists(ruta):
        return
    d = curl(f"{API}/summary?event={gid}")
    if not d or "header" not in d:
        return
    try:
        json.dump(compactar(d), open(ruta, "w"))
    except Exception as ex:  # partido raro (suspendido, sin box): se omite
        print("omitido", gid, ex, file=sys.stderr)


def main():
    os.makedirs(os.path.join(DIR, "sb"), exist_ok=True)
    os.makedirs(os.path.join(DIR, "box"), exist_ok=True)
    hoy = datetime.date.today()
    juegos = []
    for temp, (ini, fin, box) in TEMPORADAS.items():
        d0, d1 = datetime.date.fromisoformat(ini), min(datetime.date.fromisoformat(fin), hoy + datetime.timedelta(1))
        dias = [d0 + datetime.timedelta(i) for i in range((d1 - d0).days + 1)]

        def dia(d):
            ruta = os.path.join(DIR, "sb", d.strftime("%Y%m%d") + ".json")
            # los días recientes se vuelven a pedir: el marcador pudo cambiar
            if os.path.exists(ruta) and (hoy - d).days > 2:
                return json.load(open(ruta))
            sb = curl(f"{API}/scoreboard?dates={d.strftime('%Y%m%d')}")
            if sb is None:
                return []
            js = partidos_del_dia(sb, temp)
            json.dump(js, open(ruta, "w"))
            return js

        with ThreadPoolExecutor(12) as ex:
            for js in ex.map(dia, dias):
                juegos += [dict(j, box=box) for j in js]
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(bajar_box, [j["id"] for j in juegos if j["terminado"] and j["box"]]))
    json.dump(juegos, open(os.path.join(DIR, "juegos.json"), "w"))
    n = sum(j["terminado"] for j in juegos)
    print(f"{n} partidos terminados, {len(os.listdir(os.path.join(DIR, 'box')))} box scores")


if __name__ == "__main__":
    main()
