"""Pronóstico de los partidos NBA de un día.  Uso: python3 pronostico.py [AAAA-MM-DD]

Actualiza datos, reconstruye el modelo con todo lo jugado antes de ese día y aplica el
reporte de lesiones de ESPN a las plantillas. Las cuotas solo se leen al final, para comparar.
"""
import datetime, json, re, sys
import descargar, modelo as M

P = M.P


def lesiones():
    """id de jugador -> ('fuera' | 'duda' | 'probable', comentario) según el reporte de ESPN."""
    d = descargar.curl(descargar.API + "/injuries") or {}
    out = {}
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            m = re.search(r"/id/(\d+)/", " ".join(l.get("href", "") for l in i["athlete"].get("links", [])))
            if not m:
                continue
            txt = (i.get("shortComment") or "").lower()
            if i.get("status") == "Out" or "doubtful" in txt or "ruled out" in txt or "won't play" in txt:
                est = "fuera"
            elif "probable" in txt or "expected to play" in txt or "available" in txt:
                est = "probable"
            else:
                est = "duda"
            out[m.group(1)] = (est, i.get("shortComment") or "")
    return out


def plantilla(jug, G, equipo, dia):
    """Jugadores cuyo último partido fue con este equipo y que jugaron alguno de sus últimos 4."""
    dias = sorted({g["dia"] for g in G if g["dia"] < dia and equipo in (g["local"], g["visita"])})[-4:]
    if not dias:
        return []
    return [pid for pid, s in jug.d.items() if s[3] == equipo and s[4] >= dias[0]]


def sin_vig(a, b):
    imp = lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100)
    return imp(a) / (imp(a) + imp(b))


def main():
    dia = datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else datetime.date.today()
    descargar.main()
    G = M.cargar()
    _, (jug, eq, tot) = M.recorrer(G, P, hasta=dia)
    les = lesiones()
    sb = descargar.curl(f"{descargar.API}/scoreboard?dates={dia.strftime('%Y%m%d')}") or {}
    filas = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
        if t["home"] not in M.IX or t["away"] not in M.IX:
            continue
        g = dict(local=t["home"], visita=t["away"], dia=dia, neutral=bool(c.get("neutralSite")))
        info, tal, dudas = {}, {}, []
        for lado in ("local", "visita"):
            eqp = g[lado]
            rost = plantilla(jug, G, eqp, dia)
            disp = [p for p in rost if les.get(p, ("",))[0] != "fuera"]
            tal[lado] = jug.talento(disp)
            bajas = []
            for p in rost:
                est = les.get(p, ("",))[0]
                if est in ("fuera", "duda"):
                    # impacto en puntos de margen de quitar (o devolver) a este jugador
                    con = jug.talento(disp if p in disp else disp + [p])
                    sin = jug.talento([x for x in disp if x != p])
                    imp = P["esc"] * P["beta"] * (con - sin)
                    bajas.append((jug.d[p][5], est, imp))
                    if est == "duda":
                        dudas.append((lado, jug.d[p][5], imp))
            # lesionados de larga duración: ya no están en la plantilla reciente
            largas = [jug.d[p][5] for p, (est, _) in les.items() if est == "fuera" and p in jug.d
                      and jug.d[p][3] == eqp and p not in rost and jug.valor(p) * jug.minutos(p) > 6]
            info[lado] = dict(bajas=sorted(bajas, key=lambda x: -x[2]), largas=largas,
                              b2b=bool(eq.b2b(eqp, dia)), U=float(eq.resolver()[M.IX[eqp]]), tal=tal[lado])
        margen = eq.predecir(g, tal["local"], tal["visita"])
        f = dict(partido=f"{g['visita']} @ {g['local']}", local=g["local"], visita=g["visita"], margen=margen,
                 p_local=M.Phi(margen / P["sigma"]), total=tot.predecir(g),
                 p_q1=M.Phi(margen * P["k_q1"] / P["s_q1"]), p_h1=M.Phi(margen * P["k_h1"] / P["s_h1"]),
                 info=info, dudas=[(l, n, i) for l, n, i in dudas if i >= 1.0], mercado=None,
                 estado=c["status"]["type"]["name"])
        for o in c.get("odds") or []:
            ml = o.get("moneyline") or {}
            try:
                a = int(str(ml["home"]["close"]["odds"]).replace("EVEN", "100")); b = int(str(ml["away"]["close"]["odds"]).replace("EVEN", "100"))
                f["mercado"] = dict(p_local=sin_vig(a, b), ml_local=a, ml_visita=b, linea=o.get("details"), total=o.get("overUnder"))
            except (KeyError, ValueError, TypeError):
                pass
            break
        filas.append(f)

    filas.sort(key=lambda f: -abs(f["p_local"] - .5))
    print(f"\nNBA {dia} — modelo v2 (sigma {P['sigma']}, {len(G)} partidos de historia)\n")
    for f in filas:
        fav, p = (f["local"], f["p_local"]) if f["p_local"] >= .5 else (f["visita"], 1 - f["p_local"])
        conf = "ALTA" if p >= .80 else "MEDIA" if p >= .65 else "BAJA"
        if f["dudas"] and conf != "BAJA":
            conf += " (pendiente de dudas)"
        pl = (f["total"] + f["margen"]) / 2
        print(f"{f['partido']:12} gana {fav:4} {p:5.1%}  margen local {f['margen']:+5.1f}  marcador {f['visita']} {f['total'] - pl:.0f} - {f['local']} {pl:.0f}  total {f['total']:.0f}  confianza {conf}")
        q1 = f["p_q1"] if f["p_local"] >= .5 else 1 - f["p_q1"]; h1 = f["p_h1"] if f["p_local"] >= .5 else 1 - f["p_h1"]
        print(f"{'':12} {fav} gana 1er cuarto {q1:.0%}, 1a mitad {h1:.0%}  | rango 80% del margen: {f['margen'] - 1.28 * 14.2:+.0f} a {f['margen'] + 1.28 * 14.2:+.0f}")
        for lado in ("visita", "local"):
            i = f["info"][lado]
            b = ", ".join(f"{n} ({'fuera' if e == 'fuera' else 'duda'}, {imp:.1f} pts)" for n, e, imp in i["bajas"]) or "sin bajas en la rotación"
            print(f"{'':12} {f[lado]:4} residual {i['U']:+.1f}  talento {i['tal']:.1f}{'  BACK-TO-BACK' if i['b2b'] else ''}  | {b}"
                  + (f"  | larga duración: {', '.join(i['largas'])}" if i["largas"] else ""))
        for lado, n, imp in f["dudas"]:
            m2 = f["margen"] + (-imp if lado == "local" else imp)
            print(f"{'':12} si {n} no juega: {f['local']} {M.Phi(m2 / P['sigma']):.0%}")
        if f["mercado"]:
            mk = f["mercado"]
            dif = f["p_local"] - mk["p_local"]
            aviso = "  << DISCREPANCIA: revisar noticias; en el backtest el mercado acierta más cuando difieren" if abs(dif) >= .08 else ""
            print(f"{'':12} mercado: {mk['linea']}  {f['local']} {mk['p_local']:.0%} sin margen  total {mk['total']}  | modelo - mercado {dif:+.0%}{aviso}")
        print()
    json.dump(filas, open("ultimo_pronostico.json", "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
