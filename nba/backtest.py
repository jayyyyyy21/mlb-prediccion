"""Ajusta los parámetros del modelo y lo evalúa sin hacer trampa (walk-forward).

Entrena en 2023-24 y 2024-25 (2022-23 sirve de calentamiento) y evalúa en 2025-26 y 2026-27,
temporadas que el ajuste nunca vio. Uso: python3 backtest.py [--ajustar]
"""
import json, math, sys
import numpy as np
import modelo as M

G = M.cargar()


def mae(R, temps, clave="pred"):
    e = [abs(r[clave] - r["real"]) for r in R if r["g"]["temporada"] in temps]
    return sum(e) / len(e)


def ajustar():
    P = dict(M.P)
    rejilla = dict(beta=[0, .3, .5, .7, .9, 1.1, 1.3], carry=[0, .3, .5, .7, .85, 1], lam=[2, 4, 8, 12, 20, 35],
                   hca=[1.5, 2, 2.5, 3], cap=[18, 22, 28, 40, 99], phi=[.985, .99, .995, 1], w3=[0, .2, .4, .6, .8],
                   b2b=[0, .75, 1.5, 2.25], km=[150, 300, 500, 800, 1200], delta=[.98, .99, .995, .998],
                   vrepl=[.18, .22, .26, .30], alpha=[.15, .25, .4])
    f = lambda P: mae(M.recorrer(G, P)[0], (2024, 2025))
    mejor = f(P)
    for vuelta in range(3):
        for k, vals in rejilla.items():
            for v in vals:
                Q = dict(P); Q[k] = v
                m = f(Q)
                if m < mejor - 1e-4:
                    mejor, P = m, Q
            print(f"vuelta {vuelta + 1} {k:6} -> {P[k]:<6} MAE entrenamiento {mejor:.3f}", flush=True)
    P["esc"] = 1.0
    calibrar(P)
    json.dump(P, open("params.json", "w"), indent=1)
    M.P.update(P)
    print("guardado", P)


def mejor_sigma(pares):
    """Sigma que minimiza el log loss de Phi(pred/sigma) sobre pares (pred, ganó el local)."""
    f = lambda s: -sum(math.log(max(1e-9, M.Phi(p / s) if w else 1 - M.Phi(p / s))) for p, w in pares)
    return min(np.arange(4, 20, 0.1), key=f)


def calibrar(P):
    """Con los parámetros estructurales fijos, ajusta en entrenamiento el estiramiento del margen,
    las sigmas de probabilidad, la relación con el primer cuarto / primera mitad y los totales."""
    T = (2024, 2025)
    R = [r for r in M.recorrer(G, P)[0] if r["g"]["temporada"] in T]
    p = np.array([r["pred"] for r in R]); y = np.array([r["real"] for r in R])
    P["esc"] = round(float(p @ y / (p @ p)), 3)
    p = p * P["esc"]
    P["sigma"] = round(float(mejor_sigma([(a, b > 0) for a, b in zip(p, y) if b != 0])), 1)
    for clave, n in (("q1", 1), ("h1", 2)):
        q = np.array([sum(r["g"]["h"]["q"][:n]) - sum(r["g"]["a"]["q"][:n]) for r in R])
        P["k_" + clave] = round(float(p @ q / (p @ p)), 3)
        P["s_" + clave] = round(float(mejor_sigma([(a * P["k_" + clave], b > 0) for a, b in zip(p, q) if b != 0])), 1)
    mejor = None
    for lam in (5, 10, 20, 40):
        for c in (.3, .5, .7, .9):
            Q = dict(P, t_lam=lam, t_carry=c)
            e = np.mean([abs(r["total"] - (r["g"]["pl"] + r["g"]["pv"])) for r in M.recorrer(G, Q)[0] if r["g"]["temporada"] in T])
            if mejor is None or e < mejor[0]:
                mejor = (e, lam, c)
    P["t_lam"], P["t_carry"] = mejor[1], mejor[2]
    R = [r for r in M.recorrer(G, P)[0] if r["g"]["temporada"] in T]
    P["s_tot"] = round(float(np.std([r["total"] - (r["g"]["pl"] + r["g"]["pv"]) for r in R])), 1)


def sin_vig(ml_l, ml_v):
    imp = lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100)
    a, b = imp(ml_l), imp(ml_v)
    return a / (a + b)


def evaluar():
    P = M.P
    R, _ = M.recorrer(G, P)
    base = M.recorrer(G, dict(P, beta=0.0))[0]  # mismo modelo sin capa de jugadores
    print(f"\n{'temporada':10} {'n':>5} {'MAE':>6} {'sin jug.':>8} {'acierto':>8} {'logloss':>8} | {'n línea':>7} {'MAE':>6} {'MAE mkt':>8} {'acierto':>8} {'ac. mkt':>8}")
    for t in (2024, 2025, 2026, 2027):
        r = [x for x in R if x["g"]["temporada"] == t]
        b = [x for x in base if x["g"]["temporada"] == t]
        ac = np.mean([(x["pred"] > 0) == (x["real"] > 0) for x in r])
        ll = -np.mean([math.log(max(1e-9, M.Phi(x["pred"] / P["sigma"]) if x["real"] > 0 else 1 - M.Phi(x["pred"] / P["sigma"]))) for x in r])
        m = [x for x in r if x["g"]["mercado"]]
        fila = f"{t - 1}-{t % 100:<5} {len(r):5d} {mae(r, (t,)):6.2f} {mae(b, (t,)):8.2f} {ac:8.1%} {ll:8.3f} | "
        if m:
            mk = np.array([x["g"]["mercado"]["margen_local"] for x in m]); re = np.array([x["real"] for x in m]); pr = np.array([x["pred"] for x in m])
            fila += f"{len(m):7d} {np.abs(pr - re).mean():6.2f} {np.abs(mk - re).mean():8.2f} {((pr > 0) == (re > 0)).mean():8.1%} {((mk > 0) == (re > 0))[mk != 0].mean():8.1%}"
        print(fila)

    H = [x for x in R if x["g"]["temporada"] in (2026, 2027)]
    print("\nCalibración fuera de muestra (2025-26 y 2026-27): prob. del favorito del modelo vs frecuencia real")
    for lo, hi in ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        s = [(max(p, 1 - p), (p > .5) == (x["real"] > 0)) for x in H for p in [M.Phi(x["pred"] / P["sigma"])] if lo <= max(p, 1 - p) < hi]
        if s:
            print(f"  {lo:.0%}-{min(hi, 1):.0%}: n={len(s):4d}  modelo {np.mean([a for a, _ in s]):.1%}  real {np.mean([b for _, b in s]):.1%}")
    print("\nError por tramo de temporada (partidos jugados por el equipo con menos partidos):")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 83)):
        s = [x for x in H if lo <= x["nj"] < hi]
        mk = [x for x in s if x["g"]["mercado"]]
        print(f"  {lo:2d}-{hi:2d}: n={len(s):4d} MAE {np.mean([abs(x['pred'] - x['real']) for x in s]):.2f}  sd error {np.std([x['pred'] - x['real'] for x in s]):.2f}"
              + (f"  | con línea n={len(mk)} modelo {np.mean([abs(x['pred'] - x['real']) for x in mk]):.2f} mercado {np.mean([abs(x['g']['mercado']['margen_local'] - x['real']) for x in mk]):.2f}" if mk else ""))

    m = [x for x in H if x["g"]["mercado"]]
    X = np.array([[x["pred"], x["g"]["mercado"]["margen_local"]] for x in m]); y = np.array([x["real"] for x in m])
    co = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"\nMezcla óptima fuera de muestra: margen real ≈ {co[0]:.2f} x modelo + {co[1]:.2f} x mercado  (n={len(m)})")
    print(f"correlación modelo-mercado {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.3f}")

    print("\nSimulación de apuestas a ganador (1 unidad cada una) cuando el modelo ve ventaja sobre la cuota:")
    dec = lambda o: 1 + o / 100 if o > 0 else 1 + 100 / -o
    for umbral in (0.0, 0.03, 0.06, 0.10):
        n = gan = 0.0
        for x in m:
            mk = x["g"]["mercado"]
            if not mk.get("ml_local") or not mk.get("ml_visita"):
                continue
            p = M.Phi(x["pred"] / P["sigma"])
            for prob, cuota, gano in ((p, mk["ml_local"], x["real"] > 0), (1 - p, mk["ml_visita"], x["real"] < 0)):
                if prob * dec(cuota) - 1 > umbral:
                    n += 1; gan += (dec(cuota) - 1) if gano else -1
        print(f"  ventaja > {umbral:.0%}: {int(n):4d} apuestas, resultado {gan:+.1f} u, ROI {gan / max(n, 1):+.1%}")
    print("\nPrimer cuarto y primera mitad fuera de muestra (acierto del lado que el modelo da por favorito):")
    for clave, n in (("q1", 1), ("h1", 2)):
        s = [(x["pred"] * P["k_" + clave], sum(x["g"]["h"]["q"][:n]) - sum(x["g"]["a"]["q"][:n])) for x in H]
        s = [(a, b) for a, b in s if b != 0]
        for lo, hi in ((.5, .6), (.6, .7), (.7, 1.01)):
            z = [(max(pr, 1 - pr), (pr > .5) == (b > 0)) for a, b in s for pr in [M.Phi(a / P["s_" + clave])] if lo <= max(pr, 1 - pr) < hi]
            if z:
                print(f"  {clave} {lo:.0%}-{min(hi, 1):.0%}: n={len(z):4d} modelo {np.mean([a for a, _ in z]):.1%} real {np.mean([b for _, b in z]):.1%}")
    t = [x for x in H if x["g"]["mercado"] and x["g"]["mercado"].get("total")]
    re = np.array([x["g"]["pl"] + x["g"]["pv"] for x in t]); mo = np.array([x["total"] for x in t]); mk = np.array([x["g"]["mercado"]["total"] for x in t])
    co = np.linalg.lstsq(np.c_[mo - mk, np.ones(len(t))], re - mk, rcond=None)[0]
    print(f"\nTotales fuera de muestra (n={len(t)}): MAE modelo {np.abs(mo - re).mean():.2f}, MAE mercado {np.abs(mk - re).mean():.2f}; "
          f"cuando el modelo difiere del mercado en X puntos, el resultado se mueve {co[0]:+.2f}·X")
    lado = [(m_ > k_) == (r_ > k_) for m_, k_, r_ in zip(mo, mk, re) if abs(m_ - k_) >= 4 and r_ != k_]
    print(f"  over/under siguiendo al modelo cuando difiere 4+ puntos: {np.mean(lado):.1%} de acierto en {len(lado)} (hace falta 52.4% para no perder)")
    return R


if __name__ == "__main__":
    if "--ajustar" in sys.argv:
        ajustar()
    elif "--calibrar" in sys.argv:
        P = dict(M.P); calibrar(P); json.dump(P, open("params.json", "w"), indent=1); M.P.update(P); print(P)
    evaluar()
