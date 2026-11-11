"""Modelo NBA v2: valor de jugadores disponibles + rating residual de equipo.

margen_local = LOCALIA + BETA * (talento_local - talento_visita) + (U_local - U_visita)
               - B2B * (b2b_local - b2b_visita)

- talento: suma de minutos esperados x valor por minuto de los jugadores DISPONIBLES
  (así las lesiones y los cambios de plantilla entran solos, sin ajustes a mano).
- U: lo que el equipo rinde por encima o por debajo de su talento (defensa, sistema),
  estimado con regresión ridge y heredado en parte de la temporada anterior.
Todos los parámetros de P se eligieron por backtest (ver backtest.py), no a mano.
"""
import datetime, json, math, os
import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NBA = ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU', 'IND', 'LAC', 'LAL', 'MEM',
       'MIA', 'MIL', 'MIN', 'NO', 'NY', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SA', 'SAC', 'TOR', 'UTAH', 'WSH']
IX = {t: i for i, t in enumerate(NBA)}

# Parámetros por defecto; backtest.py los ajusta y los guarda en params.json
P = dict(beta=0.8, carry=0.6, lam=8.0, hca=2.3, cap=25.0, phi=1.0, w3=0.3, b2b=1.0,
         km=400.0, delta=0.99, vrepl=0.26, alpha=0.25,
         esc=1.0, sigma=12.5,              # estiramiento del margen y sigma para pasar a probabilidad
         k_q1=0.25, s_q1=7.0, k_h1=0.5, s_h1=9.5,   # primer cuarto y primera mitad
         t_lam=10.0, t_carry=0.6, s_tot=18.0)        # totales
_ruta_params = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
if os.path.exists(_ruta_params):
    P.update(json.load(open(_ruta_params)))


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def cargar():
    """Partidos terminados de fase regular entre equipos NBA, en orden cronológico."""
    G = []
    for j in json.load(open(os.path.join(DIR, "juegos.json"))):
        if not j["terminado"] or j["local"] not in IX or j["visita"] not in IX:
            continue
        ruta = os.path.join(DIR, "box", j["id"] + ".json")
        if not os.path.exists(ruta):
            continue
        b = json.load(open(ruta))
        utc = datetime.datetime.fromisoformat(j["fecha"].replace("Z", "+00:00"))
        j["dia"] = (utc - datetime.timedelta(hours=6)).date()  # día del partido en EE. UU.
        j["h"], j["a"] = b["equipos"][j["local"]], b["equipos"][j["visita"]]
        j["mercado"] = b["mercado"]
        G.append(j)
    G.sort(key=lambda g: (g["dia"], g["id"]))
    return G


def game_score(p):
    return (p["pts"] + 0.4 * p["fgm"] - 0.7 * p["fga"] - 0.4 * (p["fta"] - p["ftm"]) + 0.7 * p["orb"]
            + 0.3 * p["drb"] + p["stl"] + 0.7 * p["ast"] + 0.7 * p["blk"] - 0.4 * p["pf"] - p["tov"])


class Jugadores:
    """Valor por minuto y minutos esperados de cada jugador, actualizados partido a partido."""

    def __init__(self, P):
        self.P, self.d = P, {}

    def valor(self, pid):
        s = self.d.get(pid)
        km, vr = self.P["km"], self.P["vrepl"]
        return vr if s is None else (s[0] + km * vr) / (s[1] + km)

    def minutos(self, pid):
        s = self.d.get(pid)
        return 10.0 if s is None else s[2]

    def talento(self, pids):
        """Game Score esperado del equipo con estos jugadores repartiéndose 240 minutos."""
        if not pids:
            return 240 * self.P["vrepl"]
        m = np.array([max(self.minutos(p), 1.0) for p in pids])
        v = np.array([self.valor(p) for p in pids])
        for _ in range(3):  # reescala a 240 con tope de 38 minutos por jugador
            m = np.minimum(m * 240 / m.sum(), 38.0)
        return float((m * v).sum() * 240 / m.sum())

    def actualizar(self, equipo, abrev, dia):
        dl, al = self.P["delta"], self.P["alpha"]
        for p in equipo["jug"]:
            if p["min"] <= 0:
                continue
            s = self.d.get(p["id"])
            if s is None:
                s = self.d[p["id"]] = [0.0, 0.0, float(p["min"]), abrev, dia, p["n"]]
            else:
                s[2] += al * (p["min"] - s[2])
            s[0] = s[0] * dl + game_score(p)
            s[1] = s[1] * dl + p["min"]
            s[3], s[4], s[5] = abrev, dia, p["n"]


class Equipos:
    """Rating residual U por equipo (ridge con prior heredado), acumulado dentro de la temporada."""

    def __init__(self, P):
        self.P = P
        self.U = np.zeros(30)
        self.prior = np.zeros(30)
        self.temporada = None
        self._reset()

    def _reset(self):
        self.A, self.b, self.sucio = np.zeros((30, 30)), np.zeros(30), False
        self.ultimo = {}  # equipo -> último día jugado

    def nueva_temporada(self, temporada):
        if self.temporada is not None:
            self.prior = self.P["carry"] * self.resolver()
        self.temporada = temporada
        self._reset()
        self.U = self.prior.copy()

    def resolver(self):
        if self.sucio:
            lam = self.P["lam"]
            self.U = np.linalg.solve(self.A + lam * np.eye(30), self.b + lam * self.prior)
            self.sucio = False
        return self.U

    def b2b(self, t, dia):
        u = self.ultimo.get(t)
        return 1.0 if u is not None and (dia - u).days == 1 else 0.0

    def base(self, g, tal_h, tal_a):
        """Parte del margen que no depende de U: localía, talento y descanso."""
        P = self.P
        return ((0.0 if g.get("neutral") else P["hca"]) + P["beta"] * (tal_h - tal_a)
                - P["b2b"] * (self.b2b(g["local"], g["dia"]) - self.b2b(g["visita"], g["dia"])))

    def predecir(self, g, tal_h, tal_a):
        U = self.resolver()
        return self.P["esc"] * (self.base(g, tal_h, tal_a) + U[IX[g["local"]]] - U[IX[g["visita"]]])

    def actualizar(self, g, base, p3):
        P = self.P
        h, a = g["h"], g["a"]
        # margen corregido por suerte en el triple: se devuelve parte del exceso sobre la media
        m = (g["pl"] - 3 * P["w3"] * (h["tpm"] - h["tpa"] * p3)) - (g["pv"] - 3 * P["w3"] * (a["tpm"] - a["tpa"] * p3))
        y = max(-P["cap"], min(P["cap"], m)) - base
        i, k = IX[g["local"]], IX[g["visita"]]
        self.A[i, i] += 1; self.A[k, k] += 1; self.A[i, k] -= 1; self.A[k, i] -= 1
        self.b[i] += y; self.b[k] -= y
        self.sucio = True
        self.ultimo[g["local"]] = self.ultimo[g["visita"]] = g["dia"]

    def envejecer(self):
        """Olvido diario: da más peso a los partidos recientes (phi=1 no olvida)."""
        if self.P["phi"] < 1:
            self.A *= self.P["phi"]; self.b *= self.P["phi"]


class Totales:
    """Total de puntos: media de la liga + tendencia de cada equipo (ridge con prior heredado)."""

    def __init__(self, P):
        self.P, self.T, self.prior, self.temporada = P, np.zeros(30), np.zeros(30), None
        self.mu, self.n_mu = 228.0, 0.0
        self._reset()

    def _reset(self):
        self.A, self.b = np.zeros((30, 30)), np.zeros(30)

    def nueva_temporada(self, temporada):
        if self.temporada is not None:
            self.prior = self.P["t_carry"] * self.resolver()
        self.temporada, self.n_mu = temporada, 150.0  # la media de la liga arranca con peso de 150 partidos
        self._reset()

    def resolver(self):
        lam = self.P["t_lam"]
        self.T = np.linalg.solve(self.A + lam * np.eye(30), self.b + lam * self.prior)
        return self.T

    def predecir(self, g):
        T = self.resolver()
        return self.mu + T[IX[g["local"]]] + T[IX[g["visita"]]]

    def actualizar(self, g):
        # se usa el marcador al final del tiempo regular para que las prórrogas no inflen el total
        reg = sum(g["h"]["q"][:4]) + sum(g["a"]["q"][:4]) if len(g["h"]["q"]) >= 4 else g["pl"] + g["pv"]
        i, k = IX[g["local"]], IX[g["visita"]]
        y = reg - self.mu
        self.A[i, i] += 1; self.A[k, k] += 1; self.A[i, k] += 1; self.A[k, i] += 1
        self.b[i] += y; self.b[k] += y
        self.mu = (self.mu * self.n_mu + reg) / (self.n_mu + 1); self.n_mu += 1


def recorrer(G, P, hasta=None):
    """Recorre los partidos en orden; antes de cada uno guarda la predicción hecha solo con el pasado.

    Devuelve (lista de predicciones, estado final) para pronosticar partidos futuros.
    """
    jug, eq, tot = Jugadores(P), Equipos(P), Totales(P)
    out, dia_act, p3 = [], None, 0.36
    tpm = tpa = 0
    for g in G:
        if hasta and g["dia"] >= hasta:
            break
        if g["temporada"] != eq.temporada:
            eq.nueva_temporada(g["temporada"]); tot.nueva_temporada(g["temporada"])
            cuenta = {}
        if g["dia"] != dia_act:
            eq.envejecer(); dia_act = g["dia"]
        tal_h = jug.talento([p["id"] for p in g["h"]["jug"] if p["min"] > 0])
        tal_a = jug.talento([p["id"] for p in g["a"]["jug"] if p["min"] > 0])
        base = eq.base(g, tal_h, tal_a)
        pred = eq.predecir(g, tal_h, tal_a)
        out.append(dict(g=g, pred=pred, real=g["pl"] - g["pv"], tal=tal_h - tal_a,
                        total=tot.predecir(g), nj=min(cuenta.get(g["local"], 0), cuenta.get(g["visita"], 0))))
        cuenta[g["local"]] = cuenta.get(g["local"], 0) + 1; cuenta[g["visita"]] = cuenta.get(g["visita"], 0) + 1
        tot.actualizar(g)
        eq.actualizar(g, base, p3)
        jug.actualizar(g["h"], g["local"], g["dia"]); jug.actualizar(g["a"], g["visita"], g["dia"])
        tpm += g["h"]["tpm"] + g["a"]["tpm"]; tpa += g["h"]["tpa"] + g["a"]["tpa"]
        if tpa > 20000:
            p3 = tpm / tpa
    return out, (jug, eq, tot)
