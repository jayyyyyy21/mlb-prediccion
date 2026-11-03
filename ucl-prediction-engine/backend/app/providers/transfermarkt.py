"""Lesiones y sanciones desde las páginas públicas de Transfermarkt (sin clave).

Fuente: https://www.transfermarkt.com/<club>/sperrenundverletzungen/verein/<id>
Cada fila trae: jugador, posición, edad, motivo, desde, regreso previsto y partidos
perdidos. Las sanciones indican la competición a la que aplican (p. ej. "UEFA Champions
League · No eligibility" = jugador no inscrito en la lista de Champions).

Uso responsable: una petición por club con pausa entre peticiones y como mucho una
descarga cada `MIN_HOURS` horas. Es una fuente periodística/comunitaria: lo que
publica es una baja *reportada*, no un parte médico oficial del club.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime

from .http import HttpClient

SOURCE = "Transfermarkt (transfermarkt.com)"
BASE = "https://www.transfermarkt.com"
MIN_HOURS = 6

_client: HttpClient | None = None


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient("transfermarkt", min_interval=1.5, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _client


def _get_html(url: str) -> str:
    c = client()
    import time
    wait = c.min_interval - (time.monotonic() - c._last)
    if wait > 0:
        time.sleep(wait)
    c._last = time.monotonic()
    r = c.client.get(url)
    c._record(url, r.status_code, r.status_code == 200, None if r.status_code == 200 else f"HTTP {r.status_code}")
    r.raise_for_status()
    return r.text


@dataclass
class Absence:
    player: str
    tm_player_id: str | None
    position: str | None
    age: str | None
    section: str  # Injuries | Suspensions | ...
    reason: str
    competition: str | None  # competición a la que aplica (sanciones)
    since: date | None
    expected_return: date | None
    missed_matches: int | None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    stop = {"fc", "cf", "sc", "ac", "afc", "fk", "sk", "kv", "rc", "ssc", "losc", "de", "1907", "cp", "balompie"}
    return " ".join(w for w in s.split() if w not in stop)


def participants(season_start: int) -> list[dict]:
    """Los clubes de la fase de liga UCL con su id y slug de Transfermarkt."""
    s = _get_html(f"{BASE}/uefa-champions-league/teilnehmer/pokalwettbewerb/CL/saison_id/{season_start}")
    found: dict[str, dict] = {}
    for m in re.finditer(r'<a[^>]+href="/([^"/]+)/[a-z]+/verein/(\d+)[^"]*"[^>]*>', s):
        tag = m.group(0)
        t = re.search(r'title="([^"]+)"', tag)
        if t and m.group(2) not in found:
            found[m.group(2)] = {"tm_id": m.group(2), "slug": m.group(1), "name": html.unescape(t.group(1))}
    return list(found.values())


def _date(s: str) -> date | None:
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%b %d, %Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_absences(page: str) -> list[Absence]:
    i = page.find('class="items"')
    if i < 0:
        return []
    j = page.find("</table>\n", i)
    end = page.find('<div class="keys"', i)
    body = page[i:end if end > 0 else j]
    out: list[Absence] = []
    section = "Injuries"
    # Partimos por filas de primer nivel: cabeceras de sección o filas de jugador
    for m in re.finditer(r'<td class="extrarow[^"]*" colspan="\d+">([^<]+)</td>|<tr class="(?:odd|even)">(.*?)</td></tr>',
                         body, re.S):
        if m.group(1):
            section = m.group(1).strip()
            continue
        row = m.group(2)
        a = re.search(r'<a title="([^"]+)" href="/[^"]+/profil/spieler/(\d+)"', row)
        if not a:
            continue
        pos = re.search(r'</table>', row)
        position = None
        pm = re.findall(r"<tr>\s*<td>([^<]+)</td>\s*</tr>", row)
        if pm:
            position = pm[0].strip()
        rest = row[pos.end():] if pos else row
        cells = re.findall(r"<td[^>]*>(.*?)(?=</td>|$)", rest, re.S)
        cells = [c for c in cells]
        comp = None
        if len(cells) >= 2:
            cm = re.search(r'<img[^>]+title="([^"]+)"', cells[1])
            comp = html.unescape(cm.group(1)) if cm else None
        txt = [_text(c) for c in cells]
        while len(txt) < 5:
            txt.append("")
        missed = re.search(r"\d+", txt[4] if len(txt) > 4 else "")
        out.append(Absence(
            player=html.unescape(a.group(1)), tm_player_id=a.group(2), position=position,
            age=txt[0] or None, section=section, reason=txt[1], competition=comp,
            since=_date(txt[2]), expected_return=_date(txt[3]),
            missed_matches=int(missed.group()) if missed else None,
        ))
    return out


def club_absences(slug: str, tm_id: str) -> list[Absence]:
    return parse_absences(_get_html(f"{BASE}/{slug}/sperrenundverletzungen/verein/{tm_id}"))


def status_for_match(a: Absence, match_day: date, competition: str = "UEFA Champions League") -> str | None:
    """Estado de la baja para un partido concreto, o None si no aplica.

    * Lesión con regreso previsto posterior al partido → 'confirmed' (baja reportada).
    * Lesión sin fecha de regreso → 'probable'.
    * Regreso previsto en los 5 días anteriores al partido → 'probable' (puede llegar justo).
    * Sanción de otra competición → no aplica.
    """
    if a.competition and a.section.lower().startswith("susp") and a.competition != competition:
        return None
    if a.expected_return is None:
        if a.section.lower().startswith("susp"):
            return "confirmed"
        # Lesión larga sin fecha (≥14 días y ≥2 partidos perdidos) → baja; reciente → duda
        long_term = a.since and (match_day - a.since).days >= 14 and (a.missed_matches or 0) >= 2
        return "confirmed" if long_term else "probable"
    if a.expected_return > match_day:
        return "confirmed"
    if (match_day - a.expected_return).days <= 5:
        return "probable"
    return None


SEVERITY = {"confirmed": 2, "probable": 1}


def merge_for_match(items: list[Absence], match_day: date,
                    competition: str = "UEFA Champions League") -> list[dict]:
    """Una entrada por jugador con el estado más severo y todos los motivos."""
    by_player: dict[str, dict] = {}
    for a in items:
        st = status_for_match(a, match_day, competition)
        if st is None:
            continue
        key = a.tm_player_id or a.player
        reason = a.reason + (f" ({a.competition})" if a.competition else "")
        if a.expected_return:
            reason += f" · regreso previsto {a.expected_return.strftime('%d/%m/%Y')}"
        cur = by_player.get(key)
        if cur is None:
            by_player[key] = {"player": a.player, "position": a.position, "status": st, "reason": reason}
        else:
            cur["reason"] += "; " + reason
            if SEVERITY[st] > SEVERITY[cur["status"]]:
                cur["status"] = st
    return list(by_player.values())


ALIASES = {"internazionale": "inter milan", "bayern munich": "bayern munich", "atletico madrid": "atletico madrid",
           "bodo glimt": "bodo glimt", "lask linz": "lask", "slavia prague": "slavia prague",
           "real betis": "real betis", "club brugge": "club brugge", "como": "como"}


def match_teams(espn_teams: dict[str, str], tm_clubs: list[dict]) -> dict[str, dict]:
    """Empareja equipos ESPN (id -> nombre) con clubes Transfermarkt por nombre normalizado."""
    import difflib
    tm_norm = {norm(c["name"]): c for c in tm_clubs}
    out = {}
    for tid, name in espn_teams.items():
        n = norm(name)
        n = ALIASES.get(n, n)
        if n in tm_norm:
            out[tid] = tm_norm[n]
            continue
        cand = [k for k in tm_norm if n in k or k in n]
        if len(cand) == 1:
            out[tid] = tm_norm[cand[0]]
            continue
        best = difflib.get_close_matches(n, list(tm_norm), n=1, cutoff=0.6)
        if best:
            out[tid] = tm_norm[best[0]]
    return out
