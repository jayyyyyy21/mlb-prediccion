"""Pipeline de recopilación de datos.

1. Calendario y resultados UCL (temporada actual y anterior).
2. Descubrimiento de la liga doméstica de cada equipo UCL.
3. Resultados + estadísticas por partido de esas ligas y de UEL/UECL
   (las competiciones europeas conectan la fuerza relativa entre ligas).
4. Boxscore detallado + alineaciones de los partidos de equipos UCL.
5. Clima/altitud para partidos UCL próximos.
6. Bajas (solo si hay API_FOOTBALL_KEY).

Todo queda registrado en fetch_log. Si una fuente falla se registra y se sigue:
nunca se rellena con datos inventados.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import or_, select

from ..config import EURO_SLUGS, UCL_SLUG, settings
from ..db import Appearance, Availability, Match, SessionLocal, Team, TeamMatchStats, Weather, utcnow
from ..providers import api_football, espn, openmeteo, transfermarkt
from ..providers.http import FetchError

log = logging.getLogger("ingest")


def month_range(start: str, end: datetime) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def upsert_team(s, info: dict) -> Team:
    t = s.get(Team, info["id"])
    if t is None:
        t = Team(id=info["id"], name=info["name"])
        s.add(t)
    t.name = info["name"] or t.name
    t.short_name = info.get("short_name") or t.short_name
    t.abbreviation = info.get("abbreviation") or t.abbreviation
    t.logo = info.get("logo") or t.logo
    t.color = info.get("color") or t.color
    return t


def _set_stats(s, match_id: str, team_id: str, is_home: bool, goals, stats: dict) -> None:
    row = s.execute(select(TeamMatchStats).where(
        TeamMatchStats.match_id == match_id, TeamMatchStats.team_id == team_id)).scalar_one_or_none()
    if row is None:
        row = TeamMatchStats(match_id=match_id, team_id=team_id, is_home=is_home)
        s.add(row)
    row.goals = int(goals) if goals is not None else None
    for k, v in stats.items():
        if v is not None:
            setattr(row, k, v)


def upsert_event(s, ev: dict) -> Match:
    home = upsert_team(s, ev["home"]["team"])
    away = upsert_team(s, ev["away"]["team"])
    m = s.get(Match, ev["id"])
    if m is None:
        m = Match(id=ev["id"], league=ev["league"], kickoff=ev["kickoff"], status=ev["status"],
                  home_id=home.id, away_id=away.id)
        s.add(m)
    m.league = ev["league"]
    m.season = ev["season"]
    m.phase = ev["phase"]
    m.kickoff = ev["kickoff"]
    m.status = ev["status"]
    m.completed = ev["completed"]
    m.neutral = ev["neutral"]
    m.venue, m.city, m.country = ev["venue"], ev["city"], ev["country"]
    if ev.get("odds"):
        m.odds = ev["odds"]
    if ev["completed"]:
        m.home_goals = int(ev["home"]["score"]) if ev["home"]["score"] is not None else None
        m.away_goals = int(ev["away"]["score"]) if ev["away"]["score"] is not None else None
        s.flush()
        _set_stats(s, m.id, home.id, True, m.home_goals, ev["home"]["stats"])
        _set_stats(s, m.id, away.id, False, m.away_goals, ev["away"]["stats"])
    return m


def ingest_league_months(league: str, months: list[str]) -> int:
    n = 0
    for ym in months:
        try:
            js = espn.scoreboard(league, ym)
        except FetchError as e:
            log.warning("scoreboard %s %s: %s", league, ym, e)
            continue
        with SessionLocal() as s:
            for raw in js.get("events") or []:
                ev = espn.parse_event(raw, league)
                if ev:
                    upsert_event(s, ev)
                    n += 1
            s.commit()
    return n


def ucl_months(now: datetime) -> list[str]:
    end = datetime(settings.ucl_season_start + 1, 6, 30)
    hist_start = settings.history_from
    return month_range(hist_start, max(end, now))


def discover_domestic_leagues() -> set[str]:
    leagues: set[str] = set()
    with SessionLocal() as s:
        season_start = datetime(settings.ucl_season_start, 7, 1)
        ids = {r for (r,) in s.execute(select(Match.home_id).where(
            Match.league == UCL_SLUG, Match.kickoff >= season_start))} | {
            r for (r,) in s.execute(select(Match.away_id).where(
                Match.league == UCL_SLUG, Match.kickoff >= season_start))}
        for t in s.execute(select(Team).where(Team.id.in_(ids))).scalars():
            t.in_ucl = True
        s.commit()
    for tid in sorted(ids):
        try:
            js = espn.team_schedule(tid)
        except FetchError as e:
            log.warning("schedule %s: %s", tid, e)
            continue
        c = Counter()
        for ev in js.get("events") or []:
            slug = (ev.get("league") or {}).get("slug") or ""
            if (slug and not slug.startswith(("uefa", "club.", "fifa")) and "cup" not in slug
                    and slug.count(".") == 1):
                c[slug] += 1
        if c:
            dom = c.most_common(1)[0][0]
            leagues.add(dom)
            with SessionLocal() as s:
                t = s.get(Team, tid)
                if t:
                    t.domestic_league = dom
                    s.commit()
    return leagues


def fetch_details(match_ids: list[str], force: bool = False) -> int:
    n = 0
    for mid in match_ids:
        with SessionLocal() as s:
            m = s.get(Match, mid)
            if m is None or (m.detail_fetched and not force):
                continue
            league = m.league
        try:
            js = espn.summary(league, mid)
        except FetchError as e:
            log.warning("summary %s: %s", mid, e)
            continue
        det = espn.parse_summary(js)
        with SessionLocal() as s:
            m = s.get(Match, mid)
            if det.get("odds") and not m.completed:
                m.odds = det["odds"]
            for tid, stats in det["teams"].items():
                if m.completed and tid in (m.home_id, m.away_id):
                    goals = m.home_goals if tid == m.home_id else m.away_goals
                    _set_stats(s, mid, tid, tid == m.home_id, goals, stats)
            for tid, ro in det["rosters"].items():
                if tid not in (m.home_id, m.away_id):
                    continue
                if m.completed:
                    s.flush()
                    row = s.execute(select(TeamMatchStats).where(
                        TeamMatchStats.match_id == mid, TeamMatchStats.team_id == tid)).scalar_one_or_none()
                    if row:
                        row.formation = ro.get("formation")
                existing = {a.player_id: a for a in s.execute(
                    select(Appearance).where(Appearance.match_id == mid)).scalars()}
                for p in ro["players"]:
                    a = existing.get(p["player_id"])
                    if a is None:
                        a = Appearance(match_id=mid, team_id=tid, player_id=p["player_id"],
                                       player_name=p["player_name"], starter=p["starter"])
                        s.add(a)
                        existing[p["player_id"]] = a
                    for k in ("player_name", "position", "starter", "subbed_in", "goals", "assists",
                              "shots", "shots_on_target"):
                        setattr(a, k, p[k])
            if m.completed:
                m.detail_fetched = True
            s.commit()
            n += 1
    return n


def recent_matches_of_ucl_teams(limit_per_team: int = 20) -> list[str]:
    ids: list[str] = []
    with SessionLocal() as s:
        teams = [t.id for t in s.execute(select(Team).where(Team.in_ucl)).scalars()]
        for tid in teams:
            q = (select(Match.id).where(Match.completed, or_(Match.home_id == tid, Match.away_id == tid))
                 .order_by(Match.kickoff.desc()).limit(limit_per_team))
            ids.extend(r for (r,) in s.execute(q))
        ids.extend(r for (r,) in s.execute(select(Match.id).where(Match.league == UCL_SLUG, Match.completed)))
    return list(dict.fromkeys(ids))


def upcoming_ucl(days: int = 15) -> list[Match]:
    now = utcnow()
    with SessionLocal() as s:
        return list(s.execute(select(Match).where(
            Match.league == UCL_SLUG, Match.completed.is_(False),
            Match.kickoff >= now - timedelta(hours=3), Match.kickoff <= now + timedelta(days=days),
        ).order_by(Match.kickoff)).scalars())


def refresh_weather(matches: list[Match]) -> int:
    n = 0
    for m in matches:
        if not m.city:
            continue
        try:
            w = openmeteo.match_weather(m.city, m.country, m.kickoff)
        except FetchError as e:
            log.warning("weather %s: %s", m.id, e)
            continue
        if not w:
            continue
        with SessionLocal() as s:
            row = s.get(Weather, m.id) or Weather(match_id=m.id, source=openmeteo.SOURCE)
            for k, v in w.items():
                setattr(row, k, v)
            row.source = openmeteo.SOURCE
            row.fetched_at = utcnow()
            s.merge(row)
            s.commit()
            n += 1
    return n


def _state(key):
    from ..db import ModelState
    with SessionLocal() as s:
        st = s.get(ModelState, key)
        return st.value if st else None


def _set_state(key, value):
    from ..db import ModelState
    with SessionLocal() as s:
        row = s.get(ModelState, key) or ModelState(key=key, value=value)
        row.value = value
        s.merge(row)
        s.commit()


def refresh_injuries_transfermarkt(matches: list[Match], force: bool = False) -> int:
    """Bajas reportadas en Transfermarkt para los equipos de los próximos partidos UCL."""
    from dataclasses import asdict
    if not matches:
        return 0
    now = utcnow()
    tm_map = _state("tm_map") or {}
    team_ids = {m.home_id for m in matches} | {m.away_id for m in matches}
    if not all(t in tm_map for t in team_ids):
        try:
            clubs = transfermarkt.participants(settings.ucl_season_start)
        except Exception as e:  # noqa: BLE001 - se registra y se sigue
            log.warning("transfermarkt participantes: %s", e)
            clubs = []
        with SessionLocal() as s:
            names = {t.id: t.name for t in s.execute(select(Team).where(Team.in_ucl)).scalars()}
        tm_map.update(transfermarkt.match_teams(names, clubs))
        _set_state("tm_map", tm_map)
    cache = _state("tm_absences") or {}
    for tid in team_ids:
        club = tm_map.get(tid)
        if not club:
            continue
        entry = cache.get(tid)
        fresh = entry and (now - datetime.fromisoformat(entry["fetched_at"])).total_seconds() < transfermarkt.MIN_HOURS * 3600
        if fresh and not force:
            continue
        try:
            items = transfermarkt.club_absences(club["slug"], club["tm_id"])
        except Exception as e:  # noqa: BLE001
            log.warning("transfermarkt %s: %s", club["name"], e)
            continue
        cache[tid] = {"fetched_at": now.isoformat(), "club": club["name"],
                      "items": [{**asdict(a), "since": a.since and a.since.isoformat(),
                                 "expected_return": a.expected_return and a.expected_return.isoformat()} for a in items]}
    _set_state("tm_absences", cache)

    n = 0
    for m in matches:
        with SessionLocal() as s:
            for old in s.execute(select(Availability).where(
                    Availability.match_id == m.id, Availability.source == transfermarkt.SOURCE)).scalars():
                s.delete(old)
            for tid in (m.home_id, m.away_id):
                entry = cache.get(tid)
                if not entry:
                    continue
                items = [transfermarkt.Absence(**{**it,
                         "since": it["since"] and datetime.fromisoformat(it["since"]).date(),
                         "expected_return": it["expected_return"] and datetime.fromisoformat(it["expected_return"]).date()})
                         for it in entry["items"]]
                for a in transfermarkt.merge_for_match(items, m.kickoff.date()):
                    s.add(Availability(team_id=tid, match_id=m.id, player_name=a["player"], position=a["position"],
                                       reason=a["reason"][:300], status=a["status"], source=transfermarkt.SOURCE,
                                       fetched_at=datetime.fromisoformat(entry["fetched_at"])))
                    n += 1
            s.commit()
    return n


def injury_sources_status(team_ids: list[str]) -> dict:
    """Qué fuente de bajas cubre a estos equipos y con qué antigüedad."""
    cache = _state("tm_absences") or {}
    out = {}
    for t in team_ids:
        e = cache.get(t)
        out[t] = e and e["fetched_at"]
    return out


def refresh_injuries(matches: list[Match]) -> int:
    n = refresh_injuries_transfermarkt(matches)
    if not api_football.available():
        return n
    for m in matches:
        with SessionLocal() as s:
            home, away = s.get(Team, m.home_id), s.get(Team, m.away_id)
        try:
            fx = api_football.find_fixture(home.name, away.name, m.kickoff, settings.ucl_season_start)
            if not fx:
                continue
            items = api_football.injuries_for_fixture(fx["fixture"]["id"])
        except FetchError as e:
            log.warning("injuries %s: %s", m.id, e)
            continue
        with SessionLocal() as s:
            for old in s.execute(select(Availability).where(
                    Availability.match_id == m.id, Availability.source == api_football.SOURCE)).scalars():
                s.delete(old)
            for it in items:
                hn = api_football._norm(home.name)
                tn = api_football._norm(it["team_name"])
                tid = home.id if (hn in tn or tn in hn) else away.id
                s.add(Availability(team_id=tid, match_id=m.id, player_name=it["player_name"],
                                   reason=it["reason"], status=it["status"], source=api_football.SOURCE))
                n += 1
            s.commit()
    return n


def full_refresh(history: bool = True) -> dict:
    """Ingesta completa. `history=False` solo actualiza UCL próximo (rápido)."""
    now = utcnow()
    report: dict = {"started_at": now.isoformat()}
    if history:
        report["ucl_events"] = ingest_league_months(UCL_SLUG, ucl_months(now))
        leagues = discover_domestic_leagues()
        report["domestic_leagues"] = sorted(leagues)
        months = month_range(settings.history_from, now)
        n = 0
        for lg in sorted(leagues | set(EURO_SLUGS) - {UCL_SLUG}):
            n += ingest_league_months(lg, months)
        report["other_events"] = n
        report["details"] = fetch_details(recent_matches_of_ucl_teams())
    else:
        cur = f"{now.year:04d}{now.month:02d}"
        prev_dt = now.replace(day=1) - timedelta(days=1)
        months = [f"{prev_dt.year:04d}{prev_dt.month:02d}", cur] if now.day <= 7 else [cur]
        with SessionLocal() as s:
            leagues = {r for (r,) in s.execute(select(Match.league).distinct())} | {UCL_SLUG}
        report["events"] = sum(ingest_league_months(lg, months) for lg in sorted(leagues))
        report["details"] = fetch_details(recent_matches_of_ucl_teams(limit_per_team=6))
    up = upcoming_ucl()
    report["upcoming_detail"] = fetch_details([m.id for m in up if m.kickoff <= now + timedelta(days=3)], force=True)
    report["weather"] = refresh_weather(up)
    report["injuries"] = refresh_injuries(up)
    report["finished_at"] = utcnow().isoformat()
    return report
