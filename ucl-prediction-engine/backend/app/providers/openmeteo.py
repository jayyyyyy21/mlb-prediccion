"""Clima y altitud del estadio con Open-Meteo (sin clave)."""
from __future__ import annotations

from datetime import datetime, timedelta

from .http import FetchError, HttpClient

SOURCE = "Open-Meteo (open-meteo.com)"
_client: HttpClient | None = None
_geo_cache: dict[str, dict | None] = {}


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient("open-meteo", min_interval=0.2)
    return _client


def geocode(city: str, country: str | None = None) -> dict | None:
    key = f"{city}|{country}"
    if key in _geo_cache:
        return _geo_cache[key]
    js = client().get_json("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 5, "language": "en"})
    res = js.get("results") or []
    pick = None
    if country:
        c = country.lower()
        for r in res:
            if c in (r.get("country") or "").lower() or (r.get("country") or "").lower() in c:
                pick = r
                break
    pick = pick or (res[0] if res else None)
    _geo_cache[key] = pick
    return pick


def match_weather(city: str, country: str | None, kickoff_utc: datetime) -> dict | None:
    """Pronóstico (o dato observado si ya pasó) para la hora de inicio. None si no hay datos."""
    geo = geocode(city, country)
    if not geo:
        return None
    day = kickoff_utc.date()
    if kickoff_utc - datetime.utcnow() > timedelta(days=15):
        return {"elevation_m": geo.get("elevation"), "temperature_c": None,
                "precipitation_mm": None, "wind_kmh": None}
    try:
        js = client().get_json("https://api.open-meteo.com/v1/forecast", params={
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "start_date": day.isoformat(), "end_date": day.isoformat(), "timezone": "GMT",
        })
    except FetchError:
        return {"elevation_m": geo.get("elevation"), "temperature_c": None,
                "precipitation_mm": None, "wind_kmh": None}
    h = js.get("hourly") or {}
    times = h.get("time") or []
    target = kickoff_utc.strftime("%Y-%m-%dT%H:00")
    idx = times.index(target) if target in times else None
    def at(name):
        arr = h.get(name) or []
        return arr[idx] if idx is not None and idx < len(arr) else None
    return {
        "elevation_m": js.get("elevation", geo.get("elevation")),
        "temperature_c": at("temperature_2m"),
        "precipitation_mm": at("precipitation"),
        "wind_kmh": at("wind_speed_10m"),
    }
