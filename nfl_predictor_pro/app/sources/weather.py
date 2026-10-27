"""Pronóstico del clima para partidos al aire libre (Open-Meteo, sin API key).

Open-Meteo sólo ofrece pronóstico ~16 días hacia adelante. Para partidos más
lejanos o si el servicio no responde se devuelve None y la aplicación lo
indica como dato faltante (no se inventa clima).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

from ..config import HTTP_TIMEOUT, OPEN_METEO_URL, USER_AGENT, WEATHER_ENABLED

_cache: dict[tuple, dict | None] = {}
_down_until: float = 0.0  # si el servicio no responde, se deja de consultar 10 minutos


def forecast(lat: float, lon: float, kickoff_utc: datetime) -> dict | None:
    if not WEATHER_ENABLED:
        return None
    now = datetime.now(timezone.utc)
    days_ahead = (kickoff_utc - now).total_seconds() / 86400
    if days_ahead > 15 or days_ahead < -1:
        return None
    key = (round(lat, 3), round(lon, 3), kickoff_utc.strftime("%Y-%m-%dT%H"))
    if key in _cache:
        return _cache[key]
    global _down_until
    if time.time() < _down_until:
        return None
    day = kickoff_utc.strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": day,
        "end_date": day,
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=min(HTTP_TIMEOUT, 8), headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        hourly = r.json()["hourly"]
        times = hourly["time"]
        target = kickoff_utc.strftime("%Y-%m-%dT%H:00")
        # promedio de las ~3 horas de juego
        idx = times.index(target) if target in times else 0
        sl = slice(idx, min(idx + 3, len(times)))

        def avg(name: str) -> float | None:
            vals = [v for v in hourly.get(name, [])[sl] if v is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        result = {
            "temp_f": avg("temperature_2m"),
            "wind_mph": avg("wind_speed_10m"),
            "gust_mph": avg("wind_gusts_10m"),
            "precip_prob": avg("precipitation_probability"),
            "precip_mm": avg("precipitation"),
            "source": "Open-Meteo (pronóstico)",
            "fetched_at": now.replace(microsecond=0).isoformat(),
        }
    except requests.ConnectionError:
        _down_until = time.time() + 600
        return None
    except requests.Timeout:
        _down_until = time.time() + 600
        return None
    except (requests.RequestException, KeyError, ValueError):
        result = None
    _cache[key] = result
    return result
