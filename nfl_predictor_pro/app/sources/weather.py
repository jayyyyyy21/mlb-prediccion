"""Pronóstico del clima para partidos al aire libre (sin API key).

Fuente principal: Open-Meteo (~16 días hacia adelante, cualquier país).
Respaldos, en orden: Servicio Meteorológico Nacional de EE.UU. (api.weather.gov,
~7 días, sólo estadios en EE.UU.) y MET Norway (api.met.no, ~9 días, cualquier
país; datos CC BY 4.0), usados cuando Open-Meteo falla o limita consultas.
Si ninguno responde se devuelve None y la aplicación lo indica como dato
faltante (no se inventa clima).
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone

import requests

from ..config import HTTP_TIMEOUT, OPEN_METEO_URL, USER_AGENT, WEATHER_ENABLED

NWS_POINTS_URL = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
# met.no pide identificar la aplicación; NFLPP_CONTACT añade un contacto opcional (URL o correo).
METNO_USER_AGENT = USER_AGENT + (f" {os.environ['NFLPP_CONTACT']}" if os.environ.get("NFLPP_CONTACT") else "")

_cache: dict[tuple, dict | None] = {}
# Si un servicio no responde, se deja de consultar 10 minutos.
_down_until: dict[str, float] = {"open_meteo": 0.0, "nws": 0.0, "metno": 0.0}


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
    result = _guarded("open_meteo", _open_meteo, lat, lon, kickoff_utc, now)
    if result is None and days_ahead <= 7:
        result = _guarded("nws", _nws, lat, lon, kickoff_utc, now)
    if result is None and days_ahead <= 9:
        result = _guarded("metno", _metno, lat, lon, kickoff_utc, now)
    if result is not None:  # los fallos no se guardan: se reintenta en la próxima consulta
        _cache[key] = result
    return result


def _guarded(name, fn, lat, lon, kickoff_utc, now) -> dict | None:
    if time.time() < _down_until[name]:
        return None
    try:
        return fn(lat, lon, kickoff_utc, now)
    except (requests.ConnectionError, requests.Timeout):
        _down_until[name] = time.time() + 600
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:  # límite de consultas
            _down_until[name] = time.time() + 600
    except (requests.RequestException, KeyError, ValueError, IndexError):
        pass
    return None


def _avg(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _open_meteo(lat: float, lon: float, kickoff_utc: datetime, now: datetime) -> dict | None:
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
    r = requests.get(OPEN_METEO_URL, params=params, timeout=min(HTTP_TIMEOUT, 8), headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    hourly = r.json()["hourly"]
    times = hourly["time"]
    target = kickoff_utc.strftime("%Y-%m-%dT%H:00")
    # promedio de las ~3 horas de juego
    idx = times.index(target) if target in times else 0
    sl = slice(idx, min(idx + 3, len(times)))
    return {
        "temp_f": _avg(hourly.get("temperature_2m", [])[sl]),
        "wind_mph": _avg(hourly.get("wind_speed_10m", [])[sl]),
        "gust_mph": _avg(hourly.get("wind_gusts_10m", [])[sl]),
        "precip_prob": _avg(hourly.get("precipitation_probability", [])[sl]),
        "precip_mm": _avg(hourly.get("precipitation", [])[sl]),
        "source": "Open-Meteo (pronóstico)",
        "fetched_at": now.replace(microsecond=0).isoformat(),
    }


def _nws(lat: float, lon: float, kickoff_utc: datetime, now: datetime) -> dict | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    timeout = min(HTTP_TIMEOUT, 10)
    r = requests.get(NWS_POINTS_URL.format(lat=lat, lon=lon), timeout=timeout, headers=headers)
    if r.status_code == 404:  # fuera de EE.UU.
        return None
    r.raise_for_status()
    hourly_url = r.json()["properties"]["forecastHourly"]
    r = requests.get(hourly_url, timeout=timeout, headers=headers)
    r.raise_for_status()
    periods = r.json()["properties"]["periods"]
    game = []
    for p in periods:
        start = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
        hours = (start - kickoff_utc).total_seconds() / 3600
        if -1 < hours < 3:  # las ~3 horas de juego
            game.append(p)
    if not game:
        return None

    def temp_f(p):
        t = p.get("temperature")
        return t if p.get("temperatureUnit") == "F" or t is None else t * 9 / 5 + 32

    def wind(p):  # "10 mph" o "10 to 15 mph"
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", p.get("windSpeed") or "")]
        return sum(nums) / len(nums) if nums else None

    return {
        "temp_f": _avg([temp_f(p) for p in game]),
        "wind_mph": _avg([wind(p) for p in game]),
        "gust_mph": None,
        "precip_prob": _avg([(p.get("probabilityOfPrecipitation") or {}).get("value") for p in game]),
        "precip_mm": None,
        "conditions": game[0].get("shortForecast"),
        "source": "Servicio Meteorológico Nacional de EE.UU. (pronóstico)",
        "fetched_at": now.replace(microsecond=0).isoformat(),
    }


def _metno(lat: float, lon: float, kickoff_utc: datetime, now: datetime) -> dict | None:
    r = requests.get(METNO_URL, params={"lat": round(lat, 4), "lon": round(lon, 4)},
                     timeout=min(HTTP_TIMEOUT, 10), headers={"User-Agent": METNO_USER_AGENT})
    r.raise_for_status()
    game = []
    for step in r.json()["properties"]["timeseries"]:
        t = datetime.fromisoformat(step["time"].replace("Z", "+00:00"))
        hours = (t - kickoff_utc).total_seconds() / 3600
        if -1 < hours < 3:  # las ~3 horas de juego
            game.append(step["data"])
    if not game:
        return None
    inst = [g["instant"]["details"] for g in game]
    rain = [(g.get("next_1_hours") or {}).get("details", {}).get("precipitation_amount") for g in game]
    temp_c = _avg([d.get("air_temperature") for d in inst])
    wind_ms = _avg([d.get("wind_speed") for d in inst])
    return {
        "temp_f": round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None,
        "wind_mph": round(wind_ms * 2.23694, 1) if wind_ms is not None else None,
        "gust_mph": None,
        "precip_prob": None,
        "precip_mm": _avg(rain),
        "conditions": (game[0].get("next_1_hours") or {}).get("summary", {}).get("symbol_code"),
        "source": "MET Norway (pronóstico, CC BY 4.0)",
        "fetched_at": now.replace(microsecond=0).isoformat(),
    }
