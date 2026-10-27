"""Ubicación de estadios (coordenadas públicas y zona horaria IANA).

Se usa para estimar distancia de viaje, cambio de zona horaria y para pedir el
pronóstico del clima. Las claves son los `stadium_id` del calendario de nflverse.
Si aparece un estadio desconocido, el sistema lo marca como dato faltante en
lugar de inventar una ubicación.
"""
from __future__ import annotations

import math

# stadium_id: (nombre, latitud, longitud, zona horaria)
STADIUMS: dict[str, tuple[str, float, float, str]] = {
    "ATL00": ("Georgia Dome", 33.7577, -84.4008, "America/New_York"),
    "ATL97": ("Mercedes-Benz Stadium", 33.7554, -84.4008, "America/New_York"),
    "BAL00": ("M&T Bank Stadium", 39.2780, -76.6227, "America/New_York"),
    "BOS00": ("Gillette Stadium", 42.0909, -71.2643, "America/New_York"),
    "BUF00": ("Highmark Stadium", 42.7738, -78.7870, "America/New_York"),
    "CAR00": ("Bank of America Stadium", 35.2258, -80.8528, "America/New_York"),
    "CHI98": ("Soldier Field", 41.8623, -87.6167, "America/Chicago"),
    "CIN00": ("Paycor Stadium", 39.0955, -84.5161, "America/New_York"),
    "CLE00": ("Huntington Bank Field", 41.5061, -81.6995, "America/New_York"),
    "DAL00": ("AT&T Stadium", 32.7473, -97.0945, "America/Chicago"),
    "DEN00": ("Empower Field at Mile High", 39.7439, -105.0201, "America/Denver"),
    "DET00": ("Ford Field", 42.3400, -83.0456, "America/Detroit"),
    "GNB00": ("Lambeau Field", 44.5013, -88.0622, "America/Chicago"),
    "HOU00": ("NRG Stadium", 29.6847, -95.4107, "America/Chicago"),
    "IND00": ("Lucas Oil Stadium", 39.7601, -86.1639, "America/Indiana/Indianapolis"),
    "JAX00": ("EverBank Stadium", 30.3239, -81.6373, "America/New_York"),
    "KAN00": ("Arrowhead Stadium", 39.0489, -94.4839, "America/Chicago"),
    "LAX01": ("SoFi Stadium", 33.9535, -118.3392, "America/Los_Angeles"),
    "LAX97": ("StubHub Center", 33.8644, -118.2611, "America/Los_Angeles"),
    "LAX99": ("Los Angeles Memorial Coliseum", 34.0141, -118.2879, "America/Los_Angeles"),
    "MIA00": ("Hard Rock Stadium", 25.9580, -80.2389, "America/New_York"),
    "MIN01": ("U.S. Bank Stadium", 44.9737, -93.2581, "America/Chicago"),
    "NAS00": ("Nissan Stadium", 36.1665, -86.7713, "America/Chicago"),
    "NOR00": ("Caesars Superdome", 29.9511, -90.0812, "America/Chicago"),
    "NYC01": ("MetLife Stadium", 40.8135, -74.0745, "America/New_York"),
    "OAK00": ("Oakland Coliseum", 37.7516, -122.2005, "America/Los_Angeles"),
    "PHI00": ("Lincoln Financial Field", 39.9008, -75.1675, "America/New_York"),
    "PHO00": ("State Farm Stadium", 33.5276, -112.2626, "America/Phoenix"),
    "PIT00": ("Acrisure Stadium", 40.4468, -80.0158, "America/New_York"),
    "SDG00": ("Qualcomm Stadium", 32.7831, -117.1196, "America/Los_Angeles"),
    "SEA00": ("Lumen Field", 47.5952, -122.3316, "America/Los_Angeles"),
    "SFO01": ("Levi's Stadium", 37.4030, -121.9700, "America/Los_Angeles"),
    "TAM00": ("Raymond James Stadium", 27.9759, -82.5033, "America/New_York"),
    "VEG00": ("Allegiant Stadium", 36.0909, -115.1833, "America/Los_Angeles"),
    "WAS00": ("Northwest Stadium", 38.9078, -76.8645, "America/New_York"),
    # Sedes internacionales / neutrales
    "LON00": ("Wembley Stadium", 51.5560, -0.2796, "Europe/London"),
    "LON01": ("Twickenham Stadium", 51.4560, -0.3415, "Europe/London"),
    "LON02": ("Tottenham Hotspur Stadium", 51.6043, -0.0664, "Europe/London"),
    "FRA00": ("Deutsche Bank Park", 50.0686, 8.6455, "Europe/Berlin"),
    "GER00": ("Allianz Arena", 48.2188, 11.6247, "Europe/Berlin"),
    "MUN01": ("FC Bayern Munich Stadium", 48.2188, 11.6247, "Europe/Berlin"),
    "MEX00": ("Estadio Azteca", 19.3029, -99.1505, "America/Mexico_City"),
    "SAO00": ("Arena Corinthians", -23.5453, -46.4742, "America/Sao_Paulo"),
    "RIO00": ("Maracana Stadium", -22.9121, -43.2302, "America/Sao_Paulo"),
    "MAD01": ("Estadio Santiago Bernabeu", 40.4531, -3.6883, "Europe/Madrid"),
    "PAR00": ("Stade de France", 48.9245, 2.3602, "Europe/Paris"),
    "MEL00": ("Melbourne Cricket Ground", -37.8200, 144.9834, "Australia/Melbourne"),
}

# Estadio local de cada franquicia (para estimar el origen del viaje).
TEAM_HOME: dict[str, str] = {
    "ARI": "PHO00", "ATL": "ATL97", "BAL": "BAL00", "BUF": "BUF00", "CAR": "CAR00",
    "CHI": "CHI98", "CIN": "CIN00", "CLE": "CLE00", "DAL": "DAL00", "DEN": "DEN00",
    "DET": "DET00", "GB": "GNB00", "HOU": "HOU00", "IND": "IND00", "JAX": "JAX00",
    "KC": "KAN00", "LV": "VEG00", "LAC": "LAX01", "LA": "LAX01", "MIA": "MIA00",
    "MIN": "MIN01", "NE": "BOS00", "NO": "NOR00", "NYG": "NYC01", "NYJ": "NYC01",
    "PHI": "PHI00", "PIT": "PIT00", "SF": "SFO01", "SEA": "SEA00", "TB": "TAM00",
    "TEN": "NAS00", "WAS": "WAS00",
}

# Sedes históricas (para el backtest) de franquicias que cambiaron de estadio.
HISTORIC_HOME: dict[tuple[str, int], str] = {}
for _s in range(2016, 2020):
    HISTORIC_HOME[("LV", _s)] = "OAK00"
for _s in range(2016, 2020):
    HISTORIC_HOME[("LA", _s)] = "LAX99"
HISTORIC_HOME[("LAC", 2016)] = "SDG00"
for _s in range(2017, 2020):
    HISTORIC_HOME[("LAC", _s)] = "LAX97"
HISTORIC_HOME[("ATL", 2016)] = "ATL00"


def home_stadium(team: str, season: int) -> str | None:
    return HISTORIC_HOME.get((team, season), TEAM_HOME.get(team))


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))
