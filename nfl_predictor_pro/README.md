# NFL PREDICTOR PRO

Aplicación web para pronosticar partidos de la NFL con estadísticas avanzadas, ratings ajustados por rival,
simulación Monte Carlo y validación histórica fuera de muestra. **No usa cuotas de apuestas**: las columnas de
mercado se eliminan en la ingesta y un test verifica que ninguna entra al modelo.

Responde cuatro preguntas por partido:

1. ¿Cuántos puntos debería anotar cada equipo contra la defensa específica de su rival?
2. ¿Qué equipo tiene mayor probabilidad estadística de ganar?
3. ¿Qué ventajas tácticas y lesiones pueden decidir el resultado?
4. ¿Qué tan confiable es el pronóstico según los datos disponibles y el desempeño histórico del modelo?

## Arquitectura

| Capa | Implementación |
|---|---|
| Frontend | HTML + JS sin build (`app/static`), Chart.js; oscuro, responsive, en español |
| Backend | FastAPI (`app/main.py`), API REST |
| Base de datos | SQLite (`data/nfl.db`): datos agregados, fecha de actualización por fuente, registro de pronósticos |
| Motor | `ratings.py` (ridge ajustado por rival) → `model.py` (proyección de puntos) → `injuries.py` → `simulate.py` (Monte Carlo) |
| Validación | `backtest.py` (walk-forward, validación/prueba separadas, comparación con modelos de referencia) |

## Instalación y uso

```bash
cd nfl_predictor_pro
pip install -r requirements.txt
python -m app.cli init      # descarga 2016→hoy (~200 MB) y calibra el modelo (~3 min)
python -m app.cli serve     # http://localhost:8000
```

Otros comandos: `python -m app.cli update` (datos nuevos), `python -m app.cli backtest` (recalibrar),
`python -m app.cli predict --week 9` (pronósticos en consola), `python -m pytest tests`.

El servidor actualiza los datos cada 3 horas (`NFLPP_AUTO_UPDATE_HOURS`) y recalcula los pronósticos de la
semana en curso; también hay un botón **Actualizar** en la interfaz (`POST /api/update`).

## Fuentes de datos

Ninguna fuente requiere API key ni credenciales.

| Fuente | Uso | Condiciones |
|---|---|---|
| [nflverse-data](https://github.com/nflverse/nflverse-data) | Calendario, resultados, play-by-play (EPA, success rate), reporte oficial de lesiones, snaps, rosters | CC-BY 4.0 (atribuir a nflverse) |
| FTN Data vía nflverse | Blitz y pass rushers (desde 2022) | CC-BY-SA 4.0 |
| ESPN (endpoint público) | Estado de lesiones entre reportes oficiales; marcado como **no oficial** | Endpoint no documentado; uso personal, puede cambiar |
| [Open-Meteo](https://open-meteo.com) | Pronóstico del clima (≤15 días) en estadios abiertos | Gratis para uso no comercial |

NFL.com y Pro Football Reference no ofrecen API abierta; sus términos prohíben el scraping, así que no se
consultan directamente (nflverse publica datos equivalentes con licencia abierta).

**Escudos:** los logos de la NFL son marcas registradas. Por defecto se muestran insignias con las siglas y
colores del equipo. Si cuenta con autorización, active `NFLPP_SHOW_LOGOS=true`.

### Variables de entorno

| Variable | Defecto | Descripción |
|---|---|---|
| `NFLPP_N_SIMS` | 10000 | Simulaciones por partido (mínimo 10,000) |
| `NFLPP_SEASON` | automática | Temporada actual |
| `NFLPP_FIRST_SEASON` | 2016 | Primera temporada descargada |
| `NFLPP_WEATHER` | true | Consultar Open-Meteo |
| `NFLPP_SHOW_LOGOS` | false | Mostrar escudos (sólo con autorización) |
| `NFLPP_AUTO_UPDATE_HOURS` | 3 | Frecuencia de actualización automática (0 = desactivada) |
| `NFLPP_DATA_DIR` | `./data` | Carpeta de datos |

## Metodología (resumen)

La explicación completa, con fórmulas, está en la pestaña **Metodología** (`app/methodology.py`).

1. **Ratings ajustados por rival.** Para cada métrica `y(ofensiva A, defensa B) = μ + off[A] + def[B] + h·localía`,
   regresión ridge ponderada (decaimiento exponencial, vida media de 10 semanas; λ = 4 partidos de regresión a la media).
2. **Proyección de puntos.** Regresión ridge sobre 15 variables: puntos ajustados, forma reciente, EPA esperado
   (pase/carrera de A contra lo que permite B, ponderado por tendencia de pase), puntos por posesión × posesiones,
   success rate, capturas, pérdidas, explosivas, zona roja, localía, descanso, zona horaria, distancia, viento y frío.
   La localía no se asume: su efecto es un coeficiente estimado (~+0.8 pts por equipo).
3. **Lesiones.** Sólo bajas confirmadas (reporte oficial *Out*, IR/PUP/NFI, suspensión) entran a la proyección
   principal. *Doubtful/Questionable* se muestran como escenario. QB: diferencia de EPA/dropback titular vs suplente
   × dropbacks por partido. Otras posiciones: valor posicional a priori × % de snaps recientes (no calibrado; se indica).
4. **Monte Carlo.** 10,000 partidos simulados por posesiones (TD/FG/nada) con ruido calibrado con residuos históricos;
   incluye prórroga y empates.
5. **Validación walk-forward.** Ratings con datos previos a cada semana, coeficientes con temporadas previas.
   Hiperparámetros elegidos en 2019–2021; métricas de prueba en 2022–2026:

| Modelo (prueba, 1,246 partidos) | MAE pts/equipo | MAE margen | Brier | Acierto |
|---|---|---|---|---|
| NFL Predictor Pro | 7.33 | 9.89 | 0.223 | 63.7% |
| Promedios simples PF/PC | 7.68 | 10.38 | 0.236 | 60.0% |
| Promedio de la liga | 7.80 | 10.79 | 0.247 | 54.8% |

Limitaciones: el backtest no incluye ajustes por lesiones; usa el clima registrado como proxy del pronóstico;
los valores posicionales de lesiones no-QB son supuestos a priori; el suplente del QB se estima por experiencia
(no hay depth chart oficial abierto confiable).

## API

`GET /api/status`, `/api/weeks`, `/api/predictions?season=&week=`, `/api/game/{game_id}`, `/api/compare?a=&b=`,
`/api/team/{team}`, `/api/ratings`, `/api/history`, `/api/methodology`, `/api/export?fmt=csv|json`,
`/api/export/game/{game_id}`; `POST /api/simulate`, `POST /api/update`. Documentación interactiva en `/docs`.
