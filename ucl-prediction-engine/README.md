# UCL Prediction Engine

Aplicación web de pronósticos estadísticos para la **UEFA Champions League**, basada en datos reales y
verificables, con incertidumbre explícita y validación cronológica. No elige favoritos por reputación:
todo sale de un modelo cuantitativo ajustado por calidad de rivales, y cada número indica si es
**observado**, **estimado por el modelo** o **no confirmado**.

```
ucl-prediction-engine/
├── backend/            FastAPI + SQLAlchemy + modelos (Python)
│   ├── app/providers/  ESPN, Open-Meteo, API-Football (opcional)
│   ├── app/models/     Modelo jerárquico, Dixon-Coles, binomial negativa, Monte Carlo, backtest
│   ├── app/services/   Ingesta, perfiles, táctica, bajas, oportunidades, pronósticos
│   └── tests/
├── frontend/           React + TypeScript + Vite + Recharts
├── Dockerfile          Imagen única (frontend compilado servido por FastAPI)
└── docker-compose.yml  App + PostgreSQL
```

## Puesta en marcha

### Con Docker (PostgreSQL)

```bash
cd ucl-prediction-engine
API_FOOTBALL_KEY=xxxx ADMIN_TOKEN=secreto docker compose up -d --build
# Primera carga de datos (≈5 min) y validación del modelo (≈10–15 min):
docker compose exec app python -m app.cli ingest
docker compose exec app python -m app.cli backtest
docker compose exec app python -m app.cli predict
# → http://localhost:8000
```

### Local (SQLite por defecto)

```bash
cd ucl-prediction-engine/backend
pip install -r requirements.txt
cp .env.example .env              # edita DATABASE_URL / API_FOOTBALL_KEY / ADMIN_TOKEN
python -m app.cli ingest          # descarga histórico desde HISTORY_FROM (2024-08)
python -m app.cli backtest        # elige hiperparámetros en validación y evalúa en prueba
python -m app.cli predict         # pronósticos de los próximos 60 días
uvicorn app.main:app --port 8000

cd ../frontend
npm install
npm run dev                       # http://localhost:5173 (proxy /api → :8000)
```

`python -m app.cli ingest --quick` hace una actualización rápida (mes actual, alineaciones, clima,
cuotas, bajas). Con `REFRESH_INTERVAL_MIN>0` el servidor la ejecuta automáticamente y recalcula los
pronósticos; también `POST /api/admin/refresh` con cabecera `X-Admin-Token`.

## Fuentes de datos

| Fuente | Estado | Qué aporta |
|---|---|---|
| ESPN (`site.api.espn.com`, sin clave) | Conectada | Calendario UCL, resultados, tiros, tiros a puerta, córners, posesión, faltas; boxscore detallado (pases, centros, balones largos, entradas, intercepciones, despejes); formaciones; alineaciones por partido; cuotas publicadas (DraftKings). Ligas: las domésticas de los 36 equipos que cubre ESPN + UEL, UECL y fases previas. |
| Open-Meteo (sin clave) | Conectada | Temperatura, lluvia, viento y altitud del estadio. |
| Transfermarkt (páginas públicas, sin clave) | Conectada | Lesiones (motivo, desde, regreso previsto), sanciones con la competición a la que aplican y jugadores no inscritos en la lista UCL. Se descarga como mucho cada 6 h, con pausas entre peticiones. Son bajas *reportadas*, no partes médicos oficiales. |
| API-Football (api-sports.io) | Opcional (`API_FOOTBALL_KEY`) | Segunda fuente de lesiones y sanciones. |
| xG real (Opta / StatsBomb / Understat / FBref) | **No conectado** | Requieren licencia o bloquean el acceso automatizado. Se usa un **xG‑proxy** basado en tiros a puerta y se etiqueta como tal. |

Ligas domésticas no cubiertas por ESPN (p. ej. Chequia, Eslovaquia, Azerbaiyán, Ucrania): esos equipos
solo tienen datos europeos, se agrupan por país y su confianza se marca como **Baja**.

Control de calidad: ESPN publica a veces estadísticas ausentes como ceros (frecuente en fases previas);
esos partidos conservan el resultado pero sus estadísticas se descartan.

## Modelo

1. **Fuerza de equipos (GLM de Poisson penalizado, jerárquico)** para goles, córners, tiros a puerta y
   tiros: `log E[y] = c + h·local + (A_liga + a_equipo) − (D_liga + d_equipo) + γ·covariables`.
   Los niveles de liga se identifican con los cruces europeos; las desviaciones de equipo se encogen
   hacia su liga (L2) para evitar sobreajuste. Ponderación temporal exponencial.
2. **Goles**: objetivo = mezcla de goles reales y xG‑proxy (peso elegido en validación) + corrección
   **Dixon‑Coles** de marcadores bajos (ρ estimado por máxima verosimilitud).
3. **Covariables estimadas de los datos** (no supuestos): descanso corto (≤3 días) de cada equipo,
   partido de competición europea, fase previa.
4. **Córners y tiros a puerta**: binomial negativa con sobredispersión estimada; el total se calibra
   hacia la media europea reciente con un peso elegido en validación (el reparto entre equipos se
   conserva, que es donde está la señal predictiva).
5. **Posesión**: regresión ridge sobre logit(posesión).
6. **Incertidumbre**: aproximación de Laplace (inversa del hessiano) → bandas p10–p90.
7. **Monte Carlo**: 20.000 simulaciones por partido (goles, córners, tiros a puerta).
8. **Bajas** (Transfermarkt + alineaciones ESPN): «Baja» si el regreso previsto es posterior al partido o
   la lesión es larga; «Duda» si no hay fecha o el regreso es justo antes (cuenta al 50%); las sanciones de
   otras competiciones no aplican en Champions. Multiplicador ofensivo según la cuota de goles+asistencias y tiros del jugador ausente
   (supuesto explícito: el sustituto recupera el 50%). Rumores nunca alteran el modelo.
9. **Ajuste táctico por estilo** (rendimiento frente a rivales de perfil de posesión similar, encogido):
   solo se activa si mejora la validación.

### Validación

`python -m app.cli backtest` hace walk‑forward semanal sobre partidos de UCL/UEL/UECL:
hiperparámetros elegidos en **validación (2025)**, métricas reportadas en **prueba (2026)** sin tocar.
Log loss, Brier, RPS, calibración (ECE y gráficas) y MAE, contra una referencia ingenua y contra el
mercado cuando hay cuotas verificadas. Resultados en la pantalla **Validación del modelo**.

Resultado del backtest ejecutado el 03/11/2026 (prueba: 351 partidos de UCL/UEL/UECL, 20/01/2026 → 22/10/2026):

| Mercado | Modelo | Referencia ingenua |
|---|---|---|
| 1X2 · log loss / Brier / RPS | **0,919 / 0,542 / 0,200** | 1,018 / 0,615 / 0,237 |
| 1X2 · error de calibración (ECE) | 4,1 % | — |
| Más de 2,5 goles · log loss | **0,660** | 0,674 |
| Ambos marcan · log loss | 0,693 | **0,690** (el modelo no mejora → mercado marcado como inestable) |
| Córners +9,5 · log loss | **0,680** | 0,688 |
| Tiros a puerta +8,5 · log loss | **0,654** | 0,668 |
| Posesión · MAE | **6,9 pp** | 9,0 pp |

Hiperparámetros elegidos en validación: vida media 400 días, peso del xG‑proxy 0,5, λ equipo 4,
λ conteos 20, peso del modelo en el total de córners/tiros 0,25. El ajuste táctico por estilo **no**
mejoró la validación y está desactivado. No hay suficientes cuotas históricas para compararse con el mercado.

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/matches?from=&to=&team=` | Calendario UCL con resumen de pronóstico |
| GET | `/api/matches/{id}` | Pronóstico completo (predicción, táctica, estadísticas, bajas, oportunidades, transparencia) |
| GET | `/api/matches/{id}/history` | Versiones del pronóstico y motivo de cada actualización |
| GET | `/api/teams`, `/api/teams/{id}` | Equipos UCL y perfil estadístico |
| GET | `/api/standings` | Clasificación de la fase de liga |
| GET | `/api/backtest` | Último backtest |
| GET | `/api/status` | Integraciones, frescura de datos, errores |
| POST | `/api/admin/refresh?full=` | Actualización (cabecera `X-Admin-Token`) |
| POST | `/api/admin/availability` | Alta manual de una baja (`confirmed`/`probable`/`rumor`) con fuente |

## Limitaciones conocidas

- Sin xG real, sin datos de balón parado, errores defensivos, PPDA real ni minutos por jugador.
- El total de córners tiene poca señal predecible; el backtest lo muestra y el detector de
  oportunidades lo penaliza.
- Cuotas solo cuando ESPN las publica (1X2 y una línea de goles). El valor nunca se evalúa sin cuota.
- Pronósticos estadísticos, no garantías ni consejo de apuesta.

## Página estática (sin servidor)

```bash
cd backend && python -m app.export_static --out /tmp/snap
cd ../frontend && VITE_STATIC=1 npx vite build --outDir /tmp/dist
python ../scripts/single_file.py /tmp/snap /tmp/dist /tmp/ucl.html   # un solo HTML con datos y escudos
```
La página muestra siempre la fecha de la instantánea: no se actualiza en vivo.
