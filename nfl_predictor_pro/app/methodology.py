"""Texto de metodología mostrado en la aplicación (fórmulas y supuestos)."""

METHODOLOGY = [
    {
        "title": "1. Datos y fuentes",
        "body": [
            "Calendario, marcadores, sede, techo, superficie, descanso y clima registrado: nflverse (schedules/games.csv).",
            "Estadísticas jugada por jugada (EPA, success rate, posesiones, zona roja, capturas, pérdidas): nflverse play-by-play (modelo EP de nflfastR).",
            "Blitz y número de pass rushers: charting abierto de FTN Data publicado por nflverse (desde 2022, licencia CC-BY-SA 4.0).",
            "Reporte oficial de lesiones de la NFL, snaps por jugador y rosters semanales: nflverse.",
            "Estado de lesiones actualizado entre reportes oficiales: endpoint público de ESPN (se etiqueta como no oficial).",
            "Pronóstico del clima: Open-Meteo (sin API key), sólo para partidos al aire libre dentro de ~15 días.",
            "Las columnas de mercado (spread, total, moneyline) se eliminan al descargar el calendario: el modelo NUNCA usa cuotas.",
            "Cada fuente guarda fecha y hora de su última actualización (pestaña Metodología y barra superior).",
        ],
    },
    {
        "title": "2. Ratings ajustados por rival",
        "body": [
            "Para cada métrica y (puntos, EPA/jugada, EPA por pase, EPA por carrera, success rate, posesiones, puntos por posesión, tasa de capturas, pérdidas, explosivas, TD en zona roja, terceros downs) se ajusta:",
            "y(ofensiva A, defensa B) = μ + off[A] + def[B] + h·localía + ε",
            "por mínimos cuadrados ponderados con penalización ridge λ·Σ(off² + def²). La penalización equivale a añadir λ partidos 'promedio' a cada equipo: con pocos partidos el rating se acerca al promedio de la liga (control de muestras pequeñas).",
            "Peso de cada partido: w = (jugadas/jugadas promedio) · 0.5^(antigüedad en semanas / vida media). Así los partidos recientes pesan más y la temporada anterior aporta información al inicio de la nueva.",
            "Al ajustar por rival, 30 puntos contra la mejor defensa valen más que 30 contra la peor.",
            "Se calcula además una versión de 'forma reciente' con vida media corta (3 semanas).",
        ],
    },
    {
        "title": "3. Proyección de puntos",
        "body": [
            "Los puntos del equipo A contra la defensa de B se estiman con una regresión ridge sobre variables estandarizadas:",
            "Puntos_A = β0 + β1·[μ + off_pts(A) + def_pts(B)] + β2·forma reciente + β3·EPA esperado + β4·(pts por posesión × posesiones esperadas) + β5·success rate + β6·capturas + β7·pérdidas + β8·explosivas + β9·zona roja + β10·localía + β11·descanso + β12·zona horaria + β13·distancia + β14·viento + β15·frío",
            "EPA esperado = tasa de pase esperada × (EPA pase ofensiva A + EPA pase permitido por B) + (1 − tasa de pase) × (EPA carrera A + EPA carrera permitido por B): combina el estilo de A con las debilidades de B.",
            "Posesiones esperadas = promedio de las tendencias de ritmo de ambos equipos (ajustadas por rival).",
            "El mismo cálculo se hace para B contra la defensa de A.",
            "Los coeficientes β se estiman con temporadas anteriores (walk-forward). La localía NO se asume: su efecto es el coeficiente estimado con datos.",
        ],
    },
    {
        "title": "4. Lesiones",
        "body": [
            "Sólo se aplican en la proyección principal las bajas confirmadas: reporte oficial 'Out', listas de reserva (IR/PUP/NFI) o suspensiones.",
            "'Doubtful' y 'Questionable' no se tratan como bajas: se muestran como escenario ('si no juega…') y bajan la confianza si se trata del QB titular. El simulador permite incluirlos.",
            "QB: impacto = (EPA/dropback del suplente − EPA/dropback del titular) × dropbacks por partido del equipo. El EPA de cada QB se encoge hacia el nivel de reemplazo con 250 dropbacks a priori (los QB con poca muestra se asumen de nivel suplente). Tope ±10 pts.",
            "Otras posiciones: impacto = valor posicional a priori × % de snaps del jugador en los últimos 3 partidos del equipo (máx. 4 pts por unidad). Valores a priori (pts por partido para un titular a tiempo completo): WR 0.55, TE 0.40, RB 0.30, T 0.45, G 0.30, C 0.35, DE 0.45, DT 0.30, LB 0.25, CB 0.45, S 0.30.",
            "Estos valores a priori NO están calibrados con el backtest (no existen reportes históricos tal como se conocían antes de cada partido). Si un jugador ya faltó en partidos recientes, su % de snaps reciente es bajo y su impacto también: los ratings ya reflejan su ausencia.",
        ],
    },
    {
        "title": "5. Simulación Monte Carlo",
        "body": [
            "10,000 escenarios por partido (configurable, mínimo 10,000).",
            "En cada escenario: (1) se perturba la expectativa de puntos de cada equipo con ruido normal bivariado; (2) se simula el número de posesiones; (3) cada posesión termina en TD (6 + PAT 94% o conversión de 2), gol de campo (3) o nada, con probabilidades que reproducen los puntos por posesión esperados; (4) si hay empate se simula la prórroga (en temporada regular puede terminar empatado).",
            "La varianza total y la correlación entre los puntos de ambos equipos se calibran con los residuos históricos del modelo (por semana de temporada: al inicio hay más incertidumbre). Un factor de escala adicional se elige por log-loss en las temporadas de validación.",
            "Resultados: probabilidad de victoria/empate, distribución del margen y del total, percentiles 10–90 y marcadores más probables.",
        ],
    },
    {
        "title": "6. Validación y precisión",
        "body": [
            "Backtest walk-forward: para cada semana los ratings usan sólo partidos anteriores; para cada temporada los coeficientes usan sólo temporadas anteriores.",
            "Hiperparámetros (vida media, λ, regularización, escala de dispersión) se eligen con temporadas de validación; las temporadas de prueba no intervienen en ninguna decisión.",
            "Métricas: error absoluto medio (MAE) de puntos por equipo, del margen y del total; Brier score y log-loss de la probabilidad de victoria; tabla de calibración (probabilidad predicha vs frecuencia observada).",
            "Comparación con modelos ingenuos: promedio de la liga y promedios simples de puntos anotados/permitidos.",
            "Pronósticos en vivo: cada pronóstico de un partido futuro se guarda con fecha y hora; el historial usa el último guardado ANTES del inicio del partido.",
            "Limitaciones: el backtest no incluye ajustes por lesiones y usa el clima registrado del día como aproximación del pronóstico.",
        ],
    },
    {
        "title": "7. Nivel de confianza y estado",
        "body": [
            "Confianza base según la probabilidad del favorito: Alto ≥ 72%, Moderado ≥ 60%, Bajo < 60%. Baja un nivel si hay muestra pequeña (< 4 partidos en la temporada), QB titular en duda, clima desconocido o estadísticas faltantes.",
            "Se muestra junto a la frecuencia histórica de acierto de favoritos con probabilidad similar (fuera de muestra).",
            "Estado 'Confirmado': reporte oficial de lesiones publicado para ambos equipos, QB titulares sin duda y clima conocido. En otro caso, 'Preliminar'.",
        ],
    },
]
