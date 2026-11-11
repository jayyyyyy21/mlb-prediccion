# Modelo NBA v2

Pronóstico de partidos NBA con datos públicos de ESPN.

    python3 pronostico.py              # partidos de hoy (descarga lo que falte)
    python3 pronostico.py 2026-11-10   # otro día
    python3 backtest.py                # evaluación fuera de muestra
    python3 backtest.py --ajustar      # reajusta los parámetros (≈1 min) y guarda params.json

## Cómo funciona

    margen_local = localía + beta·(talento_local − talento_visita) + (U_local − U_visita) − b2b

- **Talento**: minutos esperados × valor por minuto (Game Score) de los jugadores disponibles.
  Lesiones y cambios de plantilla entran por aquí, sin ajustes a mano.
- **U**: rendimiento del equipo por encima de su talento; ridge con prior heredado de la temporada anterior.
- Probabilidad: normal con sigma ajustada por log loss. Primer cuarto, primera mitad y totales
  tienen su propia calibración.

Los parámetros se ajustan en 2023-24 y 2024-25 y se evalúan en 2025-26 y 2026-27.

## Límites comprobados en el backtest

- El margen del modelo falla por ~11.0 puntos de media; la línea de cierre, ~11.05 en los mismos partidos donde el modelo falla 11.27.
- Cuando el modelo y el mercado difieren, acierta más el mercado. Apostar las "ventajas" del modelo pierde dinero.
  Por eso el pronóstico no recomienda apuestas de valor: marca las discrepancias para revisar noticias.
- En totales el modelo no aporta nada sobre la línea.
