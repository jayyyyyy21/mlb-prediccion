"""Línea de comandos.

    python -m app.cli init        # descarga todo + calibra el modelo (primera vez)
    python -m app.cli update      # actualiza datos (temporada actual, lesiones)
    python -m app.cli backtest    # re-calibra y valida el modelo
    python -m app.cli predict     # imprime los pronósticos de la próxima semana
    python -m app.cli serve       # inicia la aplicación web
"""
from __future__ import annotations

import argparse
import json
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="nfl-predictor-pro")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("update")
    sub.add_parser("backtest")
    p = sub.add_parser("predict")
    p.add_argument("--season", type=int)
    p.add_argument("--week", type=int)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd in ("init", "update"):
        from .ingest import update_all
        print(json.dumps(update_all(), indent=1, ensure_ascii=False))
    if args.cmd in ("init", "backtest"):
        from .backtest import run
        r = run()
        print(json.dumps({k: r[k] for k in ("params", "validation", "test", "calibration_test", "baselines_test")},
                         indent=1, ensure_ascii=False))
    if args.cmd == "predict":
        from .config import CURRENT_SEASON
        from .service import engine
        e = engine()
        season = args.season or CURRENT_SEASON
        week = args.week or e.default_week(season)
        for g in e.predict_week(season, week):
            gm, pr = g["game"], g["projection"]
            print(f"{gm['away']:>3} @ {gm['home']:<3}  {pr['away_points']:5.1f} - {pr['home_points']:5.1f}  "
                  f"P(local)={pr['p_home']:.0%}  total={pr['total']:.1f}  [{g['confidence']}, {g['status']}]")
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
