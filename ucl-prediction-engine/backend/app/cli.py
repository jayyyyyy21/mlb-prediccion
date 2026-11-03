"""Línea de comandos.

    python -m app.cli init                  # crea tablas
    python -m app.cli ingest [--quick]      # descarga datos (histórico completo o actualización rápida)
    python -m app.cli backtest              # valida hiperparámetros + backtest cronológico
    python -m app.cli predict               # recalcula pronósticos de los próximos partidos UCL
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

from .db import init_db


def main() -> None:
    ap = argparse.ArgumentParser(prog="ucl-engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--quick", action="store_true")
    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--val-start", default="2025-01-01")
    p_bt.add_argument("--val-end", default="2025-12-31")
    p_bt.add_argument("--test-start", default="2026-01-01")
    p_bt.add_argument("--step-weeks", type=int, default=2)
    sub.add_parser("predict")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    init_db()

    if args.cmd == "ingest":
        from .services.ingest import full_refresh
        print(json.dumps(full_refresh(history=not args.quick), indent=2, default=str))
    elif args.cmd == "backtest":
        from .services.modeling import run_tuning_and_backtest
        res = run_tuning_and_backtest(
            datetime.fromisoformat(args.val_start), datetime.fromisoformat(args.val_end),
            datetime.fromisoformat(args.test_start), step_weeks=args.step_weeks)
        t = res["test"]
        print(json.dumps({k: t.get(k) for k in ("n", "period", "result_1x2", "over_2_5", "btts",
                                                "corners_total_mae", "sot_total_mae", "possession_mae",
                                                "vs_market")}, indent=2, default=str))
    elif args.cmd == "predict":
        from .services.predictions import refresh_predictions
        print(json.dumps(refresh_predictions("cli"), indent=2, default=str))


if __name__ == "__main__":
    main()
