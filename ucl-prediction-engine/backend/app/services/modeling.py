"""Orquestación del modelo: hiperparámetros, backtest persistido y motor en caché."""
from __future__ import annotations

import logging
import threading
from datetime import datetime

from sqlalchemy import select

from ..db import BacktestRun, ModelState, SessionLocal, utcnow
from ..models import backtest as bt
from ..models.dataset import load_rows
from ..models.engine import MODEL_VERSION, Engine, Hyper

log = logging.getLogger("modeling")
_lock = threading.Lock()
_engine: Engine | None = None
_engine_key: tuple | None = None


def get_hyper() -> Hyper:
    with SessionLocal() as s:
        st = s.get(ModelState, "hyper")
        if st:
            return Hyper(**st.value)
    return Hyper()


def set_state(key: str, value: dict) -> None:
    with SessionLocal() as s:
        row = s.get(ModelState, key) or ModelState(key=key, value=value)
        row.value = value
        s.merge(row)
        s.commit()


def get_state(key: str) -> dict | None:
    with SessionLocal() as s:
        st = s.get(ModelState, key)
        return st.value if st else None


def get_engine(force: bool = False) -> Engine:
    """Motor ajustado con todos los datos hasta ahora. Se reajusta si cambian los datos."""
    global _engine, _engine_key
    with _lock:
        rows = load_rows()
        key = (len(rows), rows[-1].match_id if rows else None, tuple(get_hyper().as_dict().items()))
        if _engine is None or force or key != _engine_key:
            log.info("Ajustando motor con %d partidos", len(rows))
            _engine = Engine(hyper=get_hyper()).fit(rows, utcnow())
            _engine_key = key
        return _engine


def market_probs_from_db() -> dict[str, list[float]]:
    """Probabilidades implícitas (sin margen) de cuotas verificadas guardadas."""
    from ..db import Match
    out = {}
    with SessionLocal() as s:
        for m in s.execute(select(Match).where(Match.odds.is_not(None))).scalars():
            o = m.odds or {}
            if o.get("home") and o.get("draw") and o.get("away"):
                inv = [1 / o["home"], 1 / o["draw"], 1 / o["away"]]
                tot = sum(inv)
                out[m.id] = [x / tot for x in inv]
    return out


def run_tuning_and_backtest(val_start: datetime, val_end: datetime, test_start: datetime,
                            test_end: datetime | None = None, grid: list[Hyper] | None = None,
                            step_weeks: int = 2, log_fn=print) -> dict:
    rows = load_rows()
    test_end = test_end or utcnow()
    log_fn(f"Validación {val_start.date()} → {val_end.date()} | {len(rows)} partidos en BD")
    best, table = bt.tune(rows, val_start, val_end, grid or bt.DEFAULT_GRID, step_weeks=step_weeks, log=log_fn)
    log_fn(f"Mejor configuración: {best.as_dict()}")

    # ¿Aporta el ajuste táctico por estilo? Se decide también en validación.
    # Segunda etapa: modelos de conteo (córners / tiros a puerta), con los mejores parámetros de goles.
    from dataclasses import replace as _r
    count_table = []
    best_c, best_c_score = best, float("inf")
    for k in (1.0, 0.6):
        for w in (1.0, 0.5, 0.25):
            h = _r(best, count_shrink=k, count_blend=w)
            sc = bt.summarize(bt.walk_forward(rows, h, val_start, val_end, step_weeks=step_weeks))
            score = sc["corners_over_9_5"]["model"]["log_loss"] + sc["sot_over_8_5"]["model"]["log_loss"]
            count_table.append({"count_shrink": k, "count_blend": w, "score": score,
                                "corners_ll": sc["corners_over_9_5"]["model"]["log_loss"],
                                "sot_ll": sc["sot_over_8_5"]["model"]["log_loss"],
                                "corners_baseline_ll": sc["corners_over_9_5"]["baseline"]["log_loss"],
                                "sot_baseline_ll": sc["sot_over_8_5"]["baseline"]["log_loss"]})
            log_fn(f"  conteos shrink={k} blend={w} -> {score:.4f}")
            if score < best_c_score:
                best_c, best_c_score = h, score
    best = best_c
    log_fn(f"Mejor configuración (con conteos): {best.as_dict()}")

    style_h = bt.with_style(best)
    s_val = bt.summarize(bt.walk_forward(rows, style_h, val_start, val_end, step_weeks=step_weeks))
    style_score = s_val["result_1x2"]["model"]["log_loss"] + s_val["over_2_5"]["model"]["log_loss"]
    s_best = bt.summarize(bt.walk_forward(rows, best, val_start, val_end, step_weeks=step_weeks))
    best_score = s_best["result_1x2"]["model"]["log_loss"] + s_best["over_2_5"]["model"]["log_loss"]
    use_style = style_score < best_score - 0.002
    log_fn(f"Ajuste por estilo: {style_score:.4f} vs {best_score:.4f} → {'SE USA' if use_style else 'no se usa'}")
    chosen = style_h if use_style else best

    market = market_probs_from_db()
    recs = bt.walk_forward(rows, chosen, test_start, test_end, step_weeks=1, market=market)
    summary = bt.summarize(recs)
    payload = {
        "model_version": MODEL_VERSION,
        "chosen_hyper": chosen.as_dict(),
        "validation": {"period": [val_start.isoformat(), val_end.isoformat()], "grid": table,
                       "count_grid": count_table,
                       "style_adjust_score": style_score, "style_adjust_used": use_style},
        "test": summary,
        "test_records": recs[-400:],
        "notes": [
            "Hiperparámetros elegidos solo con el periodo de validación; el periodo de prueba es posterior y no se usó para elegirlos.",
            "Evaluado sobre partidos de UCL, UEL y UECL (cruces entre ligas).",
            "Referencia ingenua = frecuencias observadas en competiciones europeas del último año.",
        ],
    }
    set_state("hyper", chosen.as_dict())
    with SessionLocal() as s:
        s.add(BacktestRun(model_version=MODEL_VERSION, payload=payload))
        s.commit()
    return payload


def latest_backtest() -> dict | None:
    with SessionLocal() as s:
        r = s.execute(select(BacktestRun).order_by(BacktestRun.id.desc()).limit(1)).scalar_one_or_none()
        if r is None:
            return None
        p = dict(r.payload)
        p["created_at"] = r.created_at.isoformat()
        return p
