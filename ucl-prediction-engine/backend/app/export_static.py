"""Exporta una INSTANTÁNEA estática (JSON + escudos) para servir el frontend sin backend.

    python -m app.export_static --out ../frontend/public/snapshot

La instantánea lleva su fecha de generación; el frontend en modo estático la muestra
siempre para no aparentar datos en vivo.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx
from sqlalchemy import select

from .db import Match, Prediction, SessionLocal, Team, init_db, utcnow
from .main import status as api_status
from .services import modeling, predictions


def _crest(client: httpx.Client, url: str, dest: Path) -> bool:
    try:
        from PIL import Image
        r = client.get(url)
        if r.status_code != 200:
            return False
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        im.thumbnail((96, 96))
        im.save(dest, "PNG", optimize=True)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    init_db()
    out = Path(args.out)
    (out / "crests").mkdir(parents=True, exist_ok=True)

    with SessionLocal() as s:
        ucl_ids = {r for (r,) in s.execute(select(Match.home_id).where(Match.league == "uefa.champions"))} | {
            r for (r,) in s.execute(select(Match.away_id).where(Match.league == "uefa.champions"))}
        teams = {t.id: t for t in s.execute(select(Team).where(Team.id.in_(ucl_ids))).scalars()}
        pred_ids = sorted({p for (p,) in s.execute(select(Prediction.match_id))})

    local_logo: dict[str, str] = {}
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for tid, t in teams.items():
            if t.logo and _crest(client, t.logo, out / "crests" / f"{tid}.png"):
                local_logo[tid] = f"snapshot/crests/{tid}.png"  # relativo a la página

    def fix(obj):
        """Sustituye las URLs de escudos por los ficheros locales."""
        if isinstance(obj, dict):
            if "logo" in obj and obj.get("id") in local_logo:
                obj["logo"] = local_logo[obj["id"]]
            elif "logo" in obj and obj.get("team_id") in local_logo:
                obj["logo"] = local_logo[obj["team_id"]]
            elif "logo" in obj:
                obj["logo"] = None
            for v in obj.values():
                fix(v)
        elif isinstance(obj, list):
            for v in obj:
                fix(v)
        return obj

    def dump(name, data):
        (out / name).write_text(json.dumps(fix(data), ensure_ascii=False, default=str, separators=(",", ":")))

    dump("matches.json", predictions.list_matches(None, None, None))
    for mid in pred_ids:
        p = predictions.latest_prediction(mid)
        p["_history"] = predictions.prediction_history(mid)
        dump(f"match-{mid}.json", p)
    dump("teams.json", [{"id": t.id, "name": t.name, "logo": t.logo, "abbreviation": t.abbreviation,
                         "domestic_league": t.domestic_league} for t in sorted(teams.values(), key=lambda t: t.name)
                        if t.in_ucl])
    st = predictions.standings(utcnow())
    for r in st:
        r["name"] = teams[r["team_id"]].name if r["team_id"] in teams else r["team_id"]
        r["logo"] = None
    dump("standings.json", st)
    bt = modeling.latest_backtest()
    if bt:
        bt.pop("test_records", None)
        dump("backtest.json", bt)
    stt = api_status()
    stt["snapshot"] = {"generated_at": utcnow().isoformat(), "predictions": len(pred_ids)}
    dump("status.json", stt)
    print(f"Instantánea: {len(pred_ids)} pronósticos, {len(local_logo)} escudos → {out}")


if __name__ == "__main__":
    main()
