"""Genera una página HTML autocontenida (código + datos + escudos) a partir de una
instantánea exportada y de un build estático del frontend.

    cd backend && python -m app.export_static --out /tmp/snap
    cd ../frontend && VITE_STATIC=1 npx vite build --outDir /tmp/dist
    python ../scripts/single_file.py /tmp/snap /tmp/dist /tmp/ucl.html
"""
import base64
import glob
import json
import os
import sys

snap_dir, dist_dir, out = sys.argv[1:4]
crests = {f"snapshot/crests/{os.path.basename(f)}": "data:image/png;base64," + base64.b64encode(open(f, "rb").read()).decode()
          for f in glob.glob(os.path.join(snap_dir, "crests", "*.png"))}
snap = {os.path.basename(f): json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(snap_dir, "*.json")))}


def esc(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


js = open(glob.glob(os.path.join(dist_dir, "assets", "*.js"))[0], encoding="utf-8").read().replace("</script", "<\\/script")
css = open(glob.glob(os.path.join(dist_dir, "assets", "*.css"))[0], encoding="utf-8").read()
page = f"""<title>UCL Prediction Engine</title>
<meta name="theme-color" content="#050914">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{css}
:root{{color-scheme:dark}}
</style>
<div id="root"><div style="padding:60px 16px;text-align:center;color:#8fa3c8;font-family:system-ui,sans-serif">Cargando UCL Prediction Engine…</div></div>
<script type="application/json" id="ucl-snapshot">{esc(snap)}</script>
<script type="application/json" id="ucl-crests">{esc(crests)}</script>
<script>try{{window.__SNAPSHOT__=JSON.parse(document.getElementById("ucl-snapshot").textContent);window.__CRESTS__=JSON.parse(document.getElementById("ucl-crests").textContent)}}catch(e){{document.getElementById("root").textContent="No se pudieron leer los datos de la instantánea: "+e.message}}</script>
<script type="module">
{js}
</script>
"""
open(out, "w", encoding="utf-8").write(page)
print(f"{out}: {len(page.encode()) / 1e6:.2f} MB")
