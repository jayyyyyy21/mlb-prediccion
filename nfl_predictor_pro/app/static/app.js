/* NFL Predictor Pro — interfaz */
"use strict";

const S = { status: null, teams: {}, season: null, week: null, charts: {}, lastView: "dashboard" };

// ---------------------------------------------------------------- utilidades
const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v, d = 0) => (v === null || v === undefined ? "—" : (v * 100).toFixed(d) + "%");
const sign = (v, d = 1) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* sin cuerpo JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

function teamColor(t) { return (S.teams[t] && S.teams[t].color) || "#334155"; }
function teamName(t) { return (S.teams[t] && S.teams[t].name) || t; }
function badge(t) {
  const info = S.teams[t] || {};
  if (S.status && S.status.show_logos && info.logo) {
    return `<span class="badge" style="background:${esc(info.color)}"><img alt="${esc(t)}" src="${esc(info.logo)}"></span>`;
  }
  return `<span class="badge" style="background:${esc(info.color || "#334155")};border-color:${esc(info.color2 || "#fff3")}">${esc(t)}</span>`;
}
function alertsHtml(list) {
  return (list || []).map((a) => `<div class="alert ${esc(a.level)}">${esc(a.text)}</div>`).join("");
}
function confPill(c) { return `<span class="pill ${esc((c || "").toLowerCase())}">Confianza: ${esc(c)}</span>`; }
function statusPill(s) {
  const cls = s === "Confirmado" ? "conf" : s && s.startsWith("Finalizado") ? "final" : "prelim";
  return `<span class="pill ${cls}">${esc(s)}</span>`;
}
function fmtDate(g) {
  if (!g.kickoff_utc) return esc(g.gameday || "");
  const d = new Date(g.kickoff_utc);
  return d.toLocaleString("es", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}
function destroyChart(id) { if (S.charts[id]) { S.charts[id].destroy(); delete S.charts[id]; } }
function chart(id, cfg) {
  destroyChart(id);
  const el = document.getElementById(id);
  if (!el || !window.Chart) return;
  Chart.defaults.color = "#8b98b3";
  Chart.defaults.borderColor = "#26324a";
  S.charts[id] = new Chart(el, cfg);
}

// ---------------------------------------------------------------- navegación
function show(view) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  if (view !== "game") S.lastView = view;
  window.scrollTo(0, 0);
  if (view === "history") loadHistory();
  if (view === "method") loadMethod();
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  show(b.dataset.view);
  history.replaceState(null, "", "#" + b.dataset.view);
}));

// ---------------------------------------------------------------- arranque
async function init() {
  try {
    [S.status, S.teams] = await Promise.all([api("/api/status"), api("/api/teams")]);
  } catch (e) {
    $("#global-alerts").innerHTML = alertsHtml([{ level: "error", text: "No se pudo cargar el estado: " + e.message }]);
    return;
  }
  renderStatus();
  S.season = S.status.season;
  const seasonOpts = S.status.seasons.slice().reverse().map((s) => `<option value="${s}">Temporada ${s}</option>`).join("");
  $("#season-select").innerHTML = seasonOpts;
  $("#cmp-season").innerHTML = seasonOpts;
  const teamOpts = Object.keys(S.teams).sort().map((t) => `<option value="${t}">${esc(teamName(t))}</option>`).join("");
  ["#cmp-a", "#cmp-b", "#sim-home", "#sim-away"].forEach((s) => ($(s).innerHTML = teamOpts));
  $("#cmp-a").value = "KC"; $("#cmp-b").value = "BUF";
  $("#sim-home").value = "KC"; $("#sim-away").value = "BUF";
  await loadWeeks(S.season, S.status.default_week);
  const hash = location.hash.replace("#", "");
  if (hash.startsWith("game/")) openGame(hash.slice(5));
  else if (["compare", "simulator", "history", "method"].includes(hash)) show(hash);
}

function renderStatus() {
  const st = S.status;
  const upd = st.last_update ? new Date(st.last_update).toLocaleString("es") : "nunca";
  $("#updated").innerHTML = `Datos actualizados: <b>${esc(upd)}</b><button class="btn secondary" id="btn-update" title="Descargar datos nuevos">Actualizar</button>`;
  $("#btn-update").addEventListener("click", runUpdate);
  $("#global-alerts").innerHTML = alertsHtml(st.alerts);
}

async function runUpdate() {
  const btn = $("#btn-update");
  btn.disabled = true; btn.textContent = "Actualizando…";
  try {
    await api("/api/update", { method: "POST" });
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      const st = await api("/api/status");
      if (!st.update.running) { S.status = st; break; }
    }
    renderStatus();
    loadWeek();
  } catch (e) {
    alert("Error al actualizar: " + e.message);
    btn.disabled = false;
  }
}

async function loadWeeks(season, selected) {
  const weeks = await api(`/api/weeks?season=${season}`);
  $("#week-select").innerHTML = weeks.map((w) =>
    `<option value="${w.week}">${esc(w.label)} · ${w.played}/${w.games} jugados</option>`).join("");
  $("#week-select").value = selected || weeks[weeks.length - 1].week;
  $("#season-select").value = season;
  await loadWeek();
}

$("#season-select").addEventListener("change", async (e) => {
  S.season = +e.target.value;
  const weeks = await api(`/api/weeks?season=${S.season}`);
  const pending = weeks.find((w) => w.played < w.games);
  loadWeeks(S.season, pending ? pending.week : weeks[weeks.length - 1].week);
});
$("#week-select").addEventListener("change", loadWeek);
$("#export-csv").addEventListener("click", () => { location.href = `/api/export?season=${S.season}&week=${$("#week-select").value}&fmt=csv`; });
$("#export-json").addEventListener("click", () => { location.href = `/api/export?season=${S.season}&week=${$("#week-select").value}&fmt=json`; });

// ---------------------------------------------------------------- dashboard
async function loadWeek() {
  const week = +$("#week-select").value;
  S.week = week;
  $("#games").innerHTML = `<div class="loading">Calculando pronósticos (10,000 simulaciones por partido)…</div>`;
  try {
    const data = await api(`/api/predictions?season=${S.season}&week=${week}`);
    const games = data.games;
    const played = games.filter((g) => g.game.played);
    let summary = `${games.length} partidos.`;
    if (played.length) {
      const hits = played.filter((g) => {
        const p = g.projection, gm = g.game;
        if (gm.home_score === gm.away_score) return false;
        return (p.p_home >= p.p_away) === (gm.home_score > gm.away_score);
      }).length;
      const mae = played.reduce((s, g) => s + Math.abs(g.projection.home_points - g.game.home_score) + Math.abs(g.projection.away_points - g.game.away_score), 0) / (2 * played.length);
      summary += ` Partidos jugados: ${played.length} · favorito acertado en ${hits} · error absoluto medio ${mae.toFixed(1)} pts (pronóstico retrospectivo con datos previos a la semana).`;
    }
    $("#week-summary").textContent = summary;
    $("#games").innerHTML = games.map(gameCard).join("") || `<div class="muted">Sin partidos.</div>`;
    document.querySelectorAll(".game").forEach((el) => el.addEventListener("click", () => openGame(el.dataset.id)));
  } catch (e) {
    $("#games").innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

function gameCard(r) {
  const g = r.game, p = r.projection;
  const real = g.played ? `<span class="real">Final: ${esc(g.away)} ${g.away_score} - ${g.home_score} ${esc(g.home)}</span>` : "";
  const row = (t, pts, pr) => `<div class="team-row"><div class="team">${badge(t)}<span class="team-name">${esc(teamName(t))}</span></div>
      <div class="pts">${fmt(pts)}</div><div class="pct">${pct(pr)}</div></div>`;
  return `<div class="card game" data-id="${esc(g.game_id)}" role="button" tabindex="0">
    <div class="game-head"><span>${esc(fmtDate(g))}${g.neutral ? " · sede neutral" : ""}</span><span>${esc(g.stadium || "")}</span></div>
    ${row(g.away, p.away_points, p.p_away)}
    ${row(g.home, p.home_points, p.p_home)}
    <div class="probbar"><div style="width:${p.p_away * 100}%;background:${esc(teamColor(g.away))}"></div><div style="width:${p.p_tie * 100}%;background:#64748b"></div><div style="width:${p.p_home * 100}%;background:${esc(teamColor(g.home))}"></div></div>
    <div class="game-foot">${confPill(r.confidence)} ${statusPill(r.status)}
      <span>Marcador: ${esc(g.away)} ${p.score_away}-${p.score_home} ${esc(g.home)} · Total ${fmt(p.total)}</span>${real}</div>
  </div>`;
}

// ---------------------------------------------------------------- detalle
async function openGame(id) {
  show("game");
  history.replaceState(null, "", "#game/" + id);
  const view = $("#view-game");
  view.innerHTML = `<div class="loading">Analizando partido…</div>`;
  try {
    const r = await api(`/api/game/${encodeURIComponent(id)}`);
    view.innerHTML = `<button class="btn secondary back" id="back">← Volver</button>
      <button class="btn secondary back" onclick="location.href='/api/export/game/${encodeURIComponent(id)}'">Exportar análisis (JSON)</button>
      <button class="btn secondary back" onclick="window.print()">Imprimir / PDF</button>
      <div id="game-detail"></div>`;
    $("#back").addEventListener("click", () => { show(S.lastView); history.replaceState(null, "", "#" + S.lastView); });
    renderPrediction($("#game-detail"), r, "gd");
  } catch (e) {
    view.innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

function contribHtml(items, color) {
  const max = Math.max(1, ...items.map((i) => Math.abs(i.points)));
  return `<div class="contrib">${items.map((i) => {
    const w = (Math.abs(i.points) / max) * 50;
    const left = i.points >= 0 ? 50 : 50 - w;
    return `<div class="row"><span>${esc(i.label)}</span><div class="bar"><span style="left:${left}%;width:${w}%;background:${i.points >= 0 ? color : "#f87171"}"></span><i class="mid"></i></div><span class="val">${sign(i.points, 2)}</span></div>`;
  }).join("")}</div>`;
}

function splitsTable(summary, side) {
  const s = summary && summary[side];
  if (!s) return `<div class="muted small">Sin datos.</div>`;
  const cols = [["temporada", "Temporada"], ["ult3", "Últ. 3"], ["ult5", "Últ. 5"], ["ult10", "Últ. 10"], ["local", "Local"], ["visitante", "Visitante"]];
  const rows = side === "offense"
    ? [["Puntos/partido", "ppg", 1], ["Yardas pase/p", "pass_ypg", 0], ["Yardas carrera/p", "rush_ypg", 0], ["EPA/jugada", "epa_play", 3],
       ["Success rate", "success_rate", "%"], ["TD en zona roja", "rz_td_pct", "%"], ["Pérdidas/p", "turnovers_pg", 1],
       ["Capturas permitidas/p", "sacks_pg", 1], ["3er down", "third_pct", "%"], ["Posesiones/p", "drives_pg", 1], ["Jugadas/p", "plays_pg", 1]]
    : [["Puntos permitidos/p", "ppg", 1], ["Yardas pase perm./p", "pass_ypg", 0], ["Yardas carrera perm./p", "rush_ypg", 0], ["EPA/jugada perm.", "epa_play", 3],
       ["Success rate perm.", "success_rate", "%"], ["TD zona roja perm.", "rz_td_pct", "%"], ["Capturas/p", "sacks_pg", 1],
       ["Golpes al QB/p", "qb_hits_pg", 1], ["Intercepciones/p", "ints_pg", 2], ["Balones recuperados/p", "turnovers_pg", 1],
       ["% blitz (FTN)", "blitz_rate", "%"], ["3er down perm.", "third_pct", "%"]];
  const cell = (v, d) => (d === "%" ? pct(v, 1) : fmt(v, d));
  return `<div class="table-wrap"><table><thead><tr><th></th>${cols.map((c) => `<th>${c[1]}<div class="muted small">${s[c[0]] ? s[c[0]].games : 0} PJ</div></th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr><td>${r[0]}</td>${cols.map((c) => `<td>${s[c[0]] && s[c[0]].games ? cell(s[c[0]][r[1]], r[2]) : "—"}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function injuriesHtml(inj) {
  if (!inj) return `<div class="muted small">Sin información de lesiones aplicable (partido jugado o temporada sin reportes).</div>`;
  const qb = inj.qb ? `<div class="small" style="margin-bottom:6px">QB titular: <b>${esc(inj.qb.starter)}</b> (EPA/dropback encogido ${fmt(inj.qb.starter_epa_db, 3)}, ${fmt(inj.qb.dropbacks_pg, 0)} dropbacks/p)${inj.qb.replacement ? ` → reemplazo: <b>${esc(inj.qb.replacement)}</b> (${fmt(inj.qb.replacement_epa_db, 3)}), impacto ${sign(inj.qb.impact)} pts` : ""}</div>` : "";
  const all = (inj.players || []).filter((p) => p.category !== "disponible");
  // Jugadores en reserva que no jugaban recientemente: sin impacto (los ratings ya reflejan su ausencia).
  const quiet = all.filter((p) => p.category === "confirmada" && !p.impact && !p.snap_share && p.position !== "QB");
  const rows = all.filter((p) => !quiet.includes(p)).slice(0, 25);
  const catEs = { confirmada: "Baja confirmada", probable: "Baja probable (no confirmada)", cuestionable: "Cuestionable", simulada: "Baja simulada" };
  return `${qb}<div class="small muted" style="margin-bottom:6px">${inj.official_report ? "Reporte oficial de la semana publicado." : "Reporte oficial de la semana aún no publicado: se usan listas de reserva y reportes de ESPN (no oficiales)."}
    Ajuste aplicado: ofensiva ${sign(inj.off_adj)} pts · defensa ${sign(inj.def_adj)} pts permitidos.</div>
    ${rows.length ? `<div class="table-wrap"><table><thead><tr><th>Jugador</th><th>Pos.</th><th>Estado</th><th>Snaps recientes</th><th>Impacto (pts)</th><th>Aplicado</th></tr></thead><tbody>
    ${rows.map((p) => `<tr><td class="wrap">${esc(p.name)}<div class="muted small">${esc(p.source)}${p.detail ? " · " + esc(p.detail) : ""}</div></td><td>${esc(p.position)}</td><td class="wrap">${esc(catEs[p.category] || p.category)}<div class="muted small">${esc(p.status)}</div></td><td>${p.snap_share != null ? pct(p.snap_share) : "—"}</td><td>${sign(p.impact, 2)}</td><td>${p.applied ? "Sí" : "No (escenario)"}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="muted small">Sin bajas relevantes reportadas.</div>`}
    ${quiet.length ? `<div class="small muted" style="margin-top:6px">Además, ${quiet.length} jugadores en lista de reserva sin snaps recientes (sin impacto adicional): ${quiet.slice(0, 12).map((p) => esc(p.name) + " (" + esc(p.position) + ")").join(", ")}${quiet.length > 12 ? "…" : ""}.</div>` : ""}`;
}

function matchupsHtml(list, off, def) {
  if (!list || !list.length) return `<div class="muted small">Sin datos suficientes.</div>`;
  return list.map((m) => {
    const color = m.advantage ? teamColor(m.advantage) : "#475569";
    return `<div class="matchup" style="border-left-color:${esc(color)}"><b>${esc(m.title)} — ${esc(m.summary)}</b>
      <span class="small muted">${esc(m.description)}. ${esc(m.detail || "")}</span></div>`;
  }).join("");
}

function haTable(ha, team) {
  if (!ha) return "";
  const row = (lbl, d) => d ? `<tr><td>${lbl}</td><td>${d.games}</td><td>${esc(d.record || "—")}</td><td>${fmt(d.ppg)}</td><td>${fmt(d.papg)}</td><td>${sign(d.margin)}</td><td>${sign(d.opp_quality)}</td></tr>` : "";
  return `<div class="table-wrap"><table><thead><tr><th>${esc(team)} (3 temp.)</th><th>PJ</th><th>Récord</th><th>PF/p</th><th>PC/p</th><th>Dif.</th><th>Calidad rival</th></tr></thead><tbody>
    ${row("Local", ha.local)}${row("Visitante", ha.visitante)}
    ${ha.estadio_cerrado ? `<tr><td>Estadio cerrado</td><td>${ha.estadio_cerrado.games}</td><td></td><td>${fmt(ha.estadio_cerrado.ppg)}</td><td>${fmt(ha.estadio_cerrado.papg)}</td><td></td><td></td></tr>` : ""}
    ${ha.aire_libre ? `<tr><td>Aire libre</td><td>${ha.aire_libre.games}</td><td></td><td>${fmt(ha.aire_libre.ppg)}</td><td>${fmt(ha.aire_libre.papg)}</td><td></td><td></td></tr>` : ""}
    </tbody></table></div><div class="small muted">Efecto local propio: bruto ${sign(ha.home_edge_raw)} · encogido ${sign(ha.home_edge_shrunk)} pts. ${esc(ha.sample_note || "")}</div>`;
}

function renderPrediction(el, r, key) {
  const g = r.game, p = r.projection, sim = r.simulation, ex = r.explanation || {};
  const home = g.home, away = g.away;
  const fav = p.p_home >= p.p_away ? home : away;
  const favP = Math.max(p.p_home, p.p_away);
  const wx = r.weather || {};
  const wxTxt = wx.outdoor
    ? (wx.known ? `${fmt(wx.temp, 0)}°F, viento ${fmt(wx.wind, 0)} mph (${esc(wx.source)})` : "Clima no disponible")
    : "Estadio cerrado";
  el.innerHTML = `
  <div class="card">
    <div class="game-head"><span>${esc(g.game_type_es || "")} · ${esc(fmtDate(g))}${g.neutral ? " · sede neutral" : ""}</span><span>${esc(g.stadium || "")}</span></div>
    <div class="hero">
      <div class="side">${badge(away)}<div class="big" style="--tc:${esc(teamColor(away))}">${fmt(p.away_points)}</div><div class="nm">${esc(teamName(away))}</div><div class="muted">${pct(p.p_away, 1)}</div></div>
      <div class="vs">@</div>
      <div class="side">${badge(home)}<div class="big" style="--tc:${esc(teamColor(home))}">${fmt(p.home_points)}</div><div class="nm">${esc(teamName(home))}</div><div class="muted">${pct(p.p_home, 1)}</div></div>
    </div>
    <div class="probbar" style="height:12px"><div style="width:${p.p_away * 100}%;background:${esc(teamColor(away))}"></div><div style="width:${p.p_tie * 100}%;background:#64748b"></div><div style="width:${p.p_home * 100}%;background:${esc(teamColor(home))}"></div></div>
    <div class="game-foot">${confPill(r.confidence)} ${statusPill(r.status)} ${r.confidence_reasons && r.confidence_reasons.length ? `<span>Reduce la confianza: ${esc(r.confidence_reasons.join(", "))}</span>` : ""}
      ${g.played ? `<span class="real">Resultado real: ${esc(away)} ${g.away_score} - ${g.home_score} ${esc(home)}</span>` : ""}</div>
    <div class="kpis">
      <div class="kpi"><div class="l">Marcador estimado</div><div class="v">${esc(away)} ${p.score_away} - ${p.score_home} ${esc(home)}</div></div>
      <div class="kpi"><div class="l">Diferencia proyectada</div><div class="v">${esc(fav)} ${sign(Math.abs(p.spread_home))}</div></div>
      <div class="kpi"><div class="l">Total proyectado</div><div class="v">${fmt(p.total)} pts</div></div>
      <div class="kpi"><div class="l">Prob. empate</div><div class="v">${pct(p.p_tie, 1)}</div></div>
      <div class="kpi"><div class="l">Rango 80% margen (${esc(home)})</div><div class="v">${sign(sim.margin_p10, 0)} a ${sign(sim.margin_p90, 0)}</div></div>
      <div class="kpi"><div class="l">Rango 80% total</div><div class="v">${fmt(sim.total_p10, 0)}–${fmt(sim.total_p90, 0)}</div></div>
      <div class="kpi"><div class="l">Clima</div><div class="v" style="font-size:13px">${wxTxt}</div></div>
      <div class="kpi"><div class="l">Simulaciones</div><div class="v">${sim.n_sims.toLocaleString("es")}</div></div>
    </div>
    ${r.historical_bucket ? `<div class="small muted" style="margin-top:10px">${esc(r.historical_bucket.text)}</div>` : ""}
  </div>
  <div class="alerts section">${alertsHtml(r.alerts)}</div>

  <div class="grid g2 section">
    <div class="card"><h2>Distribución del margen (${esc(home)} − ${esc(away)})</h2><div class="chart-box"><canvas id="${key}-margin"></canvas></div></div>
    <div class="card"><h2>Distribución del total de puntos</h2><div class="chart-box"><canvas id="${key}-total"></canvas></div>
      <div class="small muted">Marcadores más probables: ${sim.top_scores.slice(0, 5).map((s) => `${esc(away)} ${s.b}-${s.a} ${esc(home)} (${pct(s.p, 1)})`).join(" · ")}</div></div>
  </div>

  ${ex.home_baseline ? `
  <div class="card section"><h2>¿Por qué se proyectan esos puntos?</h2>
    <div class="grid g2">
      <div><h3>${esc(teamName(away))}: ${fmt(p.away_points)} pts</h3><p>${esc(ex.away_baseline)}</p>
        <p class="small muted">Modelo base ${fmt(p.away_points_base)} pts · ajuste por lesiones ${sign(p.away_injury_adj)} pts. Aporte de cada factor respecto a un partido promedio:</p>
        ${contribHtml(ex.away_contributions.slice(0, 10), teamColor(away))}</div>
      <div><h3>${esc(teamName(home))}: ${fmt(p.home_points)} pts</h3><p>${esc(ex.home_baseline)}</p>
        <p class="small muted">Modelo base ${fmt(p.home_points_base)} pts · ajuste por lesiones ${sign(p.home_injury_adj)} pts. Aporte de cada factor respecto a un partido promedio:</p>
        ${contribHtml(ex.home_contributions.slice(0, 10), teamColor(home))}</div>
    </div>
    <p class="small muted">Las variables están correlacionadas entre sí (p. ej. puntos y EPA): interprete los aportes en conjunto. Coeficientes calibrados con temporadas anteriores.</p>
  </div>

  <div class="grid g2 section">
    <div class="card"><h2>Ofensiva ${esc(away)} vs defensa ${esc(home)}</h2>${matchupsHtml(ex.matchups.away_offense)}
      ${ex.similar_opponents.home_def_vs_away_style ? `<p class="small muted">${esc(ex.similar_opponents.home_def_vs_away_style.note)}</p>` : ""}</div>
    <div class="card"><h2>Ofensiva ${esc(home)} vs defensa ${esc(away)}</h2>${matchupsHtml(ex.matchups.home_offense)}
      ${ex.similar_opponents.away_def_vs_home_style ? `<p class="small muted">${esc(ex.similar_opponents.away_def_vs_home_style.note)}</p>` : ""}</div>
  </div>

  <div class="grid g2 section">
    <div class="card"><h2>Lesiones · ${esc(teamName(away))}</h2>${injuriesHtml(r.injuries && r.injuries.away)}</div>
    <div class="card"><h2>Lesiones · ${esc(teamName(home))}</h2>${injuriesHtml(r.injuries && r.injuries.home)}</div>
  </div>

  <div class="card section"><h2>Localía y viaje</h2><p>${esc(ex.home_away.text)}</p>
    <div class="grid g2">${haTable(ex.home_away.away, away)}${haTable(ex.home_away.home, home)}</div>
    <p class="small muted">Contexto: ${esc(away)} viaja ${fmt(ex.context.away.miles, 0)} millas (cambio horario ${sign(ex.context.away.tz_shift, 0)} h).
      Diferencia de descanso local−visitante: ${sign(ex.context.home.rest_diff, 0)} días.</p></div>

  <div class="grid g2 section">
    <div class="card"><h2>Ofensiva · puntos anotados y estadísticas</h2><h3>${esc(away)}</h3>${splitsTable(ex.away_summary, "offense")}<h3 class="section">${esc(home)}</h3>${splitsTable(ex.home_summary, "offense")}</div>
    <div class="card"><h2>Defensa · puntos y yardas permitidas</h2><h3>${esc(away)}</h3>${splitsTable(ex.away_summary, "defense")}<h3 class="section">${esc(home)}</h3>${splitsTable(ex.home_summary, "defense")}</div>
  </div>

  <div class="card section"><h2>Factores de mayor incertidumbre</h2><ul class="clean">${ex.uncertainty.map((u) => `<li>${esc(u)}</li>`).join("") || "<li>Sin factores especiales.</li>"}</ul></div>` : ""}
  `;
  const labels = sim.margin_hist.edges.slice(0, -1).map((e) => `${e}`);
  chart(`${key}-margin`, {
    type: "bar",
    data: { labels, datasets: [{ data: sim.margin_hist.counts.map((c) => c * 100), backgroundColor: sim.margin_hist.edges.slice(0, -1).map((e) => (e + 1.5 > 0 ? teamColor(home) : teamColor(away))) }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw.toFixed(1)}% de los escenarios` } } },
      scales: { y: { title: { display: true, text: "% escenarios" } }, x: { title: { display: true, text: `Margen ${home}` } } } },
  });
  chart(`${key}-total`, {
    type: "bar",
    data: { labels: sim.total_hist.edges.slice(0, -1).map((e) => `${e}`), datasets: [{ data: sim.total_hist.counts.map((c) => c * 100), backgroundColor: "#3fa7ff" }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "% escenarios" } }, x: { title: { display: true, text: "Total de puntos" } } } },
  });
}

// ---------------------------------------------------------------- comparador
$("#cmp-go").addEventListener("click", loadCompare);
async function loadCompare() {
  const a = $("#cmp-a").value, b = $("#cmp-b").value, season = $("#cmp-season").value || S.season;
  const out = $("#compare-out");
  out.innerHTML = `<div class="loading">Cargando…</div>`;
  try {
    const d = await api(`/api/compare?a=${a}&b=${b}&season=${season}`);
    const adjRows = [["Puntos (ofensiva)", "pts_off", 1], ["Puntos permitidos (defensa)", "pts_def", 1], ["EPA/jugada ofensiva", "epa_off", 3],
      ["EPA/jugada permitido", "epa_def", 3], ["EPA pase ofensiva", "pepa_off", 3], ["EPA pase permitido", "pepa_def", 3],
      ["EPA carrera ofensiva", "repa_off", 3], ["EPA carrera permitido", "repa_def", 3], ["Success rate ofensiva", "sr_off", 3], ["Success rate permitido", "sr_def", 3]];
    out.innerHTML = `
      <div class="grid g2">
        <div class="card"><h2>${badge(a)} ${esc(teamName(a))} <span class="muted">${esc(d.a.summary.record || "")}</span></h2>
          <div class="chart-box"><canvas id="cmp-log-a"></canvas></div></div>
        <div class="card"><h2>${badge(b)} ${esc(teamName(b))} <span class="muted">${esc(d.b.summary.record || "")}</span></h2>
          <div class="chart-box"><canvas id="cmp-log-b"></canvas></div></div>
      </div>
      <div class="grid g2 section">
        <div class="card"><h2>Perfil ajustado por rival (percentil, 100 = mejor)</h2><div class="chart-box tall"><canvas id="cmp-radar"></canvas></div></div>
        <div class="card"><h2>Ratings ajustados por rival</h2><div class="table-wrap"><table><thead><tr><th>Métrica</th><th>${esc(a)}</th><th>Puesto</th><th>${esc(b)}</th><th>Puesto</th></tr></thead><tbody>
          ${adjRows.map((r) => { const x = d.a.adjusted[r[1]] || {}, y = d.b.adjusted[r[1]] || {}; return `<tr><td>${r[0]}</td><td>${fmt(x.value, r[2])}</td><td>${x.rank || "—"}</td><td>${fmt(y.value, r[2])}</td><td>${y.rank || "—"}</td></tr>`; }).join("")}
          </tbody></table></div><p class="small muted">Puesto 1 = mejor de la liga. Ajustado por la calidad de los rivales y ponderando más lo reciente.</p></div>
      </div>
      <div class="grid g2 section">
        <div class="card"><h2>Ofensiva ${esc(a)} vs defensa ${esc(b)}</h2>${matchupsHtml(d.matchups.a_offense)}</div>
        <div class="card"><h2>Ofensiva ${esc(b)} vs defensa ${esc(a)}</h2>${matchupsHtml(d.matchups.b_offense)}</div>
      </div>
      <div class="grid g2 section">
        <div class="card"><h2>Ofensiva</h2><h3>${esc(a)}</h3>${splitsTable(d.a.summary, "offense")}<h3 class="section">${esc(b)}</h3>${splitsTable(d.b.summary, "offense")}</div>
        <div class="card"><h2>Defensa</h2><h3>${esc(a)}</h3>${splitsTable(d.a.summary, "defense")}<h3 class="section">${esc(b)}</h3>${splitsTable(d.b.summary, "defense")}</div>
      </div>
      <div class="card section"><h2>Local / visitante</h2><div class="grid g2">${haTable(d.a.home_away, a)}${haTable(d.b.home_away, b)}</div></div>
      ${d.injuries ? `<div class="grid g2 section"><div class="card"><h2>Lesiones · ${esc(a)}</h2>${injuriesHtml(d.injuries.a)}</div><div class="card"><h2>Lesiones · ${esc(b)}</h2>${injuriesHtml(d.injuries.b)}</div></div>` : ""}
    `;
    for (const [id, t, s] of [["cmp-log-a", a, d.a.summary], ["cmp-log-b", b, d.b.summary]]) {
      const log = s.games_log || [];
      chart(id, {
        type: "line",
        data: { labels: log.map((g) => `S${g.week} ${g.loc === 1 ? "vs" : "@"} ${g.opp}`),
          datasets: [{ label: "Puntos anotados", data: log.map((g) => g.pf), borderColor: teamColor(t), backgroundColor: teamColor(t), tension: 0.25 },
                     { label: "Puntos permitidos", data: log.map((g) => g.pa), borderColor: "#f87171", backgroundColor: "#f87171", borderDash: [5, 4], tension: 0.25 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true } } },
      });
    }
    const keys = [["pts_off", "Puntos"], ["epa_off", "EPA of."], ["pepa_off", "Pase of."], ["repa_off", "Carrera of."], ["sr_off", "Éxito of."],
      ["pts_def", "Puntos def."], ["epa_def", "EPA def."], ["pepa_def", "Pase def."], ["repa_def", "Carrera def."], ["sr_def", "Éxito def."]];
    const pctl = (x) => (x && x.rank ? Math.round(((32 - x.rank) / 31) * 100) : null);
    chart("cmp-radar", {
      type: "radar",
      data: { labels: keys.map((k) => k[1]), datasets: [
        { label: a, data: keys.map((k) => pctl(d.a.adjusted[k[0]])), borderColor: teamColor(a), backgroundColor: teamColor(a) + "33" },
        { label: b, data: keys.map((k) => pctl(d.b.adjusted[k[0]])), borderColor: teamColor(b), backgroundColor: teamColor(b) + "33" }] },
      options: { maintainAspectRatio: false, scales: { r: { min: 0, max: 100, grid: { color: "#26324a" }, angleLines: { color: "#26324a" }, pointLabels: { color: "#c7d0e0" }, ticks: { display: false } } } },
    });
  } catch (e) {
    out.innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

// ---------------------------------------------------------------- simulador
$("#sim-swap").addEventListener("click", () => {
  const h = $("#sim-home").value; $("#sim-home").value = $("#sim-away").value; $("#sim-away").value = h;
  $("#sim-players").innerHTML = "";
});
$("#sim-load").addEventListener("click", () => runSim(true));
$("#sim-run").addEventListener("click", () => runSim(false));

function simBody() {
  const num = (id) => ($(id).value === "" ? null : +$(id).value);
  const out = (side) => Array.from(document.querySelectorAll(`#sim-players input[data-side=${side}]:checked`)).map((i) => i.value);
  return {
    home: $("#sim-home").value, away: $("#sim-away").value, neutral: $("#sim-neutral").value === "1",
    roof: $("#sim-roof").value || null, temp: num("#sim-temp"), wind: num("#sim-wind"), rest_diff: num("#sim-rest"),
    include_probable: $("#sim-prob").checked, out_home: out("home"), out_away: out("away"),
  };
}

async function runSim(onlyPlayers) {
  const body = simBody();
  if (body.home === body.away) { alert("Elija dos equipos distintos."); return; }
  const outEl = $("#sim-out");
  outEl.innerHTML = `<div class="loading">Simulando 10,000 partidos…</div>`;
  try {
    const r = await api("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const kp = r.key_players || {};
    const prev = new Set([...body.out_home.map((x) => "home:" + x), ...body.out_away.map((x) => "away:" + x)]);
    const list = (side, team) => `<div><h3>${esc(teamName(team))}: marcar ausencias</h3><div class="players">${(kp[side] || []).map((p) =>
      `<label><input type="checkbox" data-side="${side}" value="${esc(p.gsis_id)}" ${prev.has(side + ":" + p.gsis_id) ? "checked" : ""}>
        <b>${esc(p.position)}</b> ${esc(p.name)} <span class="muted small">${p.snap_share != null ? pct(p.snap_share) + " snaps" : esc(p.note)}</span></label>`).join("") || `<span class="muted small">Sin datos de jugadores.</span>`}</div></div>`;
    $("#sim-players").innerHTML = list("away", body.away) + list("home", body.home);
    if (onlyPlayers && !body.out_home.length && !body.out_away.length) {
      outEl.innerHTML = `<div class="muted small">Jugadores cargados. Marque ausencias y ejecute el pronóstico.</div>`;
      return;
    }
    renderPrediction(outEl, r, "sim");
  } catch (e) {
    outEl.innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

// ---------------------------------------------------------------- historial
async function loadHistory() {
  const out = $("#history-out");
  try {
    const h = await api("/api/history");
    const bt = h.backtest || {};
    const m = (x) => x || {};
    const metricRow = (name, x) => `<tr><td>${esc(name)}</td><td>${x.games ?? "—"}</td><td>${fmt(x.mae_points, 2)}</td><td>${fmt(x.mae_margin, 2)}</td><td>${fmt(x.mae_total, 2)}</td><td>${fmt(x.brier, 4)}</td><td>${fmt(x.log_loss, 4)}</td><td>${pct(x.accuracy, 1)}</td></tr>`;
    const live = h.live || {};
    out.innerHTML = `
      <div class="card"><h2>Pronósticos guardados antes de cada partido</h2>
        ${live.games ? `<div class="kpis"><div class="kpi"><div class="l">Partidos evaluados</div><div class="v">${live.games}</div></div>
          <div class="kpi"><div class="l">Error absoluto medio (pts)</div><div class="v">${fmt(live.mae_points, 2)}</div></div>
          <div class="kpi"><div class="l">Brier score</div><div class="v">${fmt(live.brier, 4)}</div></div>
          <div class="kpi"><div class="l">Acierto del favorito</div><div class="v">${pct(live.accuracy, 1)}</div></div></div>`
        : `<p class="muted">Aún no hay partidos finalizados con pronóstico guardado previamente. Cada vez que se consulta un partido futuro se registra el pronóstico con fecha y hora; al terminar el partido se compara aquí con el resultado real.</p>`}
        ${(live.rows || []).length ? `<div class="table-wrap section"><table><thead><tr><th>Partido</th><th>Semana</th><th>Guardado</th><th>Proyección</th><th>P(local)</th><th>Real</th><th>Estado</th></tr></thead><tbody>
          ${live.rows.slice().reverse().slice(0, 60).map((r) => `<tr><td>${esc(r.away)} @ ${esc(r.home)}</td><td>${r.week}</td><td>${esc(new Date(r.created_at).toLocaleString("es"))}</td><td>${fmt(r.pred_away)}-${fmt(r.pred_home)}</td><td>${pct(r.p_home)}</td><td>${r.home_score != null ? `${r.away_score}-${r.home_score}` : "pendiente"}</td><td>${esc(r.status)}</td></tr>`).join("")}
          </tbody></table></div>` : ""}
      </div>
      <div class="card section"><h2>Validación histórica fuera de muestra (walk-forward)</h2>
        <p class="small muted">Validación (selección de hiperparámetros): ${esc((bt.validation_seasons || []).join(", "))} · Prueba (no usada en ninguna decisión): ${esc((bt.test_seasons || []).join(", "))} · ${bt.n_sims ? bt.n_sims.toLocaleString("es") : "—"} simulaciones por partido · calibrado ${esc(bt.created_at ? new Date(bt.created_at).toLocaleString("es") : "—")}</p>
        <div class="table-wrap"><table><thead><tr><th>Modelo</th><th>Partidos</th><th>MAE pts/equipo</th><th>MAE margen</th><th>MAE total</th><th>Brier</th><th>Log-loss</th><th>Acierto</th></tr></thead><tbody>
          ${metricRow("NFL Predictor Pro · prueba", m(bt.test))}
          ${metricRow("NFL Predictor Pro · validación", m(bt.validation))}
          ${metricRow("Referencia: promedio de la liga", m((bt.baselines_test || {}).promedio_liga))}
          ${metricRow("Referencia: promedios simples PF/PC", m((bt.baselines_test || {}).promedios_simples))}
        </tbody></table></div>
        <p class="small muted">Brier score: error cuadrático de la probabilidad (0 = perfecto, 0.25 = moneda al aire). Menor es mejor en todas las columnas excepto Acierto.</p>
      </div>
      <div class="grid g2 section">
        <div class="card"><h2>Calibración (temporadas de prueba)</h2><div class="chart-box tall"><canvas id="calib"></canvas></div>
          <p class="small muted">Cada punto compara la probabilidad predicha del favorito con la frecuencia real de victoria. Sobre la diagonal = el favorito gana más de lo predicho.</p></div>
        <div class="card"><h2>Por temporada</h2><div class="table-wrap"><table><thead><tr><th>Temporada</th><th>Partidos</th><th>MAE pts</th><th>Brier</th><th>Acierto</th></tr></thead><tbody>
          ${Object.entries(bt.by_season || {}).map(([s, x]) => `<tr><td>${s}${(bt.test_seasons || []).includes(+s) ? "" : " (val.)"}</td><td>${x.games}</td><td>${fmt(x.mae_points, 2)}</td><td>${fmt(x.brier, 4)}</td><td>${pct(x.accuracy, 1)}</td></tr>`).join("")}
        </tbody></table></div></div>
      </div>`;
    const cal = bt.calibration_test || [];
    chart("calib", {
      type: "scatter",
      data: { datasets: [
        { label: "Observado", data: cal.map((c) => ({ x: c.pred * 100, y: c.obs * 100, n: c.n })), backgroundColor: "#3fa7ff", pointRadius: cal.map((c) => 4 + Math.sqrt(c.n) / 3) },
        { label: "Calibración perfecta", type: "line", data: [{ x: 50, y: 50 }, { x: 100, y: 100 }], borderColor: "#8b98b3", borderDash: [4, 4], pointRadius: 0 }] },
      options: { maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (c) => `pred ${c.raw.x.toFixed(0)}% · obs ${c.raw.y.toFixed(0)}%${c.raw.n ? " · n=" + c.raw.n : ""}` } } },
        scales: { x: { min: 50, max: 100, title: { display: true, text: "Probabilidad predicha (%)" } }, y: { min: 40, max: 100, title: { display: true, text: "Frecuencia observada (%)" } } } },
    });
  } catch (e) {
    out.innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

// ---------------------------------------------------------------- metodología
async function loadMethod() {
  const out = $("#method-out");
  try {
    const [m, st] = await Promise.all([api("/api/methodology"), api("/api/status")]);
    out.innerHTML = `
      ${m.sections.map((s) => `<div class="card section"><h2>${esc(s.title)}</h2><ul class="clean">${s.body.map((b) => `<li>${esc(b)}</li>`).join("")}</ul></div>`).join("")}
      <div class="card section"><h2>Coeficientes del modelo en producción</h2><div class="table-wrap"><table><thead><tr><th>Variable</th><th>Coef. (pts por 1 DE)</th><th>Pts por unidad</th><th>Media</th></tr></thead><tbody>
        ${m.coefficients.map((c) => `<tr><td class="wrap">${esc(c.label)} <span class="tag">${esc(c.feature)}</span></td><td>${sign(c.coef_std, 3)}</td><td>${sign(c.per_unit, 4)}</td><td>${fmt(c.mean, 3)}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Hiperparámetros: ${esc(Object.entries(m.params).map(([k, v]) => `${k}=${v}`).join(" · "))}</p></div>
      <div class="card section"><h2>Fuentes de datos y última actualización</h2><div class="table-wrap"><table><thead><tr><th>Fuente</th><th>Actualizado (UTC)</th><th>Filas</th><th>Estado</th><th class="wrap">Nota</th></tr></thead><tbody>
        ${st.sources.map((s) => `<tr><td>${esc(s.source)}</td><td>${esc(s.updated_at)}</td><td>${s.rows}</td><td>${esc(s.status)}</td><td class="wrap">${esc(s.message || "")}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Ninguna fuente requiere API key. Variables de entorno opcionales: NFLPP_SHOW_LOGOS (mostrar escudos sólo con autorización), NFLPP_WEATHER, NFLPP_N_SIMS, NFLPP_SEASON, NFLPP_DATA_DIR. Ver README.</p></div>`;
  } catch (e) {
    out.innerHTML = alertsHtml([{ level: "error", text: e.message }]);
  }
}

init();
