#!/usr/bin/env python3
"""Build a dependency-free HTML review dashboard from aggregate OOF metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "airport_calendar": "Airport calendar",
    "catboost_operational": "CatBoost operational",
}


def _aggregate(frame: pd.DataFrame, keys: Iterable[str]) -> list[dict[str, Any]]:
    key_columns = list(keys)
    working = frame.copy()
    working["squared_error"] = working["error_sec"].pow(2)
    grouped = working.groupby(key_columns, observed=True, sort=True)
    result = grouped.agg(
        rows=("error_sec", "size"),
        mean_squared_error=("squared_error", "mean"),
        mae_sec=("abs_error_sec", "mean"),
        p90_abs_error_sec=("abs_error_sec", lambda values: values.quantile(0.90)),
        bias_sec=("error_sec", "mean"),
    ).reset_index()
    result["rmse_sec"] = np.sqrt(result.pop("mean_squared_error"))
    metric_columns = ["rmse_sec", "mae_sec", "p90_abs_error_sec", "bias_sec"]
    result[metric_columns] = result[metric_columns].round(2)
    result["rows"] = result["rows"].astype(int)
    return result.to_dict(orient="records")


def _dashboard_payload(
    oof: pd.DataFrame,
    importance: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target_bands = pd.cut(
        oof["actual_sec"],
        bins=[-np.inf, 0, 300, 600, 900, 1200, 1800, np.inf],
        labels=["Negative", "0–5 min", "5–10 min", "10–15 min", "15–20 min", "20–30 min", "30+ min"],
        right=False,
    )
    banded = oof.assign(target_band=target_bands)

    daily_all = pd.DataFrame(_aggregate(oof, ["model", "date"]))
    daily_all.insert(2, "airport", "ALL")
    daily_airport = pd.DataFrame(_aggregate(oof, ["model", "date", "airport"]))
    daily = pd.concat([daily_all, daily_airport], ignore_index=True)

    if importance.empty:
        feature_importance: list[dict[str, Any]] = []
    else:
        summarized = (
            importance.groupby("feature", observed=True)["importance"]
            .agg(mean_importance="mean", min_importance="min", max_importance="max")
            .sort_values("mean_importance", ascending=False)
            .head(20)
            .reset_index()
        )
        summarized[["mean_importance", "min_importance", "max_importance"]] = summarized[
            ["mean_importance", "min_importance", "max_importance"]
        ].round(3)
        feature_importance = summarized.to_dict(orient="records")

    return {
        "title": "PRC 2026 taxi-time model review",
        "generated_at_utc": metadata["generated_at_utc"],
        "smoke": bool(metadata["smoke"]),
        "models": [
            {"id": model, "label": MODEL_LABELS.get(model, model)}
            for model in sorted(oof["model"].unique())
        ],
        "airports": sorted(oof["airport"].dropna().unique().tolist()),
        "folds": metadata["folds"],
        "overall": _aggregate(oof, ["model"]),
        "by_fold": _aggregate(oof, ["model", "fold"]),
        "by_airport": _aggregate(oof, ["model", "airport"]),
        "by_target_band": _aggregate(banded.dropna(subset=["target_band"]), ["model", "target_band"]),
        "daily": daily.to_dict(orient="records"),
        "feature_importance": feature_importance,
        "definitions": {
            "metric": "Root mean squared error (RMSE), in seconds. Lower is better.",
            "population": "All departure rows with a scheduled time and a nonmissing taxi-time target; no rows are removed by target value.",
            "validation": "Forward folds: train only on dates before each validation month.",
            "calendar": "Each cell is one UTC scheduled-departure date. Empty cells were not validation dates.",
        },
        "unsafe_fields_excluded": metadata["unsafe_fields_excluded"],
        "training_rows_after_quality_filter": metadata["training_rows_after_quality_filter"],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PRC 2026 model review</title>
  <style>
    :root {
      color-scheme: light;
      --canvas: #f6f8fa;
      --card: #ffffff;
      --ink: #1f2328;
      --muted: #59636e;
      --line: #d0d7de;
      --line-soft: #eaeef2;
      --blue: #0969da;
      --blue-soft: #ddf4ff;
      --gold: #9a6700;
      --gold-soft: #fff8c5;
      --purple: #8250df;
      --shadow: 0 1px 2px rgba(31, 35, 40, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--canvas);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, select { font: inherit; }
    .shell { width: min(1480px, 100%); margin: 0 auto; padding: 28px clamp(16px, 3vw, 42px) 52px; }
    header { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 22px; }
    h1 { margin: 0 0 6px; font-size: clamp(24px, 3vw, 34px); line-height: 1.18; letter-spacing: -0.025em; }
    h2 { margin: 0; font-size: 17px; line-height: 1.3; }
    h3 { margin: 0; font-size: 14px; }
    p { margin: 0; }
    .subtitle { color: var(--muted); max-width: 760px; }
    .badge { flex: 0 0 auto; border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; color: var(--muted); background: var(--card); font-size: 12px; font-weight: 600; }
    .badge.smoke { border-color: #d4a72c; background: var(--gold-soft); color: #633c01; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; padding: 14px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); margin-bottom: 16px; }
    .control { display: grid; gap: 4px; min-width: min(220px, 100%); }
    .control label { color: var(--muted); font-size: 12px; font-weight: 600; }
    select { height: 34px; border: 1px solid #afb8c1; border-radius: 6px; background: white; color: var(--ink); padding: 0 30px 0 10px; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(12, minmax(0, 1fr)); }
    .card { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; }
    .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 16px 18px 12px; border-bottom: 1px solid var(--line-soft); }
    .card-head p { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .card-body { padding: 16px 18px 18px; min-width: 0; }
    .scorecards { grid-column: span 12; display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .score { padding: 17px 18px; position: relative; }
    .score::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--blue); }
    .score.secondary::before { background: var(--purple); }
    .score-name { color: var(--muted); font-weight: 600; }
    .score-value { margin: 4px 0 1px; font-size: clamp(27px, 3vw, 38px); line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -0.025em; }
    .score-value small { font-size: 13px; color: var(--muted); letter-spacing: 0; }
    .score-meta { display: flex; flex-wrap: wrap; gap: 8px 16px; color: var(--muted); font-size: 12px; }
    .delta { display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 7px; font-weight: 650; }
    .delta.better { color: #0550ae; background: var(--blue-soft); }
    .delta.worse { color: #633c01; background: var(--gold-soft); }
    .calendar-card { grid-column: span 12; }
    .calendar-scroll { overflow-x: auto; padding-bottom: 4px; }
    .calendar-wrap { min-width: 720px; position: relative; }
    .month-labels { display: grid; grid-template-columns: 28px repeat(53, 11px); gap: 2px; height: 20px; color: var(--muted); font-size: 11px; }
    .month-label { white-space: nowrap; }
    .calendar-grid { display: grid; grid-template-columns: 28px repeat(53, 11px); grid-template-rows: repeat(7, 11px); gap: 2px; }
    .day-label { color: var(--muted); font-size: 10px; line-height: 11px; }
    .day { width: 11px; height: 11px; border-radius: 2px; border: 1px solid rgba(27, 31, 36, 0.06); background: #ebedf0; padding: 0; cursor: default; }
    .day.has-data { cursor: pointer; }
    .day:focus-visible { outline: 2px solid var(--blue); outline-offset: 1px; }
    .day.selected { outline: 2px solid var(--ink); outline-offset: 1px; }
    .legend { display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 5px; margin-top: 12px; color: var(--muted); font-size: 11px; }
    .legend-cell { width: 13px; height: 13px; border-radius: 2px; border: 1px solid rgba(27, 31, 36, 0.06); }
    .tooltip { position: fixed; z-index: 20; pointer-events: none; max-width: 270px; padding: 8px 10px; border-radius: 6px; background: #24292f; color: white; font-size: 12px; box-shadow: 0 8px 24px rgba(140,149,159,.25); opacity: 0; transform: translateY(4px); transition: opacity 100ms, transform 100ms; }
    .tooltip.visible { opacity: 1; transform: translateY(0); }
    .tooltip strong { display: block; }
    .split { grid-column: span 7; }
    .side { grid-column: span 5; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th { color: var(--muted); font-size: 11px; font-weight: 650; text-align: left; white-space: nowrap; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--line-soft); }
    th:first-child, td:first-child { padding-left: 0; }
    th:last-child, td:last-child { padding-right: 0; text-align: right; }
    tbody tr:last-child td { border-bottom: 0; }
    .number { text-align: right; }
    .bar-row { display: grid; grid-template-columns: minmax(80px, 130px) minmax(100px, 1fr) 58px; align-items: center; gap: 10px; margin: 10px 0; }
    .bar-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { height: 10px; background: #eaeef2; border-radius: 999px; overflow: hidden; }
    .bar { height: 100%; border-radius: inherit; background: var(--blue); }
    .bar.purple { background: var(--purple); }
    .bar.gold { background: #bf8700; }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; }
    .two-bars { display: grid; gap: 3px; }
    .two-bars .bar-track { height: 7px; }
    .feature-note, .empty { color: var(--muted); font-size: 12px; }
    .method { grid-column: span 12; }
    .method-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
    .method-grid dt { color: var(--muted); font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: .04em; }
    .method-grid dd { margin: 4px 0 0; }
    code { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--canvas); padding: 2px 5px; border-radius: 4px; overflow-wrap: anywhere; }
    @media (max-width: 900px) {
      .split, .side { grid-column: span 12; }
      .method-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 620px) {
      .shell { padding-top: 18px; }
      header { display: block; }
      header .badge { display: inline-block; margin-top: 12px; }
      .scorecards { grid-template-columns: 1fr; }
      .method-grid { grid-template-columns: 1fr; }
      .control { width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>PRC 2026 taxi-time model review</h1>
        <p class="subtitle">Forward validation for a simple airport-calendar baseline and an operational model that uses planned and schedule fields. Use the calendar to find error clusters, then use the comparisons to choose the next experiment.</p>
      </div>
      <span id="runBadge" class="badge"></span>
    </header>

    <section class="controls" aria-label="Dashboard filters">
      <div class="control">
        <label for="modelSelect">Calendar model</label>
        <select id="modelSelect"></select>
      </div>
      <div class="control">
        <label for="airportSelect">Airport</label>
        <select id="airportSelect"></select>
      </div>
    </section>

    <div class="grid">
      <section id="scorecards" class="scorecards" aria-label="Overall model scores"></section>

      <section class="card calendar-card">
        <div class="card-head">
          <div><h2>Daily error calendar</h2><p id="calendarSubtitle"></p></div>
          <div id="selectedDay" class="badge">Select a date</div>
        </div>
        <div class="card-body">
          <div class="calendar-scroll"><div class="calendar-wrap">
            <div id="monthLabels" class="month-labels"></div>
            <div id="calendar" class="calendar-grid" role="grid" aria-label="Daily RMSE calendar"></div>
          </div></div>
          <div class="legend"><span>Lower RMSE</span><span class="legend-cell" style="background:#ddf4ff"></span><span class="legend-cell" style="background:#80ccff"></span><span class="legend-cell" style="background:#218bff"></span><span class="legend-cell" style="background:#0969da"></span><span class="legend-cell" style="background:#8250df"></span><span>Higher RMSE</span></div>
        </div>
      </section>

      <section class="card split">
        <div class="card-head"><div><h2>Airport comparison</h2><p>Same validation rows, ordered by the selected model's RMSE.</p></div></div>
        <div class="card-body"><div id="airportComparison" class="table-wrap"></div></div>
      </section>

      <section class="card side">
        <div class="card-head"><div><h2>Error by taxi-time band</h2><p>Checks whether long taxi times dominate model error.</p></div></div>
        <div class="card-body" id="targetBands"></div>
      </section>

      <section class="card split">
        <div class="card-head"><div><h2>Worst validation days</h2><p id="worstDaysSubtitle"></p></div></div>
        <div class="card-body"><div id="worstDays" class="table-wrap"></div></div>
      </section>

      <section class="card side">
        <div class="card-head"><div><h2>Operational feature importance</h2><p>Mean CatBoost importance across completed folds.</p></div></div>
        <div class="card-body" id="featureImportance"></div>
      </section>

      <section class="card method">
        <div class="card-head"><div><h2>Validation contract</h2><p>Definitions and provenance for this review surface.</p></div></div>
        <div class="card-body">
          <dl class="method-grid">
            <div><dt>Metric</dt><dd id="metricDefinition"></dd></div>
            <div><dt>Population</dt><dd id="populationDefinition"></dd></div>
            <div><dt>Validation</dt><dd id="validationDefinition"></dd></div>
            <div><dt>Generated</dt><dd id="generatedAt"></dd></div>
          </dl>
          <div class="table-wrap" style="margin-top:16px"><table id="foldTable"></table></div>
          <p class="feature-note" style="margin-top:14px">Quarantined fields: <span id="unsafeFields"></span></p>
        </div>
      </section>
    </div>
  </main>
  <div id="tooltip" class="tooltip" role="tooltip"></div>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    const $ = (id) => document.getElementById(id);
    const modelLabel = Object.fromEntries(DATA.models.map(d => [d.id, d.label]));
    const palette = ['#ddf4ff', '#80ccff', '#218bff', '#0969da', '#8250df'];
    const bandOrder = ['Negative', '0–5 min', '5–10 min', '10–15 min', '15–20 min', '20–30 min', '30+ min'];
    let state = { model: DATA.models.at(-1).id, airport: 'ALL', selectedDate: null };
    const fmt = (value, digits = 0) => Number(value).toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
    const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    const metricFor = (records, model, key, value) => records.find(d => d.model === model && (!key || d[key] === value));

    function setup() {
      const badge = $('runBadge');
      badge.textContent = DATA.smoke ? 'SMOKE RUN · bounded sample' : 'FULL RUN · three forward folds';
      if (DATA.smoke) badge.classList.add('smoke');
      DATA.models.forEach(model => $('modelSelect').add(new Option(model.label, model.id)));
      $('modelSelect').value = state.model;
      $('airportSelect').add(new Option('All airports', 'ALL'));
      DATA.airports.forEach(airport => $('airportSelect').add(new Option(airport, airport)));
      $('modelSelect').addEventListener('change', event => { state.model = event.target.value; state.selectedDate = null; render(); });
      $('airportSelect').addEventListener('change', event => { state.airport = event.target.value; state.selectedDate = null; render(); });
      $('metricDefinition').textContent = DATA.definitions.metric;
      $('populationDefinition').textContent = DATA.definitions.population;
      $('validationDefinition').textContent = DATA.definitions.validation;
      $('generatedAt').textContent = new Date(DATA.generated_at_utc).toLocaleString();
      $('unsafeFields').innerHTML = DATA.unsafe_fields_excluded.map(value => `<code>${escapeHtml(value)}</code>`).join(' ');
      renderFoldTable();
      renderScorecards();
      renderFeatureImportance();
      render();
    }

    function renderScorecards() {
      const baseline = DATA.overall.find(d => d.model === 'airport_calendar');
      $('scorecards').innerHTML = DATA.models.map((model, index) => {
        const metric = DATA.overall.find(d => d.model === model.id);
        let delta = '';
        if (baseline && model.id !== 'airport_calendar') {
          const pct = 100 * (baseline.rmse_sec - metric.rmse_sec) / baseline.rmse_sec;
          const cls = pct >= 0 ? 'better' : 'worse';
          const word = pct >= 0 ? 'lower' : 'higher';
          delta = `<span class="delta ${cls}">${fmt(Math.abs(pct), 1)}% ${word} than baseline</span>`;
        }
        return `<article class="card score ${index ? 'secondary' : ''}">
          <div class="score-name">${escapeHtml(model.label)}</div>
          <div class="score-value">${fmt(metric.rmse_sec, 1)} <small>sec RMSE</small></div>
          <div class="score-meta"><span>${fmt(metric.mae_sec, 1)} sec MAE</span><span>${fmt(metric.p90_abs_error_sec, 0)} sec P90</span><span>${fmt(metric.rows)} rows</span>${delta}</div>
        </article>`;
      }).join('');
    }

    function render() {
      renderCalendar();
      renderAirportComparison();
      renderTargetBands();
      renderWorstDays();
    }

    function renderCalendar() {
      const rows = DATA.daily.filter(d => d.model === state.model && d.airport === state.airport);
      const byDate = new Map(rows.map(d => [d.date, d]));
      const values = rows.map(d => d.rmse_sec).sort((a,b) => a-b);
      const quantile = q => values.length ? values[Math.min(values.length - 1, Math.floor((values.length - 1) * q))] : 0;
      const cuts = [quantile(.2), quantile(.4), quantile(.6), quantile(.8)];
      const airportText = state.airport === 'ALL' ? 'all airports' : state.airport;
      $('calendarSubtitle').textContent = `${modelLabel[state.model]} · ${airportText} · ${rows.length} validation days`;
      const year = 2025;
      const jan1 = new Date(Date.UTC(year, 0, 1));
      const mondayOffset = (jan1.getUTCDay() + 6) % 7;
      const start = new Date(jan1); start.setUTCDate(start.getUTCDate() - mondayOffset);
      const monthLabels = [''];
      let lastMonth = -1;
      for (let week = 0; week < 53; week++) {
        const date = new Date(start); date.setUTCDate(date.getUTCDate() + week * 7);
        const month = date.getUTCMonth();
        monthLabels.push(month !== lastMonth && date.getUTCDate() <= 7 ? date.toLocaleString(undefined, {month:'short', timeZone:'UTC'}) : '');
        lastMonth = month;
      }
      $('monthLabels').innerHTML = monthLabels.map((label, index) => `<span class="month-label" style="grid-column:${index + 1}">${label}</span>`).join('');
      const dayNames = ['Mon', '', 'Wed', '', 'Fri', '', 'Sun'];
      let html = dayNames.map((name, row) => `<span class="day-label" style="grid-column:1;grid-row:${row+1}">${name}</span>`).join('');
      for (let week = 0; week < 53; week++) {
        for (let day = 0; day < 7; day++) {
          const date = new Date(start); date.setUTCDate(date.getUTCDate() + week * 7 + day);
          const iso = date.toISOString().slice(0,10);
          const record = date.getUTCFullYear() === year ? byDate.get(iso) : null;
          let color = '#ebedf0';
          if (record) color = palette[cuts.findIndex(cut => record.rmse_sec <= cut) === -1 ? 4 : cuts.findIndex(cut => record.rmse_sec <= cut)];
          const selected = state.selectedDate === iso ? ' selected' : '';
          const label = record ? `${iso}: ${fmt(record.rmse_sec, 1)} seconds RMSE, ${fmt(record.rows)} rows` : `${iso}: no validation data`;
          html += `<button class="day${record ? ' has-data' : ''}${selected}" style="grid-column:${week+2};grid-row:${day+1};background:${color}" data-date="${iso}" aria-label="${label}" ${record ? '' : 'disabled'}></button>`;
        }
      }
      $('calendar').innerHTML = html;
      document.querySelectorAll('.day.has-data').forEach(cell => {
        cell.addEventListener('mouseenter', event => showTooltip(event, byDate.get(cell.dataset.date)));
        cell.addEventListener('mousemove', moveTooltip);
        cell.addEventListener('mouseleave', hideTooltip);
        cell.addEventListener('focus', event => showTooltip(event, byDate.get(cell.dataset.date)));
        cell.addEventListener('blur', hideTooltip);
        cell.addEventListener('click', () => { state.selectedDate = cell.dataset.date; renderCalendar(); renderWorstDays(); });
      });
      const chosen = state.selectedDate ? byDate.get(state.selectedDate) : null;
      $('selectedDay').textContent = chosen ? `${chosen.date} · ${fmt(chosen.rmse_sec, 1)} sec RMSE` : 'Select a date';
    }

    function showTooltip(event, record) {
      if (!record) return;
      const other = DATA.models.find(model => model.id !== state.model);
      const peer = other && DATA.daily.find(d => d.model === other.id && d.airport === state.airport && d.date === record.date);
      const comparison = peer ? `<br>${escapeHtml(other.label)}: ${fmt(peer.rmse_sec, 1)} sec` : '';
      $('tooltip').innerHTML = `<strong>${record.date}</strong>${fmt(record.rmse_sec, 1)} sec RMSE · ${fmt(record.mae_sec, 1)} sec MAE<br>${fmt(record.rows)} departures${comparison}`;
      $('tooltip').classList.add('visible'); moveTooltip(event);
    }
    function moveTooltip(event) {
      const x = Math.min(window.innerWidth - 285, Math.max(8, (event.clientX || event.target.getBoundingClientRect().right) + 12));
      const y = Math.min(window.innerHeight - 110, Math.max(8, (event.clientY || event.target.getBoundingClientRect().bottom) + 12));
      $('tooltip').style.left = `${x}px`; $('tooltip').style.top = `${y}px`;
    }
    function hideTooltip() { $('tooltip').classList.remove('visible'); }

    function renderAirportComparison() {
      const models = DATA.models.map(d => d.id);
      const selected = DATA.by_airport.filter(d => d.model === state.model).sort((a,b) => b.rmse_sec-a.rmse_sec);
      const rows = selected.map(item => {
        const values = Object.fromEntries(models.map(model => [model, metricFor(DATA.by_airport, model, 'airport', item.airport)]));
        const baseline = values.airport_calendar;
        const boosted = values.catboost_operational;
        const change = baseline && boosted ? 100 * (baseline.rmse_sec - boosted.rmse_sec) / baseline.rmse_sec : null;
        return `<tr><td><strong>${escapeHtml(item.airport)}</strong></td>${models.map(model => `<td class="number">${fmt(values[model].rmse_sec, 1)}</td>`).join('')}<td class="number">${change === null ? '—' : `<span class="delta ${change >= 0 ? 'better':'worse'}">${change >= 0 ? '↓':'↑'} ${fmt(Math.abs(change),1)}%</span>`}</td></tr>`;
      }).join('');
      $('airportComparison').innerHTML = `<table><thead><tr><th>Airport</th>${models.map(model => `<th class="number">${escapeHtml(modelLabel[model])}</th>`).join('')}<th class="number">Tree vs base</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderTargetBands() {
      const models = DATA.models.map(d => d.id);
      const all = DATA.by_target_band;
      const max = Math.max(...all.map(d => d.rmse_sec), 1);
      $('targetBands').innerHTML = bandOrder.map(band => {
        const entries = models.map(model => metricFor(all, model, 'target_band', band)).filter(Boolean);
        if (!entries.length) return '';
        return `<div class="bar-row"><div class="bar-label">${band}</div><div class="two-bars">${entries.map((entry, index) => `<div class="bar-track" title="${escapeHtml(modelLabel[entry.model])}: ${fmt(entry.rmse_sec,1)} sec"><div class="bar ${index ? 'purple':''}" style="width:${100*entry.rmse_sec/max}%"></div></div>`).join('')}</div><div class="bar-value">${fmt(entries.find(d => d.model === state.model)?.rmse_sec ?? entries[0].rmse_sec,0)}s</div></div>`;
      }).join('') + `<p class="feature-note" style="margin-top:14px"><span style="color:var(--blue)">■</span> Airport calendar &nbsp; <span style="color:var(--purple)">■</span> CatBoost operational</p>`;
    }

    function renderWorstDays() {
      let rows = DATA.daily.filter(d => d.model === state.model && d.airport === state.airport);
      if (state.selectedDate) rows = rows.filter(d => d.date === state.selectedDate);
      rows = rows.sort((a,b) => b.rmse_sec-a.rmse_sec).slice(0, state.selectedDate ? 1 : 12);
      const scope = state.airport === 'ALL' ? 'all airports' : state.airport;
      $('worstDaysSubtitle').textContent = state.selectedDate ? `${modelLabel[state.model]} · ${scope} · selected day` : `${modelLabel[state.model]} · ${scope} · highest RMSE first`;
      $('worstDays').innerHTML = rows.length ? `<table><thead><tr><th>Date</th><th class="number">Rows</th><th class="number">RMSE</th><th class="number">MAE</th><th class="number">Bias</th></tr></thead><tbody>${rows.map(row => `<tr><td>${row.date}</td><td class="number">${fmt(row.rows)}</td><td class="number"><strong>${fmt(row.rmse_sec,1)}s</strong></td><td class="number">${fmt(row.mae_sec,1)}s</td><td class="number">${row.bias_sec >= 0 ? '+':''}${fmt(row.bias_sec,1)}s</td></tr>`).join('')}</tbody></table>` : '<p class="empty">No matching days.</p>';
    }

    function renderFeatureImportance() {
      if (!DATA.feature_importance.length) {
        $('featureImportance').innerHTML = '<p class="empty">The boosted model was not part of this run.</p>'; return;
      }
      const top = DATA.feature_importance.slice(0, 12);
      const max = Math.max(...top.map(d => d.mean_importance), 1);
      $('featureImportance').innerHTML = top.map(item => `<div class="bar-row"><div class="bar-label" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</div><div class="bar-track"><div class="bar purple" style="width:${100*item.mean_importance/max}%"></div></div><div class="bar-value">${fmt(item.mean_importance,1)}</div></div>`).join('');
    }

    function renderFoldTable() {
      const metrics = DATA.by_fold;
      $('foldTable').innerHTML = `<thead><tr><th>Fold</th><th>Training window</th><th>Validation window</th>${DATA.models.map(model => `<th class="number">${escapeHtml(model.label)} RMSE</th>`).join('')}</tr></thead><tbody>${DATA.folds.map(fold => `<tr><td><strong>${escapeHtml(fold.name)}</strong></td><td>${fold.train_start} to ${fold.train_end}</td><td>${fold.valid_start} to ${fold.valid_end}</td>${DATA.models.map(model => { const row = metrics.find(d => d.model === model.id && d.fold === fold.name); return `<td class="number">${row ? fmt(row.rmse_sec,1)+'s' : '—'}</td>`; }).join('')}</tr>`).join('')}</tbody>`;
    }
    setup();
  </script>
</body>
</html>
'''


def build_dashboard(
    oof: pd.DataFrame,
    importance: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path,
    results_dir: Path,
) -> None:
    """Write the aggregate JSON contract and one portable HTML dashboard."""
    payload = _dashboard_payload(oof, importance, metadata)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (results_dir / "dashboard_data.json").write_text(payload_text + "\n", encoding="utf-8")
    safe_payload = payload_text.replace("</", "<\\/")
    output_path.write_text(
        HTML_TEMPLATE.replace("__DASHBOARD_DATA__", safe_payload),
        encoding="utf-8",
    )


def main() -> None:
    """Regenerate the dashboard from saved full-run results."""
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "modeling" / "results"
    oof = pd.read_parquet(results_dir / "oof_predictions.parquet")
    importance = pd.read_csv(results_dir / "feature_importance.csv")
    metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    build_dashboard(
        oof,
        importance,
        metadata,
        root / "modeling" / "dashboard.html",
        results_dir,
    )


if __name__ == "__main__":
    main()
