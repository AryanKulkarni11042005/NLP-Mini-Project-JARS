import os
import html
import tempfile
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sentiment
from evaluation.text_metrics import calculate_text_metrics
from evaluation.transforms import add_noise, to_hinglish, NOISE_LEVELS
from model_recommender import (
    analyze_dataset,
    detect_text_column,
    detect_label_column,
    detect_task,
    recommend_from_dataset,
)

MODELS = sentiment.available_models()

CLASSES = sentiment.classes
COLUMNS = ["Model", "Prediction", "Confidence"] + [c.capitalize() for c in CLASSES] + ["Time (ms)"]

# ---------------------------------------------------------------------
# Design system (CSS only). Dark analytics theme.
# Tokens -> Gradio variable overrides -> components -> product chrome
# ---------------------------------------------------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');

/* ===== 1. Tokens ===== */
:root, .gradio-container, .dark {
  --bg:#0b0f17; --bg-deep:#080b12; --surface:#111722; --surface-2:#151c28; --surface-3:#192231;
  --line:#263142; --line-soft:#1d2635;
  --text:#f1f5f9; --text-2:#94a3b8; --text-3:#64748b;
  --accent:#22c55e; --accent-dim:rgba(34,197,94,.14); --accent-line:rgba(34,197,94,.38);
  --indigo:#6366f1; --indigo-dim:rgba(99,102,241,.16); --cyan:#22d3ee; --cyan-dim:rgba(34,211,238,.14);
  --warn:#f59e0b; --warn-dim:rgba(245,158,11,.14); --danger:#ef4444; --danger-dim:rgba(239,68,68,.14);
  --r:12px; --r-sm:8px; --gap:12px;
  --ease:.18s ease;
}

/* ===== 2. Gradio variable overrides (forces dark everywhere) ===== */
:root, .gradio-container, .gradio-container.dark, .dark, body.dark {
  color-scheme:dark;
  --body-background-fill:var(--bg); --body-text-color:var(--text); --body-text-color-subdued:var(--text-2);
  --background-fill-primary:var(--surface); --background-fill-secondary:var(--surface-2);
  --block-background-fill:var(--surface); --block-border-color:var(--line); --block-border-width:1px; --block-radius:var(--r);
  --block-label-background-fill:var(--surface-2); --block-label-text-color:var(--text-2); --block-title-text-color:var(--text-2);
  --block-info-text-color:var(--text-3); --block-shadow:none; --block-label-border-color:var(--line);
  --panel-background-fill:var(--surface); --panel-border-color:var(--line);
  --border-color-primary:var(--line); --border-color-accent:var(--accent); --color-accent:var(--accent); --color-accent-soft:var(--accent-dim);
  --input-background-fill:var(--surface); --input-background-fill-focus:var(--surface); --input-background-fill-hover:var(--surface-2);
  --input-border-color:var(--line); --input-border-color-focus:var(--accent); --input-border-color-hover:#334155; --input-radius:var(--r-sm);
  --input-placeholder-color:var(--text-3); --input-shadow:none; --input-shadow-focus:0 0 0 3px var(--accent-dim);
  --table-even-background-fill:var(--surface); --table-odd-background-fill:var(--surface-2); --table-border-color:var(--line);
  --table-row-focus:var(--surface-3); --table-text-color:var(--text); --table-radius:var(--r-sm);
  --button-primary-background-fill:var(--accent); --button-primary-background-fill-hover:#16a34a; --button-primary-text-color:#04130a;
  --button-primary-border-color:var(--accent); --button-secondary-background-fill:var(--surface-2); --button-secondary-background-fill-hover:var(--surface-3);
  --button-secondary-text-color:var(--text); --button-secondary-border-color:var(--line); --button-large-radius:var(--r-sm);
  --checkbox-background-color:var(--surface); --checkbox-border-color:var(--line); --checkbox-label-background-fill:var(--surface-2);
  --checkbox-label-background-fill-hover:var(--surface-3); --checkbox-label-background-fill-selected:var(--accent-dim);
  --checkbox-label-text-color:var(--text-2); --checkbox-label-text-color-selected:var(--accent); --checkbox-label-border-color:var(--line);
  --checkbox-label-border-color-selected:var(--accent-line); --checkbox-background-color-selected:var(--accent);
  --shadow-drop:none; --shadow-drop-lg:0 8px 24px rgba(0,0,0,.4); --shadow-spread:0px;
  --neutral-50:#151c28; --neutral-100:#192231; --neutral-200:#263142; --neutral-300:#334155;
  --layout-gap:var(--gap); --block-padding:12px; --text-sm:13px; --text-md:14px;
  --code-background-fill:var(--surface-2); --link-text-color:var(--cyan);
  --stat-background-fill:var(--accent);
}

/* ===== 3. Base ===== */
body, .gradio-container, .gradio-container * { font-family:'Inter','Noto Sans Devanagari','Segoe UI',system-ui,sans-serif; }
body, .gradio-container { background:var(--bg) !important; color:var(--text); font-size:13.5px; }
.gradio-container {
  width:min(1140px, calc(100vw - 48px)) !important;
  max-width:1140px !important;
  box-sizing:border-box !important;
  margin:0 auto !important;
  padding:14px 0 28px !important;
}
.gradio-container > main, .gradio-container main.fillable, .gradio-container .app { width:100% !important; max-width:100% !important; padding:0 !important; margin:0 auto !important; }
.gradio-container .hero, .gradio-container .panel, .gradio-container .control-panel { margin-left:0 !important; margin-right:0 !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 { color:var(--text); letter-spacing:-.01em; }
.gradio-container .prose, .gradio-container .md, .gradio-container .markdown { color:var(--text-2); font-size:13.5px; }
.gradio-container a { color:var(--cyan); }
.gradio-container footer { display:none !important; }
.sink { display:none !important; }
::-webkit-scrollbar { width:9px; height:9px; } ::-webkit-scrollbar-thumb { background:#2a3648; border-radius:6px; } ::-webkit-scrollbar-track { background:transparent; }

/* ===== 4. Header ===== */
.app-header { display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:62px; padding:10px 16px;
  background:var(--surface); border:1px solid var(--line); border-radius:var(--r); margin-bottom:10px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:36px; height:36px; border-radius:10px; display:grid; place-items:center; font-weight:700; font-size:17px; color:#04130a;
  background:linear-gradient(135deg,#22c55e,#22d3ee); }
.brand-name { font-size:15.5px; font-weight:650; color:var(--text); line-height:1.2; }
.brand-sub { font-size:12px; color:var(--text-2); margin-top:2px; }
.header-chips { display:flex; gap:8px; }
.chip { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:500; padding:5px 11px; border-radius:999px;
  background:var(--surface-2); border:1px solid var(--line); color:var(--text-2); }
.chip::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--text-3); }
.chip.ok { color:#86efac; border-color:var(--accent-line); background:var(--accent-dim); } .chip.ok::before { background:var(--accent); }
.chip.warn { color:#fcd34d; border-color:rgba(245,158,11,.4); background:var(--warn-dim); } .chip.warn::before { background:var(--warn); }

/* ===== 5. Navigation (tabs) ===== */
.gradio-container .tab-wrapper, .gradio-container .tab-container, .gradio-container .tab-nav { background:transparent !important; border-color:var(--line) !important; }
.gradio-container .tab-wrapper { margin-bottom:12px; border-bottom:1px solid var(--line) !important; padding:0 !important; }
.gradio-container .tab-container { gap:2px; border:0 !important; padding:0 !important; }
.gradio-container .tab-container::after { display:none !important; }
.gradio-container button[role="tab"], .gradio-container .tab-nav button { height:38px; padding:0 16px !important; margin:0; border:0 !important; border-radius:var(--r-sm) var(--r-sm) 0 0 !important;
  background:transparent !important; color:var(--text-2) !important; font-size:13.5px !important; font-weight:550 !important; transition:color var(--ease), background var(--ease); }
.gradio-container button[role="tab"]:hover, .gradio-container .tab-nav button:hover { color:var(--text) !important; background:var(--surface) !important; }
.gradio-container button[role="tab"].selected, .gradio-container button[role="tab"][aria-selected="true"], .gradio-container .tab-nav button.selected {
  color:var(--accent) !important; background:var(--surface) !important; box-shadow:inset 0 -2px 0 var(--accent); }
.gradio-container button[role="tab"]::after { display:none !important; }
.gradio-container .tabitem { padding:0 !important; border:0 !important; background:transparent !important; }

/* ===== 6. Surfaces & section heads ===== */
.panel, .control-panel { background:var(--surface-2) !important; border:1px solid var(--line) !important; border-radius:var(--r) !important; padding:14px !important; }
.control-panel { background:var(--surface-3) !important; }
.gradio-container .block, .gradio-container .form { border-radius:var(--r) !important; }
.gradio-container .gap, .gradio-container .row, .gradio-container .column { gap:var(--gap) !important; }
.sec-head { margin:20px 0 8px; display:flex; flex-direction:column; gap:2px; }
.sec-head h2 { margin:0; font-size:17px; font-weight:650; color:var(--text); }
.sec-head p { margin:0; font-size:12.5px; color:var(--text-3); }
.sec-head.first { margin-top:4px; }
.empty { padding:18px; text-align:center; color:var(--text-3); font-size:13px; background:var(--surface); border:1px dashed var(--line); border-radius:var(--r); }

/* ===== 7. Inputs & buttons ===== */
.gradio-container input, .gradio-container textarea, .gradio-container select, .gradio-container .wrap-inner, .gradio-container .secondary-wrap {
  background:var(--surface) !important; color:var(--text) !important; border-color:var(--line) !important; font-size:13.5px !important; }
.gradio-container input:focus, .gradio-container textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-dim) !important; }
.gradio-container label > span, .gradio-container .block-title, .gradio-container .block-label, .gradio-container [data-testid="block-label"] {
  color:var(--text-2) !important; font-size:12px !important; font-weight:550 !important; }
.gradio-container ul.options, .gradio-container .options { background:var(--surface-2) !important; border:1px solid var(--line) !important; box-shadow:var(--shadow-drop-lg) !important; }
.gradio-container ul.options li, .gradio-container .options .item { color:var(--text) !important; font-size:13px !important; }
.gradio-container ul.options li:hover, .gradio-container ul.options li.active, .gradio-container .options .item:hover { background:var(--surface-3) !important; }
.gradio-container button { transition:background var(--ease), border-color var(--ease), transform var(--ease); font-size:13.5px; }
.gradio-container button.primary, .gradio-container .primary { background:var(--accent) !important; border:1px solid var(--accent) !important; color:#04130a !important;
  font-weight:650 !important; min-height:40px; border-radius:var(--r-sm) !important; }
.gradio-container button.primary:hover { background:#4ade80 !important; transform:translateY(-1px); }
.gradio-container button.secondary, .gradio-container button.ghost { background:var(--surface-2) !important; color:var(--text) !important; border:1px solid var(--line) !important;
  min-height:40px; font-weight:550 !important; border-radius:var(--r-sm) !important; }
.gradio-container button.secondary:hover, .gradio-container button.ghost:hover { background:var(--surface-3) !important; border-color:#3b4a60 !important; }
.gradio-container button:focus-visible, .gradio-container [role="tab"]:focus-visible { outline:2px solid var(--accent) !important; outline-offset:2px; }
.gradio-container .accordion, .gradio-container .block.accordion, .gradio-container button.label-wrap { background:var(--surface-2) !important; color:var(--text) !important; border-color:var(--line) !important; }
.gradio-container .label-wrap span { color:var(--text) !important; font-weight:600; }
.gradio-container .file-preview, .gradio-container .upload-container, .gradio-container [data-testid="file-upload"] { background:var(--surface) !important; border-color:var(--line) !important; color:var(--text-2) !important; }
.upload-compact .wrap, .upload-compact [data-testid="file-upload"] { min-height:0 !important; }
.upload-compact { max-height:130px; }

/* ===== 8. Metric cards ===== */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.metric-grid.cols-4 { grid-template-columns:repeat(4,1fr); }
.metric-card { min-height:82px; padding:12px 14px; background:var(--surface); border:1px solid var(--line); border-radius:var(--r);
  transition:border-color var(--ease), background var(--ease); display:flex; flex-direction:column; justify-content:space-between; }
.metric-card:hover { border-color:#3b4a60; background:var(--surface-2); }
.metric-label { font-size:11.5px; font-weight:500; color:var(--text-2); }
.metric-value { font-size:1.45rem; font-weight:650; color:var(--text); letter-spacing:-.02em; line-height:1.15; margin:4px 0 2px; font-variant-numeric:tabular-nums; }
.metric-value.sm { font-size:1.15rem; }
.metric-delta { font-size:11.5px; color:var(--text-3); }
.metric-positive { color:#4ade80 !important; } .metric-warning { color:#fbbf24 !important; } .metric-negative { color:#f87171 !important; }
.metric-card.accent { border-color:var(--accent-line); background:linear-gradient(180deg,var(--accent-dim),var(--surface)); }
.metric-tag { font-size:10.5px; color:var(--text-3); margin-left:6px; font-weight:400; }
.group-title { font-size:12px; font-weight:600; color:var(--text-2); margin:2px 0 8px; display:flex; align-items:center; gap:6px; }
.group-title .dir { font-size:10.5px; padding:1px 7px; border-radius:999px; background:var(--surface-3); color:var(--text-3); font-weight:500; }
.fact-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.fact { font-size:12px; color:var(--text-2); background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:3px 10px; }
.fact b { color:var(--text); font-weight:550; }
.tok-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.tok { font-size:11.5px; color:var(--text-2); background:var(--surface-3); border-radius:6px; padding:2px 8px; } .tok b { color:var(--text); font-weight:550; }

/* ===== 9. Recommendation ===== */
.reco { background:linear-gradient(180deg,rgba(34,197,94,.09),var(--surface) 70%); border:1px solid var(--accent-line); border-radius:var(--r);
  padding:16px 18px; box-shadow:0 0 0 1px rgba(34,197,94,.05), 0 8px 28px rgba(34,197,94,.06); }
.reco-tag { display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:650; letter-spacing:.06em; color:#4ade80; text-transform:uppercase; }
.reco-tag svg { width:13px; height:13px; fill:none; stroke:currentColor; stroke-width:2; }
.reco-name { font-size:1.55rem; font-weight:650; color:var(--text); margin:6px 0 4px; letter-spacing:-.02em; }
.reco-why { font-size:13px; color:var(--text-2); max-width:820px; line-height:1.55; }
.reco-stats { display:flex; flex-wrap:wrap; gap:22px; margin-top:12px; padding-top:12px; border-top:1px solid var(--line); }
.reco-stats div { font-size:11.5px; color:var(--text-3); } .reco-stats b { display:block; font-size:1.05rem; color:var(--text); font-weight:650; margin-top:1px; }

/* ===== 10. Tables ===== */
.tbl-wrap { overflow:auto; max-height:360px; border:1px solid var(--line); border-radius:var(--r); background:var(--surface); }
table.dt { width:100%; border-collapse:separate; border-spacing:0; font-size:12px; font-variant-numeric:tabular-nums; }
table.dt th { position:sticky; top:0; z-index:1; background:var(--surface-3); color:var(--text-2); font-size:11px; font-weight:600; text-transform:uppercase;
  letter-spacing:.04em; text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
table.dt td { padding:6px 10px; color:var(--text); border-bottom:1px solid var(--line-soft); white-space:nowrap; }
table.dt tr:last-child td { border-bottom:0; }
table.dt tbody tr:hover td { background:var(--surface-2); }
table.dt, table.dt th, table.dt td { border-left:0 !important; border-right:0 !important; border-top:0 !important; }
table.dt th { border-bottom:1px solid var(--line) !important; } table.dt td { border-bottom:1px solid var(--line-soft) !important; background:transparent; }
.tbl-wrap table { margin:0 !important; }
.metric-grid.tight { grid-template-columns:repeat(auto-fill,minmax(118px,1fr)); }
table.dt td.num { text-align:right; } table.dt th.num { text-align:right; }
table.dt td.best { color:#4ade80; font-weight:650; background:var(--accent-dim); }
table.dt td.fast { color:var(--cyan); font-weight:600; }
table.dt tr.sel td { background:rgba(34,197,94,.06); }
.badge { display:inline-block; font-size:10.5px; font-weight:600; padding:1px 7px; border-radius:999px; margin-left:6px; vertical-align:1px; }
.badge.g { background:var(--accent-dim); color:#4ade80; } .badge.c { background:var(--cyan-dim); color:var(--cyan); }
.badge.i { background:var(--indigo-dim); color:#a5b4fc; } .badge.a { background:var(--warn-dim); color:#fbbf24; }
.gradio-container .table-wrap, .gradio-container .dataframe, .gradio-container [data-testid="dataframe"] { max-width:100% !important; overflow-x:auto !important; background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:var(--r-sm) !important; }
.gradio-container table { font-size:12px !important; }
.gradio-container table th, .gradio-container table td, .gradio-container .cell-wrap { font-size:12px !important; padding-top:5px !important; padding-bottom:5px !important; color:var(--text) !important; }
.gradio-container table thead th, .gradio-container thead { background:var(--surface-3) !important; color:var(--text-2) !important; font-weight:600 !important; }

/* ===== 11. Prediction console ===== */
.pred { background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:16px 18px; }
.pred-k { font-size:11px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--text-3); }
.pred-v { font-size:2rem; font-weight:650; letter-spacing:-.02em; margin:6px 0 2px; }
.pred-v.pos { color:#4ade80; } .pred-v.neg { color:#f87171; } .pred-v.neu { color:#fbbf24; } .pred-v.oth { color:#a5b4fc; }
.pred-c { font-size:13px; color:var(--text-2); }
.bar { height:6px; background:var(--surface-3); border-radius:999px; overflow:hidden; margin:10px 0 12px; }
.bar > i { display:block; height:100%; border-radius:999px; background:var(--accent); }
.bar.neg > i { background:#f87171; } .bar.neu > i { background:#fbbf24; } .bar.oth > i { background:var(--indigo); }
.probs { display:grid; gap:6px; }
.prob { display:grid; grid-template-columns:78px 1fr 48px; align-items:center; gap:10px; font-size:12px; color:var(--text-2); }
.prob .bar { margin:0; height:5px; } .prob span:last-child { text-align:right; font-variant-numeric:tabular-nums; color:var(--text); }
.pred-meta { font-size:11.5px; color:var(--text-3); margin-top:10px; }

/* ===== 12. Charts ===== */
.chart-card { width:100%; max-width:550px !important; min-width:0 !important; background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:var(--r) !important; padding:6px !important; overflow:hidden; }
.chart-card img, .chart-card canvas, .chart-card svg { max-width:100%; height:auto; border-radius:8px; }
.chart-card .plot-container { border:0 !important; background:transparent !important; }
.chart-card { min-height:150px; }
.chart-card svg.empty, .chart-card .empty svg, .chart-card [class*="unpadded_box"] svg { width:34px !important; height:34px !important; opacity:.35; }
.chart-card [class*="unpadded_box"], .chart-card .empty { min-height:150px !important; display:grid; place-items:center; }

/* ===== 13. Home ===== */
.hero { background:radial-gradient(700px 240px at 92% -10%, rgba(34,197,94,.13), transparent 70%), var(--surface) !important; border:1px solid var(--line) !important;
  border-radius:var(--r) !important; padding:20px 24px !important; }
.hero-title { font-size:clamp(1.7rem,3vw,2.15rem); font-weight:650; letter-spacing:-.025em; line-height:1.12; margin:0 0 10px; color:var(--text); max-width:720px; }
.hero-text { font-size:14.5px; line-height:1.6; color:var(--text-2); max-width:620px; margin:0 0 4px; }
.cta-row { gap:10px !important; margin-top:10px; flex-wrap:wrap !important; }
.cta-row > * { flex:0 0 auto !important; min-width:0 !important; } .cta-row button { min-width:180px; }
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.feature { background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:14px 16px; transition:border-color var(--ease), background var(--ease); }
.feature:hover { border-color:#3b4a60; background:var(--surface-2); }
.ico { width:30px; height:30px; border-radius:8px; background:var(--accent-dim); color:#4ade80; display:grid; place-items:center; margin-bottom:10px; }
.ico.alt { background:var(--indigo-dim); color:#a5b4fc; } .ico.cy { background:var(--cyan-dim); color:var(--cyan); }
.ico svg { width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.feature h3 { margin:0 0 3px; font-size:13.5px; font-weight:600; } .feature p { margin:0; color:var(--text-2); font-size:12.5px; line-height:1.5; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.step { display:flex; gap:12px; align-items:flex-start; background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:12px 14px; }
.step i { flex:0 0 24px; height:24px; border-radius:50%; display:grid; place-items:center; font-style:normal; font-size:12px; font-weight:650; background:var(--surface-3); color:#4ade80; border:1px solid var(--accent-line); }
.step h3 { margin:0 0 2px; font-size:13px; } .step p { margin:0; font-size:12px; color:var(--text-2); line-height:1.45; }

/* ===== 14. About ===== */
.about-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
.about-card { background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:14px 16px; }
.about-card h3 { margin:0 0 6px; font-size:13.5px; font-weight:600; color:var(--text); }
.about-card p, .about-card li { margin:0; font-size:12.5px; line-height:1.6; color:var(--text-2); } .about-card ul { margin:0; padding-left:16px; }
.about-card code { background:var(--surface-3); color:#a5b4fc; padding:1px 6px; border-radius:5px; font-size:11.5px; }

/* ===== 15. Footer ===== */
.app-footer { margin-top:22px; padding:16px 18px 12px; background:var(--bg-deep); border:1px solid var(--line); border-radius:var(--r); }
.footer-grid { display:grid; grid-template-columns:1.5fr repeat(4,1fr); gap:18px; align-items:start; }
.footer-brand { display:flex; align-items:center; gap:10px; color:var(--text); font-weight:600; font-size:13.5px; }
.footer-brand .logo { width:28px; height:28px; font-size:13px; border-radius:8px; }
.app-footer p, .app-footer li { font-size:12px; color:var(--text-3); line-height:1.7; margin:0; } .app-footer ul { list-style:none; padding:0; margin:0; }
.app-footer h4 { font-size:11.5px; font-weight:600; color:var(--text-2); margin:0 0 6px; }
.footer-bottom { margin-top:12px; padding-top:10px; border-top:1px solid var(--line-soft); font-size:11.5px; color:var(--text-3); display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }

/* ===== 16. Responsive ===== */
@media (max-width:1100px) { .metric-grid.cols-4 { grid-template-columns:repeat(2,1fr); } .footer-grid { grid-template-columns:1fr 1fr; } }
@media (max-width:900px) {
  .gradio-container { width:calc(100vw - 40px) !important; }
  .feature-grid, .steps, .about-grid { grid-template-columns:1fr; }
  .gradio-container .row { flex-wrap:wrap !important; }
}
@media (max-width:720px) {
  .gradio-container { width:calc(100vw - 28px) !important; padding:8px 0 20px !important; }
  .app-header { flex-wrap:wrap; } .cta-row button { min-width:100%; }
  .gradio-container button[role="tab"] { padding:0 11px !important; font-size:12.5px !important; }
  .footer-grid { grid-template-columns:1fr; } .prob { grid-template-columns:64px 1fr 42px; }
}
@media (max-width:480px) { .gradio-container { width:calc(100vw - 20px) !important; } }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; animation:none !important; } }
"""


def run(text, model):
    if not MODELS:
        return {}, "Models not installed", "Add trained files under model/ to enable prediction.", pd.DataFrame(columns=COLUMNS)
    if not text or not text.strip():
        return {}, "", "", pd.DataFrame(columns=COLUMNS)

    rows = sentiment.compare(text)
    table = pd.DataFrame(
        [
            [r["model"], r["label"], r["confidence"]]
            + [r["probs"][c] for c in CLASSES]
            + [r["ms"]]
            for r in rows
        ],
        columns=COLUMNS,
    )

    chosen = next((r for r in rows if r["model"] == model), rows[0])
    if chosen.get("error"):
        return {}, chosen["script"], chosen["text_used"], table

    return chosen["probs"], chosen["script"], chosen["text_used"], table


examples = [
    ["kimat men ye phone achha hai", MODELS[0] if MODELS else None],
    ["camera bahut kharab hai", MODELS[0] if MODELS else None],
    ["kimmat mein ye fon acha hai", MODELS[0] if MODELS else None],
]


# ---------------------------------------------------------------------
# Dataset analyzer
# ---------------------------------------------------------------------

def get_columns(file):
    if not file:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(file)
        text_col = detect_text_column(df)
        label_col = detect_label_column(df, text_col)

        return (
            gr.update(choices=list(df.columns), value=text_col),
            gr.update(choices=["(None)"] + list(df.columns), value=label_col or "(None)"),
        )
    except Exception:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)


def _profile_rows(profile):
    rows = [
        ["Documents", profile["documents"]],
        ["Average characters", profile["avg_characters"]],
        ["Average tokens", profile["avg_tokens"]],
        ["Vocabulary size", profile["vocabulary_size"]],
        ["Unique token ratio (%)", profile["unique_token_ratio"]],
        ["Duplicate documents (%)", profile["duplicate_percentage"]],
        ["Emoji count", profile["emoji_count"]],
        ["Punctuation density (%)", profile["punctuation_density"]],
        ["Primary language", profile["primary_language"]],
        ["Code-mixed (%)", profile["code_mixed_percentage"]],
        ["Romanized (%)", profile["romanized_percentage"]],
    ]
    return pd.DataFrame(rows, columns=["NLP Metric", "Value"])


def _distribution_df(d, name):
    return pd.DataFrame(
        [{name: k, "Percentage": v} for k, v in d.items()]
    )


def _robustness_results(df, text_column, label_column):
    """Measure text restoration and sentiment robustness on generated pairs."""
    from model_recommender import _normalize_label
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    source = df[text_column].fillna("").astype(str).tolist()
    # Hindi rows use the project transliterator; Roman rows are already the clean
    # Hinglish reference. These are controlled synthetic corruptions, not human refs.
    refs = [to_hinglish(x) if any("\u0900" <= c <= "\u097f" for c in x) else x for x in source]
    if len(source) > 500:
        # Keep the interactive robustness analysis responsive and deterministic.
        positions = [i * (len(source) - 1) // 499 for i in range(500)]
        refs = [refs[i] for i in positions]
        source = [source[i] for i in positions]
        labels = [_normalize_label(df[label_column].iloc[i]) for i in positions] if label_column else []
    elif label_column:
        labels = [_normalize_label(x) for x in df[label_column].tolist()]
    else:
        labels = []
    model_names = sentiment.available_models()
    metric_rows, class_rows = [], []
    conditions = [("Clean Hinglish", refs)]
    metric_rows.append({"Condition": "Clean Hinglish", "Noise level": "None",
                        **calculate_text_metrics(refs, refs)})
    for level_name, probability in NOISE_LEVELS.items():
        noisy = [add_noise(text, probability, seed=41) for text in refs]
        conditions.extend([(f"{level_name} noise", noisy),
                           (f"{level_name} normalized", [sentiment.prepare(t)[0] for t in noisy])])
        for cond, candidates in conditions[-2:]:
            metric_rows.append({"Condition": cond, "Noise level": level_name,
                                **calculate_text_metrics(refs, candidates)})

    if label_column and model_names:
        for cond, texts in conditions:
            for model in model_names:
                preds, times = [], []
                for text in texts:
                    import time
                    started = time.perf_counter()
                    probabilities = sentiment._get(model)(text)
                    times.append((time.perf_counter() - started) * 1000)
                    preds.append(_normalize_label(max(probabilities, key=probabilities.get)))
                class_rows.append({
                    "Condition": cond, "Model": model,
                    "Accuracy": accuracy_score(labels, preds),
                    "Precision (macro)": precision_score(labels, preds, average="macro", zero_division=0),
                    "Recall (macro)": recall_score(labels, preds, average="macro", zero_division=0),
                    "Macro F1": f1_score(labels, preds, average="macro", zero_division=0),
                    "Weighted F1": f1_score(labels, preds, average="weighted", zero_division=0),
                    "Avg Time (ms)": sum(times) / max(1, len(times)),
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(class_rows)


def _report_file(summary, profile, benchmark, ranking, text_scores, robustness_scores, figure):
    """Build a self-contained temporary HTML report for the browser download."""
    import base64
    from io import BytesIO
    image = BytesIO()
    figure.savefig(image, format="png", bbox_inches="tight", dpi=130)
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    sections = [f"<h1>Uploaded dataset analysis</h1><p>{html.escape(summary)}</p>"]
    for title, table in [("Dataset profile", profile), ("Model benchmark", benchmark),
                         ("Model ranking", ranking), ("Text transformation metrics", text_scores),
                         ("Robustness classification metrics", robustness_scores)]:
        if table is not None and not table.empty:
            sections.append(f"<h2>{title}</h2>{table.to_html(index=False, border=0, float_format=lambda x: f'{x:.4f}')}")
    sections.append(f'<h2>Interactive analysis plot</h2><img style="max-width:100%" src="data:image/png;base64,{encoded}">')
    page = "<!doctype html><meta charset='utf-8'><title>Dataset analysis report</title><style>body{font:15px Arial;max-width:1200px;margin:32px auto;color:#222}table{border-collapse:collapse;width:100%;margin-bottom:24px}th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#eee}h2{margin-top:32px}</style>" + "".join(sections)
    fd, path = tempfile.mkstemp(prefix="nlp_dataset_report_", suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(page)
    return path


def analyze_uploaded_dataset(file, text_column, label_column, priority):
    empty = pd.DataFrame()

    if not file:
        return (
            "Upload a CSV dataset.",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            "No recommendation yet.", "", empty, empty, None, None,
        )

    try:
        df = pd.read_csv(file)

        if not text_column or text_column not in df.columns:
            text_column = detect_text_column(df)

        if label_column == "(None)":
            label_column = None
        elif label_column not in df.columns:
            label_column = detect_label_column(df, text_column)

        if label_column and not MODELS:
            result = {
                "profile": analyze_dataset(df, text_column), "task": detect_task(df, label_column),
                "text_column": text_column, "label_column": label_column,
                "benchmark": None, "ranking": None, "recommendation": "Models unavailable",
                "reason": "Dataset profiling and text metrics are available. Add trained model files under model/ for classification scores and recommendations.",
            }
        else:
            result = recommend_from_dataset(
                df, text_column=text_column, label_column=label_column, priority=priority,
            )

        profile = result["profile"]
        task = result["task"]
        recommendation = result["recommendation"]
        reason = result["reason"] or ""

        summary = (
            f"**Dataset:** {len(df):,} rows  |  "
            f"**Task:** {task}  |  "
            f"**Text column:** `{result['text_column']}`  |  "
            f"**Label column:** `{result['label_column'] or 'None'}`  |  "
            f"**Primary language:** {profile['primary_language']}"
        )

        if result["benchmark"] is not None:
            benchmark = result["benchmark"].copy()
            ranking = result["ranking"].copy()

            ranking_display = ranking[
                [
                    "Rank", "Model", "Accuracy", "Precision", "Recall",
                    "Macro F1", "Weighted F1", "Avg Time (ms)",
                    "Recommendation Score"
                ]
            ]

            lang_df = _distribution_df(profile["language_distribution"], "Language")
            script_df = _distribution_df(profile["script_distribution"], "Script")

            text_scores, robustness_scores = _robustness_results(df, result["text_column"], result["label_column"])
            figure, ax = plt.subplots(figsize=(10, 4))
            if not robustness_scores.empty:
                pivot = robustness_scores.pivot(index="Condition", columns="Model", values="Macro F1")
                pivot.plot(kind="bar", ax=ax, ylim=(0, 1), title="Macro F1 by model and robustness condition")
                ax.set_ylabel("Macro F1")
                ax.tick_params(axis="x", rotation=25)
                figure.tight_layout()
            elif not text_scores.empty:
                text_scores.set_index("Condition")[["bleu", "rougeL", "chrf"]].plot(kind="bar", ax=ax, title="Text overlap under spelling noise")
                ax.set_ylabel("Score (0–100)")
                ax.tick_params(axis="x", rotation=25)
                figure.tight_layout()
            report_path = _report_file(summary, _profile_rows(profile), benchmark, ranking_display, text_scores, robustness_scores, figure)
            return (
                summary,
                _profile_rows(profile),
                lang_df,
                script_df,
                benchmark,
                ranking_display,
                lang_df,
                script_df,
                f"### Recommended model: {recommendation}\n\n{reason}",
                "\n".join(
                    f"- **{x[0]}**: {x[1]}"
                    for x in profile["top_tokens"]
                ), text_scores, robustness_scores, figure, report_path,
            )

        lang_df = _distribution_df(profile["language_distribution"], "Language")
        script_df = _distribution_df(profile["script_distribution"], "Script")

        text_scores = empty
        robustness_scores = empty
        text_scores, robustness_scores = _robustness_results(
            df, result["text_column"], result["label_column"]
        )
        figure, ax = plt.subplots(figsize=(8, 3))
        if not text_scores.empty:
            text_scores.set_index("Condition")[["bleu", "rougeL", "chrf"]].plot(kind="bar", ax=ax, title="Text overlap under spelling noise")
            ax.set_ylabel("Score (0–100)")
            ax.tick_params(axis="x", rotation=25)
            figure.tight_layout()
        else:
            ax.text(.5, .5, "Classification model files required for model robustness scores", ha="center", va="center")
            ax.axis("off")
        report_path = _report_file(summary, _profile_rows(profile), empty, empty, text_scores, robustness_scores, figure)
        return (
            summary,
            _profile_rows(profile),
            lang_df,
            script_df,
            empty,
            empty,
            lang_df,
            script_df,
            f"### Recommendation: {recommendation}\n\n{reason}",
            "\n".join(f"- **{x[0]}**: {x[1]}" for x in profile["top_tokens"]),
            text_scores, robustness_scores, figure, report_path,
        )

    except Exception as e:
        return (
            f"Error: {e}",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            "Analysis failed. Check the CSV and selected columns.",
            "", empty, empty, None, None,
        )


# ---------------------------------------------------------------------
# Dashboard helper (ADDITIVE, display only).
# It only reads the robustness table that analyze_uploaded_dataset already
# produced and reshapes it for the per-model charts. Nothing above calls it.
# ---------------------------------------------------------------------

def model_dashboard(model, scores):
    try:
        if (
            scores is None
            or not isinstance(scores, pd.DataFrame)
            or scores.empty
            or "Model" not in scores.columns
            or not model
        ):
            return None, None
        sub = scores[scores["Model"] == model]
        if sub.empty:
            return None, None
        metric_cols = [c for c in ["Accuracy", "Macro F1", "Weighted F1"] if c in sub.columns]
        quality = sub.melt(
            id_vars=["Condition"], value_vars=metric_cols,
            var_name="Metric", value_name="Score",
        )
        speed = sub[["Condition", "Avg Time (ms)"]].copy() if "Avg Time (ms)" in sub.columns else None
        return quality, speed
    except Exception:
        return None, None


# ---------------------------------------------------------------------
# PDF report (ADDITIVE, display/export only).
# Reads the tables/markdown the analysis already put on screen and writes
# a professional PDF. Uses reportlab when installed, otherwise falls back
# to matplotlib (already a dependency) so it always works.
# ---------------------------------------------------------------------
_PDF_COLORS = ["#0f766e", "#4f46e5", "#f59e0b", "#ec4899", "#64748b"]


def _has(d):
    return isinstance(d, pd.DataFrame) and not d.empty


def _plain(s):
    import re
    return re.sub(r"[*`#]", "", s or "").strip()


def _latin(s):
    """Built-in PDF fonts are Latin-only; drop anything they cannot draw."""
    return "".join(ch if ord(ch) < 256 else "?" for ch in str(s))


def _pdf_figures(lang, script, robustness):
    from matplotlib.figure import Figure
    figs = []

    def style(ax, title):
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color="#0f172a")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)

    panels = [(d, c, t) for d, c, t in ((lang, "Language", "Language distribution (%)"),
                                         (script, "Script", "Script distribution (%)"))
              if _has(d) and c in d.columns and "Percentage" in d.columns]
    if panels:
        fig = Figure(figsize=(10, 3.1))
        for i, (d, col, title) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1)
            ax.bar(d[col].astype(str), pd.to_numeric(d["Percentage"], errors="coerce"), color=_PDF_COLORS[i])
            style(ax, title)
        fig.tight_layout()
        figs.append(fig)

    if _has(robustness) and {"Condition", "Model"} <= set(robustness.columns):
        rb = robustness.copy()
        order = rb["Condition"].drop_duplicates().tolist()
        for metric, title, ylim in (("Macro F1", "Macro F1 by model and input condition", (0, 1)),
                                    ("Avg Time (ms)", "Average inference time (ms)", None)):
            if metric not in rb.columns:
                continue
            rb[metric] = pd.to_numeric(rb[metric], errors="coerce")
            pv = rb.pivot_table(index="Condition", columns="Model", values=metric, sort=False).reindex(order)
            fig = Figure(figsize=(10, 3.4))
            ax = fig.add_subplot(111)
            pv.plot(kind="bar", ax=ax, color=_PDF_COLORS[: len(pv.columns)], width=.78, rot=20)
            style(ax, title)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xlabel("")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            figs.append(fig)
    return figs


def build_pdf_report(summary, rec, tokens, profile, lang, script, benchmark, ranking, text_scores, robustness):
    import datetime
    if not summary or not _has(profile):
        return None
    try:
        figs = _pdf_figures(lang, script, robustness)
        fd, path = tempfile.mkstemp(prefix="nlp_dataset_report_", suffix=".pdf")
        os.close(fd)
        try:
            _pdf_reportlab(path, summary, rec, tokens, profile, benchmark, ranking, text_scores, robustness, figs)
        except ImportError:
            _pdf_matplotlib(path, summary, rec, profile, figs)
        return path
    except Exception:
        return None


def _pdf_reportlab(path, summary, rec, tokens, profile, benchmark, ranking, text_scores, robustness, figs):
    import datetime
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, KeepTogether)

    BRAND, INK, MUTED, LINE = colors.HexColor("#0f766e"), colors.HexColor("#0f172a"), colors.HexColor("#64748b"), colors.HexColor("#e2e8f0")
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, textColor=INK, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"))
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.6, leading=9.5)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=INK)
    page_w, page_h = landscape(A4)
    margin = 15 * mm
    width = page_w - 2 * margin
    stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(margin, 11 * mm, page_w - margin, 11 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 7 * mm, "Hindi / Hinglish NLP Studio  |  Dataset analysis report  |  " + stamp)
        canvas.drawRightString(page_w - margin, 7 * mm, "Page %d" % doc.page)
        canvas.restoreState()

    def fmt(v):
        if isinstance(v, float):
            return "%.3f" % v
        return html.escape(_latin(v))

    def df_table(df, max_rows=40):
        df = df.head(max_rows)
        data = [[Paragraph(html.escape(_latin(c)), head) for c in df.columns]]
        data += [[Paragraph(fmt(v), cell) for v in row] for row in df.itertuples(index=False)]
        t = Table(data, colWidths=[width / len(df.columns)] * len(df.columns), repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("LINEBELOW", (0, 0), (-1, -1), .4, LINE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    story = []
    title = Table([[Paragraph('<font color="white" size="19"><b>Dataset analysis report</b></font><br/>'
                              '<font color="#cdeae6" size="9.5">Hindi / Hinglish NLP Studio</font>', body),
                    Paragraph('<font color="#cdeae6" size="9">%s</font>' % stamp,
                              ParagraphStyle("r", parent=body, alignment=2))]],
                  colWidths=[width * .75, width * .25])
    title.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRAND), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                               ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    story += [title, Spacer(1, 10)]

    facts = [p.strip() for p in _plain(summary).split("|") if ":" in p]
    if facts:
        cells = [[Paragraph("<b>%s</b><br/><font color='#64748b'>%s</font>" % (
            html.escape(_latin(v.strip())), html.escape(_latin(k.strip()))), body)]
            for k, v in (f.split(":", 1) for f in facts)]
        row = Table([[c[0] for c in cells]], colWidths=[width / len(cells)] * len(cells))
        row.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .6, LINE), ("INNERGRID", (0, 0), (-1, -1), .6, LINE),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [row, Spacer(1, 6)]

    if rec:
        lines = [l for l in _plain(rec).splitlines() if l.strip()]
        box = Table([[Paragraph("<b>%s</b>" % html.escape(_latin(lines[0])), ParagraphStyle("rb", parent=body, fontSize=12, textColor=colors.HexColor("#115e59"))) if lines else ""],
                     [Paragraph(html.escape(_latin(" ".join(lines[1:]))), body)]], colWidths=[width])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e6f4f2")),
                                 ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#b7e0da")),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [Spacer(1, 4), box]

    def section(name, df):
        if _has(df):
            story.append(KeepTogether([Paragraph(name, h2), df_table(df)]))

    section("Dataset profile", profile)
    if tokens:
        toks = [l.strip("- ").strip() for l in _plain(tokens).splitlines() if l.strip() and all(ord(c) < 256 for c in l)]
        if toks:
            story += [Paragraph("Top tokens", h2), Paragraph(html.escape("   |   ".join(toks[:15])), body)]
    section("Model ranking", ranking)
    section("Model benchmark", benchmark)

    if figs:
        story.append(Paragraph("Dashboard", h2))
        for fig in figs:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
            buf.seek(0)
            iw, ih = ImageReader(buf).getSize()
            buf.seek(0)
            w = min(width, 235 * mm)
            story += [Image(buf, width=w, height=w * ih / iw), Spacer(1, 6)]

    section("Text transformation metrics", text_scores)
    section("Robustness classification metrics", robustness)
    story += [Spacer(1, 10), Paragraph(
        "Robustness scores use deterministic synthetic spelling corruptions on at most 500 rows and a "
        "generated Hinglish reference. They are controlled benchmarks, not human judgments.",
        ParagraphStyle("note", parent=body, fontSize=8, textColor=MUTED))]

    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=16 * mm,
                      title="Dataset analysis report", author="Hindi / Hinglish NLP Studio").build(
        story, onFirstPage=on_page, onLaterPages=on_page)


def _pdf_matplotlib(path, summary, rec, profile, figs):
    """Fallback when reportlab is not installed: text page + charts."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    with PdfPages(path) as pdf:
        page = Figure(figsize=(11.69, 8.27))
        page.text(.06, .93, "Dataset analysis report", fontsize=20, fontweight="bold", color="#0f766e")
        y = .87
        for line in [_plain(summary).replace("|", "\n")] + [_plain(rec)] + \
                    ["%s: %s" % (r[0], r[1]) for r in profile.itertuples(index=False)]:
            for sub in str(line).splitlines():
                page.text(.06, y, _latin(sub)[:150], fontsize=10, color="#334155")
                y -= .033
        pdf.savefig(page)
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


# ---------------------------------------------------------------------
# Display layer (ADDITIVE, presentation only).
# Everything below reads values the analysis already produced and turns
# them into compact HTML cards / dark charts. No metric is recomputed
# except trivial differences between existing scores (e.g. F1 drop).
# ---------------------------------------------------------------------
import re as _re

_P = dict(bg="#111722", grid="#263142", text="#94a3b8", ink="#e2e8f0", green="#22c55e",
          indigo="#6366f1", cyan="#22d3ee", amber="#f59e0b", rose="#f43f5e")
_SERIES = [_P["green"], _P["indigo"], _P["cyan"], _P["amber"], _P["rose"]]


def _esc(x):
    return html.escape(str(x))


def _flt(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _f(v, d=3):
    x = _flt(v)
    return _esc(v) if x is None else f"{x:.{d}f}"


def _kpi(value, label, delta="", tone="", accent=False, small=False):
    d = f'<div class="metric-delta metric-{tone}">{delta}</div>' if delta else ""
    return (f'<div class="metric-card{" accent" if accent else ""}"><div class="metric-label">{label}</div>'
            f'<div class="metric-value{" sm" if small else ""}">{value}</div>{d}</div>')


def _grid(cards, cls=""):
    return f'<div class="metric-grid {cls}">{"".join(cards)}</div>'


def _head(title, sub="", first=False):
    return (f'<div class="sec-head{" first" if first else ""}"><h2>{title}</h2>'
            + (f"<p>{sub}</p>" if sub else "") + "</div>")


def _empty(msg):
    return f'<div class="empty">{msg}</div>'


def _short(c):
    c = str(c)
    return c.replace(" Hinglish", "").replace(" normalized", "+norm").replace(" noise", "")


def _ncol(df, *names):
    """Find a column by loose name (case/punctuation-insensitive)."""
    norm = {_re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for n in names:
        k = _re.sub(r"[^a-z0-9]", "", n.lower())
        if k in norm:
            return norm[k]
    return None


# ----------------------------- charts -----------------------------
def _dark_fig(w=5.6, h=2.9):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), facecolor=_P["bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(_P["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_P["grid"])
    ax.tick_params(colors=_P["text"], labelsize=8, length=0)
    ax.grid(axis="y", color=_P["grid"], linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=10, fontweight="semibold", color=_P["ink"], pad=10)
    if sub:
        ax.text(1, 1.04, sub, transform=ax.transAxes, ha="right", fontsize=7.5, color=_P["text"])


def _legend(ax, n):
    if n > 1:
        ax.legend(fontsize=7.5, frameon=False, labelcolor=_P["text"], loc="best")


def _finish(fig):
    fig.tight_layout(pad=1.0)
    return fig


def _hbar(df, col, title):
    if not _has(df) or col not in df.columns or "Percentage" not in df.columns:
        return None
    d = df.assign(Percentage=pd.to_numeric(df["Percentage"], errors="coerce")).dropna().sort_values("Percentage")
    fig, ax = _dark_fig(5.6, 2.9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=_P["grid"], linewidth=.6, alpha=.7)
    cols = [_P["green"] if i == len(d) - 1 else _P["indigo"] for i in range(len(d))]
    ax.barh(d[col].astype(str), d["Percentage"], color=cols, height=.62)
    for y, v in enumerate(d["Percentage"]):
        ax.text(v + max(d["Percentage"].max() * .015, .4), y, f"{v:.1f}%", va="center", fontsize=8, color=_P["ink"])
    ax.set_xlim(0, d["Percentage"].max() * 1.15 if d["Percentage"].max() > 0 else 1)
    _title(ax, title)
    return _finish(fig)


def _line_plot(pivot, title, ylim=None, sub=None, dashed=()):
    if pivot is None or pivot.empty:
        return None
    fig, ax = _dark_fig()
    x = list(range(len(pivot.index)))
    for i, c in enumerate(pivot.columns):
        ax.plot(x, pivot[c].values, marker="o", ms=4, lw=1.8, color=_SERIES[i % len(_SERIES)],
                ls="--" if c in dashed else "-", label=str(c))
    ax.set_xticks(x)
    ax.set_xticklabels([_short(c) for c in pivot.index], rotation=22, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    _title(ax, title, sub)
    fig.tight_layout(pad=1.0)
    if len(pivot.columns) > 1:
        ax.legend(fontsize=7.5, frameon=False, labelcolor=_P["text"], loc="center left", bbox_to_anchor=(1.01, .5))
        fig.subplots_adjust(right=.80)
    return fig


def _cond_bars(conds, values, title, unit="", ylim=None, fmt="{:.2f}"):
    if fmt == "{:.1f}" and values and max(values) < 1:
        fmt = "{:.3f}"
    if not len(values):
        return None
    fig, ax = _dark_fig()
    colors = [_P["green"] if "Clean" in str(c) else (_P["cyan"] if "normalized" in str(c) else _P["amber"]) for c in conds]
    ax.bar(range(len(values)), values, color=colors, width=.62)
    for i, v in enumerate(values):
        ax.text(i, v + (ax.get_ylim()[1] * .015), fmt.format(v), ha="center", fontsize=7.5, color=_P["ink"])
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([_short(c) for c in conds], rotation=22, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    _title(ax, title, unit)
    return _finish(fig)


def _model_compare(df_clean, selected):
    if df_clean is None or df_clean.empty:
        return None
    fig, ax = _dark_fig()
    models = df_clean["Model"].astype(str).tolist()
    w = .36
    for j, (col, color) in enumerate((("Macro F1", _P["green"]), ("Accuracy", _P["indigo"]))):
        if col not in df_clean.columns:
            continue
        vals = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).tolist()
        bars = ax.bar([i + (j - .5) * w for i in range(len(models))], vals, width=w, color=color, label=col)
        for b, m in zip(bars, models):
            if m != str(selected):
                b.set_alpha(.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.2f}", ha="center", fontsize=7, color=_P["ink"])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    _title(ax, "Model comparison (clean input)", "selected model highlighted")
    _legend(ax, 2)
    return _finish(fig)


# ----------------------------- text-quality helpers -----------------------------
_TM_DEFS = [("BLEU", ["bleu"], 1), ("BLEU-1", ["bleu1"], 1), ("BLEU-2", ["bleu2"], 1), ("BLEU-3", ["bleu3"], 1),
            ("BLEU-4", ["bleu4"], 1), ("ROUGE-1", ["rouge1"], 1), ("ROUGE-2", ["rouge2"], 1),
            ("ROUGE-L", ["rougeL", "rougel"], 1), ("METEOR", ["meteor"], 1), ("chrF", ["chrf"], 1), ("TER", ["ter"], -1)]


def _tm_metrics(ts):
    out = []
    for label, keys, direction in _TM_DEFS:
        c = _ncol(ts, *keys)
        if c:
            out.append((label, c, direction))
    return out


def _text_plot(ts):
    if not _has(ts) or "Condition" not in ts.columns:
        return None
    ts = ts.copy()
    conds = ts["Condition"].drop_duplicates().tolist()
    wanted = [m for m in _tm_metrics(ts) if m[0] in ("BLEU", "ROUGE-L", "METEOR", "chrF", "TER")]
    if not wanted:
        return None
    data = {}
    for label, col, _d in wanted:
        s = pd.to_numeric(ts[col], errors="coerce")
        data[label] = s.groupby(ts["Condition"], sort=False).first().reindex(conds)
    pv = pd.DataFrame(data)
    return _line_plot(pv, "Text quality by noise level", (0, 105), "TER: lower is better", dashed=("TER",))


def _text_cards(ts):
    if not _has(ts) or "Condition" not in ts.columns:
        return _empty("Text transformation metrics appear after analysis.")
    mets = _tm_metrics(ts)
    if not mets:
        return _empty("No text metrics available.")
    conds = ts["Condition"].astype(str).tolist()
    noisy = [c for c in conds if c.endswith(" noise")]
    ref = noisy[-1] if noisy else conds[-1]
    norm = ref.replace(" noise", " normalized")
    row = ts[ts["Condition"].astype(str) == ref].iloc[0]
    nrow = ts[ts["Condition"].astype(str) == norm]
    nrow = nrow.iloc[0] if not nrow.empty else None

    def card(label, col, direction):
        v = _flt(row[col])
        delta, tone = "", ""
        if nrow is not None and v is not None and _flt(nrow[col]) is not None:
            nv = _flt(nrow[col])
            diff = (nv - v) * direction
            tone = "positive" if diff > 0.005 else ("negative" if diff < -0.005 else "")
            delta = f"{nv:.1f} normalized"
        return _kpi("n/a" if v is None else f"{v:.1f}", label, delta, tone, small=False)

    up = [card(*m) for m in mets if m[2] == 1]
    dn = [card(*m) for m in mets if m[2] == -1]
    html_ = (f'<div class="group-title">Text quality <span class="dir">higher is better</span></div>' + _grid(up, 'tight')
             + (f'<div class="group-title" style="margin-top:12px">Edit distance <span class="dir">lower is better</span></div>' + _grid(dn, 'tight') if dn else "")
             + f'<div class="metric-delta" style="margin-top:8px">Scores 0-100 at <b>{_esc(ref)}</b>; the second line is the score after normalization.</div>')
    return html_


# ----------------------------- table helper -----------------------------
def _table(df, num_cols=(), best_max=(), best_min=(), sel_col=None, sel_val=None, badges=None, limit=80):
    d = df.head(limit)
    best = {}
    for c in best_max:
        s = pd.to_numeric(d[c], errors="coerce")
        if s.notna().any():
            best[c] = s.idxmax()
    for c in best_min:
        s = pd.to_numeric(d[c], errors="coerce")
        if s.notna().any():
            best[c] = s.idxmin()
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{_esc(c)}</th>' for c in d.columns)
    rows = []
    for idx, r in d.iterrows():
        sel = ' class="sel"' if sel_col and str(r[sel_col]) == str(sel_val) else ""
        tds = []
        for c in d.columns:
            cls = ["num"] if c in num_cols else []
            if c in best_max and best.get(c) == idx:
                cls.append("best")
            if c in best_min and best.get(c) == idx:
                cls.append("fast")
            val = _f(r[c], 3) if c in num_cols else _esc(r[c])
            if badges and (c, idx) in badges:
                val += badges[(c, idx)]
            tds.append(f'<td class="{" ".join(cls)}">{val}</td>')
        rows.append(f"<tr{sel}>{''.join(tds)}</tr>")
    return f'<div class="tbl-wrap"><table class="dt"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ----------------------------- dataset-page renderer -----------------------------
_STAR = '<svg viewBox="0 0 24 24"><path d="M12 3l2.4 5.6L20 9.5l-4.3 3.9L17 19l-5-3-5 3 1.3-5.6L4 9.5l5.6-.9z"/></svg>'


def _reco_html(rec, ranking):
    lines = [l for l in _plain(rec).splitlines() if l.strip()]
    if not lines:
        return _empty("The model recommendation appears here after analysis.")
    name = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0]
    why = " ".join(lines[1:])
    stats = ""
    if _has(ranking) and "Model" in ranking.columns:
        row = ranking[ranking["Model"].astype(str) == name]
        if not row.empty:
            r = row.iloc[0]
            items = [(k, _f(r[k], 3) if k != "Avg Time (ms)" else _f(r[k], 1) + " ms") for k in
                     ("Macro F1", "Accuracy", "Weighted F1", "Avg Time (ms)") if k in row.columns]
            stats = '<div class="reco-stats">' + "".join(f"<div>{_esc(k.replace('Avg Time (ms)', 'Latency'))}<b>{v}</b></div>" for k, v in items) + "</div>"
    return (f'<div class="reco"><div class="reco-tag">{_STAR} Recommended model</div><div class="reco-name">{_esc(name)}</div>'
            f'<div class="reco-why">{_esc(why)}</div>{stats}</div>'), name


def _overview_html(summary, profile, tokens):
    p = dict(zip(profile.iloc[:, 0].astype(str), profile.iloc[:, 1]))

    def num(k, pct=False, d=1):
        x = _flt(p.get(k))
        if x is None:
            return _esc(p.get(k, "-"))
        return (f"{int(x):,}" if float(x).is_integer() and not pct else f"{x:,.{d}f}") + ("%" if pct else "")

    cards = [_kpi(num("Documents"), "Documents", accent=True),
             _kpi(_esc(p.get("Primary language", "-")), "Primary language", small=True),
             _kpi(num("Romanized (%)", True), "Romanized"),
             _kpi(num("Code-mixed (%)", True), "Code-mixed"),
             _kpi(num("Vocabulary size"), "Vocabulary"),
             _kpi(num("Average tokens"), "Avg tokens / doc"),
             _kpi(num("Unique token ratio (%)", True), "Unique tokens"),
             _kpi(num("Duplicate documents (%)", True), "Duplicates")]
    facts = [x.strip() for x in _plain(summary).split("|") if ":" in x]
    chips = "".join(f'<span class="fact">{_esc(k.strip())}: <b>{_esc(v.strip())}</b></span>'
                    for k, v in (f.split(":", 1) for f in facts) if k.strip() in ("Task", "Text column", "Label column"))
    toks = [l.strip("- ").strip() for l in _plain(tokens).splitlines() if l.strip()]
    tok_html = ('<div class="tok-row">' + "".join(f'<span class="tok">{_esc(t)}</span>' for t in toks[:12]) + "</div>") if toks else ""
    return f'<div class="fact-row">{chips}</div>' + _grid(cards) + tok_html


def _robust_kpis(robustness):
    if not _has(robustness) or not {"Condition", "Model", "Macro F1"} <= set(robustness.columns):
        return _empty("Robustness scores need a labeled dataset and installed models.")
    rb = robustness.copy()
    rb["Macro F1"] = pd.to_numeric(rb["Macro F1"], errors="coerce")
    conds = rb["Condition"].drop_duplicates().astype(str).tolist()
    clean = conds[0]
    best = rb[rb["Condition"].astype(str) == clean].sort_values("Macro F1", ascending=False).iloc[0]["Model"]
    sub = rb[rb["Model"] == best].set_index(rb[rb["Model"] == best]["Condition"].astype(str))["Macro F1"]
    noisy = [c for c in conds if c.endswith(" noise")]
    cards = [_kpi(_f(sub.get(clean), 3), "Clean Macro F1", _esc(best), accent=True)]
    if noisy:
        hi = noisy[-1]
        hn = hi.replace(" noise", " normalized")
        drop = sub.get(hi, float("nan")) - sub.get(clean, float("nan"))
        cards.append(_kpi(f"{drop:+.3f}", "F1 drop", f"at {_esc(_short(hi))} noise", "negative" if drop < 0 else "positive"))
        if hn in sub.index:
            rec_ = sub[hn] - sub[hi]
            cards.append(_kpi(f"{rec_:+.3f}", "F1 recovery", "from normalization", "positive" if rec_ > 0 else "warning"))
            cards.append(_kpi(f"{sub[hn] - sub[clean]:+.3f}", "Residual gap", "normalized vs clean", "warning"))
    return _grid(cards, "cols-4")


def render_results(summary, rec, tokens, profile, lang, script, ranking, text_scores, robustness):
    blank = (_empty("Upload a CSV and run the analysis to see dataset KPIs."), _empty("The model recommendation appears here after analysis."),
             None, None, _empty("Model benchmark appears for labeled datasets."), _empty("Text transformation metrics appear after analysis."),
             None, _empty("Robustness scores appear after analysis."), None, _empty(""))
    if not _has(profile):
        msg = _plain(summary)
        return (_empty(_esc(msg)) if msg else blank[0],) + blank[1:]
    try:
        overview = _overview_html(summary, profile, tokens)
        reco = _reco_html(rec, ranking)
        reco_html, reco_name = reco if isinstance(reco, tuple) else (reco, "")
        lang_fig = _hbar(lang, "Language", "Language distribution")
        script_fig = _hbar(script, "Script", "Script distribution")

        if _has(ranking):
            show = [c for c in ("Rank", "Model", "Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1", "Avg Time (ms)", "Recommendation Score") if c in ranking.columns]
            rk = ranking[show].rename(columns={"Avg Time (ms)": "Latency (ms)"})
            nums = [c for c in rk.columns if c not in ("Rank", "Model")]
            badges = {("Model", i): '<span class="badge g">Recommended</span>' for i, m in rk["Model"].astype(str).items() if m == reco_name}
            bench = _table(rk, nums, best_max=[c for c in ("Macro F1",) if c in rk.columns],
                           best_min=[c for c in ("Latency (ms)",) if c in rk.columns], sel_col="Model", sel_val=reco_name, badges=badges)
        else:
            bench = _empty("Model benchmark appears for labeled datasets with installed models.")

        text_cards = _text_cards(text_scores)
        text_fig = _text_plot(text_scores)

        r_kpis = _robust_kpis(robustness)
        r_fig, r_table = None, _empty("")
        if _has(robustness) and {"Condition", "Model", "Macro F1"} <= set(robustness.columns):
            rb = robustness.copy()
            rb["Macro F1"] = pd.to_numeric(rb["Macro F1"], errors="coerce")
            order = rb["Condition"].drop_duplicates().tolist()
            pv = rb.pivot_table(index="Condition", columns="Model", values="Macro F1", sort=False).reindex(order)
            r_fig = _line_plot(pv, "Macro F1 by condition", (0, 1.02), "per model")
            cols = [c for c in ("Condition", "Model", "Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1", "Weighted F1", "Avg Time (ms)") if c in rb.columns]
            rt = robustness[cols].rename(columns={"Avg Time (ms)": "Latency (ms)", "Precision (macro)": "Precision", "Recall (macro)": "Recall"})
            r_table = _table(rt, [c for c in rt.columns if c not in ("Condition", "Model")], best_max=["Macro F1"])
        return (overview, reco_html, lang_fig, script_fig, bench, text_cards, text_fig, r_kpis, r_fig, r_table)
    except Exception as e:  # never break the page because of a display problem
        return (_empty("Could not render results: " + _esc(e)),) + blank[1:]


# ----------------------------- dashboard renderer -----------------------------
def render_dashboard(model, lang, script, ranking, text_scores, robustness):
    blank = (_empty("Run a dataset analysis, then pick a model."),) + (None,) * 6
    try:
        lang_fig = _hbar(lang, "Language", "Language distribution") if _has(lang) else None
        script_fig = _hbar(script, "Script", "Script distribution") if _has(script) else None
        text_fig = _text_plot(text_scores) if _has(text_scores) else None
        if not _has(robustness) or "Model" not in robustness.columns:
            return (blank[0], None, None, text_fig, lang_fig, script_fig, None)
        if not model or model not in set(robustness["Model"].astype(str)):
            model = str(robustness["Model"].iloc[0])
        quality, speed = model_dashboard(model, robustness)
        sub = robustness[robustness["Model"].astype(str) == model].reset_index(drop=True)
        conds = sub["Condition"].astype(str).tolist()
        f1 = pd.to_numeric(sub["Macro F1"], errors="coerce").tolist()
        acc = pd.to_numeric(sub["Accuracy"], errors="coerce").tolist()
        wf1 = pd.to_numeric(sub["Weighted F1"], errors="coerce").tolist()
        lat = pd.to_numeric(sub["Avg Time (ms)"], errors="coerce").tolist()
        noisy = [i for i, c in enumerate(conds) if c.endswith(" noise")]
        d_f1 = (f1[noisy[-1]] - f1[0]) if noisy else None
        kpis = _grid([
            _kpi(f"{f1[0]:.3f}", "Macro F1 (clean)", f"{d_f1:+.3f} at highest noise" if d_f1 is not None else "", "negative" if (d_f1 or 0) < 0 else "positive", accent=True),
            _kpi(f"{acc[0]:.3f}", "Accuracy (clean)"),
            _kpi(f"{wf1[0]:.3f}", "Weighted F1 (clean)"),
            _kpi((f"{sum(lat) / len(lat):.1f}" if sum(lat) / len(lat) >= 1 else f"{sum(lat) / len(lat):.3f}") + " ms", "Avg latency", "mean across conditions"),
        ], "cols-4")
        f1_fig = _cond_bars(conds, f1, f"{model}: Macro F1 by condition", "clean / noisy / normalized", (0, 1.08), "{:.2f}")
        t_fig = _cond_bars(conds, lat, f"{model}: inference time", "ms per review", None, "{:.1f}")
        clean_cond = robustness["Condition"].iloc[0]
        cmp_df = robustness[robustness["Condition"] == clean_cond]
        return (kpis, f1_fig, t_fig, text_fig, lang_fig, script_fig, _model_compare(cmp_df, model))
    except Exception as e:
        return (_empty("Could not render dashboard: " + _esc(e)),) + (None,) * 6


# ----------------------------- single-review renderers -----------------------------
def _tone(label):
    l = str(label).lower()
    return "pos" if l.startswith("pos") else "neg" if l.startswith("neg") else "neu" if l.startswith("neu") else "oth"


def _pick(table, model):
    row = table[table["Model"].astype(str) == str(model)] if "Model" in table.columns else table.iloc[0:0]
    return row.iloc[0] if not row.empty else table.iloc[0]


def _pct(v):
    x = _flt(v)
    return None if x is None else (x * 100 if x <= 1.0 else x)


def render_prediction(table, model):
    if not _has(table) or "Prediction" not in table.columns:
        return _empty("Enter a review and press Analyze Sentiment.")
    r = _pick(table, model)
    t = _tone(r["Prediction"])
    c = _pct(r["Confidence"]) or 0
    probs = ""
    for cls in CLASSES:
        col = cls.capitalize()
        if col in table.columns:
            pv = _pct(r[col]) or 0
            probs += f'<div class="prob"><span>{_esc(cls.capitalize())}</span><div class="bar {_tone(cls)}"><i style="width:{pv:.1f}%"></i></div><span>{pv:.1f}%</span></div>'
    return (f'<div class="pred"><div class="pred-k">Sentiment &middot; {_esc(r["Model"])}</div>'
            f'<div class="pred-v {t}">{_esc(str(r["Prediction"]).capitalize())}</div><div class="pred-c">{c:.1f}% confidence</div>'
            f'<div class="bar {t}"><i style="width:{min(c, 100):.1f}%"></i></div><div class="probs">{probs}</div>'
            f'<div class="pred-meta">Inference {_f(r["Time (ms)"], 1)} ms</div></div>')


def render_compare(table, model):
    if not _has(table) or "Prediction" not in table.columns:
        return _empty("Model comparison appears here.")
    d = table.copy()
    conf = d["Confidence"].map(_pct)
    time_ = pd.to_numeric(d["Time (ms)"], errors="coerce")
    top_i, fast_i = conf.idxmax(), time_.idxmin()
    majority = d["Prediction"].mode().iloc[0]
    n_major = int((d["Prediction"] == majority).sum())
    badges = {}
    for i in d.index:
        b = ""
        if str(d.loc[i, "Model"]) == str(model):
            b += '<span class="badge i">Selected</span>'
        if i == top_i:
            b += '<span class="badge g">Top confidence</span>'
        if i == fast_i:
            b += '<span class="badge c">Fastest</span>'
        badges[("Model", i)] = b
    d["Confidence"] = conf.map(lambda v: "" if v is None else f"{v:.1f}%")
    d["Prediction"] = d["Prediction"].astype(str)
    nums = [c for c in d.columns if c not in ("Model", "Prediction", "Confidence")]
    tbl = _table(d, nums, sel_col="Model", sel_val=model, badges=badges)
    head = (f'<div class="metric-delta" style="margin:0 0 8px">{n_major} of {len(d)} models predict <b>{_esc(majority)}</b>.</div>')
    return head + tbl


# ---------------------------------------------------------------------
# Page chrome (header, home, about, footer) - presentation only
# ---------------------------------------------------------------------
def _icon(paths):
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{paths}</svg>'


ICONS = {
    "chat": _icon('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
    "db": _icon('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'),
    "shield": _icon('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
    "award": _icon('<circle cx="12" cy="8" r="7"/><polyline points="8.2 13.9 7 23 12 20 17 23 15.8 13.9"/>'),
    "layout": _icon('<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>'),
    "file": _icon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'),
}

_n_models = len(MODELS)
HEADER_HTML = (
    '<div class="app-header"><div class="brand"><div class="logo">हि</div><div>'
    '<div class="brand-name">Hindi / Hinglish NLP Studio</div>'
    '<div class="brand-sub">NLP robustness &amp; dataset intelligence</div></div></div>'
    '<div class="header-chips">'
    + (f'<span class="chip">{_n_models} Model{"s" if _n_models != 1 else ""}</span><span class="chip ok">Ready</span>'
       if _n_models else '<span class="chip warn">No models installed</span>')
    + '</div></div>'
)

HERO_HTML = """
<h1 class="hero-title">Understand Hindi &amp; Hinglish NLP robustness.</h1>
<p class="hero-text">Analyze sentiment, script variation, spelling noise, model performance and text transformation quality.</p>
"""

HOME_KPIS = _grid([
    _kpi(str(_n_models), "Models"), _kpi("3", "Noise levels"), _kpi("10+", "NLP metrics"), _kpi("PDF + HTML", "Reports", small=True),
], "cols-4")


def _feature(icon, title, text, cls=""):
    return f'<div class="feature"><div class="ico {cls}">{ICONS[icon]}</div><h3>{title}</h3><p>{text}</p></div>'


FEATURES_HTML = (
    _head("What you can do", "Runs on the installed models. Nothing is retrained.")
    + '<div class="feature-grid">'
    + _feature("chat", "Single Review", "Compare every model's prediction, confidence and latency on one review.")
    + _feature("db", "Dataset Intelligence", "Language, script, code-mixing, vocabulary and duplicates at a glance.", "alt")
    + _feature("shield", "Robustness Testing", "Low, medium and high spelling noise, raw and normalized.", "cy")
    + _feature("award", "Model Recommendation", "Ranks models for accuracy, speed or a balance of both.")
    + _feature("layout", "Interactive Dashboard", "Per-model KPIs and charts in one view.", "alt")
    + _feature("file", "Exportable Reports", "Download a formatted PDF or a self-contained HTML report.", "cy")
    + '</div>'
)

STEPS_HTML = (
    _head("How it works")
    + '<div class="steps">'
    '<div class="step"><i>1</i><div><h3>Upload a CSV</h3><p>Text and label columns are detected automatically.</p></div></div>'
    '<div class="step"><i>2</i><div><h3>Pick a priority</h3><p>Accuracy, speed or balanced, then analyze.</p></div></div>'
    '<div class="step"><i>3</i><div><h3>Review and export</h3><p>Explore the dashboard, download the PDF.</p></div></div>'
    '</div>'
)

_models_li = "".join(f"<li><code>{html.escape(str(m))}</code></li>" for m in MODELS) or "<li>No trained models found under <code>model/</code>.</li>"
ABOUT_HTML = _head("About", "Documentation for the metrics and methodology used in this app.", True) + f"""
<div class="about-grid">
  <div class="about-card"><h3>What this application does</h3><p>Predicts sentiment for Hindi and Hinglish text, profiles datasets by language and script, benchmarks the installed models, tests them under spelling noise, and recommends a model for your priority. Existing models are evaluated as-is; nothing is retrained.</p></div>
  <div class="about-card"><h3>Models</h3><ul>{_models_li}</ul></div>
  <div class="about-card"><h3>Classification metrics</h3><p><b>Accuracy</b>, macro <b>Precision</b> and <b>Recall</b>, <b>Macro F1</b> (all classes equal), <b>Weighted F1</b> (weighted by class frequency) and average inference latency in milliseconds.</p></div>
  <div class="about-card"><h3>Text transformation metrics</h3><p><b>BLEU</b>, <b>ROUGE-1/2/L</b>, <b>METEOR</b> and <b>chrF</b> measure overlap with the clean reference (higher is better, 0-100). <b>TER</b> is an edit rate (lower is better).</p></div>
  <div class="about-card"><h3>Robustness methodology</h3><p>Clean Hinglish, then Low / Medium / High spelling corruption, each shown raw and after normalization. Corruption is deterministic (fixed seed) and uses at most 500 evenly spaced rows. Hindi rows are transliterated to build the clean reference.</p></div>
  <div class="about-card"><h3>Limitations</h3><p>Robustness scores are synthetic benchmarks, not human judgments. Unlabeled datasets get a profile only. Latency depends on your hardware.</p></div>
</div>
"""


def _footer_html():
    import datetime
    return f"""
<div class="app-footer"><div class="footer-grid">
  <div><div class="footer-brand"><div class="logo">हि</div>Hindi / Hinglish NLP Studio</div><p style="margin-top:6px">NLP robustness &amp; dataset intelligence.</p></div>
  <div><h4>Sentiment Analysis</h4><ul><li>Single review</li><li>Model comparison</li></ul></div>
  <div><h4>Dataset Intelligence</h4><ul><li>Language &amp; script</li><li>NLP profile</li></ul></div>
  <div><h4>Robustness Testing</h4><ul><li>Spelling noise</li><li>Text metrics</li></ul></div>
  <div><h4>Model Recommendation</h4><ul><li>Benchmark</li><li>PDF &amp; HTML reports</li></ul></div>
</div><div class="footer-bottom"><span>&copy; {datetime.date.today().year} Hindi / Hinglish NLP Studio</span><span>Built with Gradio</span></div></div>
"""


demo = gr.Blocks(
    title="Hindi / Hinglish NLP Studio",
)

with demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_id="main-tabs") as main_tabs:

        # ------------------------- HOME -------------------------
        with gr.Tab("Home", id="home"):
            with gr.Column(elem_classes=["hero"]):
                gr.HTML(HERO_HTML)
                with gr.Row(elem_classes=["cta-row"]):
                    go_dataset = gr.Button("Analyze a Dataset", variant="primary")
                    go_single = gr.Button("Try a Single Review", variant="secondary", elem_classes=["ghost"])
            gr.HTML(HOME_KPIS)
            gr.HTML(FEATURES_HTML)
            gr.HTML(STEPS_HTML)

        # ------------------------- SINGLE REVIEW -------------------------
        with gr.Tab("Single Review", id="single"):
            gr.HTML(_head("Inference console", "Type a Hindi or Hinglish review and compare every model on it.", True))
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    review_text = gr.Textbox(
                        lines=3,
                        label="Review (Hindi or Hinglish)",
                    )
                    selected_model = gr.Dropdown(
                        choices=MODELS,
                        value=(MODELS[0] if MODELS else None),
                        label="Model shown in the main result",
                    )
                    predict_btn = gr.Button(
                        "Analyze Sentiment", variant="primary", elem_classes=["action-button"]
                    )
                    gr.Examples(
                        examples=examples,
                        inputs=[review_text, selected_model],
                    )

                with gr.Column(scale=6):
                    pred_card = gr.HTML(_empty("Enter a review and press Analyze Sentiment."))
                    with gr.Row():
                        detected_script = gr.Textbox(label="Detected script")
                        normalized_text = gr.Textbox(
                            label="Text sent to the models (after normalization for Hinglish)"
                        )
                    # raw components kept so run() keeps its exact outputs (hidden by CSS)
                    sentiment_output = gr.Label(
                        num_top_classes=len(CLASSES),
                        label="Sentiment (selected model)",
                        elem_classes=["sink"],
                    )

            gr.HTML(_head("Model comparison", "All models on this review."))
            compare_html = gr.HTML(_empty("Model comparison appears here."))
            comparison_table = gr.Dataframe(
                label="All models on this line",
                interactive=False,
                elem_classes=["sink"],
            )

        # ------------------------- DATASET ANALYSIS -------------------------
        with gr.Tab("Dataset Analysis", id="dataset"):
            # 1. Upload / configuration
            with gr.Column(elem_classes=["control-panel"]):
                gr.HTML(_head("Dataset Analysis", "Analyze your dataset's linguistic and model characteristics.", True))
                with gr.Row(equal_height=True):
                    dataset_file = gr.File(
                        label="Upload CSV dataset",
                        file_types=[".csv"],
                        type="filepath",
                        elem_classes=["upload-compact"],
                        scale=2,
                    )
                    with gr.Column(scale=3):
                        with gr.Row():
                            text_column = gr.Dropdown(
                                choices=[],
                                label="Text column",
                                interactive=True,
                            )
                            label_column = gr.Dropdown(
                                choices=["(None)"],
                                value="(None)",
                                label="Label column (optional)",
                                interactive=True,
                            )
                        priority = gr.Radio(
                            choices=["Accuracy", "Speed", "Balanced"],
                            value="Balanced",
                            label="Application priority",
                        )
                analyze_btn = gr.Button(
                    "Analyze Dataset & Recommend Model",
                    variant="primary",
                )

            # 2. Overview KPIs  /  7. Recommendation (promoted: it is the headline result)
            gr.HTML(_head("Dataset overview", "Size, language mix and vocabulary of the uploaded data."))
            ds_overview = gr.HTML(_empty("Upload a CSV and run the analysis to see dataset KPIs."))
            gr.HTML(_head("Model recommendation", "Best fit for the selected application priority."))
            ds_reco = gr.HTML(_empty("The model recommendation appears here after analysis."))

            # 3. Language & script
            gr.HTML(_head("Language & script intelligence", "How the text is distributed across languages and scripts."))
            with gr.Row():
                ds_lang_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
                ds_script_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])

            # 4. Benchmark
            gr.HTML(_head("Model benchmark", "Performance on the uploaded labeled dataset."))
            ds_bench = gr.HTML(_empty("Model benchmark appears for labeled datasets."))

            # 5. Text transformation quality
            gr.HTML(_head("Text transformation quality", "How closely noisy and normalized text matches the clean Hinglish reference."))
            with gr.Row(equal_height=False):
                ds_text_cards = gr.HTML(_empty("Text transformation metrics appear after analysis."))
                ds_text_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])

            # 6. Robustness
            gr.HTML(_head("Robustness analysis", "Macro F1 across clean, noisy and normalized input."))
            ds_robust_kpis = gr.HTML(_empty("Robustness scores appear after analysis."))
            ds_robust_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
            ds_robust_table = gr.HTML()

            # 8. Detailed tables (the raw analysis outputs; also feed charts, PDF and dashboard)
            gr.HTML(_head("Detailed tables", "Full raw outputs of the analysis."))
            with gr.Accordion("Show detailed tables", open=False):
                profile_table = gr.Dataframe(
                    label="NLP Dataset Profile",
                    interactive=False,
                )
                with gr.Row():
                    language_table = gr.Dataframe(
                        label="Language Distribution",
                        interactive=False,
                    )
                    script_table = gr.Dataframe(
                        label="Script Distribution",
                        interactive=False,
                    )
                benchmark_table = gr.Dataframe(
                    label="Measured Model Performance",
                    interactive=False,
                )
                ranking_table = gr.Dataframe(
                    label="Recommendation Ranking",
                    interactive=False,
                )
                text_metrics_table = gr.Dataframe(label="Text overlap and restoration metrics", interactive=False)
                robustness_table = gr.Dataframe(label="Scores by model and input condition", interactive=False)

            # raw markdown / state sinks that the analysis function writes to
            dataset_summary = gr.Markdown(elem_classes=["sink"])
            recommendation = gr.Markdown(elem_classes=["sink"])
            top_tokens = gr.Markdown(elem_classes=["sink"])
            language_chart = gr.State()
            script_chart = gr.State()
            robustness_plot = gr.State()

            # 9. Export
            gr.HTML(_head("Export reports", "Download the full analysis."))
            with gr.Row():
                pdf_download = gr.File(label="Download PDF report", interactive=False)
                report_download = gr.File(label="Download HTML report", interactive=False)

        # ------------------------- DASHBOARD -------------------------
        with gr.Tab("Dashboard", id="dashboard"):
            gr.HTML(_head("Dashboard", "Dataset robustness & model performance.", True))
            with gr.Column(elem_classes=["panel"]):
                dashboard_model = gr.Dropdown(
                    choices=MODELS,
                    value=(MODELS[0] if MODELS else None),
                    label="Model shown in the dashboard",
                    interactive=True,
                )
            dash_kpis = gr.HTML(_empty("Run a dataset analysis, then pick a model."))
            with gr.Row():
                dash_f1_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
                dash_time_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
            with gr.Row():
                dash_text_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
                dash_lang_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
            with gr.Row():
                dash_script_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])
                dash_compare_plot = gr.Plot(show_label=False, elem_classes=["chart-card"])

        # ------------------------- ABOUT -------------------------
        with gr.Tab("About", id="about"):
            gr.HTML(ABOUT_HTML)

    gr.HTML(_footer_html())

    # ------------------------- wiring -------------------------
    # Navigation only
    go_dataset.click(lambda: gr.update(selected="dataset"), None, main_tabs)
    go_single.click(lambda: gr.update(selected="single"), None, main_tabs)

    # Single review: run() is unchanged; the HTML cards are rendered from its table afterwards.
    predict_event = predict_btn.click(
        run,
        inputs=[review_text, selected_model],
        outputs=[
            sentiment_output,
            detected_script,
            normalized_text,
            comparison_table,
        ],
    )
    predict_event.then(render_prediction, [comparison_table, selected_model], [pred_card])
    predict_event.then(render_compare, [comparison_table, selected_model], [compare_html])
    selected_model.change(render_prediction, [comparison_table, selected_model], [pred_card])

    # Dataset analysis: analyze_uploaded_dataset() is unchanged.
    dataset_file.change(
        get_columns,
        inputs=[dataset_file],
        outputs=[text_column, label_column],
    )

    analysis_event = analyze_btn.click(
        analyze_uploaded_dataset,
        inputs=[dataset_file, text_column, label_column, priority],
        outputs=[
            dataset_summary,
            profile_table,
            language_table,
            script_table,
            benchmark_table,
            ranking_table,
            language_chart,
            script_chart,
            recommendation,
            top_tokens,
            text_metrics_table,
            robustness_table,
            robustness_plot,
            report_download,
        ],
    )

    # PDF is built from what is already on screen, right after the analysis.
    analysis_event.then(
        build_pdf_report,
        inputs=[dataset_summary, recommendation, top_tokens, profile_table, language_table,
                script_table, benchmark_table, ranking_table, text_metrics_table, robustness_table],
        outputs=[pdf_download],
    )
    analysis_event.then(
        render_results,
        inputs=[dataset_summary, recommendation, top_tokens, profile_table, language_table,
                script_table, ranking_table, text_metrics_table, robustness_table],
        outputs=[ds_overview, ds_reco, ds_lang_plot, ds_script_plot, ds_bench, ds_text_cards,
                 ds_text_plot, ds_robust_kpis, ds_robust_plot, ds_robust_table],
    )
    _dash_in = [dashboard_model, language_table, script_table, ranking_table, text_metrics_table, robustness_table]
    _dash_out = [dash_kpis, dash_f1_plot, dash_time_plot, dash_text_plot, dash_lang_plot, dash_script_plot, dash_compare_plot]
    analysis_event.then(render_dashboard, _dash_in, _dash_out)
    dashboard_model.change(render_dashboard, _dash_in, _dash_out)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=APP_CSS,
    )
