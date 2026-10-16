r"""Interactive versions of the campaign figures: self-contained HTML, no deps.

One shared scaffold (nearest-point hover + rich tooltip + legend + keyboard +
table view), four pages, each reading the same CSVs the static plots use:

  compass      results/compass/compass_interactive.html      (t1 curves per vocab)
  frontier     results/pareto_synthesis/frontier_interactive.html  (4 Pareto panels)
  dial         results/pareto_synthesis/dial_interactive.html (disparity+gini vs t2)
  interaction  results/surface_8k/interaction_interactive.html    (parallel lines)

Every point carries its full record in the tooltip (and the page-specific
verdict: best-per-curve stars, on-the-front vs DOMINATED, per-column spread).

Run (from superparity/):
  ..\.venv\Scripts\python.exe interactive_figs.py            # all four
  ..\.venv\Scripts\python.exe interactive_figs.py compass frontier
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import config
import compass

# reference palette (light mode), categorical slots in validated fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#4a3aa7", "#e34948"]
GRAY = "#898781"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10); }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 18px; font-weight: 650; }
  .sub { color: var(--ink-2); margin: 4px 0 14px; }
  .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
          padding: 16px 16px 10px; position: relative; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 2px 0 8px 6px; }
  .legend .it { display: flex; align-items: center; gap: 6px; font-size: 12.5px;
                color: var(--ink-2); }
  .legend .key { width: 16px; height: 0; border-top: 3px solid; border-radius: 2px; }
  .grid2 { display: grid; grid-template-columns: repeat(__GRID__, 1fr); gap: 4px 14px; }
  .panel .pt { font-size: 13px; color: var(--ink-2); font-weight: 600; margin: 6px 0 0 8px; }
  svg { width: 100%; height: auto; display: block; }
  svg text { font: 11.5px system-ui, -apple-system, "Segoe UI", sans-serif;
             font-variant-numeric: tabular-nums; }
  .tip { position: absolute; pointer-events: none; display: none; z-index: 3;
         background: var(--surface); border: 1px solid var(--ring); border-radius: 8px;
         box-shadow: 0 4px 14px rgba(11,11,11,0.12); padding: 8px 12px; max-width: 300px; }
  .tip .head { font-weight: 650; }
  .tip .sub2 { color: var(--ink-2); font-size: 12px; margin-bottom: 5px; }
  .tip .row { display: flex; align-items: baseline; gap: 7px; margin: 2px 0; }
  .tip .key { width: 14px; height: 0; border-top: 3px solid; border-radius: 2px;
              transform: translateY(-3px); flex: none; }
  .tip .val { font-weight: 650; font-variant-numeric: tabular-nums; }
  .tip .lbl { color: var(--ink-2); font-size: 12px; }
  .tip .minor { color: var(--ink-2); font-size: 12px; margin-top: 5px; }
  .hint { color: var(--muted); font-size: 12px; margin: 8px 0 4px 6px; }
  details { margin-top: 12px; }
  summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
  table { border-collapse: collapse; margin-top: 8px; font-size: 12px;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 9px; text-align: right; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; }
</style></head>
<body><div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="card" id="card" tabindex="0" aria-label="chart; arrow keys step through points">
    <div class="legend" id="legend"></div>
    <div class="grid2" id="panels"></div>
    <div class="tip" id="tip"></div>
    <div class="hint">__HINT__</div>
    <details><summary>data table (every point)</summary>
      <div style="overflow-x:auto"><table id="tbl"></table></div></details>
  </div>
</div>
<script>
"use strict";
const SPEC = __SPEC__;

const W = 520, H = 340, ML = 56, MR = 14, MT = 24, MB = 40;
const IW = W - ML - MR, IH = H - MT - MB;
const NS = "http://www.w3.org/2000/svg";
const el = (n, at) => { const e = document.createElementNS(NS, n);
  for (const k in at) e.setAttribute(k, at[k]); return e; };

function ticks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => span / s <= n + 1);
  const dec = Math.max(0, -Math.floor(Math.log10(step)) + (step / mag === 2.5 ? 1 : 0));
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return { vals: out, dec };
}
function domain(vals, pad, forced) {
  let lo = forced && forced[0] != null ? forced[0] : Math.min(...vals);
  let hi = forced && forced[1] != null ? forced[1] : Math.max(...vals);
  if (!forced) { const p = (hi - lo) * pad || 1; lo -= p; hi += p; }
  return [lo, hi];
}

const ALL = [];   // flat registry: {panel, series, pt, cx, cy, circle}
const panelsDiv = document.getElementById("panels");

SPEC.panels.forEach((P, pi) => {
  const box = document.createElement("div"); box.className = "panel";
  const t = document.createElement("div"); t.className = "pt"; t.textContent = P.title || "";
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                          "aria-label": P.title || "chart panel" });
  box.append(t, svg); panelsDiv.append(box);

  const xs = P.series.flatMap(s => s.pts.map(p => p.x));
  const ys = P.series.flatMap(s => s.pts.map(p => p.y));
  const [xlo, xhi] = domain(xs, 0.05, P.xdom);
  const [ylo, yhi] = domain(ys, 0.08, null);
  const X = v => ML + (v - xlo) / (xhi - xlo) * IW;
  const Y = v => MT + IH - (v - ylo) / (yhi - ylo) * IH;

  for (const b of (P.bands || [])) {
    svg.appendChild(el("rect", { x: X(b.x0), y: MT, width: X(b.x1) - X(b.x0), height: IH,
      fill: "rgba(11,11,11,0.05)" }));
    const bt = el("text", { x: (X(b.x0) + X(b.x1)) / 2, y: MT + IH - 6,
      "text-anchor": "middle", fill: "#898781", "font-size": "10" });
    bt.textContent = b.label; svg.appendChild(bt);
  }
  const ty = ticks(ylo, yhi, 5);
  for (const v of ty.vals) {
    svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: Y(v), y2: Y(v),
      stroke: "#e1e0d9", "stroke-width": 1 }));
    const lt = el("text", { x: ML - 7, y: Y(v) + 4, "text-anchor": "end", fill: "#898781" });
    lt.textContent = v.toFixed(ty.dec); svg.appendChild(lt);
  }
  svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: MT + IH, y2: MT + IH,
    stroke: "#c3c2b7", "stroke-width": 1 }));
  const tx = ticks(xlo, xhi, 6);
  for (const v of tx.vals) {
    const lt = el("text", { x: X(v), y: MT + IH + 16, "text-anchor": "middle", fill: "#898781" });
    lt.textContent = v.toFixed(tx.dec); svg.appendChild(lt);
  }
  const xl = el("text", { x: ML + IW / 2, y: H - 6, "text-anchor": "middle", fill: "#52514e" });
  xl.textContent = P.xlabel || ""; svg.appendChild(xl);
  const yl = el("text", { x: 14, y: MT + IH / 2, fill: "#52514e",
    transform: "rotate(-90 14 " + (MT + IH / 2) + ")", "text-anchor": "middle" });
  yl.textContent = P.ylabel || ""; svg.appendChild(yl);

  for (const s of P.series) {
    if (s.line !== false && s.pts.length > 1) {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p.x) + " " + Y(p.y)).join("");
      svg.appendChild(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-linejoin": "round",
        "stroke-dasharray": s.dash ? "6 5" : "none" }));
    }
    if (s.markers !== false) {
      s.pts.forEach((p, i) => {
        const best = s.star === i;
        const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: best ? 7 : 4.5,
          fill: p.c || s.color, stroke: best ? "#0b0b0b" : "none",
          "stroke-width": best ? 1.4 : 0 });
        svg.appendChild(c);
        ALL.push({ pi, s, p, cx: X(p.x), cy: Y(p.y), circle: c, svg, best });
      });
    }
    if (s.end_label) {
      const last = s.pts[s.pts.length - 1];
      const lt = el("text", { x: X(last.x) + 8, y: Y(last.y) + 4, fill: s.color,
        "font-weight": "650", "font-size": "12" });
      lt.textContent = s.end_label; svg.appendChild(lt);
    }
  }
  svg.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect();
    const mx = (e.clientX - r.left) / r.width * W, my = (e.clientY - r.top) / r.height * H;
    let bi = -1, bd = Infinity;
    ALL.forEach((a, i) => { if (a.svg !== svg) return;
      const d = (a.cx - mx) ** 2 + (a.cy - my) ** 2; if (d < bd) { bd = d; bi = i; } });
    if (bi >= 0) { cur = bi; show(bi); }
  });
  svg.addEventListener("pointerleave", hide);
});

// legend (series with names, first panel's set)
const seen = new Set();
for (const P of SPEC.panels) for (const s of P.series) {
  if (!s.name || seen.has(s.name)) continue; seen.add(s.name);
  const it = document.createElement("span"); it.className = "it";
  const k = document.createElement("span"); k.className = "key";
  k.style.borderTopColor = s.color; if (s.dash) k.style.borderTopStyle = "dashed";
  const n = document.createElement("span"); n.textContent = s.name;
  it.append(k, n); document.getElementById("legend").append(it);
}

// tooltip
const tip = document.getElementById("tip"), card = document.getElementById("card");
let cur = -1;
function show(i) {
  const a = ALL[i], d = a.p.d;
  tip.replaceChildren();
  const h = document.createElement("div"); h.className = "head"; h.textContent = d.head;
  tip.append(h);
  if (d.sub) { const s2 = document.createElement("div"); s2.className = "sub2";
    s2.textContent = d.sub; tip.append(s2); }
  for (const [color, val, lbl] of d.lines) {
    const r = document.createElement("div"); r.className = "row";
    const k = document.createElement("span"); k.className = "key";
    k.style.borderTopColor = color || a.s.color;
    const v = document.createElement("span"); v.className = "val"; v.textContent = val;
    const l = document.createElement("span"); l.className = "lbl"; l.textContent = lbl;
    r.append(k, v, l); tip.append(r);
  }
  if (d.minor) { const m = document.createElement("div"); m.className = "minor";
    m.textContent = d.minor; tip.append(m); }
  ALL.forEach(b => { b.circle.setAttribute("r", b.best ? 7 : 4.5);
    if (!b.best) b.circle.setAttribute("stroke", "none"); });
  a.circle.setAttribute("r", 7);
  a.circle.setAttribute("stroke", "#fcfcfb"); a.circle.setAttribute("stroke-width", 2);
  const sr = a.svg.getBoundingClientRect(), cr = card.getBoundingClientRect();
  const px = sr.left - cr.left + a.cx / W * sr.width;
  const py = sr.top - cr.top + a.cy / H * sr.height;
  tip.style.display = "block";
  const flip = px > cr.width * 0.6;
  tip.style.left = flip ? (px - tip.offsetWidth - 14) + "px" : (px + 14) + "px";
  tip.style.top = Math.max(6, py - tip.offsetHeight / 2) + "px";
}
function hide() { tip.style.display = "none";
  ALL.forEach(b => { b.circle.setAttribute("r", b.best ? 7 : 4.5);
    if (!b.best) b.circle.setAttribute("stroke", "none"); }); }
card.addEventListener("keydown", e => {
  if (e.key === "Escape") { hide(); cur = -1; return; }
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  cur = e.key === "ArrowRight" ? Math.min(ALL.length - 1, cur < 0 ? 0 : cur + 1)
                               : Math.max(0, cur < 0 ? 0 : cur - 1);
  show(cur);
});

// table
const tbl = document.getElementById("tbl");
{
  const tr = document.createElement("tr");
  for (const h of SPEC.table.cols) { const th = document.createElement("th");
    th.textContent = h; tr.append(th); }
  tbl.append(tr);
  for (const row of SPEC.table.rows) {
    const r = document.createElement("tr");
    for (const v of row) { const td = document.createElement("td");
      td.textContent = v; r.append(td); }
    tbl.append(r);
  }
}
</script></body></html>
"""


def write_page(out: Path, title: str, subtitle: str, hint: str, grid: int, spec: dict):
    html = (PAGE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)
                .replace("__HINT__", hint).replace("__GRID__", str(grid))
                .replace("__SPEC__", json.dumps(spec)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _pareto_rows(tag: str) -> list[dict]:
    rows = []
    for r in _rows(config.RESULTS / tag / "summary.csv"):
        rows.append({"t1": int(r["t1"]), "t2": int(r["t2"]),
                     "t1_frac": float(r["t1_frac"]), "t2_frac": float(r["t2_frac"]),
                     "total": int(r["total_merges"]),
                     "tok": float(r["mean_tokens_per_line"]),
                     "bpt": float(r["mean_bytes_per_token"]),
                     "disp": float(r["disparity_ratio"]), "gini": float(r["gini"]),
                     "worst": r["worst_language"], "best": r["best_language"],
                     "secs": float(r["train_seconds"])})
    return sorted(rows, key=lambda d: d["t2_frac"])


CANON = [("pareto_8k", 8000), ("pareto_20k", 20000),
         ("pareto_50k", 50000), ("pareto_80k", 80000)]


# --- spec builders (reused by simple_html.py) --------------------------------
def spec_compass() -> dict:
    pts = compass.harvest()
    ctx = compass.scaling_context()
    vocabs = sorted({p["vocab"] for p in pts})
    series, table_rows = [], []
    for i, v in enumerate(vocabs):
        vp = [p for p in pts if p["vocab"] == v]
        best = max(range(len(vp)), key=lambda k: vp[k]["bytes_per_token"])
        spread = (max(p["bytes_per_token"] for p in vp) -
                  min(p["bytes_per_token"] for p in vp)) / max(
                      p["bytes_per_token"] for p in vp) * 100
        spts = []
        for k, p in enumerate(vp):
            d = {"head": f"vocab {v//1000}k — t1 = {p['t1']:,} (frac {p['frac']:.3f})",
                 "sub": "best measured on this curve" if k == best else "",
                 "lines": [[SLOTS[i % 8], f"{p['bytes_per_token']:.4f}", "bytes / token"],
                           ["#898781", f"{p['tokens_per_line']:.2f}", "tokens / line"],
                           ["#898781", f"{p['disparity']:.3f}", "disparity (flat in t1)"]],
                 "minor": f"curve spread {spread:.2f}% · source: {p['source']}"}
            spts.append({"x": p["frac"], "y": p["bytes_per_token"], "d": d})
            table_rows.append([f"{v//1000}k", f"{p['t1']:,}", f"{p['frac']:.3f}",
                               f"{p['bytes_per_token']:.4f}", f"{p['tokens_per_line']:.2f}",
                               f"{p['disparity']:.3f}", p["source"]])
        series.append({"name": f"vocab {v//1000}k", "color": SLOTS[i % 8],
                       "end_label": f"{v//1000}k", "star": best, "pts": spts})
    if ctx:
        series.insert(0, {"name": "scaling run (fixed head, 2 MB data — context)",
                          "color": GRAY, "dash": True,
                          "pts": [{"x": p["frac"], "y": p["bytes_per_token"],
                                   "d": {"head": f"scaling context — vocab {p['vocab']//1000}k",
                                         "sub": "fixed 7,744-merge head, 2 MB data — not comparable to the 1 MB curves",
                                         "lines": [[GRAY, f"{p['bytes_per_token']:.4f}", "bytes / token"]],
                                         "minor": ""}} for p in ctx]})
    spec = {"panels": [{
        "title": "bytes / token vs t1 fraction — one line per vocab",
        "xlabel": "t1 (fraction of merge budget, sub→super)",
        "ylabel": "bytes / token  (↑ better)", "xdom": [0, 1.02],
        "bands": [{"x0": 0.30, "x1": 0.40, "label": "SuperBPE paper peak (200k)"}],
        "series": series}],
        "table": {"cols": ["vocab", "t1", "frac", "bytes/tok", "tok/line",
                           "disparity", "source"], "rows": table_rows}}
    return spec


def build_compass():
    write_page(config.RESULTS / "compass" / "compass_interactive.html",
               "Compass — the t1 sweep, interactive",
               "Each dot = one complete tokenizer; ★ = best measured per curve. "
               "The flatness is the finding: every curve's spread is under 1%.",
               "Hover any dot (or ←/→) for its full record. Bigger dot with black ring = "
               "that curve's best. Gray band = the paper's 200k peak region.", 1,
               spec_compass())


def spec_frontier() -> dict:
    # sequential blue ordinal ramp for t2 (light→dark = later→earlier parity start)
    RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
    panels, table_rows = [], []
    for tag, vocab in CANON:
        rows = _pareto_rows(tag)
        if not rows:
            continue
        # dominance: r is dominated if some other row is <= on both and < on one
        for r in rows:
            r["dom"] = next((o for o in rows if o is not r and o["tok"] <= r["tok"]
                             and o["disp"] <= r["disp"]
                             and (o["tok"] < r["tok"] or o["disp"] < r["disp"])), None)
        front = sorted([r for r in rows if not r["dom"]], key=lambda r: r["tok"])
        fr_series = {"name": None, "color": "#e34948", "dash": True, "markers": False,
                     "pts": [{"x": r["tok"], "y": r["disp"], "d": None} for r in front]}
        spts = []
        for r in rows:
            share = round((1 - r["t2_frac"]) * 100)
            shade = RAMP[min(int(r["t2_frac"] * len(RAMP)), len(RAMP) - 1)]
            verdict = ("no parity — DOMINATED (t2=0.90 beats it on BOTH axes)"
                       if r["t2_frac"] >= 0.999 and r["dom"] else
                       ("DOMINATED by t2=%.2f" % r["dom"]["t2_frac"] if r["dom"]
                        else "on the Pareto front"))
            d = {"head": f"t2 = {r['t2_frac']:.2f}  (merge {r['t2']:,})",
                 "sub": f"parity gets the last {share}% of the budget",
                 "lines": [[shade, f"{r['tok']:.3f}", "tokens / line (↓ better)"],
                           [shade, f"{r['disp']:.4f}", "disparity (↓ fairer)"],
                           ["#898781", f"{r['gini']:.4f}", "Gini"]],
                 "minor": f"{verdict} · worst {r['worst']} · best {r['best']}"
                          + (f" · tail {r['secs']/60:.0f} min" if r["secs"] else "")}
            spts.append({"x": r["tok"], "y": r["disp"], "c": shade, "d": d})
            table_rows.append([f"{vocab//1000}k", f"{r['t2_frac']:.2f}", f"{r['tok']:.3f}",
                               f"{r['disp']:.4f}", f"{r['gini']:.4f}", r["worst"],
                               "dominated" if r["dom"] else "front"])
        panels.append({"title": f"vocab {vocab//1000}k  (t1 = {rows[0]['t1']:,})",
                       "xlabel": "tokens / line (↓ better compression)",
                       "ylabel": "disparity (↓ fairer)",
                       "series": [fr_series,
                                  {"name": None, "color": RAMP[2], "line": False,
                                   "pts": spts}]})
    return {"panels": panels,
            "table": {"cols": ["vocab", "t2", "tok/line", "disparity", "gini",
                               "worst", "status"], "rows": table_rows}}


def build_frontier():
    write_page(config.RESULTS / "pareto_synthesis" / "frontier_interactive.html",
               "Pareto frontiers — the t2 menu, interactive",
               "Each dot = one finished tokenizer; darker blue = earlier parity start "
               "(more parity). Down-left is better on both axes; the dashed red line is "
               "the menu. The pale dot off every front is the no-parity endpoint.",
               "Hover a dot: its t2, parity share, both metrics, worst language, and "
               "whether it is on the front or dominated.", 2, spec_frontier())


def spec_dial() -> dict:
    panels, table_rows = [], []
    data = [(tag, vocab, _pareto_rows(tag)) for tag, vocab in CANON]
    data = [(t, v, r) for t, v, r in data if r]
    for key, label, dec in [("disp", "disparity worst/best  (↓ fairer)", 2),
                            ("gini", "Gini  (↓ fairer)", 3)]:
        series = []
        for i, (tag, vocab, rows) in enumerate(data):
            spts = []
            for r in rows:
                share = round((1 - r["t2_frac"]) * 100)
                d = {"head": f"vocab {vocab//1000}k — t2 = {r['t2_frac']:.2f}",
                     "sub": f"parity gets the last {share}% of the budget",
                     "lines": [[SLOTS[i], f"{r['disp']:.4f}", "disparity"],
                               [SLOTS[i], f"{r['gini']:.4f}", "Gini"],
                               ["#898781", f"{r['tok']:.2f}", "tokens / line"]],
                     "minor": f"worst {r['worst']} · best {r['best']}"}
                spts.append({"x": r["t2_frac"], "y": r[key], "d": d})
                if key == "disp":
                    table_rows.append([f"{vocab//1000}k", f"{r['t2_frac']:.2f}",
                                       f"{r['disp']:.4f}", f"{r['gini']:.4f}",
                                       f"{r['tok']:.2f}", r["worst"]])
            series.append({"name": f"vocab {vocab//1000}k", "color": SLOTS[i],
                           "end_label": f"{vocab//1000}k", "pts": spts})
        panels.append({"title": label, "xlabel": "t2 — when parity starts (1.0 = never)",
                       "ylabel": label, "series": series})
    return {"panels": panels,
            "table": {"cols": ["vocab", "t2", "disparity", "gini", "tok/line", "worst"],
                      "rows": table_rows}}


def build_dial():
    write_page(config.RESULTS / "pareto_synthesis" / "dial_interactive.html",
               "The fairness dial — disparity and Gini vs t2, interactive",
               "One curve per vocab. In the working range (t2 ≈ 0.6–0.9) the curves "
               "collapse onto one line — the dial is scale-invariant; only the extremes "
               "move with vocab.",
               "Hover any dot for that tokenizer's full record.", 2, spec_dial())


def spec_interaction() -> dict:
    rows = []
    for r in _rows(config.RESULTS / "surface_8k" / "surface.csv"):
        rows.append({"t1f": float(r["t1_frac"]), "t2f": float(r["t2_frac"]),
                     "tok": float(r["mean_tokens_per_line"]),
                     "disp": float(r["disparity_ratio"]), "gini": float(r["gini"]),
                     "worst": r["worst_language"], "kind": r["kind"]})
    t1s = sorted({r["t1f"] for r in rows})
    # per-column disparity spread across t1 rows (the interaction size, per t2)
    spread = {}
    for t2f in {r["t2f"] for r in rows}:
        col = [r["disp"] for r in rows if r["t2f"] == t2f]
        if len(col) > 1:
            spread[t2f] = max(col) - min(col)
    panels, table_rows = [], []
    for key, label in [("disp", "disparity worst/best  (↓ fairer)"),
                       ("tok", "tokens / line  (↓ better compression)")]:
        series = []
        for i, t1f in enumerate(t1s):
            rp = sorted([r for r in rows if r["t1f"] == t1f], key=lambda r: r["t2f"])
            spts = []
            for r in rp:
                sp = spread.get(r["t2f"])
                kind = {"superbpe_edge": "no-parity edge (= superbpe)",
                        "bpe_then_pa_super_edge": "t2=t1 edge (no classical-super)",
                        "interior": "interior cell"}.get(r["kind"], r["kind"])
                d = {"head": f"t1 = {r['t1f']:.1f}, t2 = {r['t2f']:.1f}",
                     "sub": kind,
                     "lines": [[SLOTS[i], f"{r['disp']:.4f}", "disparity"],
                               [SLOTS[i], f"{r['tok']:.2f}", "tokens / line"]],
                     "minor": (f"spread across t1 rows at this t2: {sp:.4f} "
                               f"(bar 0.02)" if sp else "") + f" · worst {r['worst']}"}
                spts.append({"x": r["t2f"], "y": r[key], "d": d})
                if key == "disp":
                    table_rows.append([f"{r['t1f']:.1f}", f"{r['t2f']:.1f}",
                                       f"{r['disp']:.4f}", f"{r['tok']:.2f}",
                                       f"{r['gini']:.4f}", r["worst"]])
            series.append({"name": f"t1 = {t1f:.1f}", "color": SLOTS[i],
                           "pts": spts})
        panels.append({"title": label, "xlabel": "t2 (fraction of merge budget)",
                       "ylabel": label, "series": series})
    return {"panels": panels,
            "table": {"cols": ["t1", "t2", "disparity", "tok/line", "gini", "worst"],
                      "rows": table_rows}}


def build_interaction():
    write_page(config.RESULTS / "surface_8k" / "interaction_interactive.html",
               "Interaction test — does t2's effect depend on t1? (8k joint grid)",
               "One line per t1 setting, from the complete 21-cell grid. Parallel lines "
               "= no interaction: max spread 0.0090, pure-drift 0.0072, bar 0.02.",
               "Hover any dot: its cell, both metrics, and the disparity spread across "
               "t1 rows at that t2 column.", 2, spec_interaction())


BUILDERS = {"compass": build_compass, "frontier": build_frontier,
            "dial": build_dial, "interaction": build_interaction}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", default=[], choices=[[], *BUILDERS] if False else None,
                    help=f"which pages to build (default: all of {list(BUILDERS)})")
    args = ap.parse_args()
    pages = args.pages or list(BUILDERS)
    for p in pages:
        if p not in BUILDERS:
            raise SystemExit(f"unknown page {p!r}; choices: {list(BUILDERS)}")
        BUILDERS[p]()


if __name__ == "__main__":
    main()
