r"""Interactive anatomy figure: self-contained HTML (inline SVG + vanilla JS).

Reads an anatomy.csv (from anatomy_plot.py) and writes anatomy.html next to it:
two stacked panels over the merge timeline with tinted phase bands, dashed t1/t2
rules, a crosshair that snaps to the nearest checkpoint, and one tooltip showing
every recorded field for that checkpoint. No dependencies, works offline —
same workflow as compass.html: just open it in a browser.

Run (from superparity/):
  ..\.venv\Scripts\python.exe anatomy_html.py
  ..\.venv\Scripts\python.exe anatomy_html.py --csv results\anatomy\anatomy.csv --t1 20000 --t2 35900
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import config

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --c-compress: #2a78d6;   /* series slot 1 (blue)   */
    --c-fair: #4a3aa7;       /* series slot 7 (violet) */
    --c-t1: #008300;         /* green rule  */
    --c-t2: #e34948;         /* red rule    */
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         padding: 24px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 18px; font-weight: 650; }
  .sub { color: var(--ink-2); margin: 4px 0 16px; }
  .card { background: var(--surface); border: 1px solid var(--ring);
          border-radius: 10px; padding: 18px 18px 10px; position: relative; }
  .panel-title { font-size: 13px; color: var(--ink-2); font-weight: 600;
                 margin: 6px 0 2px 8px; }
  svg { width: 100%; height: auto; display: block; }
  svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif;
             font-variant-numeric: tabular-nums; }
  .tip { position: absolute; pointer-events: none; display: none; z-index: 3;
         background: var(--surface); border: 1px solid var(--ring);
         border-radius: 8px; box-shadow: 0 4px 14px rgba(11,11,11,0.12);
         padding: 8px 12px; min-width: 190px; }
  .tip .head { font-weight: 650; margin-bottom: 2px; }
  .tip .phase { color: var(--ink-2); font-size: 12px; margin-bottom: 6px; }
  .tip .row { display: flex; align-items: baseline; gap: 7px; margin: 2px 0; }
  .tip .key { width: 14px; height: 0; border-top: 3px solid; border-radius: 2px;
              transform: translateY(-3px); }
  .tip .val { font-weight: 650; font-variant-numeric: tabular-nums; }
  .tip .lbl { color: var(--ink-2); font-size: 12px; }
  .tip .minor { color: var(--ink-2); font-size: 12px; margin-top: 5px; }
  .hint { color: var(--muted); font-size: 12px; margin: 10px 0 4px 8px; }
  details { margin-top: 14px; }
  summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
  table { border-collapse: collapse; margin-top: 8px; font-size: 12px;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; }
</style></head>
<body><div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__ &mdash; hover or use &larr;/&rarr; for the numbers at any checkpoint.</div>
  <div class="card" id="card" tabindex="0" aria-label="Anatomy chart; use arrow keys to step through checkpoints">
    <div class="panel-title">bytes / token &nbsp;(&uarr; better)</div>
    <svg id="p1" viewBox="0 0 940 300" role="img" aria-label="bytes per token over merges"></svg>
    <div class="panel-title">disparity worst/best &nbsp;(&darr; fairer)</div>
    <svg id="p2" viewBox="0 0 940 300" role="img" aria-label="disparity over merges"></svg>
    <div class="tip" id="tip"></div>
    <div class="hint">Each dot is the run evaluated after that many merges (truncation checkpoint).
      Bands: the three phases. Dashed rules: the two switches.</div>
    <details><summary>data table (all checkpoints)</summary>
      <div style="overflow-x:auto"><table id="tbl"></table></div>
    </details>
  </div>
</div>
<script>
"use strict";
const DATA = __DATA__;
const CFG  = __CFG__;

const W = 940, H = 300, ML = 62, MR = 14, MT = 20, MB = 30;
const IW = W - ML - MR, IH = H - MT - MB;
const xmax = CFG.total;
const X = m => ML + (m / xmax) * IW;

function yScale(vals, pad) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const p = (hi - lo) * pad || 1; lo -= p; hi += p;
  return { lo, hi, y: v => MT + IH - ((v - lo) / (hi - lo)) * IH };
}
function ticks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => span / s <= n + 1);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}
const fmtK = m => (m / 1000) + "k";
const el = (n, at) => { const e = document.createElementNS("http://www.w3.org/2000/svg", n);
  for (const k in at) e.setAttribute(k, at[k]); return e; };

const PHASES = [
  { from: 0,      to: CFG.t1,  name: "BPE",             fill: "rgba(11,11,11,0.035)" },
  { from: CFG.t1, to: CFG.t2,  name: "classical super", fill: "transparent" },
  { from: CFG.t2, to: xmax,    name: "parity super",    fill: "rgba(74,58,167,0.055)" },
];

function drawPanel(svg, key, color, decimals, showX) {
  const S = yScale(DATA.map(d => d[key]), 0.08);
  // phase bands + labels (top panel only labels)
  for (const ph of PHASES) {
    svg.appendChild(el("rect", { x: X(ph.from), y: MT, width: X(ph.to) - X(ph.from),
      height: IH, fill: ph.fill }));
    if (showX === false) {  // labels only on the first panel
      const t = el("text", { x: (X(ph.from) + X(ph.to)) / 2, y: MT + 14,
        "text-anchor": "middle", fill: "#898781", "font-size": "11" });
      t.textContent = ph.name; svg.appendChild(t);
    }
  }
  // gridlines + y ticks
  for (const v of ticks(S.lo, S.hi, 5)) {
    svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: S.y(v), y2: S.y(v),
      stroke: "#e1e0d9", "stroke-width": 1 }));
    const t = el("text", { x: ML - 8, y: S.y(v) + 4, "text-anchor": "end", fill: "#898781" });
    t.textContent = v.toFixed(decimals); svg.appendChild(t);
  }
  // x axis
  svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: MT + IH, y2: MT + IH,
    stroke: "#c3c2b7", "stroke-width": 1 }));
  if (showX) {
    for (let m = 0; m <= xmax; m += 10000) {
      const t = el("text", { x: X(m), y: MT + IH + 18, "text-anchor": "middle", fill: "#898781" });
      t.textContent = fmtK(m); svg.appendChild(t);
    }
    const xl = el("text", { x: ML + IW / 2, y: H - 2, "text-anchor": "middle", fill: "#52514e" });
    xl.textContent = "merge number"; svg.appendChild(xl);
  }
  // t1 / t2 rules
  for (const [m, c, name] of [[CFG.t1, "#008300", "t1: wall drops"],
                              [CFG.t2, "#e34948", "t2: parity on"]]) {
    svg.appendChild(el("line", { x1: X(m), x2: X(m), y1: MT, y2: MT + IH, stroke: c,
      "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
    if (showX === false) {
      const t = el("text", { x: X(m) + 4, y: MT + 28, fill: c, "font-size": "11" });
      t.textContent = name; svg.appendChild(t);
    }
  }
  // series line + dots
  const path = DATA.map((d, i) => (i ? "L" : "M") + X(d.n_merges) + " " + S.y(d[key])).join("");
  svg.appendChild(el("path", { d: path, fill: "none", stroke: color, "stroke-width": 2,
    "stroke-linejoin": "round" }));
  const dots = DATA.map(d => {
    const c = el("circle", { cx: X(d.n_merges), cy: S.y(d[key]), r: 4, fill: color });
    svg.appendChild(c); return c;
  });
  const cross = el("line", { y1: MT, y2: MT + IH, stroke: "#898781", "stroke-width": 1,
    "stroke-dasharray": "2 3", visibility: "hidden" });
  svg.appendChild(cross);
  return { dots, cross, S };
}

const p1 = drawPanel(document.getElementById("p1"), "mean_bytes_per_token", "#2a78d6", 1, false);
const p2 = drawPanel(document.getElementById("p2"), "disparity_ratio", "#4a3aa7", 2, true);

// ---- tooltip (DOM built with textContent -- CSV fields are untrusted) ----
const tip = document.getElementById("tip"), card = document.getElementById("card");
function row(color, val, lbl) {
  const r = document.createElement("div"); r.className = "row";
  const k = document.createElement("span"); k.className = "key"; k.style.borderTopColor = color;
  const v = document.createElement("span"); v.className = "val"; v.textContent = val;
  const l = document.createElement("span"); l.className = "lbl"; l.textContent = lbl;
  r.append(k, v, l); return r;
}
function showTip(i) {
  const d = DATA[i];
  tip.replaceChildren();
  const h = document.createElement("div"); h.className = "head";
  h.textContent = "merge " + d.n_merges.toLocaleString("en-US") +
                  "  (vocab " + d.vocab.toLocaleString("en-US") + ")";
  const ph = document.createElement("div"); ph.className = "phase";
  ph.textContent = "phase: " + d.phase; tip.append(h, ph);
  tip.append(row("#2a78d6", d.mean_bytes_per_token.toFixed(4), "bytes / token"));
  tip.append(row("#4a3aa7", d.disparity_ratio.toFixed(4), "disparity worst/best"));
  const m = document.createElement("div"); m.className = "minor";
  m.textContent = d.mean_tokens_per_line.toFixed(2) + " tok/line · gini " +
                  d.gini.toFixed(4) + " · worst " + d.worst_language +
                  " · best " + d.best_language;
  tip.append(m);
  // position near the crosshair, flipping at the right edge
  const svgBox = document.getElementById("p1").getBoundingClientRect();
  const cardBox = card.getBoundingClientRect();
  const px = svgBox.left - cardBox.left + (X(d.n_merges) / W) * svgBox.width;
  tip.style.display = "block";
  const flip = px > cardBox.width * 0.62;
  tip.style.left = flip ? (px - tip.offsetWidth - 14) + "px" : (px + 14) + "px";
  tip.style.top = "70px";
  // crosshair + lifted dot on both panels
  for (const p of [p1, p2]) {
    p.cross.setAttribute("x1", X(d.n_merges)); p.cross.setAttribute("x2", X(d.n_merges));
    p.cross.setAttribute("visibility", "visible");
    p.dots.forEach((c, j) => { c.setAttribute("r", j === i ? 6 : 4);
      c.setAttribute("stroke", j === i ? "#fcfcfb" : "none");
      c.setAttribute("stroke-width", j === i ? 2 : 0); });
  }
}
function hideTip() {
  tip.style.display = "none";
  for (const p of [p1, p2]) { p.cross.setAttribute("visibility", "hidden");
    p.dots.forEach(c => { c.setAttribute("r", 4); c.setAttribute("stroke", "none"); }); }
}
function nearest(clientX) {
  const box = document.getElementById("p1").getBoundingClientRect();
  const m = ((clientX - box.left) / box.width * W - ML) / IW * xmax;
  let best = 0, bd = Infinity;
  DATA.forEach((d, i) => { const dd = Math.abs(d.n_merges - m); if (dd < bd) { bd = dd; best = i; } });
  return best;
}
let cur = -1;
for (const id of ["p1", "p2"]) {
  const svg = document.getElementById(id);
  svg.addEventListener("pointermove", e => { cur = nearest(e.clientX); showTip(cur); });
  svg.addEventListener("pointerleave", () => { cur = -1; hideTip(); });
}
card.addEventListener("keydown", e => {
  if (e.key === "Escape") { cur = -1; hideTip(); return; }
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  e.preventDefault();
  if (e.key === "Home") cur = 0;
  else if (e.key === "End") cur = DATA.length - 1;
  else if (e.key === "ArrowRight") cur = Math.min(DATA.length - 1, cur < 0 ? 0 : cur + 1);
  else cur = Math.max(0, cur < 0 ? 0 : cur - 1);
  showTip(cur);
});

// ---- table view ----
const tbl = document.getElementById("tbl");
{
  const cols = [["n_merges", "merges"], ["vocab", "vocab"], ["phase", "phase"],
    ["mean_bytes_per_token", "bytes/tok"], ["mean_tokens_per_line", "tok/line"],
    ["disparity_ratio", "disparity"], ["gini", "gini"],
    ["worst_language", "worst"], ["best_language", "best"]];
  const tr = document.createElement("tr");
  for (const [, h] of cols) { const th = document.createElement("th"); th.textContent = h; tr.append(th); }
  tbl.append(tr);
  for (const d of DATA) {
    const r = document.createElement("tr");
    for (const [k] of cols) { const td = document.createElement("td");
      const v = d[k]; td.textContent = typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : v;
      r.append(td); }
    tbl.append(r);
  }
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RESULTS / "anatomy" / "anatomy.csv"))
    ap.add_argument("--t1", type=int, default=20000)
    ap.add_argument("--t2", type=int, default=35900)
    ap.add_argument("--out", default=None, help="output html (default: anatomy.html next to the csv)")
    ap.add_argument("--title", default="Anatomy of one run — BPE → SuperBPE → parity-super")
    args = ap.parse_args()

    src = Path(args.csv)
    rows = []
    for r in csv.DictReader(open(src, encoding="utf-8")):
        rows.append({
            "n_merges": int(r["n_merges"]), "vocab": int(r["vocab"]), "phase": r["phase"],
            "mean_bytes_per_token": float(r["mean_bytes_per_token"]),
            "mean_tokens_per_line": float(r["mean_tokens_per_line"]),
            "disparity_ratio": float(r["disparity_ratio"]), "gini": float(r["gini"]),
            "worst_language": r["worst_language"], "best_language": r["best_language"],
        })
    rows.sort(key=lambda d: d["n_merges"])
    total = rows[-1]["n_merges"]
    sub = (f"vocab {total + 256:,}; t1 = {args.t1:,} (whitespace wall drops), "
           f"t2 = {args.t2:,} (parity objective on); {len(rows)} truncation checkpoints")

    # "</" cannot appear in this data (numbers + language codes), so inline JSON is safe
    html = (PAGE
            .replace("__TITLE__", args.title)
            .replace("__SUBTITLE__", sub)
            .replace("__DATA__", json.dumps(rows))
            .replace("__CFG__", json.dumps({"t1": args.t1, "t2": args.t2, "total": total})))
    out = Path(args.out) if args.out else src.parent / "anatomy.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(rows)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
