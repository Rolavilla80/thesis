r"""SIMPLE.html — the professor document with INTERACTIVE figures inline.

Takes the SIMPLE.md prose (the user's voice, verbatim) and embeds the five
figures as live charts instead of PNGs: hover any dot for its full record,
arrow keys step, every figure carries a collapsible data table. One fully
self-contained file (no libraries, no internet, no external images) — it can
be opened anywhere, projected in the meeting, or emailed as-is.

Chart specs are reused from interactive_figs.py (same CSV sources as the
static plots); the anatomy spec is built here (it needs phase bands + rules).

Run (from superparity/):
  ..\.venv\Scripts\python.exe simple_html.py        # -> notes/SIMPLE.html
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import config
import interactive_figs as ifigs

OUT = Path(__file__).resolve().parent / "notes" / "SIMPLE.html"


# --- anatomy spec (local: needs bands with fills + vertical rules) ----------
def spec_anatomy(t1: int = 20000, t2: int = 35900) -> dict:
    src = config.RESULTS / "anatomy" / "anatomy.csv"
    rows = []
    for r in csv.DictReader(open(src, encoding="utf-8")):
        rows.append({"n": int(r["n_merges"]), "vocab": int(r["vocab"]),
                     "phase": r["phase"],
                     "bpt": float(r["mean_bytes_per_token"]),
                     "tok": float(r["mean_tokens_per_line"]),
                     "disp": float(r["disparity_ratio"]), "gini": float(r["gini"]),
                     "worst": r["worst_language"], "best": r["best_language"]})
    rows.sort(key=lambda d: d["n"])
    total = rows[-1]["n"]
    bands = [{"x0": 0, "x1": t1, "label": "BPE", "fill": "rgba(11,11,11,0.035)",
              "label_pos": "top"},
             {"x0": t1, "x1": t2, "label": "classical super", "fill": "transparent",
              "label_pos": "top"},
             {"x0": t2, "x1": total, "label": "parity super",
              "fill": "rgba(74,58,167,0.055)", "label_pos": "top"}]
    vrules = [{"x": t1, "color": "#008300", "label": "t1: wall drops"},
              {"x": t2, "color": "#e34948", "label": "t2: parity on"}]

    def detail(d):
        return {"head": f"merge {d['n']:,}  (vocab {d['vocab']:,})",
                "sub": "phase: " + d["phase"],
                "lines": [["#2a78d6", f"{d['bpt']:.4f}", "bytes / token"],
                          ["#4a3aa7", f"{d['disp']:.4f}", "disparity worst/best"]],
                "minor": (f"{d['tok']:.2f} tok/line · gini {d['gini']:.4f} · "
                          f"worst {d['worst']} · best {d['best']}")}

    def panel(key, ylabel, color, top):
        return {"title": "", "ylabel": ylabel,
                "xlabel": "" if top else "merge number", "h": 300,
                "bands": bands if True else [],
                "vrules": vrules, "vrule_labels": top,
                "band_labels": top,
                "series": [{"name": None, "color": color,
                            "pts": [{"x": d["n"], "y": d[key], "d": detail(d)}
                                    for d in rows]}]}

    return {"panels": [panel("bpt", "bytes / token  (↑ better)", "#2a78d6", True),
                       panel("disp", "disparity worst/best  (↓ fairer)", "#4a3aa7", False)],
            "table": {"cols": ["merges", "vocab", "phase", "bytes/tok", "tok/line",
                               "disparity", "gini", "worst", "best"],
                      "rows": [[f"{d['n']:,}", f"{d['vocab']:,}", d["phase"],
                                f"{d['bpt']:.4f}", f"{d['tok']:.2f}", f"{d['disp']:.4f}",
                                f"{d['gini']:.4f}", d["worst"], d["best"]] for d in rows]}}


# --- assemble the five chart specs ------------------------------------------
def build_specs() -> dict:
    fr = ifigs.spec_frontier()
    fr80 = {"panels": [p for p in fr["panels"] if p["title"].startswith("vocab 80k")],
            "table": {"cols": fr["table"]["cols"],
                      "rows": [r for r in fr["table"]["rows"] if r[0] == "80k"]}}
    dial = ifigs.spec_dial()
    dial1 = {"panels": [dial["panels"][0]], "table": dial["table"]}
    inter = ifigs.spec_interaction()
    inter1 = {"panels": [inter["panels"][0]], "table": inter["table"]}
    for spec, h in [(fr80, 430), (dial1, 420), (inter1, 420)]:
        for p in spec["panels"]:
            p["h"] = h
    comp = ifigs.spec_compass()
    comp["panels"][0]["h"] = 460
    return {"anatomy": spec_anatomy(), "compass": comp, "frontier80k": fr80,
            "dial": dial1, "interaction": inter1}


# --- the document -----------------------------------------------------------
def figure(key: str, caption: str) -> str:
    return f"""
<figure class="figblock" id="fb-{key}">
  <div class="chart" id="c-{key}" tabindex="0" aria-label="interactive chart"></div>
  <div class="tip" id="tip-{key}"></div>
  <figcaption>{caption}<span class="hint"> — hover a dot for its full record; ←/→ steps, Esc clears.</span></figcaption>
  <details><summary>data table</summary>
    <div style="overflow-x:auto"><table id="t-{key}"></table></div></details>
</figure>"""


PROSE = f"""
<h1>Where should the transitions go? Results of RQ2</h1>

<blockquote>
<p><strong>We found that parity removes a roughly constant fraction of the remaining
fairness gap as its budget increases — at 80k vocabulary, every additional
quarter of the merge budget given to parity halves the gap that remains. How
much budget that takes depends mainly on vocabulary size; languages, data,
and training samples enter only through the tokenizer's measured starting
disparity E. Compression, meanwhile, barely depends on either switch — it is
set by vocabulary size: within the safe regions the transitions cost under
1–2%, and a small parity ending actually <em>improves</em> compression, so training
with no parity at all is strictly dominated.</strong></p>
</blockquote>

<p>That is the whole result. The rest of this page unpacks it into three answers,
one table, and one procedure.</p>

<h2>The question</h2>

<p>A training run makes ~80,000 merge decisions and has two switches:
<strong>t1</strong> = when merges may start crossing spaces (BPE → SuperBPE), and
<strong>t2</strong> = when the objective changes from "most frequent overall" to "help the
worst-compressed language" (classical → parity). Where should they go, and in
what order? The answer came out in <strong>three different shapes</strong>.</p>

<p>One thing to notice first: this is really <strong>two questions in one</strong>. Where the
<em>safe regions</em> are is a question of fact, with one answer. But the exact
placement of t2 is a question of <em>preference</em>, because t2 is the
compression–fairness dial, <strong>t1 tells when it is safe to become SuperBPE;
t2 tells how much fairness you want to get.</strong> So t1 gets an answer, and t2
gets a menu plus a formula that turns a fairness target into a position.</p>

{figure("anatomy", '''<em>One 80k run, watched during training via truncation checkpoints. Each switch
owns one metric: t1 (green) acts on compression (top panel) and is invisible to
fairness; t2 (red) barely touches compression but triggers the disparity drop
(bottom panel, 1.40 → 1.09). This single picture is the mechanism — the paper's
Figure&nbsp;1 extended with the fairness panel it never had; everything below is
about where to put the two dashed lines.</em>''')}

<h2>Answer 1: t1 is a RANGE, but not really relevant</h2>

<p>Anywhere in <strong>10–50% of the budget (the total number of merges the run gets to
make)</strong> is within &lt;1% of the best compression, at every scale we measured
(8k → 80k; the paper's own 200k curve agrees). Only the extremes hurt: before
~1–2k merges the multi-byte characters aren't assembled yet; very late starves
the superword phase. t1 does <strong>nothing</strong> for fairness. It is not a
sensitive hyperparameter: pick 0.2·V and move on.</p>

{figure("compass", '''<em>Bytes/token vs t1, one line per vocabulary (8k–80k). The lines are nearly flat
(total spread under 1% per curve) which is the finding: on this axis there is
nothing to optimize. ★ (ringed dot) = best measured point per curve.</em>''')}

<h2>Answer 2: ordering is a RULE (parity goes last)</h2>

<p>Fairness has recency. In a budget-matched test at 80k (identical phase sizes,
only the <em>position</em> of the parity block moved) placing parity in the middle and
finishing classical <strong>erased the parity benefit completely</strong>: the final
tokenizer was 99.95% merge-identical to one trained with no parity at all.
Mechanism: early in a run, the two objectives pick nearly the same merges, and
a classical finish re-derives whatever parity preempted; only at the <strong>end</strong> of
a run does parity pick merges classical never would, and nothing comes after
to undo them. <strong>Either way, it is better to finish on parity.</strong></p>

<p>(This also settles the original BPE → parity → SuperBPE ordering: it stays a
legitimate <em>compression-first</em> menu item (in the June runs it sat on the
frontier's compressed end at every scale) but its middle parity block gets
partially re-derived by the classical finish, so it cannot reach the fairness
floor. Its strong June numbers came from its large parity share, one third of
the budget, not from the ordering; a well-placed parity <em>ending</em> buys more
fairness with a smaller share.)</p>

<h2>Answer 3: t2 is a DIAL</h2>

<p>Each t2 gives one tokenizer = one trade-off. The 80k menu (10 languages, FLORES;
lower is better on both axes; plain BPE shown for scale):</p>

<table class="menu">
<tr><th>configuration</th><th>tokens/line</th><th>disparity</th><th>reading</th></tr>
<tr><td>plain BPE (no superwords)</td><td>45.9</td><td>2.00</td><td>not in the race</td></tr>
<tr><td>t2 = 1.00 — no parity</td><td>33.77</td><td>1.366</td><td><strong>dominated (do not do this)</strong></td></tr>
<tr><td>t2 = 0.90 — last 10% parity</td><td><strong>33.66</strong></td><td>1.274</td><td><strong>compression optimum</strong></td></tr>
<tr><td>t2 = 0.75 — last 25%</td><td>33.79</td><td>1.202</td><td>the ~1.20 landmark</td></tr>
<tr><td>t2 = 0.60</td><td>34.14</td><td>1.148</td><td>balanced</td></tr>
<tr><td>t2 = 0.45</td><td>34.47</td><td>1.085</td><td>near the fairness floor</td></tr>
<tr><td>t2 = 0.30</td><td>34.69</td><td>1.075</td><td>paying a lot for very little</td></tr>
</table>

{figure("frontier80k", '''<em>The same menu, drawn: each dot is one finished 80k tokenizer, labeled by its
t2 in the tooltip. Down-left is better on both axes. Note the pale dot (t2 = 1.00,
no parity) hanging off the frontier: dominated.</em>''')}

<p>Three rules of thumb, stable at every scale we tested (8k/20k/50k/80k):</p>

<ol>
<li><strong>Never 0% parity.</strong> Giving parity the last ~10% improves compression <em>and</em>
fairness at once; the no-parity endpoint is strictly dominated (confirmed at
4 scales, 3 language subsets, 3 data sizes, and an independent data draw).</li>
<li><strong>Last ~25% of the budget → disparity ≈ 1.20</strong> at every vocabulary we
measured (the "one half-life" point). A vocab-free landmark. (Two scale-trends
cancel here: bigger vocabs start fairer but respond to parity more slowly; at
80k this point is exactly one half-life.)</li>
<li><strong>Last ~half → near the floor (~1.05)</strong>; beyond half is nearly pure cost.</li>
</ol>

{figure("dial", '''<em>Disparity vs t2, one curve per vocabulary. In the working range (t2 ≈ 0.6–0.9)
the four curves lie on top of each other — the dial is scale-invariant, which is
what makes the rules of thumb portable. Only the extremes move with vocab.</em>''')}

<h2>Using it on a new setting (the procedure)</h2>

<p>Nothing extra needs to be trained to apply this:</p>

<ol>
<li>Train the classical spine (BPE → SuperBPE, t1 ≈ 0.2·V, no parity): it is
needed anyway as phases 1–2 of the final tokenizer.</li>
<li>Evaluate it → that disparity <strong>is E</strong>, the setting's starting unfairness.
Languages, data size, and corpus draw all enter <em>only</em> through E.</li>
<li>Choose a fairness target d, read off (or compute) the parity share b.</li>
<li>Slice the spine at t2 = 1 − b (BPE is deterministic — a slice is a valid
run) and train only the parity tail. The measurement costs nothing.</li>
</ol>

<h2>How much to trust it</h2>

<p>~60 trained tokenizers across 4 vocab scales, 3 language subsets (3/6/10), and
3 data sizes; the full 21-cell (t1, t2) grid at 8k plus production-scale checks
at 80k show the two switches don't interact (the sliced design loses nothing);
key results reproduce out-of-domain (NTREX news) and on an <strong>independent
training-data draw</strong>; all headline deltas carry bootstrap CIs. One notable
bonus: the parity region of the menu is nearly invariant to the training draw —
parity absorbs data-sampling variance; only the no-parity baseline E moves.</p>

<p>And the effect is visible in the tokens themselves, not just the curves: with a
classical objective, ~62% of learned superwords are Latin-script; the parity
phase redistributes them (Latin → ~29%, Cyrillic 13→38%, Devanagari 7→23%).
Accordingly, the binding language changes: without parity the worst-treated
language is Thai or Russian; at the fairness floor it becomes German, i.e. the
weak scripts have been lifted until a high-resource European language is the
constraint. That is what "parity" looks like up close.</p>

{figure("interaction", '''<em>The separability evidence, visually: disparity vs t2, one line per t1 setting,
from the complete 21-combination grid. The lines lie on top of each other —
where you drop the whitespace wall does not change what the parity dial does.
This is what licenses sweeping the two switches independently.</em>''')}

<h2>Limits (volunteered up front)</h2>

<p>~1 MB/language training data (curves may sharpen with more); intrinsic metrics
only (no downstream LM); the vocab-scaling of the dial is fitted on 8k–80k —
using it at 150k+ is an extrapolation with one falsifiable prediction on record
(τ(150k) ≈ 0.39), testable when cluster/Rust compute is available.</p>

<hr>

<p class="appx"><em>Appendix, one line:</em> the dial is disparity(b) = 1.02 + (E − 1.02)·e<sup>−b/τ</sup>,
τ(V) = 0.037·log₂V − 0.25, with b = 1 − t2; fitted on 8k/20k/50k it predicts the
held-out 80k frontier with RMSE 0.010. Details: <code>rung2_synthesis.md</code> §8,
<code>transition_findings.md</code>.</p>
"""

SHELL = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ2 — Where should the transitions go?</title>
<style>
  :root { color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10); }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink);
         font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         padding: 32px 20px 60px; }
  .doc { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 26px; font-weight: 700; margin: 0 0 18px; line-height: 1.25; }
  h2 { font-size: 19px; font-weight: 650; margin: 30px 0 10px; }
  p { margin: 10px 0; }
  ol { margin: 10px 0 10px 24px; }
  li { margin: 6px 0; }
  blockquote { background: var(--surface); border-left: 4px solid #2a78d6;
               border-radius: 6px; padding: 14px 18px; margin: 16px 0; }
  blockquote p { margin: 6px 0; }
  hr { border: none; border-top: 1px solid var(--grid); margin: 28px 0 16px; }
  code { background: rgba(11,11,11,0.05); border-radius: 4px; padding: 1px 5px;
         font-size: 13px; }
  .appx { color: var(--ink-2); font-size: 14px; }
  table.menu { border-collapse: collapse; margin: 14px 0; font-size: 14px; }
  table.menu th, table.menu td { border: 1px solid var(--grid); padding: 5px 12px;
    text-align: left; }
  table.menu td:nth-child(2), table.menu td:nth-child(3) { text-align: right;
    font-variant-numeric: tabular-nums; }
  table.menu th { background: rgba(11,11,11,0.035); font-weight: 650; }
  .figblock { background: var(--surface); border: 1px solid var(--ring);
              border-radius: 10px; padding: 14px 14px 8px; margin: 18px 0;
              position: relative; }
  .figblock figcaption { color: var(--ink-2); font-size: 13px; margin: 6px 6px 8px; }
  .figblock .hint { color: var(--muted); }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 6px 6px; }
  .legend .it { display: flex; align-items: center; gap: 6px; font-size: 12.5px;
                color: var(--ink-2); }
  .legend .key { width: 16px; height: 0; border-top: 3px solid; border-radius: 2px; }
  .chart { outline: none; }
  svg { width: 100%; height: auto; display: block; }
  svg text { font: 11.5px system-ui, -apple-system, "Segoe UI", sans-serif;
             font-variant-numeric: tabular-nums; }
  .pt { font-size: 13px; color: var(--ink-2); font-weight: 600; margin: 4px 0 0 6px; }
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
  details { margin: 4px 6px 8px; }
  summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
  table:not(.menu) { border-collapse: collapse; margin-top: 8px; font-size: 12px;
                     font-variant-numeric: tabular-nums; }
  table:not(.menu) th, table:not(.menu) td { padding: 3px 9px; text-align: right;
    border-bottom: 1px solid var(--grid); }
  table:not(.menu) th:first-child, table:not(.menu) td:first-child { text-align: left; }
  table:not(.menu) th { color: var(--ink-2); font-weight: 600; }
</style></head>
<body><div class="doc">
__PROSE__
</div>
<script>
"use strict";
const SPECS = __SPECS__;

const W = 940, ML = 60, MR = 16, MT = 26, MB = 40;
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
const fx = (v, dec, big) => big ? (v / 1000) + "k" : v.toFixed(dec);

for (const key of Object.keys(SPECS)) {
  const S = SPECS[key];
  const mount = document.getElementById("c-" + key);
  const tip = document.getElementById("tip-" + key);
  const block = document.getElementById("fb-" + key);
  const PTS = [];
  let cur = -1;

  // legend
  const names = [];
  for (const P of S.panels) for (const s of P.series)
    if (s.name && !names.some(n => n[0] === s.name)) names.push([s.name, s.color, s.dash]);
  if (names.length > 1) {
    const lg = document.createElement("div"); lg.className = "legend";
    for (const [n, c, dash] of names) {
      const it = document.createElement("span"); it.className = "it";
      const k = document.createElement("span"); k.className = "key";
      k.style.borderTopColor = c; if (dash) k.style.borderTopStyle = "dashed";
      const t = document.createElement("span"); t.textContent = n;
      it.append(k, t); lg.append(it);
    }
    mount.append(lg);
  }

  S.panels.forEach((P, pi) => {
    const H = P.h || 360, IW = W - ML - MR, IH = H - MT - MB;
    if (P.title && S.panels.length > 1) {
      const t = document.createElement("div"); t.className = "pt";
      t.textContent = P.title; mount.append(t);
    }
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                            "aria-label": P.title || "chart" });
    mount.append(svg);

    const xs = P.series.flatMap(s => s.pts.map(p => p.x));
    const ys = P.series.flatMap(s => s.pts.map(p => p.y));
    let xlo = P.xdom ? P.xdom[0] : Math.min(...xs), xhi = P.xdom ? P.xdom[1] : Math.max(...xs);
    if (!P.xdom) { const px = (xhi - xlo) * 0.05 || 1; xlo -= px; xhi += px; }
    let ylo = Math.min(...ys), yhi = Math.max(...ys);
    { const py = (yhi - ylo) * 0.08 || 1; ylo -= py; yhi += py; }
    const X = v => ML + (v - xlo) / (xhi - xlo) * IW;
    const Y = v => MT + IH - (v - ylo) / (yhi - ylo) * IH;
    const bigX = xhi >= 5000;

    for (const b of (P.bands || [])) {
      svg.appendChild(el("rect", { x: X(b.x0), y: MT, width: X(b.x1) - X(b.x0),
        height: IH, fill: b.fill || "rgba(11,11,11,0.05)" }));
      if (b.label && (P.band_labels !== false)) {
        const top = b.label_pos === "top";
        const bt = el("text", { x: (X(b.x0) + X(b.x1)) / 2,
          y: top ? MT + 14 : MT + IH - 6, "text-anchor": "middle",
          fill: "#898781", "font-size": "10.5" });
        bt.textContent = b.label; svg.appendChild(bt);
      }
    }
    const ty = ticks(ylo, yhi, 5);
    for (const v of ty.vals) {
      svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: Y(v), y2: Y(v),
        stroke: "#e1e0d9", "stroke-width": 1 }));
      const lt = el("text", { x: ML - 7, y: Y(v) + 4, "text-anchor": "end",
        fill: "#898781" });
      lt.textContent = v.toFixed(ty.dec); svg.appendChild(lt);
    }
    svg.appendChild(el("line", { x1: ML, x2: W - MR, y1: MT + IH, y2: MT + IH,
      stroke: "#c3c2b7", "stroke-width": 1 }));
    const tx = ticks(xlo, xhi, 7);
    for (const v of tx.vals) {
      const lt = el("text", { x: X(v), y: MT + IH + 16, "text-anchor": "middle",
        fill: "#898781" });
      lt.textContent = fx(v, tx.dec, bigX); svg.appendChild(lt);
    }
    if (P.xlabel) {
      const xl = el("text", { x: ML + IW / 2, y: H - 6, "text-anchor": "middle",
        fill: "#52514e" });
      xl.textContent = P.xlabel; svg.appendChild(xl);
    }
    if (P.ylabel) {
      const yl = el("text", { x: 14, y: MT + IH / 2, fill: "#52514e",
        "text-anchor": "middle",
        transform: "rotate(-90 14 " + (MT + IH / 2) + ")" });
      yl.textContent = P.ylabel; svg.appendChild(yl);
    }
    for (const r of (P.vrules || [])) {
      svg.appendChild(el("line", { x1: X(r.x), x2: X(r.x), y1: MT, y2: MT + IH,
        stroke: r.color, "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
      if (P.vrule_labels) {
        const lt = el("text", { x: X(r.x) + 4, y: MT + 28, fill: r.color,
          "font-size": "11" });
        lt.textContent = r.label; svg.appendChild(lt);
      }
    }
    for (const s of P.series) {
      if (s.line !== false && s.pts.length > 1) {
        const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p.x) + " " + Y(p.y)).join("");
        svg.appendChild(el("path", { d, fill: "none", stroke: s.color,
          "stroke-width": 2, "stroke-linejoin": "round",
          "stroke-dasharray": s.dash ? "6 5" : "none" }));
      }
      if (s.markers !== false) s.pts.forEach((p, i) => {
        const best = s.star === i;
        const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: best ? 7 : 4.5,
          fill: p.c || s.color, stroke: best ? "#0b0b0b" : "none",
          "stroke-width": best ? 1.4 : 0 });
        svg.appendChild(c);
        if (p.d) PTS.push({ svg, s, p, cx: X(p.x), cy: Y(p.y), circle: c, best,
                            hpx: H });
      });
      if (s.end_label) {
        const last = s.pts[s.pts.length - 1];
        const lt = el("text", { x: X(last.x) + 8, y: Y(last.y) + 4, fill: s.color,
          "font-weight": "650", "font-size": "12" });
        lt.textContent = s.end_label; svg.appendChild(lt);
      }
    }
    svg.addEventListener("pointermove", e => {
      const r = svg.getBoundingClientRect();
      const mx = (e.clientX - r.left) / r.width * W;
      const my = (e.clientY - r.top) / r.height * (P.h || 360);
      let bi = -1, bd = Infinity;
      PTS.forEach((a, i) => { if (a.svg !== svg) return;
        const d = (a.cx - mx) ** 2 + (a.cy - my) ** 2; if (d < bd) { bd = d; bi = i; } });
      if (bi >= 0) { cur = bi; show(bi); }
    });
    svg.addEventListener("pointerleave", hide);
  });

  function show(i) {
    const a = PTS[i], d = a.p.d;
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
    PTS.forEach(b => { b.circle.setAttribute("r", b.best ? 7 : 4.5);
      if (!b.best) b.circle.setAttribute("stroke", "none"); });
    a.circle.setAttribute("r", 7);
    a.circle.setAttribute("stroke", "#fcfcfb"); a.circle.setAttribute("stroke-width", 2);
    const sr = a.svg.getBoundingClientRect(), br = block.getBoundingClientRect();
    const px = sr.left - br.left + a.cx / W * sr.width;
    const py = sr.top - br.top + a.cy / a.hpx * sr.height;
    tip.style.display = "block";
    const flip = px > br.width * 0.6;
    tip.style.left = flip ? (px - tip.offsetWidth - 14) + "px" : (px + 14) + "px";
    tip.style.top = Math.max(6, py - tip.offsetHeight / 2) + "px";
  }
  function hide() { tip.style.display = "none";
    PTS.forEach(b => { b.circle.setAttribute("r", b.best ? 7 : 4.5);
      if (!b.best) b.circle.setAttribute("stroke", "none"); }); }
  mount.addEventListener("keydown", e => {
    if (e.key === "Escape") { hide(); cur = -1; return; }
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    cur = e.key === "ArrowRight" ? Math.min(PTS.length - 1, cur < 0 ? 0 : cur + 1)
                                 : Math.max(0, cur < 0 ? 0 : cur - 1);
    show(cur);
  });

  const tbl = document.getElementById("t-" + key);
  const tr = document.createElement("tr");
  for (const h of S.table.cols) { const th = document.createElement("th");
    th.textContent = h; tr.append(th); }
  tbl.append(tr);
  for (const row of S.table.rows) {
    const r = document.createElement("tr");
    for (const v of row) { const td = document.createElement("td");
      td.textContent = v; r.append(td); }
    tbl.append(r);
  }
}
</script></body></html>
"""


def main():
    specs = build_specs()
    html = SHELL.replace("__PROSE__", PROSE).replace("__SPECS__", json.dumps(specs))
    OUT.write_text(html, encoding="utf-8")
    n = sum(len(s["panels"]) for s in specs.values())
    print(f"{len(specs)} figures ({n} panels) -> {OUT}")


if __name__ == "__main__":
    main()
