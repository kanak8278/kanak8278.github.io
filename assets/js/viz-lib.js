/* ─────────────────────────────────────────────────────────────────────────
   viz-lib.js — reusable toolkit for interactive showcase pages
   (layout: showcase). Vanilla JS + hand-rolled SVG, no dependencies.

   Exposes window.Viz with DOM helpers, a tooltip, a debounce util, and two
   chart primitives (lineChart, groupedBar) that read their colors from the
   page's `.showcase` CSS custom properties — so a new showcase page gets
   on-theme, dark-mode-aware charts for free just by using this toolkit.
   ───────────────────────────────────────────────────────────────────────── */

(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pct = (x) => Math.round(x * 100);
  const J = (p) => fetch(p).then(r => r.json());
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  function getCSS(v, root) {
    const el = root || document.querySelector(".showcase") || document.documentElement;
    return getComputedStyle(el).getPropertyValue(v).trim();
  }

  function el(tag, attrs = {}, kids = []) {
    const ns = ["svg", "g", "path", "line", "circle", "rect", "text", "polyline", "tspan"].includes(tag);
    const n = ns ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.setAttribute("class", v);
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v);
    }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => c && n.appendChild(c));
    return n;
  }

  // shared tooltip — expects a `<div class="sc-tip">` somewhere on the page
  function showTip(html, x, y) {
    const tip = $(".sc-tip");
    if (!tip) return;
    tip.innerHTML = html;
    tip.style.opacity = 1;
    tip.style.left = clamp(x + 14, 8, innerWidth - 250) + "px";
    tip.style.top = (y - 10) + "px";
  }
  function hideTip() { const tip = $(".sc-tip"); if (tip) tip.style.opacity = 0; }

  function legItem(color, label) {
    return el("span", {}, [el("span", { class: "dot", style: `background:${color}` }), document.createTextNode(label)]);
  }

  // ── shared line chart ──
  function lineChart(host, { series, nLayers, yMax = 1, yLabel, xLabel, markerLayer, chance, xStart = 0, tickLabel }) {
    const COL = { muted: getCSS("--muted"), faint: getCSS("--faint"), border: getCSS("--border"),
      borderStrong: getCSS("--border-strong"), ink: getCSS("--ink"), accent: getCSS("--accent") };
    const xtick = tickLabel || (i => "L" + i);
    host.replaceChildren();
    const W = 880, H = 420, m = { t: 20, r: 20, b: 48, l: 56 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const span = Math.max(1, (nLayers - 1) - xStart);
    const x = i => m.l + ((i - xStart) / span) * iw;
    const y = v => m.t + (1 - v / yMax) * ih;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-height:440px" });

    for (let i = 0; i <= 4; i++) {
      const t = (yMax * i) / 4;
      svg.appendChild(el("line", { class: "gridline", x1: m.l, y1: y(t), x2: m.l + iw, y2: y(t) }));
      svg.appendChild(el("text", { x: m.l - 10, y: y(t) + 4, "text-anchor": "end",
        "font-size": 11, fill: COL.muted, text: pct(t / yMax * yMax) + "%" }));
    }
    const step = Math.max(1, Math.round((nLayers - xStart) / 6));
    for (let i = xStart; i < nLayers; i += step) {
      svg.appendChild(el("text", { x: x(i), y: m.t + ih + 20, "text-anchor": "middle",
        "font-size": 11, fill: COL.muted, text: String(xtick(i)) }));
    }
    if (chance != null) {
      svg.appendChild(el("line", { x1: m.l, y1: y(chance), x2: m.l + iw, y2: y(chance),
        stroke: COL.faint, "stroke-width": 1, "stroke-dasharray": "4 4" }));
      svg.appendChild(el("text", { x: m.l + iw, y: y(chance) - 6, "text-anchor": "end",
        "font-size": 11, fill: COL.faint, text: "chance" }));
    }
    if (xLabel) svg.appendChild(el("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle",
      "font-size": 12, fill: COL.muted, text: xLabel }));
    if (yLabel) svg.appendChild(el("text", { x: -(m.t + ih / 2), y: 14, "text-anchor": "middle",
      transform: "rotate(-90)", "font-size": 12, fill: COL.muted, text: yLabel }));

    let marker = null;
    if (markerLayer != null) {
      marker = el("line", { x1: x(markerLayer), y1: m.t, x2: x(markerLayer), y2: m.t + ih,
        stroke: COL.accent, "stroke-width": 1.5, "stroke-opacity": .5 });
      svg.appendChild(marker);
    }

    series.forEach(s => {
      if (!s.data) return;
      const pts = s.data.map((v, i) => [i, v])
        .filter(([i, v]) => i >= xStart && v != null && isFinite(v))
        .map(([i, v]) => `${x(i)},${y(v)}`).join(" ");
      svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: s.color,
        "stroke-width": s.width || 2.5, "stroke-dasharray": s.dash || "",
        "stroke-opacity": s.opacity == null ? 1 : s.opacity,
        "stroke-linejoin": "round" }));
    });

    const dots = [];
    if (markerLayer != null) series.forEach(s => {
      if (!s.data) return;
      const d = el("circle", { cx: x(markerLayer), cy: y(s.data[markerLayer]), r: 4.5,
        fill: s.color, stroke: "#fff", "stroke-width": 1.5,
        "fill-opacity": s.opacity == null ? 1 : s.opacity });
      svg.appendChild(d); dots.push({ s, d });
    });

    host.appendChild(svg);
    return { x, y, marker, dots, svg };
  }

  // ── shared grouped bar chart ──
  function groupedBar(host, { groups, yMax = 1, yLabel, barColors }) {
    const COL = { muted: getCSS("--muted"), ink: getCSS("--ink") };
    host.replaceChildren();
    const W = 880, H = 380, m = { t: 18, r: 16, b: 52, l: 48 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const y = v => m.t + (1 - v / yMax) * ih;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-height:400px" });
    for (let i = 0; i <= 4; i++) {
      const t = (yMax * i) / 4;
      svg.appendChild(el("line", { class: "gridline", x1: m.l, y1: y(t), x2: m.l + iw, y2: y(t) }));
      svg.appendChild(el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: COL.muted, text: pct(t / yMax) + "%" }));
    }
    const gw = iw / groups.length;
    groups.forEach((g, gi) => {
      const n = g.bars.length, pad = gw * 0.18, bw = (gw - 2 * pad) / n;
      g.bars.forEach((b, bi) => {
        const x = m.l + gi * gw + pad + bi * bw;
        svg.appendChild(el("rect", { x: x + 2, y: y(b.v), width: bw - 4, height: Math.max(0, ih - (y(b.v) - m.t)),
          rx: 3, fill: b.color, "fill-opacity": b.opacity == null ? 1 : b.opacity }));
        svg.appendChild(el("text", { x: x + bw / 2, y: y(b.v) - 5, "text-anchor": "middle", "font-size": 10,
          fill: COL.muted, text: pct(b.v) + "%" }));
      });
      svg.appendChild(el("text", { x: m.l + gi * gw + gw / 2, y: m.t + ih + 20, "text-anchor": "middle",
        "font-size": 12, fill: COL.ink, "font-weight": 600, text: g.label }));
    });
    host.appendChild(svg);
  }

  // ── scroll reveal ──
  function setupReveal(selector = "section .card, .section-head, .key-insight") {
    const nodes = $$(selector);
    if (!("IntersectionObserver" in window)) { nodes.forEach(n => n.classList.add("in")); return; }
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); } });
    }, { threshold: 0.08 });
    nodes.forEach(n => { n.classList.add("reveal"); obs.observe(n); });
    setTimeout(() => nodes.forEach(n => n.classList.add("in")), 1500);
  }

  window.Viz = { $, $$, el, clamp, pct, J, debounce, getCSS, showTip, hideTip, legItem, lineChart, groupedBar, setupReveal };
})();
