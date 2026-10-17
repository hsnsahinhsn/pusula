// Pusula project page: language choice and the results chart. No external requests, no cookies.
(() => {
  "use strict";
  const KEY = "pusula-lang";
  const store = {
    get() { try { return localStorage.getItem(KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(KEY, v); } catch { /* storage disabled */ } },
  };

  // Root page: open the page in the browser's language (Turkish browsers get Turkish, all others English).
  // A language picked with the switch link wins over the browser language.
  if (document.documentElement.dataset.root === "1") {
    let lang = store.get();
    if (lang !== "tr" && lang !== "en") {
      const first = (navigator.languages && navigator.languages[0]) || navigator.language || "";
      lang = /^tr\b/i.test(first) ? "tr" : "en";
    }
    location.replace(lang + "/" + location.hash);
    return;
  }

  for (const a of document.querySelectorAll("a[data-lang]")) {
    a.addEventListener("click", () => store.set(a.dataset.lang));
  }

  // Results chart: Decision 3.0 -> Pusula per evaluation (dumbbell), built as inline SVG.
  const host = document.querySelector("[data-chart]");
  const src = document.getElementById("chart-data");
  if (!host || !src) return;
  const data = JSON.parse(src.textContent);
  const fmt = new Intl.NumberFormat(document.documentElement.lang, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const NS = "http://www.w3.org/2000/svg";
  const [lo, hi] = data.domain;
  const ROW = 58, AXIS = 26, PAD_R = 44;
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  host.appendChild(tip);

  const el = (name, attrs, parent) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const cssVar = (name) => getComputedStyle(host).getPropertyValue(name).trim();

  function showTip(row, x, y) {
    tip.replaceChildren();
    const title = document.createElement("p");
    title.className = "t-title";
    title.textContent = row.label;
    tip.appendChild(title);
    for (const [name, value, color] of [[data.series[1], row.pusula, cssVar("--pusula")], [data.series[0], row.base, cssVar("--base")]]) {
      const line = document.createElement("div");
      line.className = "t-row";
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = color;
      const v = document.createElement("strong");
      v.textContent = fmt.format(value);
      const s = document.createElement("span");
      s.textContent = name;
      line.append(key, v, s);
      tip.appendChild(line);
    }
    tip.hidden = false;
    const w = host.clientWidth, tw = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(x - tw / 2, w - tw - 8)) + "px";
    tip.style.top = y + "px";
  }

  function render() {
    const svgOld = host.querySelector("svg");
    if (svgOld) svgOld.remove();
    const W = Math.max(280, host.clientWidth - 40);
    const H = data.rows.length * ROW + AXIS;
    const plotW = W - PAD_R;
    const sx = (v) => ((v - lo) / (hi - lo)) * plotW;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "group", "aria-label": data.title });
    host.insertBefore(svg, tip);

    for (let t = lo; t <= hi; t += data.step) {
      el("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: 0, y2: H - AXIS }, svg);
      const label = el("text", { class: "tick", x: sx(t), y: H - 6, "text-anchor": "middle" }, svg);
      label.textContent = fmt.format(t).replace(/[.,]0$/, "");
    }
    el("line", { class: "baseline", x1: 0, x2: plotW, y1: H - AXIS, y2: H - AXIS }, svg);

    data.rows.forEach((row, i) => {
      const y0 = i * ROW, cy = y0 + 38;
      const g = el("g", {
        class: "row", tabindex: 0, role: "img",
        "aria-label": `${row.label}: ${data.series[0]} ${fmt.format(row.base)}, ${data.series[1]} ${fmt.format(row.pusula)}`,
      }, svg);
      el("rect", { class: "hit", x: -6, y: y0 + 2, width: W + 6, height: ROW - 4, rx: 6 }, g);
      const name = el("text", { class: "row-label", x: 0, y: y0 + 18 }, g);
      name.textContent = row.label;
      const xb = sx(row.base), xp = sx(row.pusula);
      el("line", { class: "link", x1: Math.min(xb, xp), x2: Math.max(xb, xp), y1: cy, y2: cy }, g);
      el("circle", { class: "d-base", cx: xb, cy, r: 6 }, g);
      el("circle", { class: "d-pusula", cx: xp, cy, r: 7 }, g);
      const val = el("text", { class: "val", x: Math.max(xb, xp) + 12, y: cy + 4 }, g);
      val.textContent = fmt.format(row.pusula);
      const at = () => {
        const r = svg.getBoundingClientRect(), h = host.getBoundingClientRect();
        showTip(row, r.left - h.left + xp, r.top - h.top + cy + 14);
      };
      g.addEventListener("pointerenter", at);
      g.addEventListener("focus", at);
      g.addEventListener("pointerleave", () => { tip.hidden = true; });
      g.addEventListener("blur", () => { tip.hidden = true; });
    });
  }

  render();
  let raf = 0, lastW = host.clientWidth;
  new ResizeObserver(() => {
    if (host.clientWidth === lastW) return;
    lastW = host.clientWidth;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(render);
  }).observe(host);
})();
