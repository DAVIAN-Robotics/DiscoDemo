/* Data scale as nested circles whose area is proportional to the count. The view starts on the robot
 * dataset and zooms out until the image-text and then the language dataset come into view, the way a
 * solar-system size comparison works. Counts are the ones stated in each paper:
 *   Open X-Embodiment (arXiv 2310.08864): "1M+ robot trajectories from 22 robot embodiments"
 *   LAION-5B (arXiv 2210.08402): "5.85 billion CLIP-filtered image-text pairs"
 *   Llama 3 (arXiv 2407.21783): "a flagship model with 405B trainable parameters on 15.6T text tokens"
 * The units differ (trajectories, pairs, tokens); the note under the circles says so.
 * window.ScaleZoom.build(host) returns {draw(ms)} so a frame at any time can be drawn directly. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var RINGS = [
    { n: 1e6, label: "1M+", unit: "robot trajectories", src: "Open X-Embodiment", cls: "robot" },
    { n: 5.85e9, label: "5.85B", unit: "image–text pairs", src: "LAION-5B", cls: "image" },
    { n: 15.6e12, label: "15.6T", unit: "text tokens", src: "Llama 3", cls: "text" }
  ];
  var W = 840, H = 340, FOCUS_R = 130, BOTTOM = H - 30, X0 = 250, LEGEND_X = 540;
  var HOLD = [1500, 1500, 2400], ZOOM = 1700; // ms: pause on each scale, and each pull-back between them
  var TOTAL = HOLD[0] + ZOOM + HOLD[1] + ZOOM + HOLD[2];
  var R = RINGS.map(function (g) { return Math.sqrt(g.n / RINGS[0].n); });   // radius relative to the robot circle
  function inOut(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }
  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  // Camera scale (screen px per robot-circle radius), interpolated in log space between the three views.
  function scaleAt(t) {
    var lvl = R.map(function (r) { return Math.log(FOCUS_R / r); });
    var a = HOLD[0], b = a + ZOOM, c = b + HOLD[1], d = c + ZOOM;
    if (t < a) return Math.exp(lvl[0]);
    if (t < b) return Math.exp(lvl[0] + (lvl[1] - lvl[0]) * inOut((t - a) / ZOOM));
    if (t < c) return Math.exp(lvl[1]);
    if (t < d) return Math.exp(lvl[1] + (lvl[2] - lvl[1]) * inOut((t - c) / ZOOM));
    return Math.exp(lvl[2]);
  }

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text != null) n.textContent = text;
    return n;
  }

  function build(host) {
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "sz", role: "img",
      "aria-label": "Circles with area proportional to dataset size: about a million robot trajectories (Open X-Embodiment), 5.85 billion image-text pairs (LAION-5B) and 15.6 trillion text tokens (Llama 3)." });
    // Drawn largest first so the smaller circles sit on top.
    var circles = RINGS.map(function (g) { return el("circle", { class: "sz-ring " + g.cls }); });
    circles.slice().reverse().forEach(function (c) { svg.appendChild(c); });
    // Legend on the right: one entry joins as each circle comes into view.
    var legend = RINGS.map(function (g, i) {
      var t = el("g", { transform: "translate(" + LEGEND_X + " " + (78 + i * 82) + ")" });
      t.appendChild(el("circle", { cx: -18, cy: -10, r: 8, class: "sz-ring " + g.cls }));
      t.appendChild(el("text", { class: "sz-num " + g.cls }, g.label));
      t.appendChild(el("text", { class: "sz-unit", dy: 22 }, g.unit));
      t.appendChild(el("text", { class: "sz-src", dy: 40 }, g.src));
      svg.appendChild(t);
      return t;
    });
    svg.appendChild(el("text", { x: W - 4, y: H - 8, "text-anchor": "end", class: "sz-note" },
      "Circle area ∝ count. Units differ: trajectories, image–text pairs, tokens."));
    host.innerHTML = "";
    host.appendChild(svg);
    // A larger circle shows up when the zoom-out towards it starts (drawn from the start it would read as a grey
    // backdrop, since the smaller circles sit inside it); its legend entry joins halfway through that zoom.
    var reveal = [0, HOLD[0], HOLD[0] + ZOOM + HOLD[1]];
    var appear = [0, HOLD[0] + ZOOM * 0.5, HOLD[0] + ZOOM + HOLD[1] + ZOOM * 0.5];
    function draw(t) {
      var s = scaleAt(t);
      circles.forEach(function (c, i) {
        var r = Math.max(R[i] * s, 2.2);   // a circle too small to see still shows as a dot
        c.setAttribute("cx", X0); c.setAttribute("cy", BOTTOM - r); c.setAttribute("r", r);
        c.setAttribute("opacity", i === 0 ? 1 : clamp01((t - reveal[i]) / 600));
      });
      legend.forEach(function (g, i) { g.setAttribute("opacity", clamp01((t - appear[i]) / 400)); });
    }
    return { draw: draw };
  }

  window.ScaleZoom = { build: build, duration: TOTAL };

  var host = document.querySelector('[data-fig="scale"]');
  if (!host) return;
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var fig = build(host), raf = 0;
  fig.draw(REDUCED ? TOTAL : 0);
  function play() {
    cancelAnimationFrame(raf);
    if (REDUCED) { fig.draw(TOTAL); return; }
    var t0 = performance.now();
    (function step(now) {
      var t = now - t0;
      fig.draw(Math.min(t, TOTAL));
      raf = t < TOTAL ? requestAnimationFrame(step) : 0;
    })(t0);
  }
  // Plays once, the first time the figure is properly in view; Replay runs it again, Skip jumps to the end.
  window.InView.once(host, play);
  var btn = host.parentNode.querySelector("[data-replay]");
  if (btn) {
    btn.addEventListener("click", play);
    window.addSkip(btn, function () { cancelAnimationFrame(raf); raf = 0; fig.draw(TOTAL); });
  }
})();
