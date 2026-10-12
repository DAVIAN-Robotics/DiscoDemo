/* DiscoDemo project page. Data comes from static/data/page-data.js (window.PAGE_DATA). */
(function () {
  "use strict";

  var D = window.PAGE_DATA;
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var TASKS = [
    { id: "banana", label: "PnP-Banana" },
    { id: "cube", label: "StackCube" },
    { id: "round", label: "FMB-Round" },
    { id: "sqcirc", label: "FMB-SqCircle" }
  ];
  // Paper order; colors match the paper's bar charts.
  var METHODS = {
    ours: { label: "Ours", short: "Ours", color: "var(--m-ours)" },
    rfcl: { label: "P-RFCL", short: "P-RFCL", color: "var(--m-rfcl)" },
    eg: { label: "ExpertGen", short: "EG", color: "var(--m-eg)" },
    pld: { label: "PLD", short: "PLD", color: "var(--m-pld)" },
    pgdg: { label: "PGDG", short: "PGDG", color: "var(--m-other)" },
    mg: { label: "MimicGen", short: "MG", color: "var(--m-other)" },
    mp: { label: "Motion planning", short: "MP", color: "var(--m-other)" },
    h20: { label: "20 human demos", short: "H20", color: "var(--m-h20)" }
  };
  // Paper order everywhere (tiles and bars): human, planning, augmentation, RL, then ours last.
  var ORDER = ["h20", "mp", "mg", "pgdg", "pld", "eg", "rfcl", "ours"];

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text != null) n.textContent = text;
    return n;
  }

  function segmented(container, options, selected, onPick) {
    container.innerHTML = "";
    var buttons = options.map(function (o) {
      var b = el("button", { type: "button", role: "tab", "aria-selected": String(o.value === selected) }, o.label);
      b.addEventListener("click", function () {
        buttons.forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
        onPick(o.value);
      });
      container.appendChild(b);
      return b;
    });
    if (container.dataset.role === "task") window.decorateTaskPicker(container, options);
  }

  /* ------------------------------------------------------------------
   * Synchronized video comparison player.
   * One clock drives every tile, so all methods start from the same
   * moment and a verdict appears when that rollout ends.
   * ------------------------------------------------------------------ */
  // Comparison grids: columns are the eight data sources, rows the scenes (or rollouts) of one task.
  // Each task is one pre-composited video (static/data/grid-data.js); the column heads, the success
  // rate and the per-tile verdicts are drawn over it at each tile's end time.
  var G = window.GRID_DATA;

  function GridPlayer(root) {
    this.root = root;
    this.mode = root.getAttribute("data-mode"); // real | sft | teacher
    this.task = "banana";
    this.speed = 2;
    this.playing = !REDUCED;
    this.visible = false;
    this.finished = false;
    this.near = false;
    this.box = root.querySelector('[data-role="grid"]');
    this.clock = root.querySelector('[data-role="clock"]');
    this.playBtn = root.querySelector('[data-role="play"]');
    this.build();
  }

  GridPlayer.prototype.build = function () {
    var self = this;
    segmented(this.root.querySelector('[data-role="task"]'),
      TASKS.map(function (t) { return { value: t.id, label: t.label }; }), this.task,
      function (v) { self.task = v; self.load(); });
    segmented(this.root.querySelector('[data-role="speed"]'),
      [1, 2, 4].map(function (s) { return { value: s, label: s + "×" }; }), this.speed,
      function (v) { self.speed = v; if (self.video) self.video.playbackRate = v; });
    this.playBtn.addEventListener("click", function () { self.setPlaying(!self.playing); });
    this.root.querySelector('[data-role="restart"]').addEventListener("click", function () { self.seek(0); self.setPlaying(true); });
    // Plays while it is properly in view and pauses off screen; coming back resumes where it was.
    window.InView(this.root, function () { self.visible = true; self.sync(); },
      function () { self.visible = false; self.sync(); });
    // The grid video is fetched a screen ahead, so playback does not start on an empty buffer.
    new IntersectionObserver(function (e) {
      if (!e[0].isIntersecting || self.near) return;
      self.near = true;
      if (self.video) self.video.preload = "auto";
    }, { rootMargin: "1500px 0px" }).observe(this.root);
    this.load();
    requestAnimationFrame(function tick() { self.frame(); requestAnimationFrame(tick); });
  };

  GridPlayer.prototype.load = function () {
    var self = this;
    var g = G[this.mode + "_" + this.task];
    var tw = g.tile[0], th = g.tile[1], gap = g.gap, cols = g.rows[0].length, rows = g.rows.length;
    var vw = cols * tw + (cols - 1) * gap, vh = rows * th + (rows - 1) * gap;
    var pct = function (x, of) { return (100 * x / of) + "%"; };
    this.g = g;
    this.box.innerHTML = "";
    this.box.className = "ggrid";

    var head = el("div", { "class": "ggrid-head" });
    var perf = this.mode !== "teacher";
    var R = D.results, ti = R.tasks.indexOf(this.task), sr = perf ? R[this.mode === "real" ? "real" : "sim"].sr[ti] : null;
    g.rows[0].forEach(function (t, c) {
      var h = el("div", { "class": "gh" + (t.m === "ours" ? " is-ours" : "") });
      h.style.left = pct(c * (tw + gap), vw);
      h.style.width = pct(tw, vw);
      h.style.setProperty("--c", METHODS[t.m].color);
      var nm = el("div", { "class": "nm" });
      nm.appendChild(el("i"));
      nm.appendChild(document.createTextNode(METHODS[t.m].short));
      h.appendChild(nm);
      if (perf) {
        var v = Math.round(100 * sr[R.methods.indexOf(t.m)]);
        var row = el("div", { "class": "sr", title: "Success rate on " + TASKS[ti].label });
        var bar = el("span", { "class": "bar" });
        var fill = el("span");
        fill.style.width = Math.max(v, 0.6) + "%";
        bar.appendChild(fill);
        // The number leads its bar: at the bar's far end it would sit against the next column's name.
        row.appendChild(el("b", null, v + "%"));
        row.appendChild(bar);
        h.appendChild(row);
      }
      head.appendChild(h);
    });

    var labels = el("div", { "class": "ggrid-rows" });
    g.rows.forEach(function (_, r) {
      var l = el("div", { "class": "gr" });
      l.style.top = pct(r * (th + gap), vh);
      l.style.height = pct(th, vh);
      l.appendChild(el("span", null, (self.mode === "teacher" ? "Rollout " : "Scene ") + (r + 1)));
      labels.appendChild(l);
    });

    var body = el("div", { "class": "ggrid-body" });
    body.style.aspectRatio = vw + " / " + vh;
    var video = el("video", { muted: "", playsinline: "", preload: this.near ? "auto" : "none",
      "aria-label": TASKS[ti].label + ": " + (this.mode === "teacher" ? "demonstrations" : "policy rollouts") + " from eight data sources" });
    video.muted = true;
    video.poster = g.src.replace(/\.mp4$/, ".jpg");
    video.src = g.src;
    video.playbackRate = this.speed;
    video.addEventListener("ended", function () { self.finished = true; self.setPlaying(false); self.reveal(); });
    body.appendChild(video);
    this.tiles = [];
    g.rows.forEach(function (row, r) {
      row.forEach(function (t, c) {
        var o = el("div", { "class": "gt" + (t.m === "ours" ? " is-ours" : "") });
        o.style.left = pct(c * (tw + gap), vw);
        o.style.top = pct(r * (th + gap), vh);
        o.style.width = pct(tw, vw);
        o.style.height = pct(th, vh);
        var chip = el("span", { "class": "chip" });
        o.appendChild(chip);
        body.appendChild(o);
        self.tiles.push({ t: t, o: o, chip: chip, done: false });
      });
    });
    this.box.appendChild(head);
    this.box.appendChild(labels);
    this.box.appendChild(body);
    this.video = video;
    this.seek(0);
    // Picking a task starts it from the beginning.
    if (!REDUCED) this.setPlaying(true);
  };

  GridPlayer.prototype.seek = function (t) {
    this.finished = false;
    try { this.video.currentTime = t; } catch (e) { /* metadata not loaded yet */ }
    this.tiles.forEach(function (x) { x.done = false; x.o.classList.remove("is-ok", "is-fail"); x.chip.textContent = ""; });
  };

  GridPlayer.prototype.setPlaying = function (on) {
    if (on && this.finished) this.seek(0);
    this.playing = on;
    var label = on ? "Pause" : this.finished ? "Replay" : "Play";
    this.playBtn.textContent = label;
    this.playBtn.setAttribute("aria-label", label);
    this.sync();
  };

  GridPlayer.prototype.sync = function () {
    var v = this.video;
    if (!v) return;
    if (this.playing && this.visible) { if (v.paused && !v.ended) window.playVideo(v).catch(function () {}); }
    else if (!v.paused) v.pause();
  };

  GridPlayer.prototype.frame = function () {
    if (!this.video) return;
    var t = this.video.currentTime;
    this.clock.textContent = (this.video.readyState < 2 && this.playing && this.visible ? "Loading…" : Math.min(t, this.g.total).toFixed(1) + " s");
    this.reveal();
  };

  // Each tile's verdict appears at its end time: the measured run time for the real robot, the success
  // step for simulated policies, and the end of the clip for generated demonstrations.
  GridPlayer.prototype.reveal = function () {
    var t = this.finished ? Infinity : this.video.currentTime, teacher = this.mode === "teacher";
    this.tiles.forEach(function (x) {
      if (x.done || t < x.t.end) return;
      x.done = true;
      x.o.classList.add(x.t.success ? "is-ok" : "is-fail");
      if (!teacher) x.chip.textContent = x.t.success ? "Success " + x.t.end.toFixed(1) + " s" : "Failed";
    });
  };

  document.querySelectorAll(".player").forEach(function (p) { new GridPlayer(p); });

  /* ------------------------------------------------------------------
   * Results chart (same values as the paper's main figure).
   * ------------------------------------------------------------------ */
  var R = D.results;
  var CHART_ORDER = ORDER;
  var chartRoot = document.getElementById("results-chart");
  var state = { setting: "sim" };
  var SVGNS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, text) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text != null) n.textContent = text;
    return n;
  }

  // Metrics as in the paper's main table: SR (%), Time (s, lower is better), Succ./h.
  var METRIC = {
    sr: { label: "Success rate", unit: "%", higher: true },
    time_s: { label: "Completion time", unit: " s", higher: false },
    succ_h: { label: "Succ./h", unit: "/h", higher: true }
  };

  function series(setting, metric, ti) {
    var src = R[setting];
    return CHART_ORDER.map(function (m) {
      var mi = R.methods.indexOf(m);
      var v = src[metric][ti][mi];
      var sd = setting === "sim" ? src[metric + "_sd"][ti][mi] : null;
      if (metric === "sr") { v *= 100; if (sd != null) sd *= 100; }
      return { m: m, v: v, sd: sd, noSuccess: metric === "time_s" && src.sr[ti][mi] === 0 };
    });
  }

  // Mean over the four tasks. For time, a task with no success counts at its horizon, as in the paper.
  function average(setting, metric) {
    return CHART_ORDER.map(function (m) {
      var mi = R.methods.indexOf(m);
      var s = 0;
      for (var ti = 0; ti < TASKS.length; ti++) s += R[setting][metric][ti][mi];
      var v = s / TASKS.length;
      return { m: m, v: metric === "sr" ? v * 100 : v, sd: null, noSuccess: false };
    });
  }

  // Paper order within each panel: worst on the left, best on the right.
  function ranked(data, metric) {
    var up = METRIC[metric].higher;
    return data.slice().sort(function (a, b) { return up ? a.v - b.v : b.v - a.v; });
  }

  function niceMax(v) {
    var step = v > 200 ? 100 : v > 60 ? 50 : 10;
    return Math.ceil(v * 1.12 / step) * step;
  }

  var bars = [];   // rects and labels of the current draw, animated on reveal

  function panel(title, data, ymax, unit, isAvg, pi) {
    var W = 150, H = 138, top = 14, bottom = 30, left = 2;
    var slot = (W - left) / data.length;
    var bw = slot * 0.72;
    var box = el("div", { "class": "panel" + (isAvg ? " avg" : "") });
    box.appendChild(el("h4", null, title));
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": title });
    var plotH = H - top - bottom;
    [0, 0.5, 1].forEach(function (f) {
      var y = top + plotH * (1 - f);
      s.appendChild(svg("line", { x1: 0, x2: W, y1: y, y2: y, "class": "ch-grid", stroke: "#e3e6ea", "stroke-width": f === 0 ? 1 : 0.6 }));
    });
    data.forEach(function (d, i) {
      var x = left + i * slot + (slot - bw) / 2;
      var h = plotH * Math.min(d.v, ymax) / ymax;
      var y = top + plotH - h;
      var g = svg("g");
      var fill = d.noSuccess ? "#e9ebef" : METHODS[d.m].color;
      var rect = svg("rect", { x: x, y: y, width: bw, height: Math.max(h, 0.8), rx: 1.5, fill: fill, "class": d.noSuccess ? "ch-bar ch-empty" : "ch-bar" });
      g.appendChild(rect);
      var after = svg("g");
      if (d.sd) {
        var cx = x + bw / 2;
        var y1 = top + plotH * (1 - Math.min(d.v + d.sd, ymax) / ymax);
        var y2 = top + plotH * (1 - Math.max(d.v - d.sd, 0) / ymax);
        after.appendChild(svg("line", { x1: cx, x2: cx, y1: y1, y2: y2, "class": "ch-err", stroke: "#1f2937", "stroke-width": 0.9 }));
      }
      var label = d.noSuccess ? "–" : String(Math.round(d.v));
      after.appendChild(svg("text", {
        x: x + bw / 2, y: Math.min(y, d.sd ? top + plotH * (1 - Math.min(d.v + d.sd, ymax) / ymax) : y) - 3,
        "text-anchor": "middle", "font-size": 9, "font-weight": d.m === "ours" ? 700 : 500,
        "class": "ch-val" + (d.m === "ours" ? " is-ours" : ""), fill: d.m === "ours" ? "#a8326b" : "#1f2937"
      }, label));
      g.appendChild(after);
      var lx = x + bw / 2 + 2, ly = H - bottom + 7;
      g.appendChild(svg("text", {
        x: lx, y: ly, "text-anchor": "end", "font-size": 8, "class": "ch-lab" + (d.m === "ours" ? " is-ours" : ""), fill: d.m === "ours" ? "#a8326b" : "#5b6472",
        "font-weight": d.m === "ours" ? 700 : 400, transform: "rotate(-55 " + lx + " " + ly + ")"
      }, METHODS[d.m].short));
      var tip = METHODS[d.m].label + ": " + (d.noSuccess ? "no successful episode" : d.v.toFixed(1) + unit) +
        (d.sd ? " ± " + d.sd.toFixed(1) : "");
      g.appendChild(svg("title", null, tip));
      s.appendChild(g);
      bars.push({ rect: rect, after: after, delay: pi * 70 + i * 55 });
    });
    box.appendChild(s);
    return box;
  }

  // Bars rise left to right within each panel, panels slightly staggered; labels follow each bar.
  function animateBars(fast) {
    var k = fast ? 0.5 : 1;
    bars.forEach(function (b) {
      b.rect.style.transformBox = "fill-box";
      b.rect.style.transformOrigin = "50% 100%";
      if (REDUCED) return;
      b.rect.animate([{ transform: "scaleY(0)" }, { transform: "none" }],
        { delay: b.delay * k, duration: 520 * k, easing: "cubic-bezier(.2,.7,.2,1)", fill: "both" });
      b.after.animate([{ opacity: 0 }, { opacity: 1 }], { delay: (b.delay + 380) * k, duration: 260, fill: "both" });
    });
  }

  // All three metrics at once, one row each (SR, time, Succ./h); only the setting is a toggle.
  var ROW_HEAD = {
    sr: "Success rate (%), higher is better",
    time_s: "Time to first success (s), lower is better",
    succ_h: "Successes per hour, higher is better"
  };
  function drawChart(animate) {
    var chart = chartRoot.querySelector('[data-role="chart"]');
    var note = chartRoot.querySelector('[data-role="note"]');
    chart.innerHTML = "";
    bars = [];
    Object.keys(METRIC).forEach(function (metric, r) {
      var M = METRIC[metric];
      var row = el("div", { "class": "chart-row" });
      row.appendChild(el("h4", { "class": "chart-metric" }, ROW_HEAD[metric]));
      var grid = el("div", { "class": "chart", role: "img",
        "aria-label": (state.setting === "real" ? "Real robot " : "Simulation ") + M.label +
          " per data source and task, sorted worst to best" });
      var all = TASKS.map(function (t, ti) { return series(state.setting, metric, ti); });
      var avg = average(state.setting, metric);
      TASKS.forEach(function (t, ti) {
        var ymax = metric === "sr" ? 100 : metric === "time_s" ? R.horizon_s[ti]
          : niceMax(Math.max.apply(null, all[ti].map(function (d) { return d.v + (d.sd || 0); })));
        grid.appendChild(panel(t.label, ranked(all[ti], metric), ymax, M.unit, false, r * 6 + ti));
      });
      var amax = metric === "sr" ? 100 : metric === "time_s" ? 30
        : niceMax(Math.max.apply(null, avg.map(function (d) { return d.v; })));
      grid.appendChild(panel("Average", ranked(avg, metric), amax, M.unit, true, r * 6 + TASKS.length));
      row.appendChild(grid);
      chart.appendChild(row);
    });
    note.textContent = [state.setting === "sim"
      ? "Mean ± s.d. over three fine-tuning seeds, 50 episodes each."
      : "First fine-tuning seed, 20 real-robot trials per task; generated-data policies transfer zero-shot.",
      "Time averages successful episodes (full horizon if none succeed); successes/hour counts failures at the full 20–30 s horizon. Scene-reset time is excluded."].join(" ");
    if (animate) animateBars(animate === "fast");
  }

  segmented(chartRoot.querySelector('[data-role="setting"]'),
    [{ value: "sim", label: "Simulation" }, { value: "real", label: "Real robot" }], state.setting,
    function (v) { state.setting = v; drawChart("fast"); });
  drawChart(false);
  // Every bar rises once, when the chart is first properly in view; until then the bars wait at zero.
  function lower() {
    bars.forEach(function (b) { b.rect.style.transformBox = "fill-box"; b.rect.style.transformOrigin = "50% 100%";
      b.rect.style.transform = "scaleY(0)"; b.after.style.opacity = "0"; });
  }
  lower();
  window.InView.once(chartRoot, function () { drawChart(true); });

  var key = document.getElementById("method-key");
  CHART_ORDER.slice().reverse().forEach(function (m) {
    var s = el("span");
    s.style.setProperty("--c", METHODS[m].color);
    if (m === "ours") s.appendChild(el("b", null, METHODS[m].short));
    else s.textContent = METHODS[m].short + (METHODS[m].short !== METHODS[m].label ? " = " + METHODS[m].label : "");
    key.appendChild(s);
  });
  key.removeAttribute("aria-hidden");

  /* ------------------------------------------------------------------
   * BibTeX copy.
   * ------------------------------------------------------------------ */
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var text = document.getElementById(b.getAttribute("data-copy")).textContent;
      navigator.clipboard.writeText(text).then(function () {
        b.textContent = "Copied";
        setTimeout(function () { b.textContent = "Copy"; }, 1600);
      }, function () { b.textContent = "Select and copy"; });
    });
  });
})();
