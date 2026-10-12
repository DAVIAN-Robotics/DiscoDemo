/* "What the generated demonstrations look like": the gripper (TCP) paths of the paper's qualitative
 * figure, replayed over the scene in time. Each source has its successful rollouts from one fixed
 * placement; every path advances at the robot's own pace (sped up), so short, direct generators finish
 * first and the spread of each source builds up as the paths land. Data: static/traj/<task>.js, built
 * from the same selection the paper figure uses. */
(function () {
  "use strict";
  var root = document.getElementById("traj");
  if (!root) return;
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var TR = window.FigCore.TRAIL;
  // x real time; FMB-SqCircle's slow insertions play a little faster.
  var SPEEDS = { banana: 3, cube: 3, round: 3, sqcirc: 4 };
  var note = root.querySelector(".replay-row .tele-fact");
  function speed() { return SPEEDS[task]; }
  function showSpeed() { if (note) note.innerHTML = note.innerHTML.replace(/at \d+&times; speed|at \d+× speed/, "at " + speed() + "&times; speed"); }
  var TASKS = [["banana", "PnP-Banana"], ["cube", "StackCube"], ["round", "FMB-Round"], ["sqcirc", "FMB-SqCircle"]];
  var LABEL = { mp: "MP", mg: "MG", pgdg: "PGDG", pld: "PLD", eg: "EG", rfcl: "P-RFCL", ours: "Ours" };
  var grid = root.querySelector("[data-role=grid]");
  var seg = root.querySelector("[data-role=task]");
  var task = "banana", data = null, panels = [], t0 = 0, raf = 0, visible = false, bg = null;

  function load(t, done) {
    if (window.TRAJ && window.TRAJ[t]) { done(window.TRAJ[t]); return; }
    var s = document.createElement("script");
    s.src = "static/traj/" + t + ".js";
    s.onload = function () { done(window.TRAJ[t]); };
    document.head.appendChild(s);
  }

  function build(d) {
    data = d;
    grid.innerHTML = "";
    panels = d.order.map(function (m) {
      var fig = document.createElement("figure");
      fig.className = "traj-panel" + (m === "ours" ? " is-ours" : "");
      var c = document.createElement("canvas");
      c.width = d.width; c.height = d.height;
      var cap = document.createElement("figcaption");
      cap.textContent = LABEL[m];
      fig.appendChild(c); fig.appendChild(cap);
      grid.appendChild(fig);
      return { m: m, ctx: c.getContext("2d"), cell: d.cells[m] };
    });
    bg = new Image();
    bg.onload = function () { draw(REDUCED ? Infinity : 0); };
    bg.src = d.bg;
  }

  // Path up to (fractional) sample index k, broken where the source figure hides a stretch (null).
  function trace(ctx, pts, k) {
    var n = Math.min(pts.length - 1, Math.floor(k)), f = k - n, open = false, last = null;
    for (var i = 0; i <= n; i++) {
      var p = pts[i];
      if (!p) { open = false; continue; }
      if (!open) { ctx.moveTo(p[0], p[1]); open = true; } else ctx.lineTo(p[0], p[1]);
      last = p;
    }
    var nx = pts[n + 1];
    if (open && nx && f > 0 && last) {
      last = [last[0] + (nx[0] - last[0]) * f, last[1] + (nx[1] - last[1]) * f];
      ctx.lineTo(last[0], last[1]);
    }
    return last;
  }

  function draw(sec) {
    var k = sec * data.hz, running = false;
    panels.forEach(function (p) {
      var ctx = p.ctx;
      // Line widths are set in screen pixels: the canvas holds the source resolution and is shown smaller.
      var cw = Math.max(1, ctx.canvas.clientWidth), px = data.width / cw * TR.scale(cw);
      ctx.clearRect(0, 0, data.width, data.height);
      ctx.drawImage(bg, 0, 0, data.width, data.height);
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      // White halo under the paths, as in the paper figure (style shared: FigCore.TRAIL).
      ctx.strokeStyle = TR.halo; ctx.lineWidth = TR.haloW * px;
      ctx.beginPath();
      p.cell.trajs.forEach(function (pts) { trace(ctx, pts, k); });
      ctx.stroke();
      ctx.strokeStyle = p.cell.color; ctx.globalAlpha = TR.alpha; ctx.lineWidth = TR.line * px;
      ctx.beginPath();
      var heads = [];
      p.cell.trajs.forEach(function (pts) {
        var h = trace(ctx, pts, k);
        if (k < pts.length - 1) { running = true; if (h) heads.push(h); }
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = p.cell.color;
      heads.forEach(function (h) { ctx.beginPath(); ctx.arc(h[0], h[1], TR.dot * px, 0, Math.PI * 2); ctx.fill(); });
    });
    return running;
  }

  function frame(now) {
    var running = draw((now - t0) / 1000 * speed());
    raf = running ? requestAnimationFrame(frame) : 0;
  }
  function play() {
    if (!data || !bg || !bg.complete) return;
    cancelAnimationFrame(raf);
    if (REDUCED) { draw(Infinity); return; }
    t0 = performance.now();
    raf = requestAnimationFrame(frame);
  }

  TASKS.forEach(function (t, i) {
    var b = document.createElement("button");
    b.type = "button"; b.textContent = t[1]; b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", function () {
      seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
      task = t[0];
      showSpeed();
      load(task, function (d) { build(d); bg.addEventListener("load", function () { if (visible) play(); }); });
    });
    seg.appendChild(b);
  });
  window.decorateTaskPicker(seg, TASKS.map(function(t) { return {value:t[0], label:t[1]}; }));
  // Data for the first task comes in a screen ahead; the paths start once the figure is in view.
  var loaded = false;
  new IntersectionObserver(function (e) {
    if (e[0].isIntersecting && !loaded) { loaded = true; load(task, build); }
  }, { rootMargin: "800px 0px" }).observe(root);
  // Plays once, the first time the figure is in view, and runs to the end even if scrolled past.
  window.InView.once(root, function () {
    visible = true;
    if (bg && bg.complete) play(); else load(task, function (d) { if (!data) build(d); bg.addEventListener("load", play); });
  });
  root.querySelector("[data-replay]").addEventListener("click", play);
  window.addSkip(root.querySelector("[data-replay]"), function () {
    if (data && bg && bg.complete) { cancelAnimationFrame(raf); raf = 0; draw(Infinity); }
  });
})();
