/* Generator clips on the "how" question, as in the intro video: the three start together at 2x, each
 * holds its last frame, and once all have finished they start over together. The gripper path (TCP) is
 * traced over each clip as it plays, in the colour each source has in the trajectory figure.
 * The data-scale figure lives in scale.js; the teleoperation clip, rollout wall and real/sim comparison in opening.js. */
(function () {
  "use strict";
  var C = window.FigCore;
  var T = window.TCP_DATA;
  var TR = C.TRAIL;  // same path style as the trajectory figure
  var SPEED = 2, PAUSE_MS = 1500;
  var figs = [].slice.call(document.querySelectorAll(".gen"));
  if (!figs.length) return;

  // Trace the path up to clip time t: white halo, path in the source colour, and a dot at the gripper.
  function trail(ctx, pts, color, t) {
    var k = Math.min(pts.length - 1, t * T.fps), n = Math.floor(k), f = k - n;
    var cw = Math.max(1, ctx.canvas.clientWidth), px = T.w / cw * TR.scale(cw);
    ctx.clearRect(0, 0, T.w, T.h);
    function path() {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (var i = 1; i <= n; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      var head = pts[n];
      if (n + 1 < pts.length && f > 0) {
        head = [pts[n][0] + (pts[n + 1][0] - pts[n][0]) * f, pts[n][1] + (pts[n + 1][1] - pts[n][1]) * f];
        ctx.lineTo(head[0], head[1]);
      }
      return head;
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = TR.halo; ctx.lineWidth = TR.haloW * px; path(); ctx.stroke();
    ctx.strokeStyle = color; ctx.globalAlpha = TR.alpha; ctx.lineWidth = TR.line * px;
    var head = path(); ctx.stroke();
    ctx.globalAlpha = 1;
    // The dot marks the gripper while it moves, as in the trajectory figure.
    if (k < pts.length - 1) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(head[0], head[1], TR.dot * px, 0, Math.PI * 2); ctx.fill(); }
  }

  var clips = figs.map(function (fig) {
    var v = fig.querySelector("video");
    var src = v.getAttribute("src");
    v.loop = false; v.removeAttribute("loop");
    v.playbackRate = SPEED;
    v.addEventListener("loadedmetadata", function () { v.playbackRate = SPEED; });
    // A box over the clip holds the path canvas and the speed pill.
    var box = document.createElement("div");
    box.className = "gen-media";
    v.parentNode.insertBefore(box, v);
    box.appendChild(v);
    var canvas = document.createElement("canvas");
    canvas.width = T.w; canvas.height = T.h;
    canvas.setAttribute("aria-hidden", "true");
    box.appendChild(canvas);
    var pill = document.createElement("span");
    pill.className = "tele-speed on";
    pill.textContent = SPEED + "× speed";
    box.appendChild(pill);
    return { v: v, ctx: canvas.getContext("2d"), pts: T.clips[src], color: T.colors[src] };
  });
  if (clips.some(function (c) { return !c.pts; })) return;

  var visible = false, raf = 0, restart = 0;
  function draw() {
    clips.forEach(function (c) { trail(c.ctx, c.pts, c.color, c.v.currentTime); });
    raf = visible ? requestAnimationFrame(draw) : 0;
  }
  function playAll() { clips.forEach(function (c) { if (!c.v.ended) window.playVideo(c.v).catch(function () {}); }); }
  function pauseAll() { clips.forEach(function (c) { c.v.pause(); }); }
  // When every clip has ended, hold the finished set for a moment and start all three over together.
  clips.forEach(function (c) {
    c.v.addEventListener("ended", function () {
      if (!clips.every(function (x) { return x.v.ended; })) return;
      clearTimeout(restart);
      restart = setTimeout(function () {
        clips.forEach(function (x) { x.v.currentTime = 0; });
        if (visible) playAll();
      }, PAUSE_MS);
    });
  });

  if (C.REDUCED) { clips.forEach(function (c) { trail(c.ctx, c.pts, c.color, Infinity); }); return; }
  new IntersectionObserver(function (e) {
    visible = e[0].isIntersecting;
    if (visible) { playAll(); if (!raf) raf = requestAnimationFrame(draw); } else pauseAll();
  }, { threshold: 0.3 }).observe(document.querySelector(".gen-grid"));
})();
