/* Opening figures (paper Sec. 1): real teleoperation, a wall of generator rollouts, and a real/sim comparison. */
(function () {
  "use strict";
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var P = window.OPENING;
  function onView(node, fn, th, margin) {
    new IntersectionObserver(function (e) { fn(e[0].isIntersecting); },
      { threshold: th == null ? 0.4 : th, rootMargin: margin || "0px" }).observe(node);
  }
  function fmt(s) {
    var m = Math.floor(s / 60), r = s - m * 60;
    return m ? m + " min " + Math.round(r) + " s" : r.toFixed(1) + " s";
  }

  /* ---------- A. teleoperation ---------- */
  (function () {
    var root = document.getElementById("tele"), T = P.teleop;
    var v = root.querySelector("video");
    v.src = T.clip; v.poster = T.clip.replace(".mp4", ".jpg");
    var strip = root.querySelector("[data-role=strip]");
    var maxLen = Math.max.apply(null, T.lengths_s);
    var cells = T.lengths_s.map(function (s) {
      var c = document.createElement("span");
      c.style.width = (100 * s / maxLen * 0.3) + "%";
      c.title = s.toFixed(1) + " s";
      c.appendChild(document.createElement("i"));
      strip.appendChild(c);
      return c;
    });
    var total = T.lengths_s.reduce(function (a, b) { return a + b; }, 0);
    var mean = total / T.lengths_s.length;
    root.querySelector("[data-role=total]").textContent = fmt(total);
    root.querySelector("[data-role=k3]").textContent = "about " + Math.round(3000 * mean / 3600) + " hours";
    var clock = root.querySelector("[data-role=clock]"), ep = root.querySelector("[data-role=ep]");
    var badge = root.querySelector("[data-role=speed]");
    // As in the intro video: 2x through the first half of the first episode, then 6x for the rest; back to
    // 2x on loop. The pill always shows the current speed.
    var SLOW = 2, FAST = 6, fastFrom = T.clip_bounds_s[0] / 2;
    function tick() {
      var t = v.currentTime, k = 0, start = 0;
      var rate = t >= fastFrom ? FAST : SLOW;
      if (v.playbackRate !== rate) v.playbackRate = rate;
      var label = rate + "\u00d7 speed";
      if (badge.textContent !== label) badge.textContent = label;
      badge.classList.add("on");
      while (k < T.clip_bounds_s.length - 1 && t >= T.clip_bounds_s[k]) { start = T.clip_bounds_s[k]; k++; }
      clock.textContent = (t - start).toFixed(1) + " s";
      ep.textContent = "clip " + (k + 1) + " of " + T.clip_episodes.length;
      cells.forEach(function (c, i) {
        var idx = T.clip_episodes.indexOf(i);
        var fill = idx < 0 ? 0 : idx < k ? 1 : idx === k ? (t - start) / (T.clip_bounds_s[k] - start) : 0;
        c.firstChild.style.width = (100 * Math.min(1, fill)) + "%";
        c.classList.toggle("playing", idx === k);
      });
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    // Restarts from the first episode on every visit; fetched a screen ahead.
    onView(root, function (on) { if (on) v.preload = "auto"; }, 0, "1500px 0px");
    // Starts the first time it is in view; later visits resume where it was.
    if (!REDUCED) window.InView(root, function () { window.playVideo(v).catch(function () {}); }, function () { v.pause(); });
  })();

  /* ---------- B. rollout wall ---------- */
  (function () {
    var root = document.getElementById("wall"), M = P.mosaic;
    var v = root.querySelector("video");
    v.src = M.src; v.poster = M.src.replace(".mp4", ".jpg");
    var SPEED = 2;
    v.loop = false; v.removeAttribute("loop"); // plays once and holds the finished wall
    var speedPill = document.createElement("span");
    speedPill.className = "tele-speed";
    speedPill.textContent = SPEED + "\u00d7 speed";
    root.appendChild(speedPill);
    var tileAspect = M.tile[1] / M.tile[0];                  // one rollout (4:3)
    var wallAspect = M.rows * M.tile[1] / (M.cols * M.tile[0]);  // the whole wall
    // The box keeps one rollout's shape throughout, so the page never reflows. It starts framed on the
    // top-left rollout (wall scaled up) and pulls back to the whole wall, centred with dark bands above
    // and below. The centring shift as a fraction of the wall's own height:
    var shift = (tileAspect / wallAspect - 1) / 2;
    root.style.aspectRatio = M.tile[0] + " / " + M.tile[1];
    var band = 100 * (1 - wallAspect / tileAspect) / 2;     // band height, % of the box
    var rowsEl = root.querySelector(".wall-rows");
    rowsEl.style.top = band + "%"; rowsEl.style.bottom = band + "%";
    var ONE = "translateY(0%) scale(" + M.cols + ")", ALL = "translateY(" + (100 * shift) + "%) scale(1)";
    // Task boundaries (2 x 2 blocks) drawn on a layer that takes the same transform as the wall, so the
    // lines travel with the tiles during the pull-back; non-scaling strokes keep them 1 px wide.
    var grid = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    grid.setAttribute("class", "wall-grid");
    grid.setAttribute("viewBox", "0 0 2 2");
    grid.setAttribute("preserveAspectRatio", "none");
    grid.setAttribute("aria-hidden", "true");
    grid.innerHTML = '<path d="M1 0V2M0 1H2" vector-effect="non-scaling-stroke"/>';
    grid.style.height = (100 * wallAspect / tileAspect) + "%";
    v.parentNode.insertBefore(grid, v.nextSibling);
    function frameOne() { v.style.transform = ONE; grid.style.transform = ONE; }
    // Full-resolution copy of the first tile covers the blurry 10x-scaled wall until the pull-back starts.
    var solo = document.createElement("video");
    solo.className = "solo"; solo.muted = true; solo.loop = true; solo.playsInline = true; solo.preload = "none";
    solo.src = M.solo;
    root.appendChild(solo);
    if (REDUCED) { root.classList.add("out"); solo.remove(); v.style.transform = ALL; grid.style.transform = ALL; v.playbackRate = SPEED; return; }
    // Fetch both clips a screen ahead so the sequence never starts on a stalled video.
    onView(root, function (on) { if (on) [v, solo].forEach(function (x) { x.preload = "auto"; }); }, 0, "1500px 0px");
    // Resolves once the clip is actually playing. Playback is started first and waited on afterwards:
    // phone browsers (iOS Safari, data-saver modes) ignore preload and fetch nothing until play(), so
    // waiting for canplaythrough before play() never resolved there and the wall stayed on its poster.
    function playing(x) {
      return x.play().then(function () {
        return x.readyState >= 3 ? null : new Promise(function (res) { x.addEventListener("playing", res, { once: true }); });
      });
    }
    var anims = [], timers = [], run = 0;
    function stop() {
      anims.forEach(function (a) { a.cancel(); }); anims = [];
      timers.forEach(clearTimeout); timers = [];
    }
    // Straight to the whole wall (Skip animations): no zoom, the solo clip hidden, the wall playing.
    function final() {
      run++; stop();
      v.style.transform = ALL; grid.style.transform = ALL; solo.style.opacity = "0"; solo.pause();
      root.classList.add("out");
      v.playbackRate = SPEED; speedPill.classList.add("on");
      window.playVideo(v).catch(function () {});
    }
    // Same sequence as the intro video: the first rollout and then the whole wall stay on frame 0 while the
    // view pulls back, and once all 36 are revealed they start together at 2x and hold their last frames.
    // (Playing during the pull-back showed the short peg insertions already finished when their tiles appeared.)
    function play() {
      var my = ++run;
      stop();
      root.classList.remove("out");
      speedPill.classList.remove("on");
      solo.style.opacity = "0";
      frameOne();
      // play() then pause at 0: phone browsers fetch nothing for a paused clip until play() is called.
      Promise.all([playing(v), playing(solo)]).then(function () {
        if (my !== run) return;
        [v, solo].forEach(function (x) { x.pause(); x.currentTime = 0; });
        solo.style.opacity = "1";
        timers.push(setTimeout(function () {
          var fade = solo.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 500, fill: "forwards" });
          var opts = { duration: 2600, easing: "cubic-bezier(.6,0,.2,1)", fill: "forwards" };
          var zoom = v.animate([{ transform: ONE }, { transform: ALL }], opts);
          var lines = grid.animate([{ transform: ONE }, { transform: ALL }], opts);
          anims.push(fade, zoom, lines);
          zoom.onfinish = function () {
            if (my !== run) return;
            v.style.transform = ALL; grid.style.transform = ALL;
            v.currentTime = 0; v.playbackRate = SPEED;
            speedPill.classList.add("on");
            window.playVideo(v).catch(function () {});
          };
          timers.push(setTimeout(function () { root.classList.add("out"); }, 2200));
        }, 1000));
      }, function () {
        // Autoplay refused (iOS Low Power Mode refuses even muted clips): the sequence starts on the
        // reader's next tap anywhere, which lets the clips play.
        if (my !== run) return;
        window.whenTapped(function () { if (my === run) play(); });
      });
    }
    frameOne();
    // The pull-back plays once, the first time the wall is in view; later visits just resume the clips.
    var started = false;
    window.InView(root, function () {
      if (!started) { started = true; play(); return; }
      // Resume only once the clips have started together (not while the view is still pulling back).
      if (speedPill.classList.contains("on") && !v.ended) window.playVideo(v).catch(function () {});
    }, function () { v.pause(); solo.pause(); });
    document.querySelector('[data-replay="wall"]').addEventListener("click", play);
    window.addSkip(document.querySelector('[data-replay="wall"]'), function () { started = true; final(); });
  })();

  /* ---------- C. one recorded episode and its matched simulation replay ---------- */
  (function () {
    var root = document.getElementById('wipe'), v = root.querySelector('video');
    var canvas = root.querySelector('canvas'), ctx = canvas.getContext('2d');
    var bar = root.querySelector('.bar'), play = document.getElementById('wipe-play');
    var clock = document.getElementById('wipe-clock'), camera = 'exo', reveal = .5;
    var visible = false, paused = REDUCED, frame = null;
    // As in the intro video: 2x, and the divider sweeps on its own from half and half to real and then to
    // simulation, keyed to the clip time. The clip loops, and the divider is back at half by the loop point
    // so each pass starts where the last one ended. Dragging or the keys take over.
    var SPEED = 2, auto = !REDUCED;
    var KEYS = [[0, 0.5], [1.6, 0.5], [2.7, 1], [4.7, 1], [6.3, 0]]; // [seconds at 2x, real fraction]
    function sweep(t) {
      // Hold the simulation side a moment, then back to half and half just before the loop point
      // (clip end, in seconds at 2x).
      var end = isFinite(v.duration) ? v.duration / SPEED : 0;
      var keys = end > 7 ? KEYS.concat([[end - 0.65, 0], [end - 0.15, 0.5]]) : KEYS;
      for (var i = 0; i + 1 < keys.length; i++) {
        var a = keys[i], b = keys[i + 1];
        if (t <= b[0]) { var x = Math.max(0, (t - a[0]) / (b[0] - a[0])); x = x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; return a[1] + (b[1] - a[1]) * x; }
      }
      return keys[keys.length - 1][1];
    }
    v.loop = true; v.playbackRate = SPEED;
    v.addEventListener("loadedmetadata", function () { v.playbackRate = SPEED; });
    var pill = document.createElement("span");
    pill.className = "tele-speed on"; pill.textContent = SPEED + "\u00d7 speed";
    root.appendChild(pill);
    var poster = new Image(); poster.src = v.poster;
    var sources = { exo: [[0,36,640,444],[640,36,640,444]], wrist: [[0,516,640,444],[640,516,640,444]] };
    function draw() {
      var source = v.readyState >= 2 ? v : poster.complete && poster.naturalWidth ? poster : null;
      if (!source) return;
      if (auto) setReveal(sweep(v.currentTime / SPEED));
      var crop = sources[camera], w = canvas.width, h = canvas.height;
      ctx.drawImage(source,crop[0][0],crop[0][1],crop[0][2],crop[0][3],0,0,w,h);
      ctx.save(); ctx.beginPath(); ctx.rect(w*reveal,0,w*(1-reveal),h); ctx.clip();
      ctx.drawImage(source,crop[1][0],crop[1][1],crop[1][2],crop[1][3],0,0,w,h); ctx.restore();
      clock.textContent = v.currentTime.toFixed(1) + ' s';
    }
    function stopFrames() {
      if (frame === null) return;
      if (v.requestVideoFrameCallback) v.cancelVideoFrameCallback(frame); else cancelAnimationFrame(frame);
      frame = null;
    }
    function frames() {
      if (v.paused || frame !== null) return;
      if (v.requestVideoFrameCallback) frame = v.requestVideoFrameCallback(function () { frame=null; draw(); frames(); });
      else frame = requestAnimationFrame(function () { frame=null; draw(); frames(); });
    }
    function update() {
      play.textContent = paused ? 'Play' : 'Pause';
      if (!visible || paused) { v.pause(); return; }
      window.playVideo(v).then(function () { if (!visible || paused) v.pause(); },function (e) {
        if (e.name === 'NotAllowedError') { paused=true; play.textContent='Play'; }
      });
    }
    v.addEventListener('play',frames);
    v.addEventListener('pause',function () { stopFrames(); draw(); });
    ['loadeddata','seeked'].forEach(function (event) { v.addEventListener(event,draw); });
    poster.addEventListener('load',draw);
    v.addEventListener('error',function () { paused=true; update(); clock.textContent='Video unavailable'; });
    onView(root,function (on) { if (on) v.preload='auto'; },0,'800px 0px');
    window.InView(root,function () { visible=true; update(); },function () { visible=false; update(); });
    play.addEventListener('click',function () { paused=!paused; update(); });
    document.querySelector('[data-replay="wipe"]').addEventListener('click',function () { v.currentTime=0; auto=!REDUCED; paused=false; update(); });
    var seg=document.getElementById('wipe-camera');
    [['exo','External camera'],['wrist','Wrist camera']].forEach(function (item,i) {
      var b=document.createElement('button'); b.type='button'; b.textContent=item[1]; b.setAttribute('role','tab');
      b.setAttribute('aria-selected',String(i===0)); b.tabIndex=i===0?0:-1;
      b.addEventListener('click',function () {
        camera=item[0]; root.dataset.camera=camera;
        seg.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-selected',String(x===b)); x.tabIndex=x===b?0:-1; });
        draw();
      });
      b.addEventListener('keydown',function (e) {
        if (['ArrowLeft','ArrowRight','Home','End'].indexOf(e.key)<0) return;
        e.preventDefault(); var n=e.key==='Home'?0:e.key==='End'?1:1-i;seg.children[n].focus();seg.children[n].click();
      });
      seg.appendChild(b);
    });
    root.dataset.camera=camera;
    function setReveal(r) {
      var p=Math.max(0,Math.min(100,100*r)); reveal=p/100;
      bar.style.left=p+'%'; root.setAttribute('aria-valuenow',Math.round(p));
      root.setAttribute('aria-valuetext',Math.round(p)+'% real, '+Math.round(100-p)+'% simulation');
    }
    function set(p) { auto=false; setReveal(p/100); draw(); }
    var drag=false;
    function at(e) { var r=root.getBoundingClientRect();set(100*(e.clientX-r.left)/r.width); }
    root.addEventListener('pointerdown',function (e) { drag=true;root.focus({preventScroll:true});root.setPointerCapture(e.pointerId);at(e); });
    root.addEventListener('pointermove',function (e) { if (drag) at(e); });
    ['pointerup','pointercancel','lostpointercapture'].forEach(function (ev) { root.addEventListener(ev,function () {drag=false;}); });
    root.addEventListener('keydown',function (e) {
      var p=+root.getAttribute('aria-valuenow');
      if(e.key==='ArrowLeft')p-=5;else if(e.key==='ArrowRight')p+=5;else if(e.key==='Home')p=0;else if(e.key==='End')p=100;else return;
      e.preventDefault();set(p);
    });
    setReveal(.5);draw();update();
  })();
})();
