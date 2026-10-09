/* When an animation starts. A figure counts as arrived only when most of it is on screen (or it fills
 * the screen) and stays there for a moment, so passing over it does not start it. It counts as left
 * once it is entirely off screen. InView.once runs only on the first arrival: every animation on the
 * page plays once and then keeps its final state (Replay buttons play it again). */
(function () {
  "use strict";
  var bar = document.querySelector(".site-head");   // sticky header (one row on wide screens, two on narrow)
  var SETTLE = 250;     // ms the figure must stay in place, so a jump that passes it does not count
  var watched = [];

  function arrived(node) {
    var r = node.getBoundingClientRect(), vh = window.innerHeight, HEAD = bar ? bar.offsetHeight : 64;
    var seen = Math.min(r.bottom, vh) - Math.max(r.top, HEAD);
    return seen >= Math.min(r.height, vh - HEAD) * 0.8;
  }
  function check() {
    watched.forEach(function (w) {
      if (!w.near || w.inside) return;
      if (!arrived(w.node)) { clearTimeout(w.timer); w.timer = 0; return; }
      if (w.timer) return;
      w.timer = setTimeout(function () {
        w.timer = 0;
        if (w.near && !w.inside && arrived(w.node)) { w.inside = true; w.enter(); }
      }, SETTLE);
    });
  }
  var ticking = false;
  function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(function () { ticking = false; check(); if (note) showNote(); }); } }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);

  /* A "Skip" button next to a figure's Replay button: jumps that figure to its final state. */
  window.addSkip = function (replay, fn) {
    var b = document.createElement("button");
    b.type = "button"; b.className = replay.className + " skip"; b.textContent = "Skip";
    b.setAttribute("aria-label", "Skip the animation");
    b.addEventListener("click", fn);
    // The stylesheet orders the pair [Skip] [Replay] at the bottom right of the caption row.
    replay.parentNode.insertBefore(b, replay);
    return b;
  };

  /* Autoplay. Every clip on the page is muted and starts itself, but iOS in Low Power Mode refuses even
   * muted autoplay. A refused clip is remembered, a note asks for a tap, and the next tap anywhere
   * starts the refused clips that are on screen (a play() inside a tap is always allowed). */
  var blocked = [], onTap = [], note = null;
  function onScreen(v) {
    var r = v.getBoundingClientRect();
    return v.isConnected && r.bottom > 0 && r.top < window.innerHeight && r.width > 0;
  }
  // The note shows only while a refused clip is on screen (or a sequence waits for a tap).
  function showNote() {
    var on = onTap.length > 0 || blocked.some(onScreen);
    if (!on && !note) return;
    if (on && !note) {
      note = document.createElement("div");
      note.className = "tap-note"; note.setAttribute("role", "status");
      note.textContent = "Videos are paused by your device (e.g. Low Power Mode). Tap anywhere to play.";
      document.body.appendChild(note);
    }
    if (note) note.classList.toggle("on", on);
  }
  function unblock(v) { var i = blocked.indexOf(v); if (i >= 0) blocked.splice(i, 1); showNote(); }
  window.playVideo = function (v) {
    return v.play().then(function () { unblock(v); }, function (err) {
      if (err && err.name === "NotAllowedError" && blocked.indexOf(v) < 0) { blocked.push(v); showNote(); }
      throw err;
    });
  };
  window.videoBlocked = function (v) { return blocked.indexOf(v) >= 0; };
  window.whenTapped = function (fn) { onTap.push(fn); showNote(); };
  function tapped() {
    var fns = onTap; onTap = [];
    fns.forEach(function (f) { f(); });
    blocked.slice().forEach(function (v) {
      if (!v.isConnected) { unblock(v); return; }
      if (onScreen(v)) v.play().then(function () { unblock(v); }, function () {});
    });
    showNote();
  }
  document.addEventListener("touchend", tapped, true);
  document.addEventListener("click", tapped, true);

  /* Loading spinner. A clip that should be playing but is still waiting for data shows a spinner over
   * its frame after a short moment, so a slow network reads as loading rather than as a broken video.
   * Media events do not bubble, so they are caught on the way down (capture). */
  var SPIN_AFTER = 400;
  function spinner(v) {
    if (!v._spin) {
      var h = v.parentElement;
      if (getComputedStyle(h).position === "static") h.style.position = "relative";
      v._spin = document.createElement("span");
      v._spin.className = "vload"; v._spin.setAttribute("aria-hidden", "true");
      h.appendChild(v._spin);
    }
    // Centre on the video's own box (its figure may also hold a caption).
    v._spin.style.left = (v.offsetLeft + v.offsetWidth / 2) + "px";
    v._spin.style.top = (v.offsetTop + v.offsetHeight / 2) + "px";
    return v._spin;
  }
  function loading(v, on) {
    clearTimeout(v._spinT);
    if (on) v._spinT = setTimeout(function () { if (!v.paused && v.readyState < 3) spinner(v).classList.add("on"); }, SPIN_AFTER);
    else if (v._spin) v._spin.classList.remove("on");
  }
  document.addEventListener("play", function (e) { if (e.target.readyState < 3) loading(e.target, true); }, true);
  document.addEventListener("waiting", function (e) { loading(e.target, true); }, true);
  ["playing", "pause", "emptied"].forEach(function (k) {
    document.addEventListener(k, function (e) { loading(e.target, false); }, true);
  });

  /* Theme toggle in the header: flips between light and dark and remembers the choice; until the reader
   * picks one, the page follows the system setting. */
  var themeBtn = document.querySelector(".theme-btn");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var root = document.documentElement, cur = root.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  window.InView = function (node, enter, leave) {
    watch(node, enter, leave);
  };
  window.InView.once = function (node, enter) {
    var done = false;
    watch(node, function () { if (!done) { done = true; enter(); } });
  };
  function watch(node, enter, leave) {
    var w = { node: node, enter: enter, near: false, inside: false, timer: 0 };
    watched.push(w);
    new IntersectionObserver(function (e) {
      w.near = e[0].isIntersecting;
      if (w.near) { check(); return; }
      clearTimeout(w.timer); w.timer = 0;
      if (w.inside) { w.inside = false; if (leave) leave(); }
    }).observe(node);
  }
})();
