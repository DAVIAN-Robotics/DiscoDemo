/* Shared helpers for animated paper figures.
 * A figure is an SVG restored from the paper source (window.FIGS[key]) plus a timeline that
 * reveals its named elements. A figure plays once when it is properly in view, then holds its final
 * state; "Replay" runs it again. With reduced motion the final state shows immediately. */
(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var SVGNS = "http://www.w3.org/2000/svg";
  var uid = 0;

  function byName(svg, name) {
    return svg.querySelector('[data-name="' + name + '"]');
  }
  function allByName(svg, pattern) {
    return Array.prototype.filter.call(svg.querySelectorAll("[data-name]"), function (el) {
      return pattern.test(el.getAttribute("data-name"));
    });
  }

  /* Draw a stroked shape along its path. A mask whose stroke grows along the same path reveals
   * the original drawing underneath, so dotted lines and arrowheads keep their exact look. */
  function prepDraw(g) {
    var path = g.querySelector("path");
    var sw = parseFloat(path.getAttribute("stroke-width") || "1");
    var len = path.getTotalLength();
    var id = "fig-draw-" + (++uid);
    var mask = document.createElementNS(SVGNS, "mask");
    mask.setAttribute("id", id);
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    var m = document.createElementNS(SVGNS, "path");
    m.setAttribute("d", path.getAttribute("d"));
    m.setAttribute("fill", "none");
    m.setAttribute("stroke", "#fff");
    // Wide enough to uncover the arrowhead, which is about three stroke widths across.
    m.setAttribute("stroke-width", String(Math.max(sw * 5, 10)));
    m.setAttribute("stroke-linecap", "round");
    m.style.strokeDasharray = len + " " + (len + 40);
    m.style.strokeDashoffset = String(len + 20);
    mask.appendChild(m);
    path.parentNode.insertBefore(mask, path);
    path.setAttribute("mask", "url(#" + id + ")");
    return { m: m, len: len };
  }

  /* Step times are relative to the current phase. next() starts a new phase only after every step
   * added so far has finished, so one panel never overlaps the next. */
  function Timeline(svg) {
    this.svg = svg;
    this.steps = [];
    this.base = 0;
    this.end = 0;
  }
  Timeline.prototype.at = function (at, dur) {
    var t = this.base + at;
    this.end = Math.max(this.end, t + dur);
    return t;
  };
  Timeline.prototype.next = function (gap) {
    this.base = this.end + (gap == null ? 300 : gap);
    return this;
  };
  Timeline.prototype.draw = function (g, at, dur, easing) {
    if (!g) throw new Error("figure element missing");
    var d = prepDraw(g);
    at = this.at(at, dur);
    this.steps.push(function (play) {
      var from = { strokeDashoffset: d.len + 20 }, to = { strokeDashoffset: 0 };
      return play(d.m, [from, to], at, dur, easing || "cubic-bezier(.45,.05,.3,1)");
    });
    return this;
  };
  Timeline.prototype.fade = function (el, at, dur, fromOpacity) {
    if (!el) throw new Error("figure element missing");
    var o0 = fromOpacity == null ? 0 : fromOpacity;
    at = this.at(at, dur);
    this.steps.push(function (play) {
      return play(el, [{ opacity: o0 }, { opacity: 1 }], at, dur, "ease-out");
    });
    return this;
  };
  Timeline.prototype.grow = function (el, at, dur, axis) {
    if (!el) throw new Error("figure element missing");
    // A CSS transform replaces the SVG transform attribute that positions a restored shape, so
    // scale the drawing inside the positioned group instead of the group itself.
    if (el.getAttribute("transform")) el = el.firstElementChild;
    el.style.transformBox = "fill-box";
    el.style.transformOrigin = axis === "y" ? "50% 100%" : "0% 50%";
    var s0 = axis === "y" ? "scaleY(0)" : "scaleX(0)";
    at = this.at(at, dur);
    this.steps.push(function (play) {
      return play(el, [{ transform: s0 }, { transform: "none" }], at, dur, "cubic-bezier(.2,.7,.2,1)");
    });
    return this;
  };
  /* Reveal shapes with a single clip that grows from the top down (axis "y", default) or from the
   * left (axis "x"), so they appear in place rather than stretch. One animation for the whole family
   * instead of one mask per shape.
   * By default the shapes are moved into one group at the position of the first (keeping their order),
   * which works whatever transforms they carry. With opts.inPlace they stay where they are and each
   * gets the shared clip --- for untransformed shapes whose stacking against other layers must not
   * change (chart lines between grid lines and labels). opts.easing overrides the easing. */
  Timeline.prototype.wipe = function (els, at, dur, axis, opts) {
    opts = opts || {};
    if (!els.length || els.some(function (e) { return !e; })) throw new Error("figure elements missing");
    var box, host;
    if (opts.inPlace) {
      var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      els.forEach(function (e) {
        if (e.getAttribute("transform")) throw new Error("in-place wipe needs untransformed shapes");
        var b = e.getBBox();
        x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
        x1 = Math.max(x1, b.x + b.width); y1 = Math.max(y1, b.y + b.height);
      });
      box = { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
      host = els[0];
    } else {
      host = document.createElementNS(SVGNS, "g");
      els[0].parentNode.insertBefore(host, els[0]);
      els.forEach(function (e) { host.appendChild(e); });
      box = host.getBBox();
    }
    var pad = 4, h = box.height + 2 * pad, wd = box.width + 2 * pad;
    var clip = document.createElementNS(SVGNS, "clipPath");
    clip.setAttribute("id", "fig-wipe-" + (++uid));
    clip.setAttribute("clipPathUnits", "userSpaceOnUse");
    var r = document.createElementNS(SVGNS, "rect");
    r.setAttribute("x", box.x - pad); r.setAttribute("y", box.y - pad);
    r.setAttribute("width", wd); r.setAttribute("height", h);
    clip.appendChild(r);
    host.parentNode.insertBefore(clip, host);
    (opts.inPlace ? els : [host]).forEach(function (e) { e.setAttribute("clip-path", "url(#" + clip.id + ")"); });
    at = this.at(at, dur);
    this.steps.push(function (play) {
      var frames = axis === "x" ? [{ width: "0px" }, { width: wd + "px" }] : [{ height: "0px" }, { height: h + "px" }];
      return play(r, frames, at, dur, opts.easing || "cubic-bezier(.45,.05,.3,1)");
    });
    return this;
  };
  /* Any keyframes on any element, on the timeline (for effects the helpers above do not cover). */
  Timeline.prototype.anim = function (el, frames, at, dur, easing) {
    if (!el) throw new Error("figure element missing");
    at = this.at(at, dur);
    this.steps.push(function (play) { return play(el, frames, at, dur, easing || "cubic-bezier(.2,.7,.2,1)"); });
    return this;
  };
  Timeline.prototype.run = function () {
    var anims = [];
    function play(el, frames, at, dur, easing) {
      var a = el.animate(frames, { delay: REDUCED ? 0 : at, duration: REDUCED ? 0 : dur, easing: easing, fill: "both" });
      anims.push(a);
      // Once done, keep the end state as plain style and drop the animation. A finished animation that
      // keeps filling leaves the element on its own compositor layer, which the browser may keep at a
      // low raster scale --- labels then stay blurry after they have grown or slid into place.
      // The last keyframe is written as plain style by hand: Chrome's commitStyles() stored opacity
      // 6e-28 instead of 1 for the Fig. 1 bar group (measured), which hid the whole panel.
      // (finished rejects when Replay cancels the run early; nothing to keep then.)
      a.finished.then(function () {
        if (a.playState !== "finished") return;
        var last = frames[frames.length - 1];
        Object.keys(last).forEach(function (k) {
          if (k !== "offset" && k !== "easing" && k !== "composite") el.style[k] = last[k];
        });
        a.cancel();
      }, function () {});
      return a;
    }
    this.cancel();
    this.steps.forEach(function (s) { s(play); });
    this.anims = anims;
  };
  /* Jump to the end: every step at its final state (an unplayed figure is revealed whole). */
  Timeline.prototype.finish = function () {
    if (!this.anims) this.run();
    this.anims.forEach(function (a) { if (a.playState !== "idle") a.finish(); });
  };
  Timeline.prototype.cancel = function () {
    (this.anims || []).forEach(function (a) { a.cancel(); });
  };
  /* Put every element in its start state without playing (before the figure scrolls in). */
  Timeline.prototype.hold = function () {
    this.run();
    (this.anims || []).forEach(function (a) { a.pause(); a.currentTime = 0; });
  };

  /* Ids that a figure refers to (markers, clip paths, masks) get the figure's name in front, so two figures on
   * one page cannot pick up each other's definitions: url(#id) resolves to the first element with that id in
   * the document, and the motivation and method figures both define arrow-tail-* markers. */
  function scopeIds(svg, key) {
    var refs = {};
    svg.replace(/url\(#([^)]+)\)|href="#([^"]+)"/g, function (m, a, b) { refs[a || b] = true; return m; });
    Object.keys(refs).forEach(function (id) {
      var e = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      svg = svg.replace(new RegExp(' id="' + e + '"', "g"), ' id="' + key + "__" + id + '"')
        .replace(new RegExp("url\\(#" + e + "\\)", "g"), "url(#" + key + "__" + id + ")")
        .replace(new RegExp('href="#' + e + '"', "g"), 'href="#' + key + "__" + id + '"');
    });
    return svg;
  }

  /* Mount a restored figure into `host`, build its timeline, and wire play-on-view + replay. */
  function mount(host, key, build) {
    host.innerHTML = scopeIds(window.FIGS[key], key);
    var svg = host.querySelector("svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", host.getAttribute("data-label") || key);
    var tl = new Timeline(svg);
    build(svg, tl);
    tl.hold();
    // Plays once, the first time the figure is properly in view; afterwards it keeps its final state.
    window.InView.once(host, function () { tl.run(); });
    var btn = host.parentNode.querySelector("[data-replay]");
    if (btn) {
      btn.addEventListener("click", function () { tl.run(); });
      window.addSkip(btn, function () { tl.finish(); });
    }
    return svg;
  }

  /* One look for every gripper path on the page --- the trajectory figure, the generator clips and the α
   * figure all draw with these (widths in screen pixels): a white halo, the path in its source colour, and a
   * dot at the gripper while the path is still moving. The widths are for a panel shown about REF px wide
   * (the trajectory figure); a smaller panel gets proportionally thinner strokes, so it reads as a shrunk copy
   * instead of a solid blob. trailPx(w) = screen-pixel multiplier for a panel w px wide. */
  var TRAIL = { halo: "rgba(255,255,255,0.4)", haloW: 4, line: 1.8, alpha: 0.92, dot: 2.2, REF: 300 };
  TRAIL.scale = function (w) { return Math.min(1, w / TRAIL.REF); };
  window.FigCore = { mount: mount, byName: byName, allByName: allByName, SVGNS: SVGNS, REDUCED: REDUCED, TRAIL: TRAIL };
})();
