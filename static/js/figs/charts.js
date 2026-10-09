/* The paper's matplotlib figures (dataset analysis, alpha trade-off, RL time, on-policy correction),
 * restored as SVG from their generators with every data element named pg-a<axis>-<kind><n>
 * (generated, not hand-edited). Panels play one after another, left to right:
 *   - bars rise from the axis baseline; the pieces of a stacked bar rise together as one bar;
 *   - curves: a left-to-right sweep draws each line and its band, and the markers pop in one by one
 *     as the sweep reaches them;
 *   - scatter markers pop in one source after another;
 *   - images and axes are not animated. Value labels follow their bars. */
(function () {
  "use strict";
  var C = window.FigCore;

  var BAR_EASE = "cubic-bezier(.2,.7,.2,1)", BAR_DUR = 520;

  /* Bars rise from the axis baseline; pieces at the same x (a stacked bar) share delay and origin, so
   * they grow as one bar. Returns the columns (x range, top, baseline, delay) for the labels. */
  function bars(tl, list) {
    var boxes = list.map(function (e) { return e.getBBox(); });
    var base = Math.max.apply(null, boxes.map(function (b) { return b.y + b.height; }));
    var xs = boxes.map(function (b) { return Math.round(b.x); });
    var order = xs.slice().sort(function (a, b) { return a - b; }).filter(function (x, i, a) { return !i || x !== a[i - 1]; });
    var cols = order.map(function (x, k) { return { x0: Infinity, x1: -Infinity, top: Infinity, base: base, delay: 100 + k * 70 }; });
    list.forEach(function (e, i) {
      var c = cols[order.indexOf(xs[i])], b = boxes[i];
      c.x0 = Math.min(c.x0, b.x); c.x1 = Math.max(c.x1, b.x + b.width); c.top = Math.min(c.top, b.y);
      e.style.transformBox = "view-box";
      e.style.transformOrigin = "0px " + base + "px";
      tl.anim(e, [{ transform: "scaleY(0)" }, { transform: "none" }], c.delay, BAR_DUR, BAR_EASE);
    });
    return { cols: cols, end: 100 + order.length * 70 + 420 };
  }

  /* A label or error bar of a bar travels with its bar: above the bar it rides up with the bar's top,
   * inside the bar it shows up as the top passes it. Anything not over a bar appears at the end. */
  function ride(tl, els, B) {
    els.forEach(function (e) {
      var b = e.getBBox(), cx = b.x + b.width / 2, cy = b.y + b.height / 2;
      var c = B.cols.filter(function (k) { return cx >= k.x0 - 2 && cx <= k.x1 + 2; })[0];
      if (!c) { tl.anim(e, [{ opacity: 0 }, { opacity: 1 }], B.end, 300); return; }
      var h = c.base - c.top;
      if (cy <= c.top + 2) {
        tl.anim(e, [{ transform: "translateY(" + h + "px)", opacity: 0 }, { opacity: 1, offset: 0.12 },
          { transform: "none", opacity: 1 }], c.delay, BAR_DUR, BAR_EASE);
      } else {
        var f = Math.min(1, Math.max(0, (c.base - cy) / h));
        tl.anim(e, [{ opacity: 0 }, { opacity: 1 }], c.delay + BAR_DUR * f * 0.55, 160);
      }
    });
  }

  // Per figure: whether the panels play together, and how hard the curve markers pop.
  var CFG = {
    rl_time: { together: true, peak: 1.8, popDur: 380, sweep: 3000 },   // slower sweep: the densest figure
    alpha: { together: true, peak: 2.1, popDur: 420 },
    dataset: { together: true, peak: 1.8, popDur: 420 },
    onpolicy: { together: true, peak: 1.8, popDur: 420 }
  };
  function pop(peak) {
    return [{ transform: "scale(0)", opacity: 0 }, { transform: "scale(" + peak + ")", opacity: 1, offset: 0.45 },
      { transform: "scale(0.85)", offset: 0.75 }, { transform: "none", opacity: 1 }];
  }
  var SWEEP_DEFAULT = 2000;

  /* Curves: the line and its band are revealed left to right by one shared clip, and each marker pops
   * in as the sweep reaches it, so the curve reads as points joined one after another. Markers are moved
   * out of their line (into a group just above it) so the clip does not cut them in half. */
  function sweep(tl, lines, curves, cfg, svg) {
    var SWEEP = cfg.sweep || SWEEP_DEFAULT;
    var marks = [], boxes = [];
    lines.forEach(function (g) {
      var uses = g.querySelectorAll("use");
      if (!uses.length) return;
      var box = document.createElementNS(C.SVGNS, "g");
      var clip = uses[0].parentNode.getAttribute("clip-path");
      if (clip) box.setAttribute("clip-path", clip);
      g.parentNode.insertBefore(box, g.nextSibling);
      boxes.push(box);
      // Each marker gets its own group to scale: a CSS transform on the <use> itself combines with its
      // x/y attributes and throws the marker off its point while it grows.
      Array.prototype.forEach.call(uses, function (u) {
        var m = document.createElementNS(C.SVGNS, "g");
        m.appendChild(u); box.appendChild(m);
        marks.push({ el: m, x: +u.getAttribute("x") || 0 });
      });
    });
    var x0 = Infinity, x1 = -Infinity;
    curves.forEach(function (e) { var b = e.getBBox(); x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x + b.width); });
    // The panel's curves and markers move to their own SVG laid exactly over the figure, and that layer
    // is uncovered with a CSS clip-path. Clipping inside the figure made the browser repaint the whole
    // figure (every glyph is a path) on each frame, which stutters on high-density screens.
    var layer = document.createElementNS(C.SVGNS, "svg");
    layer.setAttribute("viewBox", svg.getAttribute("viewBox"));
    layer.setAttribute("class", "fig-layer");
    layer.setAttribute("aria-hidden", "true");
    curves.concat(boxes).sort(function (a, b) {
      return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
    }).forEach(function (e) { layer.appendChild(e); });
    svg.parentNode.appendChild(layer);
    var vbw = +svg.getAttribute("viewBox").split(" ")[2];
    var L = Math.max(0, (x0 - 4) / vbw * 100), R = Math.min(100, (x1 + 4) / vbw * 100);
    var edge = function (r) { return "inset(-20% " + (100 - r) + "% -20% " + L + "%)"; };
    tl.anim(layer, [{ clipPath: edge(L) }, { clipPath: edge(R) }], 0, SWEEP, "linear");
    marks.forEach(function (mk) {
      var f = (mk.x - x0 + 4) / (x1 - x0 + 8);
      mk.el.style.transformBox = "fill-box"; mk.el.style.transformOrigin = "50% 50%";
      tl.anim(mk.el, pop(cfg.peak), Math.max(0, SWEEP * f - 40), cfg.popDur, "ease-out");
    });
    return SWEEP;
  }

  function animate(svg, tl, cfg) {
    var axes = {};
    Array.prototype.forEach.call(svg.querySelectorAll('[id^="pg-a"]'), function (e) {
      var m = /^pg-a(\d+)-([a-z]+)(\d+)$/.exec(e.id);
      if (!m) return;
      var a = axes[m[1]] = axes[m[1]] || { bar: [], line: [], dot: [], band: [], img: [], text: [] };
      a[m[2]].push(e);
    });
    var first = true;
    Object.keys(axes).sort(function (x, y) { return x - y; }).forEach(function (k) {
      var a = axes[k], end = 0;
      var scatter = a.dot, bands = a.band;
      if (!a.bar.length && !a.line.length && !a.dot.length && !a.band.length) return;   // image-only panel: static
      if (!first && !cfg.together) tl.next(150);
      first = false;
      var B = a.bar.length ? bars(tl, a.bar) : null;
      if (B) end = B.end;
      var curves = a.line.concat(bands);
      if (curves.length && !B) end = sweep(tl, a.line, curves, cfg, svg);
      else if (curves.length) ride(tl, curves, B);   // error bars travel with their bars
      // Scatter: one source after another pops in, its name label right with it (nearest label).
      var labelled = [];
      var centre = function (e) { var b = e.getBBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
      // Order: from the bottom-left corner outwards --- further left and further down pop first.
      var dots = scatter.map(function (e) { return { e: e, c: centre(e) }; });
      if (dots.length) {
        var xs = dots.map(function (d) { return d.c[0]; }), ys = dots.map(function (d) { return d.c[1]; });
        var x0 = Math.min.apply(null, xs), xr = Math.max.apply(null, xs) - x0 || 1;
        var y1 = Math.max.apply(null, ys), yr = y1 - Math.min.apply(null, ys) || 1;
        dots.sort(function (p, q) {
          return ((p.c[0] - x0) / xr + (y1 - p.c[1]) / yr) - ((q.c[0] - x0) / xr + (y1 - q.c[1]) / yr);
        });
      }
      dots.forEach(function (d, j) {
        d.t = 120 + j * 220;
        d.e.style.transformBox = "fill-box"; d.e.style.transformOrigin = "50% 50%";
        tl.anim(d.e, pop(cfg.peak), d.t, cfg.popDur, "ease-out");
      });
      if (dots.length) {
        a.text.forEach(function (e) {
          var c = centre(e), best = null, bd = Infinity;
          dots.forEach(function (d) { var dd = Math.hypot(d.c[0] - c[0], d.c[1] - c[1]); if (dd < bd) { bd = dd; best = d; } });
          labelled.push(e);
          tl.anim(e, [{ opacity: 0 }, { opacity: 1 }], best.t + 60, 200);
        });
        end = Math.max(end, 120 + dots.length * 220 + 200);
      }
      var rest = a.text.filter(function (e) { return labelled.indexOf(e) < 0; });
      if (B) ride(tl, rest, B);
      else rest.forEach(function (e) { tl.anim(e, [{ opacity: 0 }, { opacity: 1 }], end, 300); });
    });
  }

  /* Trajectory panels of the alpha figure. The overlays come in as images with the paths baked in; with the
   * path data present (window.TRAJ.alpha) each image is swapped for the clean scene and its paths are drawn
   * on the figure timeline in the trajectory figure's style and pace. Paths are revealed with a dash offset
   * keyed to each sample (no per-path masks: two hundred animated masks repaint the whole figure). */
  var TRAJ_SPEED = 3;  // x real time, as in the trajectory figure
  var TR = C.TRAIL;
  function trajPanels(svg, tl, data) {
    data.order.forEach(function (key, i) {
      var img = svg.querySelector("#pg-a" + i + "-img0");
      if (!img) throw new Error("alpha: overlay image " + i + " missing");
      var cell = data.cells[key];
      var x = +img.getAttribute("x"), y = +img.getAttribute("y"), w = +img.getAttribute("width"), h = +img.getAttribute("height");
      // The embedded image is stored upside down and flipped back by its transform; the clean scene goes in
      // upright at the same place on the figure.
      var m = img.transform.baseVal.consolidate().matrix;
      var a = new DOMPoint(x, y).matrixTransform(m), b = new DOMPoint(x + w, y + h).matrixTransform(m);
      var X = Math.min(a.x, b.x), Y = Math.min(a.y, b.y), W = Math.abs(b.x - a.x), H = Math.abs(b.y - a.y);
      var bg = document.createElementNS(C.SVGNS, "image");
      [["href", data.bg], ["x", X], ["y", Y], ["width", W], ["height", H], ["preserveAspectRatio", "none"]].forEach(function (kv) { bg.setAttribute(kv[0], kv[1]); });
      img.parentNode.insertBefore(bg, img);
      img.parentNode.removeChild(img);
      var sx = W / data.width, sy = H / data.height;
      var trajs = cell.trajs.map(function (pts) {
        return pts.map(function (p) { return [X + p[0] * sx, Y + p[1] * sy]; });
      });
      // Same look and pace as the trajectory figure (FigCore.TRAIL): halo layer, path layer, a dot at each
      // gripper while it moves. Each path advances one sample per 1/(hz·speed) s, so short paths finish first.
      function layer(attrs) {
        var g = document.createElementNS(C.SVGNS, "g");
        Object.keys(attrs).forEach(function (k) { g.setAttribute(k, attrs[k]); });
        bg.parentNode.insertBefore(g, layers.length ? layers[layers.length - 1].nextSibling : bg.nextSibling);
        layers.push(g);
        return g;
      }
      var layers = [];
      var halo = layer({ fill: "none", stroke: TR.halo, "stroke-linecap": "round", "stroke-linejoin": "round" });
      var line = layer({ fill: "none", stroke: cell.color, "stroke-opacity": TR.alpha, "stroke-linecap": "round", "stroke-linejoin": "round" });
      var dots = layer({ fill: cell.color });
      // Widths are screen pixels, as on the canvas figures: convert through the figure's displayed scale.
      function scale() {
        var u = svg.viewBox.baseVal.width / Math.max(1, svg.clientWidth);
        u *= TR.scale(W / u);  // W / u = the panel's width on screen
        halo.setAttribute("stroke-width", TR.haloW * u);
        line.setAttribute("stroke-width", TR.line * u);
        dots.querySelectorAll("circle").forEach(function (c) { c.setAttribute("r", TR.dot * u); });
      }
      trajs.forEach(function (pts) {
        var d = "", cum = [0];
        pts.forEach(function (p, k) {
          d += (k ? "L" : "M") + p[0].toFixed(2) + " " + p[1].toFixed(2);
          if (k) cum.push(cum[k - 1] + Math.hypot(p[0] - pts[k - 1][0], p[1] - pts[k - 1][1]));
        });
        var len = cum[cum.length - 1] + 1, n = pts.length;
        var dur = (n - 1) / data.hz / TRAJ_SPEED * 1000;
        [halo, line].forEach(function (g) {
          var path = document.createElementNS(C.SVGNS, "path");
          path.setAttribute("d", d);
          path.setAttribute("stroke-dasharray", len + " " + len);
          g.appendChild(path);
          tl.anim(path, cum.map(function (c, k) { return { offset: k / (n - 1), strokeDashoffset: len - c }; }), 0, dur, "linear");
        });
        var dot = document.createElementNS(C.SVGNS, "circle");
        dots.appendChild(dot);
        tl.anim(dot, pts.map(function (p, k) {
          return { offset: k / (n - 1), transform: "translate(" + p[0].toFixed(2) + "px," + p[1].toFixed(2) + "px)", opacity: k < n - 1 ? 1 : 0 };
        }), 0, dur, "linear");
      });
      scale();
      new ResizeObserver(scale).observe(svg);
    });
  }

  ["dataset", "alpha", "rl_time", "onpolicy"].forEach(function (key) {
    var host = document.querySelector('[data-fig="' + key + '"]');
    if (host && window.FIGS && window.FIGS[key]) C.mount(host, key, function (svg, tl) {
      if (key === "alpha" && window.TRAJ && window.TRAJ.alpha) trajPanels(svg, tl, window.TRAJ.alpha);
      animate(svg, tl, CFG[key]);
    });
  });
})();
