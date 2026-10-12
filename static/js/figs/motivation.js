/* Paper Fig. 1 (motivation), restored from the paper source slides and animated.
 * Panels play strictly one after another, following the figure's argument: a single solution and the BC drift off it, augmentation
 * widening around that one path, then several distinct solutions filling the space. The source figure's
 * bar slot is cropped away: success rates are shown in the results section. */
(function () {
  "use strict";
  var C = window.FigCore;
  var host = document.querySelector('[data-fig="motivation"]');
  if (!host) return;

  C.mount(host, "motivation", function (svg, tl) {
    var N = function (n) { return C.byName(svg, n); };
    var dots = function (ids) { return ids.map(function (i) { return N("자유형: 도형 " + i); }); };

    // Panel 1 --- one solution; BC rollouts peel off it.
    tl.draw(N("자유형: 도형 123"), 0, 1100);
    dots([124, 130, 131]).forEach(function (g, i) { tl.draw(g, 1050 + i * 220, 600, "ease-out"); });
    tl.fade(N("TextBox 125"), 1750, 500);

    // Panel 2 --- the same path, widened by perturbation: guide lines spread out from it.
    tl.next();
    var p2 = 0;
    tl.draw(N("자유형: 도형 151"), p2, 1000);
    var guides = C.allByName(svg, /^파랑 가이드선 \d+$/).sort(function (a, b) {
      return Math.abs(+a.getAttribute("data-name").split(" ").pop() - 7.5) -
             Math.abs(+b.getAttribute("data-name").split(" ").pop() - 7.5);
    });
    // The augmented copies are drawn along their path like the demonstration itself, nearest first,
    // and the shaded band settles in under them once they are out.
    guides.forEach(function (g, i) { tl.draw(g, p2 + 350 + i * 60, 900); });
    tl.fade(N("파랑 채우기"), p2 + 350 + guides.length * 60 + 500, 500);
    dots([153, 154, 155]).forEach(function (g, i) { tl.draw(g, p2 + 1500 + i * 220, 600, "ease-out"); });
    tl.fade(N("TextBox 156"), p2 + 2200, 500);

    // Panel 3 --- distinct solutions, each with its halo; the field between them fills in.
    tl.next();
    var p3 = 0;
    ["skill-upper2", "skill-upper", "skill-demo", "skill-lower"].forEach(function (n, i) {
      tl.draw(N(n), p3 + i * 320, 1250);
    });
    C.allByName(svg, /^upper2-halo-\d$/).forEach(function (g, i) { tl.fade(g, p3 + 300 + i * 120, 700); });
    C.allByName(svg, /^lower-halo-\d$/).forEach(function (g, i) { tl.fade(g, p3 + 1260 + i * 120, 700); });
    C.allByName(svg, /^field-\d\d$/).forEach(function (g, i) { tl.fade(g, p3 + 500 + i * 22, 500); });
    C.allByName(svg, /^contour$/).forEach(function (g, i) { tl.draw(g, p3 + 700 + i * 60, 900); });
    dots([235, 236, 237]).forEach(function (g, i) { tl.draw(g, p3 + 1900 + i * 220, 600, "ease-out"); });
    tl.fade(N("TextBox 234"), p3 + 2500, 500);

    // Keep only the three panels: crop the viewBox at the source figure's bar slot.
    var slot = svg.querySelector("[data-slot]");
    var vb = svg.viewBox.baseVal;
    svg.setAttribute("viewBox", [vb.x, vb.y, +slot.getAttribute("data-x") - vb.x, vb.height].join(" "));
    slot.remove();
  });
})();
