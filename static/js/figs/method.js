/* Paper Fig. 3 (method overview), restored from the paper source slides and animated.
 * The timeline follows the figure: (a) the reverse curriculum strip fills from easy to hard, then
 * the training loop closes (skill -> policy -> action -> parallel simulation -> rewards -> policy);
 * (b) the trained policy fans out into many successful rollouts, which stream through the success
 * and safety gates into D_safe; (c) the retained data train pi_0.5. */
(function () {
  "use strict";
  var C = window.FigCore;
  var host = document.querySelector('[data-fig="method"]');
  if (!host) return;

  C.mount(host, "method", function (svg, tl) {
    var N = function (n) { return C.byName(svg, n); };
    var all = function (re) { return C.allByName(svg, re); };
    var fade = function (el, at, dur) { if (el) tl.fade(el, at, dur || 450); };
    var draw = function (el, at, dur) { if (el) tl.draw(el, at, dur || 500, "ease-out"); };

    // (a) Training loop first: skill z -> pi_Gen -> action -> parallel simulation -> rewards -> pi_Gen.
    var a = 0;
    fade(N("TextBox 451"), a);
    all(/^training-latent-prior$/).slice(1).forEach(function (el) { fade(el, a, 400); });
    fade(N("그룹 501"), a + 150, 500);
    fade(N("그룹 507"), a + 400, 400);
    draw(N("직선 화살표 연결선 455"), a + 600, 400);
    fade(N("사각형: 둥근 모서리 539"), a + 800, 400);
    fade(N("그룹 538"), a + 900, 300);
    draw(N("연결선: 꺾임 454"), a + 1100, 450);
    fade(N("사각형: 둥근 모서리 450"), a + 1400, 350);
    draw(N("연결선: 꺾임 462"), a + 1600, 450);
    fade(N("그림 461"), a + 1850, 500);
    fade(N("TextBox 457"), a + 2050);
    ["Picture 6", "그룹 470", "그림 466"].forEach(function (n, i) { fade(N(n), a + 2150 + i * 80); });
    fade(N("직사각형 542"), a + 2300, 250);
    fade(N("직사각형 543"), a + 2300, 250);
    draw(N("연결선: 꺾임 463"), a + 2350, 400);
    fade(N("reward-container"), a + 2600, 400);
    ["사각형: 둥근 모서리 459", "사각형: 둥근 모서리 460", "사각형: 둥근 모서리 468"].forEach(function (n, i) {
      fade(N(n), a + 2750 + i * 220, 400);
    });
    draw(N("연결선: 꺾임 453"), a + 3500, 550);

    // (a) Then the reverse curriculum that schedules the training: easy (near the goal) to hard (random initial state).
    tl.next();
    // The pink callout rising from the policy box introduces the curriculum, so it opens this phase.
    fade(N("callout"), 0, 500);
    fade(N("사각형: 둥근 모서리 578"), 0, 400);
    fade(N("TextBox 464"), 150);
    all(/^cue-text$/).forEach(function (el) { fade(el, 250); });
    draw(N("curriculum-progress"), 250, 1300);
    fade(N("Picture 10"), 150, 500);
    fade(N("curriculum-badge-2"), 500, 250);
    fade(N("curriculum-chevron-2"), 500, 250);
    fade(N("Picture 14"), 650, 500);
    fade(N("curriculum-badge"), 1000, 250);
    all(/^curriculum-chevron$/).forEach(function (el) { fade(el, 1000, 250); });
    fade(N("Picture 12"), 1150, 500);
    fade(N("TextBox 465"), 1250);

    // (b) The trained generator fans out into many successful rollouts.
    tl.next();
    var b = 0;
    fade(N("그림 718"), b, 600);
    fade(N("자유형: 도형 444"), b + 300, 400);
    fade(all(/^training-latent-prior$/)[0], b + 350, 350);
    fade(N("그룹 545"), b + 450, 400);
    fade(N("그룹 546"), b + 650, 350);
    fade(N("TextBox 446"), b + 650, 350);
    draw(N("직선 화살표 연결선 448"), b + 850, 300);
    fade(N("사각형: 둥근 모서리 445"), b + 1000, 350);
    // Rollouts: the four highlighted rollouts (white halo) shoot out one after another, then the 95 thin
    // ones fill the space between them with a single wipe (one mask per thin stroke made the page stutter).
    var scene = all(/^rollout_scene-\d+$/);
    var width = function (g) { return parseFloat(g.querySelector("path").getAttribute("stroke-width") || "0"); };
    var thin = scene.filter(function (g) { return width(g) < 1; });
    scene.forEach(function (g, i) {
      if (width(g) < 5) return;          // a highlight starts with its white halo stroke
      var k = (i - scene.indexOf(scene.filter(function (h) { return width(h) >= 5; })[0])) / 4;
      var t0 = b + 1200 + k * 420;
      draw(g, t0, 650);                  // halo
      draw(scene[i + 1], t0, 650);       // coloured line
      fade(scene[i + 2], t0, 200);       // start marker
      fade(scene[i + 3], t0 + 600, 200); // end marker
    });
    tl.wipe(thin, b + 1200 + 4 * 420 + 200, 900);
    fade(N("직사각형 424"), b + 3600, 400);
    fade(N("TextBox 426"), b + 3700, 500);

    // Success and safety filtering: skill streams merge, failures and unsafe rollouts drop out.
    tl.next();
    var f = 0;
    fade(N("header-success"), f, 400);
    all(/^stream-skill-\d$/).forEach(function (g, i) { tl.wipe([g], f + 100 + i * 70, 550, "x"); });
    fade(N("그룹 2"), f + 400, 300);
    tl.wipe([N("link-successful")], f + 650, 550, "x");
    fade(N("label-successful"), f + 900, 300);
    tl.wipe([N("link-failed")], f + 750, 550, "x");
    fade(N("x-failed"), f + 1100, 250);
    fade(N("label-failed"), f + 1100, 250);
    tl.wipe([N("link-safe")], f + 1250, 550, "x");
    fade(N("label-safe"), f + 1450, 300);
    tl.wipe([N("link-unsafe")], f + 1350, 550, "x");
    fade(N("x-unsafe"), f + 1700, 250);
    fade(N("label-unsafe"), f + 1700, 250);
    ["dsafe-body", "dsafe-top", "dsafe-label"].forEach(function (n) { fade(N(n), f + 1750, 350); });

    // (c) The retained data train the visuomotor policy.
    tl.next();
    var c = 0;
    fade(N("TextBox 428"), c, 400);
    fade(N("그룹 588"), c + 200, 500);
    fade(N("TextBox 589"), c + 400, 400);
  });
})();
