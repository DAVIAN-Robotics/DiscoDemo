/* Left-hand contents: on wide screens the header links move to a fixed column on the left and the
 * section in view is marked. The page itself scrolls normally. */
(function () {
  "use strict";
  var HEAD = 64;        // sticky header, same as scroll-padding-top
  function top(e) { return Math.round(e.getBoundingClientRect().top + window.scrollY - HEAD); }

  var links = Array.prototype.slice.call(document.querySelectorAll(".site-head nav a"));
  var toc = document.createElement("nav");
  toc.className = "toc";
  toc.setAttribute("aria-label", "Contents");
  var items = links.map(function (a) {
    var c = a.cloneNode(true);
    toc.appendChild(c);
    return { a: c, sec: document.querySelector(a.getAttribute("href")) };
  });
  document.body.appendChild(toc);
  var ticking = false;
  function mark() {
    ticking = false;
    var y = window.scrollY + window.innerHeight * 0.35, cur = null;
    items.forEach(function (it) { if (top(it.sec) + HEAD <= y) cur = it; });
    items.forEach(function (it) { it.a.classList.toggle("on", it === cur); });
  }
  window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(mark); } }, { passive: true });
  mark();

  // The More Research menu closes on a click elsewhere or Escape.
  var more = document.querySelector(".more");
  if (more) {
    document.addEventListener("click", function (e) { if (more.open && !more.contains(e.target)) more.open = false; });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") more.open = false; });
  }
})();
