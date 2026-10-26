/* Mobile navigation for the three original generation videos. */
(function () {
  'use strict';
  var phone=matchMedia('(max-width: 760px)');
  var generators=Array.from(document.querySelectorAll('.gen'));
  var tabs=Array.from(document.querySelectorAll('[data-generator]')),current=0;
  function choose(i){
    current=i;
    tabs.forEach(function(t,j){t.setAttribute('aria-selected',String(j===i));t.tabIndex=j===i?0:-1;});
    generators.forEach(function(g,j){g.hidden=phone.matches&&j!==i;if(g.hidden)g.querySelector('video').pause();});
  }
  tabs.forEach(function(tab,i){
    tab.addEventListener('click',function(){choose(i);});
    tab.addEventListener('keydown',function(e){if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();var j=(i+(e.key==='ArrowRight'?1:2))%3;tabs[j].focus();choose(j);}});
  });
  // No native controls: the three clips play in sync with their path traces (figs/intro.js).
  phone.addEventListener('change',function(){choose(current);});choose(0);
})();
