/* Animated disco lights for the final wordmark. */
(function () {
  'use strict';
  var hero=document.querySelector('.hero .wordmark'), stage=hero.querySelector('.logo-party');
  var canvas=stage.querySelector('canvas'), ctx=canvas.getContext('2d');
  var reduced=matchMedia('(prefers-reduced-motion: reduce)'), visible=false, raf=0, hover=false, boostUntil=0, previous=0;
  var width=0,height=0,dpr=1;
  function resize() {
    var box=stage.getBoundingClientRect();
    if (!box.width) return;
    width=box.width*1.32; height=box.height*1.55; dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  function draw(ms) {
    if (!width) resize();
    if (!width) return;
    ctx.clearRect(0,0,width,height);
    var strong=hover||ms<boostUntil, time=ms/1000, cx=width*.73, cy=height*.66;
    var colors=['#ef559f','#9466e5','#37bfc9','#e9ae46'];
    // Elliptical orbits imitate reflections from a turning mirrorball.
    var count=strong?28:18;
    for(var i=0;i<count;i++) {
      var angle=time*(strong?.72:.38)+i*2.39996;
      var radius=.45+.55*((i*7%13)/13);
      var x=width*.49+Math.cos(angle)*width*.46*radius;
      var y=height*.52+Math.sin(angle)*height*.43*radius;
      var pulse=.5+.5*Math.sin(time*1.7+i*1.8), size=(strong?3.3:2.5)+pulse*2.1;
      ctx.save();ctx.translate(x,y);ctx.rotate(angle*.65);
      ctx.globalAlpha=(strong?.52:.24)+pulse*.34;ctx.fillStyle=colors[i%4];
      ctx.shadowColor=colors[i%4];ctx.shadowBlur=strong?12:6;
      if(i%4===0) {
        ctx.beginPath();ctx.moveTo(0,-size*1.8);ctx.lineTo(size*.4,-size*.4);ctx.lineTo(size*1.8,0);ctx.lineTo(size*.4,size*.4);ctx.lineTo(0,size*1.8);ctx.lineTo(-size*.4,size*.4);ctx.lineTo(-size*1.8,0);ctx.lineTo(-size*.4,-size*.4);ctx.closePath();
      } else {ctx.beginPath();ctx.roundRect(-size*.5,-size*.3,size,size*.6,1);}
      ctx.fill();ctx.restore();
    }
    // Broad, translucent light fans stay behind the readable wordmark.
    for(var j=0;j<3;j++) {
      var a=time*.3+j*2.094, ex=cx+Math.cos(a)*width*.67, ey=cy+Math.sin(a)*height*.66;
      var g=ctx.createLinearGradient(cx,cy,ex,ey);g.addColorStop(0,colors[j]+'24');g.addColorStop(1,colors[j]+'00');
      ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex-13,ey-12);ctx.lineTo(ex+13,ey+12);ctx.closePath();ctx.fill();
    }
    hero.classList.toggle('party-boost',strong);
  }
  function stop() {cancelAnimationFrame(raf);raf=0;}
  function loop(ms) {
    raf=0;
    if(!visible||document.hidden||document.body.dataset.logoVariant!=='party'||reduced.matches)return;
    if(ms-previous>32){draw(ms);previous=ms;}
    raf=requestAnimationFrame(loop);
  }
  function update() {
    stop();
    var active=document.body.dataset.logoVariant==='party';
    hero.tabIndex=active?0:-1;
    if(active) {resize();if(reduced.matches)ctx.clearRect(0,0,width,height);else if(visible&&!document.hidden)raf=requestAnimationFrame(loop);}
  }
  new IntersectionObserver(function(entries){visible=entries[0].isIntersecting;hero.classList.toggle('party-visible',visible);update();}).observe(hero);
  new ResizeObserver(resize).observe(stage);
  document.addEventListener('visibilitychange',update);
  reduced.addEventListener('change',update);
  function boost(){if(document.body.dataset.logoVariant==='party'&&!reduced.matches){boostUntil=performance.now()+4500;draw(performance.now());}}
  hero.addEventListener('pointerenter',function(e){if(e.pointerType!=='touch')hover=true;});
  hero.addEventListener('pointerleave',function(){hover=false;});
  hero.addEventListener('click',boost);
  hero.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){if(document.body.dataset.logoVariant==='party')e.preventDefault();boost();}});
  update();
})();
