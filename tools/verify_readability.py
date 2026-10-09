"""Verify original figures remain visible and mobile generation tabs work."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--screenshots',default='/tmp/discodemo-figure-restore/final');a=p.parse_args();out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
 page=b.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce');errors=[];bad=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:bad.append(r.url) if r.status>=400 and r.url.startswith(a.url.rsplit('/',1)[0]) else None)
 page.goto(a.url,wait_until='domcontentloaded')
 assert page.locator('.idea-flow,.method-flow,.analysis-panel,.horizontal-bars').count()==0
 for key in ['motivation','method','dataset','alpha','rl_time','onpolicy']:
  host=page.locator('[data-fig="'+key+'"]');host.scroll_into_view_if_needed()
  assert host.locator(':scope > svg:not(.fig-layer)').count()==1,key
  assert host.evaluate('(e)=>!e.closest("details")'),key
  assert page.evaluate('key=>{const original=new DOMParser().parseFromString(FIGS[key],\'image/svg+xml\').documentElement,shown=document.querySelector(\'[data-fig="\'+key+\'"] > svg\');return original.getAttribute(\'viewBox\')===shown.getAttribute(\'viewBox\')&&[...original.querySelectorAll(\'[id]\')].every(e=>!!shown.parentNode.querySelector(\'[id="\'+e.id+\'"]\'));}',key),key
 assert page.locator('#wall').evaluate('(e)=>!e.closest("details")')
 for label in ['Simulation','Real robot']:
  page.locator('#results-chart [data-role=setting]').get_by_role('tab',name=label,exact=True).click()
  assert page.locator('#results-chart svg').count()>0
 hide='.site-head,.design-preview,.toc,body::before {visibility:hidden!important;}'
 for theme in ['light','dark']:
  page.evaluate('(t)=>document.documentElement.dataset.theme=t',theme)
  for name in ['method','results','analysis','questions']:
   page.locator('#'+name).screenshot(path=str(out/(theme+'-'+name+'.png')),style=hide)
 phone=b.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
 phone.on('pageerror',lambda e:errors.append(str(e)));phone.goto(a.url,wait_until='domcontentloaded')
 for i in range(3):
  phone.locator('[data-generator="'+str(i)+'"]').tap()
  assert phone.locator('.gen:not([hidden])').count()==1
  phone.wait_for_function('(i)=>document.querySelectorAll(".gen video")[i].currentTime>.1',arg=i)
  assert phone.locator('.gen[hidden] video').evaluate_all('(vs)=>vs.every(v=>v.paused)')
 phone.emulate_media(reduced_motion='reduce');phone.reload(wait_until='domcontentloaded')
 for name in ['method','results','analysis','questions']:
  phone.locator('#'+name).screenshot(path=str(out/('mobile-'+name+'.png')),style=hide)
  assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth'),name
 assert not errors,errors
 assert not bad,bad
 print(json.dumps({'result':'PASS','figures':'6 complete source SVGs with original viewBoxes and IDs, directly visible','results':'original charts in both settings','mobile':'no overflow; 3 generation tabs play/pause','js_errors':errors,'http_errors':bad}))
 b.close()
