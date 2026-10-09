import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser()
parser.add_argument('url');parser.add_argument('--screenshots',default='/tmp/discodemo-task-options')
args=parser.parse_args();url=args.url
out=Path(args.screenshots);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
 page=b.new_page(viewport={'width':1440,'height':1050},reduced_motion='reduce'); errors=[];bad=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:bad.append(r.url) if r.status>=400 and r.url.startswith(url.rsplit('/',1)[0]) else None)
 page.goto(url,wait_until='domcontentloaded')
 assert page.locator('.task-picker').count()==4
 for root in ['sim-player','real-player','gen-player','traj']:
  picker=page.locator('#'+root+' .task-choices')
  for task in ['banana','cube','round','sqcirc']:
   button=picker.locator('[data-task='+task+']');button.click()
   assert button.get_attribute('aria-selected')=='true'
   assert picker.locator('[aria-selected=true]').count()==1
   if root!='traj':
    videos=page.locator('#'+root+' .grid video')
    assert videos.count()==8
    assert all('/'+task+'/' in s for s in videos.evaluate_all('(vs)=>vs.map(v=>v.src)'))
   else:
    page.wait_for_function('(t)=>!!window.TRAJ[t]',arg=task)
  picker.locator('[data-task=banana]').click()
 page.locator('#sim-title').evaluate('(e)=>window.scrollTo(0,e.getBoundingClientRect().top+scrollY-115)')
 page.wait_for_timeout(400)
 page.wait_for_function("[...document.querySelectorAll('#sim-player .task-thumbnail')].every(i=>i.complete&&i.naturalWidth>0)")
 for theme in ['light','dark']:
  page.evaluate('(t)=>document.documentElement.dataset.theme=t',theme)
  page.screenshot(path=str(out/('photos-'+theme+'.png')))
 # Keyboard and touch selection on the fixed photo tabs.
 picker=page.locator('#sim-player .task-choices');picker.locator('[data-task=banana]').focus();page.keyboard.press('ArrowRight')
 assert picker.locator('[data-task=cube]').get_attribute('aria-selected')=='true'
 assert page.locator('.design-preview,.task-icon').count()==0
 phone=b.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,reduced_motion='reduce')
 phone.on('pageerror',lambda e:errors.append(str(e)))
 phone.goto(url,wait_until='domcontentloaded')
 phone.locator('#sim-title').evaluate('(e)=>window.scrollTo(0,e.getBoundingClientRect().top+scrollY-90)')
 phone.wait_for_timeout(300)
 phone.screenshot(path=str(out/'photos-mobile.png'))
 assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth')
 phone.locator('#sim-player [data-task=cube]').tap()
 assert phone.locator('#sim-player [data-task=cube]').get_attribute('aria-selected')=='true'
 assert phone.locator('#sim-player .task-thumbnail').first.is_visible()
 assert not errors,errors
 assert not bad,bad
 print('PASS: fixed photo tabs, 4 pickers × 4 tasks, actual video sources, thumbnails, keyboard/touch, themes, mobile overflow; no JS/HTTP errors.')
 b.close()
