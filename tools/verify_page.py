"""Browser review: final lighting logo, clean generator videos, and matched-frame wipe."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--screenshots',default='/tmp/discodemo-revision2');a=p.parse_args()
out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
    b=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
    page=b.new_page(viewport={'width':1440,'height':1050})
    errors=[];bad=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:bad.append(r.url) if r.status>=400 and r.url.startswith(a.url.rsplit('/',1)[0]) else None)
    page.goto(a.url,wait_until='domcontentloaded');page.wait_for_timeout(600)
    assert page.locator('.design-preview,[data-logo],.logo-mirrorball,.logo-sparkle,.logo-tiles').count()==0
    # Saved preview choices must not override the final design.
    page.evaluate("sessionStorage.setItem('discodemo-logo-preview-v3','tiles');sessionStorage.setItem('discodemo-task-style','icons')")
    page.reload(wait_until='domcontentloaded')
    assert page.locator('body').get_attribute('data-logo-variant')=='party'
    assert page.locator('body').get_attribute('data-task-style')=='cards'
    for theme in ['light','dark']:
        page.evaluate('(t)=>document.documentElement.dataset.theme=t',theme)
        assert page.locator('.hero .logo-party').is_visible()
        assert page.locator('.site-head .logo-party').is_visible()
        page.wait_for_timeout(150);page.screenshot(path=str(out/f'{theme}-logo-party.png'))
    # The canvas must change while visible, intensify on hover/tap, and stop offscreen.
    party=page.locator('.hero .logo-party canvas')
    frame1=party.evaluate('(c)=>c.toDataURL()');page.wait_for_timeout(220)
    assert frame1!=party.evaluate('(c)=>c.toDataURL()')
    page.locator('.hero .wordmark').hover();page.wait_for_timeout(80)
    assert page.locator('.hero').locator('.party-boost').count()==1
    page.mouse.move(1400,950);page.wait_for_timeout(80)
    assert page.locator('.hero').locator('.party-boost').count()==0
    page.locator('.hero .wordmark').focus();page.keyboard.press('Enter');page.wait_for_timeout(80)
    assert page.locator('.hero').locator('.party-boost').count()==1
    page.reload(wait_until='domcontentloaded')
    assert page.locator('body').get_attribute('data-logo-variant')=='party'
    assert page.locator('.scale-matrix,.gen canvas,.track-badge,#gen-trails').count()==0
    assert page.locator('.scale-fig svg rect').count()==3
    assert page.locator('.sc-label').all_text_contents()==['Robot trajectories','Image–text pairs','Language tokens']
    assert page.locator('.dataset-figure').count()==0
    for name,sel in [('scale','.scale-fig'),('generators','.gen-grid')]:
        page.locator(sel).scroll_into_view_if_needed();page.wait_for_timeout(500)
        page.screenshot(path=str(out/f'final-{name}.png'))
    page.wait_for_timeout(100)
    stopped=page.locator('.hero .party-lights').evaluate('(c)=>c.toDataURL()')
    page.wait_for_timeout(180)
    assert stopped==page.locator('.hero .party-lights').evaluate('(c)=>c.toDataURL()')
    page.wait_for_function("[...document.querySelectorAll('.gen video')].every(v=>v.currentTime>.2 && !v.paused)")
    wipe=page.locator('#wipe');wipe.scroll_into_view_if_needed()
    page.wait_for_function("document.querySelector('#wipe video').currentTime>.5")
    page.locator('#wipe-play').click();page.wait_for_timeout(150)
    assert page.locator('#wipe video').count()==1
    assert abs(page.locator('#wipe video').evaluate('(v)=>v.duration')-15.1)<.001
    t=page.locator('#wipe video').evaluate('(v)=>v.currentTime')
    for camera,label,y in [('exo','External camera',36),('wrist','Wrist camera',516)]:
        page.locator('#wipe-camera').get_by_role('tab',name=label).click()
        assert page.locator('#wipe video').evaluate('(v)=>v.currentTime')==t
        # Every pixel of each full reveal must equal the corresponding crop of the SAME paused source frame.
        for key,x in [('End',0),('Home',640)]:
            wipe.press(key)
            equal=page.evaluate('''({x,y})=>{
              const v=document.querySelector('#wipe video'),got=document.querySelector('#wipe canvas');
              const expected=document.createElement('canvas');expected.width=640;expected.height=444;
              expected.getContext('2d').drawImage(v,x,y,640,444,0,0,640,444);
              const a=expected.getContext('2d').getImageData(0,0,640,444).data;
              const b=got.getContext('2d').getImageData(0,0,640,444).data;
              return a.every((value,i)=>value===b[i]);
            }''',{'x':x,'y':y})
            assert equal,(camera,key)
        for _ in range(10):wipe.press('ArrowRight')
        page.screenshot(path=str(out/f'paired-{camera}.png'))
    rect=wipe.bounding_box();page.mouse.move(rect['x']+rect['width']*.5,rect['y']+rect['height']*.5)
    page.mouse.down();page.mouse.move(rect['x']+rect['width']*.25,rect['y']+rect['height']*.5,steps=8);page.mouse.up()
    assert wipe.get_attribute('aria-valuenow')=='25'
    page.locator('#wipe-play').click();page.wait_for_timeout(350)
    assert page.locator('#wipe video').evaluate('(v)=>!v.paused && v.currentTime')>t
    page.locator('[data-replay=wipe]').click();page.wait_for_timeout(300)
    assert page.locator('#wipe video').evaluate('(v)=>v.currentTime')<1
    page.locator('#citation').scroll_into_view_if_needed();page.wait_for_timeout(350)
    assert page.locator('#wipe video').evaluate('(v)=>v.paused')
    for sel in ['#results','#sim-player','#real-player','#traj','#gen-player','#analysis']:
        page.locator(sel).scroll_into_view_if_needed();page.wait_for_timeout(300)
    phone=b.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,reduced_motion='reduce')
    phone.on('pageerror',lambda e:errors.append(str(e)))
    phone.goto(a.url,wait_until='domcontentloaded')
    assert phone.locator('.design-preview').count()==0
    assert phone.locator('.hero .logo-party').is_visible()
    assert phone.locator('body').evaluate('(e)=>getComputedStyle(e).paddingTop')=='0px'
    phone.screenshot(path=str(out/'mobile-logo-party.png'))
    assert phone.locator('.hero .party-lights').evaluate('(e)=>getComputedStyle(e).display')=='none'
    assert phone.locator('.hero .party-reflection').evaluate('(e)=>getComputedStyle(e).animationName')=='none'
    phone.emulate_media(reduced_motion='no-preference')
    phone.locator('.hero .wordmark').tap();phone.wait_for_timeout(100)
    assert phone.locator('.hero .party-boost').count()==1
    phone.emulate_media(reduced_motion='reduce')
    for name,sel in [('scale','.scale-fig'),('generators','.gen-grid'),('wipe','#wipe')]:
        phone.locator(sel).scroll_into_view_if_needed();phone.wait_for_timeout(350)
        assert phone.evaluate('document.documentElement.scrollWidth<=innerWidth'),name
        phone.screenshot(path=str(out/f'mobile-{name}.png'))
    assert phone.locator('#wipe video').evaluate('(v)=>v.paused')
    r=phone.locator('#wipe').bounding_box();phone.touchscreen.tap(r['x']+r['width']*.75,r['y']+r['height']*.5)
    assert abs(int(phone.locator('#wipe').get_attribute('aria-valuenow'))-75)<=1
    phone.locator('#wipe-play').click();phone.wait_for_function("document.querySelector('#wipe video').currentTime>.2")
    assert not errors,errors
    assert not bad,bad
    print(json.dumps({'result':'PASS','logos':1,'themes':2,'mobile':True,'pair':'episode 10, 302 frames, both cameras','pixel_check':'all four real/sim camera crops exactly match the same source frame','controls':['mouse','touch','keyboard','pause','restart','camera change','saved preview choices ignored','reduced motion','light animation and hover/tap boost','offscreen animation pause'],'javascript_errors':errors,'local_http_errors':bad}))
    b.close()
