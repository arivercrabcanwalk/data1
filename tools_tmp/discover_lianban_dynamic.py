import re, requests
from urllib.parse import urljoin

URL='https://lianban.net/days/2026-07-01.html'
s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'
r=s.get(URL,timeout=30); print('PAGE',r.status_code,len(r.text)); html=r.text
for pat in ['热点题材','opendata','fetch(','axios','/api/','theme','limitup','lianban','days/']:
    if pat in html: print('INLINE_HAS',pat)
for src in re.findall(r'<script[^>]+src=["\']([^"\']+)',html,re.I):
    u=urljoin(URL,src); print('SCRIPT',u)
    try:
        x=s.get(u,timeout=30); print('SCRIPT_STATUS',x.status_code,len(x.text))
        for line in x.text.splitlines():
            low=line.lower()
            if any(k in low for k in ['fetch(','axios','/api/','theme','hotspot','limit_up','limitup','review','day-data','daily']):
                print('HIT',line[:1000])
    except Exception as e: print('ERR',u,repr(e))
# Also surface URL-like strings from inline HTML/JS.
for m in sorted(set(re.findall(r'["\']([^"\']*(?:api|theme|review|hot|limit|board)[^"\']*)["\']',html,re.I))):
    if len(m)<300: print('URLLIKE',m)
