import re, requests
from urllib.parse import urljoin

# Lianban raw fetch is intentionally probed only to document that the browser-rendered archive
# cannot be reproduced from a GitHub runner. We do not bypass its access controls.
URL='https://lianban.net/days/2026-07-01.html'
s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'
r=s.get(URL,timeout=30); print('LIANBAN_PAGE',r.status_code,len(r.text))

# Name->ticker resolution is metadata only, not a trading signal. Probe Eastmoney's public
# A-share list so manually audited PIT member names can be mapped to the GitHub minute-data code.
api='https://80.push2.eastmoney.com/api/qt/clist/get'
params={
    'pn':1,'pz':6000,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f3',
    'fs':'m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23','fields':'f12,f14'
}
try:
    x=s.get(api,params=params,timeout=30)
    print('EASTMONEY',x.status_code,len(x.text))
    if x.status_code==200:
        j=x.json(); diff=((j.get('data') or {}).get('diff') or [])
        mp={str(v.get('f14')):str(v.get('f12')).zfill(6) for v in diff if v.get('f14') and v.get('f12')}
        print('MAPPED',len(mp))
        for n in ['多氟多','埃斯顿','海南海药','大名城','立新能源','爱丽家居','一鸣食品','泛微网络']:
            print('NAME_CODE',n,mp.get(n))
except Exception as e:
    print('EASTMONEY_ERR',repr(e))
