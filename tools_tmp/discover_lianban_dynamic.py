import requests

s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'

def probe(name,url,headers=None):
    try:
        r=s.get(url,headers=headers or {},timeout=20)
        print(name,'STATUS',r.status_code,'LEN',len(r.content),'HEAD',r.content[:250])
    except Exception as e:
        print(name,'ERR',repr(e))

probe('TENCENT','https://qt.gtimg.cn/q=sh600000,sz000001,sz002407,sz002747')
probe('SINA','https://hq.sinajs.cn/list=sh600000,sz000001,sz002407,sz002747',{'Referer':'https://finance.sina.com.cn/'})
probe('EASTMONEY_PUSH2','https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=20&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:1+t:2&fields=f12,f14')
