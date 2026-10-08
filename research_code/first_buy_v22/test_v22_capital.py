# -*- coding: utf-8 -*-
import pandas as pd
import v22_capital_shadow as m
def x(eid,code,entry,exit):
 return {'event_id':eid,'code':code,'name':code,'d2_date':'2026-10-09','entry_clock':'09:33',
         'entry_price':entry,'t1_date':'2026-10-12','exit_clock':'15:00','exit_price':exit,
         'registered_before_day2':True}
def run():
 rows=[x('e1','000001',10,11),x('e2','000002',20,19)]
 d=pd.DataFrame([
  {'code':'000001','date':'2026-10-09','close':10.5},
  {'code':'000001','date':'2026-10-12','close':11.},
  {'code':'000002','date':'2026-10-09','close':20.},
  {'code':'000002','date':'2026-10-12','close':19.}
 ])
 a,q,l=m.simulate(rows,d,1,.0026)
 assert a['closed_executed']==1 and a['skipped']==1 and a['account_return']>0
 b,q,l=m.simulate(rows,d,2,.0026)
 assert b['closed_executed']==2 and b['skipped']==0
 assert b['account_return']>0
 # Changing future sale must not change which ticker is bought.
 rows2=[x('e1','000001',10,6),x('e2','000002',20,25)]
 _,q1,l1=m.simulate(rows,d,1,.0026)
 _,q2,l2=m.simulate(rows2,d,1,.0026)
 assert l1[l1.status=='BOUGHT'].event_id.tolist()==l2[l2.status=='BOUGHT'].event_id.tolist()
 m.verify()
 print('V22_CAPITAL_TEST_OK chronological slots, cash, fees, no future-outcome buy choices')
if __name__=='__main__':run()