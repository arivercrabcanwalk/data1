from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v18_results'
A=pd.read_parquet(ROOT/'v17_results/action_candidates.parquet').copy()
A['date']=pd.to_datetime(A.date)
A=A[~A.action.str.startswith('PRICE_')&A.t1_ret.notna()].copy()

def met(z):
    r=z.t1_ret
    if len(r)==0:
        return {'n':0}
    return {
        'n':int(len(r)),'win_rate':float((r>0).mean()),'mean':float(r.mean()),
        'median':float(r.median()),'bad3':float((r<=-.03).mean()),'worst':float(r.min())
    }

def split(z):
    return {
        'train_jan_jun':met(z[(z.date>='2026-01-01')&(z.date<='2026-06-30')]),
        'pseudo_oos_jul_sep':met(z[(z.date>='2026-07-01')&(z.date<='2026-09-30')])
    }

def main():
    rejected={}

    generic=A[(A.branch=='NORMAL_RED')&(A.bucket=='B4P')&(A.action=='HOLD1')]
    rejected['NORMAL_B4P_HOLD1_GENERIC']={
        'metrics':split(generic),
        'reason':'Train was perfect, but Jul-Sep contains a -11.76% loss. Five-minute repair remains safer.'
    }

    panic=A[(A.branch=='PANIC')&(A.bucket=='B3')&(A.action=='RECLAIM1')&A.d2_gap.between(0,.03)]
    rejected['PANIC_B3_RECLAIM_D2_GAP_0_3']={
        'metrics':split(panic),
        'reason':'Strong Jan-Jun pattern reverses in Jul-Sep; rejected for regime instability.'
    }

    base=A[(A.branch=='NORMAL_RED')&(A.bucket=='B4P')&(A.action=='HOLD1')]
    neighborhood=[]
    for amount_min in [5e8,7.5e8,8e8,9e8,9.5e8,1e9,1.05e9,1.1e9,1.2e9,1.5e9,2e9]:
        z=base[base.amount>=amount_min]
        neighborhood.append({'amount_min':amount_min,**split(z)})
    rejected['NORMAL_B4P_HOLD1_AMOUNT_FILTER']={
        'neighborhood':neighborhood,
        'reason':'The apparently perfect >=1.0bn result is threshold-fragile: a Jul-Sep loser has Day1 amount ~0.997bn, so 0.95bn includes it while 1.0bn excludes it. Not promoted.'
    }

    accepted=A[
        (A.branch=='POSITIVE_BREAK')&(A.bucket=='B4P')&(A.action=='RECLAIM1')&
        (A.gap<=.05)&(A.vol_prev>=1.5)&(A.signal_time<='09:35')
    ]
    accepted_record={
        'POS_4P_VOLUME_RECLAIM35':{
            'metrics':split(accepted),
            'reason':'Kept only as temporally-confirmed experimental; neighborhood in v18_summary tests nearby volume/gap/time cutoffs.'
        }
    }

    payload={
        'status':'ANTI_OVERFIT_AUDIT',
        'principle':'A strong Jan-Jun score is insufficient. Jul-Sep temporal behavior and threshold neighborhoods are used to reject fragile expansions.',
        'rejected':rejected,
        'accepted_experimental_reference':accepted_record
    }
    (OUT/'robustness_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()