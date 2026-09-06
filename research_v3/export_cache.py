"""Export all symbols without outcome selection. No strategy or optimization."""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

month=sys.argv[1]
root=Path('cache')/month
root.mkdir(parents=True,exist_ok=True)
receipts=[]
for p in sorted(Path(month).glob('part-*/date=*/minute1.parquet')):
    day=p.parent.name.split('=')[1]
    df=pq.ParquetFile(p).read().to_pandas()
    codes,idx=np.unique(df.code.to_numpy(),return_inverse=True)
    times=pd.to_datetime(df.datetime)
    minute=(times.dt.hour*60+times.dt.minute).to_numpy()
    am=(minute>=571)&(minute<=690)
    pm=(minute>=781)&(minute<=900)
    good=am|pm
    slots=np.where(am,minute-571,minute-781+120)
    cols=['open','high','low','close','volume','amount']
    vals=df[cols].to_numpy(float)
    assert np.isfinite(vals).all() and (vals[:,:4]>0).all()
    assert not df.duplicated(['code','datetime']).any()
    assert (times.dt.strftime('%Y-%m-%d')==day).all()
    x=np.full((len(codes),240,6),np.nan,dtype=np.float64)
    x[idx[good],slots[good]]=vals[good]
    y=x.reshape(len(codes),48,5,6)
    five=np.stack([y[:,:,0,0],np.max(y[:,:,:,1],axis=2),np.min(y[:,:,:,2],axis=2),y[:,:,-1,3],y[:,:,:,4].sum(axis=2),y[:,:,:,5].sum(axis=2)],axis=2)
    complete=np.isfinite(y).all(axis=(2,3))
    five[~complete]=np.nan
    # Bucket k executes only signals known by bucket k-1 close.
    # first[k] is the actual 1-minute bar ending 09:31+5k etc.
    out=root/(day+'.npz')
    np.savez_compressed(out,codes=codes.astype('U6'),five=five,first=x[:,::5,:])
    receipts.append(dict(date=day,source=str(p),raw_rows=len(df),codes=len(codes),quarantined=int((~good).sum()),source_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),cache_sha256=hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),cache_bytes=out.stat().st_size))
    print(day,len(df),out.stat().st_size,flush=True)
(root/'manifest.json').write_text(json.dumps(dict(source_commit='4de6d9311a961e05661c6914d6972544a812caf2',run_commit=os.getenv('GITHUB_SHA'),run_id=os.getenv('GITHUB_RUN_ID'),dtype='float64',schema=['open','high','low','close','volume','amount'],decision='completed five-minute bars',execution='first one-minute bar of subsequent bucket',receipts=receipts),indent=2))
