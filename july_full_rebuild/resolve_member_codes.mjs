import fs from 'fs';
import path from 'path';
import { StockSDK } from 'stock-sdk';

const base = path.dirname(new URL(import.meta.url).pathname);
const curated = path.join(base, 'curated');
const outDir = path.join(base, 'pit_data');
fs.mkdirSync(outDir, {recursive:true});

const rows=[];
for (const f of fs.readdirSync(curated).filter(x=>x.endsWith('.jsonl')).sort()) {
  const txt=fs.readFileSync(path.join(curated,f),'utf8');
  for (const line of txt.split(/\r?\n/).filter(Boolean)) rows.push(JSON.parse(line));
}
const names=new Set();
for (const d of rows) {
  names.add(d.dragon);
  for (const t of d.themes) { names.add(t.leader); for (const n of t.members) names.add(n); }
}
const sdk=new StockSDK();
const result=new Map();
const unresolved=[];
const all=[...names].filter(Boolean);

async function one(name) {
  try {
    const hits=await sdk.search(name);
    const exact=(hits||[]).filter(x=>x && x.name===name && x.type==='GP-A' && /^(sh|sz|bj)\d{6}$/i.test(x.code||''));
    if (exact.length) {
      const h=exact[0];
      result.set(name,{name,code:h.code.slice(-6),market:h.market||h.code.slice(0,2),source:'stock-sdk exact GP-A'});
    } else unresolved.push({name,hits:(hits||[]).slice(0,5)});
  } catch(e) { unresolved.push({name,error:String(e)}); }
}

const concurrency=8;
let cursor=0;
async function worker(){ while(true){ const i=cursor++; if(i>=all.length) return; await one(all[i]); if(i%50===0) console.log('resolved progress',i,'/',all.length); } }
await Promise.all(Array.from({length:concurrency},()=>worker()));

const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';
let csv='name,code,market,source\n';
for (const name of [...result.keys()].sort((a,b)=>a.localeCompare(b,'zh-CN'))) {
  const r=result.get(name); csv += [r.name,r.code,r.market,r.source].map(esc).join(',')+'\n';
}
fs.writeFileSync(path.join(outDir,'member_name_code.csv'),csv,'utf8');
fs.writeFileSync(path.join(outDir,'unresolved_names.json'),JSON.stringify(unresolved,null,2),'utf8');
const ratio=result.size/all.length;
console.log(JSON.stringify({unique_names:all.length,resolved:result.size,unresolved:unresolved.length,ratio},null,2));
if (ratio < 0.90) { console.error('resolution ratio below 90%'); process.exit(2); }
