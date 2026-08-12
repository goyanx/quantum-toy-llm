from __future__ import annotations
import csv, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
from quantum_toy_llm.data import CharDataset
from quantum_toy_llm.model import TinyGPT, TinyGPTConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'ansatz_mps_raw.csv'
SEEDS=[1337,2026,4242]
STEPS=300; BATCH=4; EVAL=20; LR=3e-3; WORKERS=4

CONFIGS=[
 ('bottleneck4', dict(mode='bottleneck', n_qubits=4, circuit_layers=2)),
 ('quantum_base', dict(mode='quantum', n_qubits=4, circuit_layers=2, quantum_reupload=False, quantum_topology='ring')),
 ('quantum_reupload_ring', dict(mode='quantum', n_qubits=4, circuit_layers=2, quantum_reupload=True, quantum_topology='ring')),
 ('quantum_reupload_alt', dict(mode='quantum', n_qubits=4, circuit_layers=2, quantum_reupload=True, quantum_topology='alternating')),
 ('mps_r2', dict(mode='mps', n_qubits=4, circuit_layers=2, tt_rank=2)),
 ('mps_r4', dict(mode='mps', n_qubits=4, circuit_layers=2, tt_rank=4)),
 ('mps_r8', dict(mode='mps', n_qubits=4, circuit_layers=2, tt_rank=8)),
]

def eval_model(m, ds):
 m.eval(); ls=[]
 with torch.no_grad():
  for _ in range(EVAL):
   x,y=ds.batch('val',BATCH,torch.device('cpu')); _,loss=m(x,y); ls.append(float(loss))
 loss=sum(ls)/len(ls); return loss, math.exp(min(loss,20))

def run(spec):
 torch.set_num_threads(1); torch.manual_seed(spec['seed'])
 ds=CharDataset(block_size=32)
 kw=dict(vocab_size=ds.vocab_size,block_size=32,d_model=32,n_heads=4,n_layers=1,dropout=0.0)
 kw.update(spec['cfg'])
 model=TinyGPT(TinyGPTConfig(**kw))
 opt=torch.optim.AdamW(model.parameters(),lr=LR)
 st=time.perf_counter()
 for _ in range(STEPS):
  x,y=ds.batch('train',BATCH,torch.device('cpu')); _,loss=model(x,y)
  opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
 sec=time.perf_counter()-st; vl,ppl=eval_model(model,ds)
 return dict(name=spec['name'],seed=spec['seed'],parameters=model.parameter_count(),val_loss=vl,perplexity=ppl,seconds=sec)

def main():
 specs=[dict(name=n,cfg=c,seed=s) for n,c in CONFIGS for s in SEEDS]
 rows=[]; print(f'Running {len(specs)} jobs with max_workers={WORKERS}',flush=True)
 with ProcessPoolExecutor(max_workers=WORKERS) as ex:
  fs={ex.submit(run,s):s for s in specs}
  for i,f in enumerate(as_completed(fs),1):
   r=f.result(); rows.append(r); print(f"[{i:02}/{len(specs)}] {r['name']:22s} seed={r['seed']} ppl={r['perplexity']:.3f} t={r['seconds']:.2f}s",flush=True)
 rows.sort(key=lambda r:(r['name'],r['seed']))
 with OUT.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['name','seed','parameters','val_loss','perplexity','seconds']); w.writeheader(); w.writerows(rows)
 print('Wrote',OUT)
if __name__=='__main__': main()
