"""Freeze new confirmation/audit graphs before parallel efficiency tuning."""
from pathlib import Path
import json,copy,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import runtime as p
import numpy as np
OUT=p.OUT.parent/'123bus_training_efficiency_2026-09-16'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    protocol=p.read(p.PROTOCOL);excluded=set(protocol['excluded_graphs']);reserved=set(excluded)
    historical=list(p.OUT.glob('seed*/training.jsonl'))+list((p.OUT.parent/'topology_efficiency').glob('seed*_123_stream_*/training.jsonl'))
    for path in historical:
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);scene=row.get('scene',row.get('exposure',{}).get('scene',{}))
            graph=scene.get('topology_id',row.get('exposure',{}).get('topology_id'))
            if graph:reserved.add(graph)
    env=p.Feeder();rng=np.random.default_rng(2026091699);counts=dict(graph_draws=0,reset_attempts=0,reset_failures=0);start=time.perf_counter()
    for group,size,offset in [('confirmation',128,120000000),('audit_fresh',512,140000000)]:
        rows=[]
        for i in range(size):
            topo=p.draw_graph(env,rng,reserved,counts);scene,_=p.scene_for(env,topo,rng,offset+i,counts)
            reserved.add(topo['id']);excluded.add(topo['id']);rows.append(scene)
        protocol['groups'][group]=rows;print(group,len(rows),'seconds',time.perf_counter()-start,flush=True)
    protocol['excluded_graphs']=sorted(excluded);protocol['fresh_holdout_preparation']=dict(seed=2026091699,counts=counts,historical_paths=[str(x) for x in historical])
    p.save_json(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=p.init_worker) as pool:
        for group in ('anchor','confirmation'):
            rows=p.evaluate(pool,None,protocol['groups'][group],reference=True)
            p.save_json(OUT/f'reference_{group}.json',dict(protocol_sha256=p.sha(OUT/'protocol.json'),rows=rows,metrics=p.metrics(rows)))
            print('reference',group,p.metrics(rows),flush=True)
    p.save_json(OUT/'preparation.json',dict(protocol_sha256=p.sha(OUT/'protocol.json'),seconds=time.perf_counter()-start,counts=counts,final_audit_evaluated=False))
if __name__=='__main__':main()
