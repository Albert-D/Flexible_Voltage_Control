"""Parallel joint peak/duration exploration; fixed policies and actual topology events."""
from pathlib import Path
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import maintenance as m
paths=m.paths
vis=m.vis
np=m.np
pd=m.pd
OUT=paths.DATA/'experiments/operational_path_shift/20260916_joint_search'
CURRENT=None
BUILD=m.ORIGINAL_BUILD
DUMP=paths.base.dump


def atomic_dump(path,payload):
    path=Path(path)
    temporary=path.with_name(path.name+'.partial')
    DUMP(temporary,payload)
    temporary.replace(path)


def build(runtime):
    if CURRENT['direction']=='maintenance':
        m.CURRENT=CURRENT
        return m.build_env(runtime)
    return BUILD(runtime)


def apply(env,case,k,r0,x0):
    if case['direction']=='maintenance':
        m.apply_case(env,case,k,r0,x0)
    else:
        paths.apply_case(env,case,k,r0,x0)


def install(case):
    global CURRENT
    CURRENT=case
    paths.base.build_env=build
    paths.base.apply_case=apply
    paths.base.ROOT=str(OUT/'cache')
    paths.base.dump=atomic_dump
    protocol=paths.context()[4]
    protocol['joint_entry_hash']=paths.base.digest(__file__)
    protocol['maintenance_helper_hash']=paths.base.digest(m.__file__)
    protocol['initialization']='full day: zero-injection startup under fixed scenario background; short branch-exchange screen: exact unchanged-prefix cached state'


def generate():
    answer=[]
    for rf,xf in ((1.25,1.25),(1.5,1.5),(2.,2.),(2.5,2.5),(2.5,1.5),(2.,2.5)):
        c=dict(m.cases()[1],id=f'm54_evening_r{rf:g}_x{xf:g}',r_factor=rf,x_factor=xf,full=True)
        answer.append(c)
    for line,event,tag in ((54,12279,'evening'),(48,7838,'midday'),(48,12279,'evening')):
        answer.append(dict(m.cases()[0],line=line,event=event,id=f'm{line}_{tag}_r2.5_x2',full=True))
    paths.base.build_env=BUILD
    for c in paths.cases():
        if c['direction']=='branch_exchange' and c['line']==30 and c['event']==7838:
            for fraction in (.5,.75,1.):
                answer.append(dict(c,id=f'{c["id"]}_z{fraction:g}',tie_r=c['tie_r']*fraction,
                                   tie_x=c['tie_x']*fraction,tie_fraction=fraction,full=False,
                                   assumption='Hypothetical normally-open maintenance tie; R/X is stated fraction of original endpoint path, not verified SCE asset'))
    return answer


def run_one(case,method,full):
    install(case)
    summary=paths.base.run(case,method,full_day=full)
    stage='full' if full else 'screen'
    p=vis.load(OUT/'cache'/stage/f'{case["id"]}__{method}__original.pkl.gz')
    if 'result' in p:
        summary=dict(case_id=case['id'],method=method,status='ok',window='full_day' if full else 'event_1h',
                     elapsed=p['result']['elapsed'],**vis.metrics(p['result']))
    else:
        summary=dict(summary,window=stage)
    return summary


def all_cases():
    result={}
    for file in sorted((OUT/'batches').glob('*/manifest.json')):
        for c in json.loads(file.read_text())['cases']:
            if c['id'] in result: assert result[c['id']]==c
            result[c['id']]=c
    return list(result.values())


def table():
    rows=[]
    for c in all_cases():
        for stage in ('screen','full'):
            for method in ('No control',*vis.METHODS):
                path=OUT/'cache'/stage/f'{c["id"]}__{method}__original.pkl.gz'
                if not path.exists(): continue
                p=vis.load(path)
                if 'result' not in p:
                    rows.append(dict(case_id=c['id'],method=method,window=stage,status='failed',error=p['summary']['error']))
                else:
                    rows.append(dict(case_id=c['id'],method=method,window='full_day' if stage=='full' else 'event_1h',
                                     status='ok',**vis.metrics(p['result'])))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'all_metrics.csv',index=False)
    scores=[]
    if not frame.empty:
        for (cid,window),group in frame.groupby(['case_id','window']):
            rr={r['method']:r for r in group.to_dict('records') if r['status']=='ok'}
            if not set(vis.METHODS).issubset(rr):continue
            r,l,s=rr['RLC-FT'],rr['Linear'],rr['Safe-DDPG']
            scores.append(dict(case_id=cid,window=window,rlc_peak=r['peak'],linear_peak=l['peak'],safe_peak=s['peak'],
                rlc_duration=r['duration'],linear_duration=l['duration'],safe_duration=s['duration'],
                peak_gap=min(l['peak'],s['peak'])-r['peak'],rlc_minimum=r['minimum'],
                cost_ratio_linear=r['cost']/l['cost'],cost_ratio_safe=r['cost']/s['cost'],
                joint_target=bool(r['passes'] and l['duration']>=5 and s['duration']>=5 and min(l['peak'],s['peak'])-r['peak']>=.01)))
    pd.DataFrame(scores).to_csv(OUT/'case_scores.csv',index=False)
    return frame,pd.DataFrame(scores)


def review():
    vis.style(); m.OUT=OUT
    frame,scores=table()
    for c in all_cases():
        rr={}
        for method in ('No control',*vis.METHODS):
            path=OUT/'cache/full'/f'{c["id"]}__{method}__original.pkl.gz'
            if path.exists():
                p=vis.load(path)
                if 'result'in p:rr[method]=p['result']
        if len(rr)==4:
            if c['direction']=='maintenance':
                m.figure(c['id'],c,rr,paths.context()[2])
            else:
                # Reuse the original 2x3 plot, then relabel the scenario accurately.
                import matplotlib.pyplot as plt
                oldsave=m.plt.close
                def save_transfer(fig=None):
                    if hasattr(fig,'texts'):
                        for t in fig.texts:
                            if t.get_text().startswith('Maintenance isolation:'):
                                t.set_text(f'Maintenance supply transfer: {c["event"]/600:.2f} h | Metrics: full 24 hours')
                        for ax in fig.axes[:3]:
                            for line in list(ax.lines):
                                x=np.asarray(line.get_xdata())
                                if x.size==2 and np.all(x==0) and line.get_linestyle()=='--':line.remove()
                        for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{c["id"]}_2x3.{ext}',dpi=210,facecolor='white')
                    oldsave(fig)
                m.plt.close=save_transfer
                try:m.figure(c['id'],dict(c,outage_start=0),rr,paths.context()[2])
                finally:m.plt.close=oldsave
    import nbformat as nbf
    from nbclient import NotebookClient
    nb=nbf.v4.new_notebook()
    nb.cells=[nbf.v4.new_markdown_cell('# Joint peak and duration search\n\nFixed policies and shared scenarios. All candidates and failed runs are retained. Full-day rows and one-hour branch-exchange screens are explicitly separated. Synthetic ties are scenario assumptions, not verified SCE assets.'),
              nbf.v4.new_code_cell(f'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path({str(OUT)!r})\ndisplay(pd.read_csv(root/"case_scores.csv"))\ndisplay(pd.read_csv(root/"all_metrics.csv"))')]
    for path in sorted((OUT/'figures').glob('*_2x3.png')):
        nb.cells.append(nbf.v4.new_markdown_cell('## '+path.stem))
        nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename={str(path)!r}))'))
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(HERE)}}).execute()
    nbf.write(nb,HERE/'joint_search_review.ipynb')
    print(scores.to_string(index=False),flush=True)


def verify(ids):
    checks=[]; runtime=paths.context()[0]; p=paths.context()[2]
    device=paths.torch.device('cuda' if paths.torch.cuda.is_available() else 'cpu')
    for c in all_cases():
        if c['id'] not in ids:continue
        install(c)
        for method in ('No control',*vis.METHODS):
            path=OUT/'cache/full'/f'{c["id"]}__{method}__original.pkl.gz'
            if not path.exists():continue
            payload=vis.load(path)
            if 'result' not in payload:continue
            r=payload['result']; env,_=build(runtime)
            policies=paths.old.load_policy_set(method,env,paths.DATA,*runtime[2:5],runtime[5],device)
            r0=env.network.line.r_ohm_per_km.to_numpy().copy(); x0=env.network.line.x_ohm_per_km.to_numpy().copy()
            errors=np.zeros(3)
            points=sorted(set([0,c.get('outage_start',c['event'])-1,c.get('outage_start',c['event']),c['event']-1,c['event'],c['event']+50,14399]))
            for k in points:
                # Original controller observes the preceding update's topology feature.
                apply(env,c,max(0,k-1),r0,x0)
                x=env.network.line.x_ohm_per_km.to_numpy().copy()
                apply(env,c,k,r0,x0);env.state=r['states'][k].copy()
                topo=paths.torch.as_tensor(x,dtype=paths.torch.float32,device=device)[None]
                with paths.torch.inference_mode():
                    q=np.clip(paths.old.controller_action(method,env.state,r['actions'][k-1] if k else np.zeros(5),policies,topo,device),-25,25)
                    _,_,reward,_=env.step_load(q.reshape(5,1),p['p'][k],p['q'][k],p['pv_p'][k])
                v=env.network.res_bus.vm_pu.to_numpy()
                assert np.array_equal(np.isfinite(v),np.isfinite(r['all_states'][k]))
                errors=np.maximum(errors,[np.max(np.abs(q-r['actions'][k])),np.nanmax(np.abs(v-r['all_states'][k])),abs(reward-r['rewards'][k])])
            assert max(errors)<1e-7,errors
            checks.append(dict(case_id=c['id'],method=method,errors=errors.tolist()))
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','run','full','table','review','verify'],default='prepare')
    ap.add_argument('--batch',default='initial');ap.add_argument('--cases-json',type=Path);ap.add_argument('--ids',nargs='*')
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--methods',nargs='+',default=list(vis.METHODS));args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='table':
        print(table()[1].to_string(index=False),flush=True);return
    if args.stage=='review':review();return
    if args.stage=='verify':verify(args.ids or []);return
    folder=OUT/'batches'/args.batch;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest.json'
    if manifest.exists():chosen=json.loads(manifest.read_text())['cases']
    else:
        chosen=json.loads(args.cases_json.read_text()) if args.cases_json else generate()
        payload=dict(cases=chosen,protocol=paths.context()[4],entry_hash=paths.base.digest(__file__))
        manifest.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if args.ids:chosen=[c for c in chosen if c['id'] in args.ids]
    if args.stage=='prepare':
        checks=[]
        for c in chosen:
            if c['direction']=='maintenance':checks.append(m.topology_record(c))
            else:
                env,_=BUILD(paths.context()[0]);g=paths.graph(env.network)
                g.remove_edge(c['old_parent'],c['child']);g.add_edge(c['new_parent'],c['child'])
                assert m.nx.is_tree(g)
                checks.append(dict(case_id=c['id'],radial=True,online_nodes=len(g),event=c['event']))
        (folder/'topology_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        print(json.dumps(dict(cases=len(chosen),workers=args.workers,device=paths.context()[4]['device'])),flush=True);return
    rows=[]
    if args.stage=='run':
        long=[c for c in chosen if c['full']];short=[c for c in chosen if not c['full']]
        chosen=[group[i] for i in range(max(len(long),len(short))) for group in (long,short) if i<len(group)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run_one,c,method,True if args.stage=='full' else c['full']) for c in chosen for method in args.methods]
        for f in as_completed(fs):
            rows.append(f.result());print('COMPLETED '+json.dumps(rows[-1]),flush=True)
            (folder/f'{args.stage}_last_batch.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            table()
    print('BATCH COMPLETE',flush=True)


if __name__=='__main__':main()
