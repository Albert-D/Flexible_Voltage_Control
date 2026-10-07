"""Bounded parallel screening, followed by full-day confirmation of candidates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import argparse
import gzip
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandapower as pp
import torch
import run_real_world_pv_outage_stress as old
from run_operational_priority_pilot import dump,digest

ROOT='operational_direction_screen_2026-09-15'
CONTROL=np.array([17,20,29,44,52])
CONTEXT=None


def load(path):
    with gzip.open(path,'rb') as f: return pickle.load(f)


def context():
    global CONTEXT
    if CONTEXT is not None: return CONTEXT
    torch.set_num_threads(1)
    repo=Path(__file__).resolve().parent
    runtime=old.load_runtime(repo); data=Path(runtime[-1].data_path)
    reference=load(data/'cache/load_branch_restoration_2026-09-15/combined.pkl.gz')
    assert reference['settings']['rlcft_scale']==1.
    for name,expected in reference['settings']['scripts'].items():
        assert digest(repo/name)==expected,name
    for path,expected in reference['settings']['checkpoints'].items(): assert digest(path)==expected,path
    assert digest(data/'realworld_results.pkl.gz')==reference['settings']['source_sha256']
    old.RLCFT_SCALE=1.
    baselines={r['method']:r for r in reference['results'] if not r['event']}
    profiles=reference['profiles']
    protocol=dict(source=str(data/'cache/load_branch_restoration_2026-09-15/combined.pkl.gz'),
                  source_hash=digest(data/'cache/load_branch_restoration_2026-09-15/combined.pkl.gz'),
                  runner_hash=digest(__file__),rlcft_scale=1.,safe_scale=10.,linear_gain=10.,dt_seconds=6,
                  q_cap_mvar=25.,initialization='exact method-specific unchanged-network cached state/action',
                  primary_metric='duration of >10 percent energized network buses outside 0.95-1.05',
                  criterion_note='SCE-style node proxy, not verified meter compliance',
                  checkpoint_hashes=reference['settings']['checkpoints'])
    CONTEXT=(runtime,data,profiles,baselines,protocol)
    return CONTEXT


def cases():
    _,_,p,_,_=context()
    net=1.75*p['pv_p']-.88*p['p']
    times={'midday':int(6900+np.argmax(net[6900:8400])),
           'evening':int(9600+np.argmax(-net[9600:13200]))}
    answer=[]
    for time_name,k in times.items():
        for group,lines in [('upper',[7,10,14]),('lower',[48,50,54])]:
            for operation in ('open','restore'):
                answer.append(dict(id=f'load_{group}_{operation}_{time_name}',direction='load_switch',
                                   lines=lines,operation=operation,event=k,start=k-600 if operation=='restore' else k,stop=k+600))
        for agent,bus in enumerate(CONTROL+1):
            answer.append(dict(id=f'pv{bus}_out_{time_name}',direction='pv_outage',agent=agent,event=k,start=k,stop=k+600))
        for line,label in [(0,'upstream'),(21,'shared'),(45,'distal')]:
            for factor in (1.5,2.):
                answer.append(dict(id=f'path_{label}_z{factor:g}_{time_name}',direction='path_impedance',
                                   line=line,factor=factor,event=k,start=k,stop=k+600,
                                   assumption='hypothetical equivalent-path impedance change, not an original SCE asset claim'))
    return answer


def build_env(runtime):
    env,state,_=old.build_environment(*runtime[:2])
    assert len(env.network.bus)==56 and len(env.network.line)==55
    return env,state


def apply_case(env,case,k,r0,x0):
    if case['direction']=='load_switch':
        closed=(k>=case['event']) if case['operation']=='restore' else (k<case['event'])
        # A restoration case's preceding outage begins at the cached restart.
        if case['operation']=='restore' and k<case.get('outage_start',case['start']): closed=True
        mask=env.network.switch.element.isin(case['lines']) & (env.network.switch.et=='l')
        assert mask.sum()==len(case['lines'])
        env.network.switch.loc[mask,'closed']=closed
    elif case['direction']=='pv_outage':
        active=not (case['event']<=k<case.get('restore_step',10**9))
        env.network.sgen.at[case['agent']+1,'in_service']=active
    elif case['direction']=='path_impedance':
        factor=case['factor'] if k>=case['event'] else 1.
        line=case['line']
        env.network.line.at[line,'r_ohm_per_km']=r0[line]*factor
        env.network.line.at[line,'x_ohm_per_km']=x0[line]*factor
    else: raise ValueError(case['direction'])


def metrics(r,event=None):
    start=max(0,(event if event is not None else r['case']['event'])-r['start'])
    v=r['all_states'][start:]; c=v[:,CONTROL]
    bad=(v<.95)|(v>1.05); finite=np.isfinite(v)
    fractions=bad.sum(axis=1)/finite.sum(axis=1)
    any_c=((c<.95)|(c>1.05)).any(axis=1)
    longest=old.longest_true_run(fractions>.1)*.1
    return dict(peak=float(np.nanmax(v)),minimum=float(np.nanmin(v)),
                fraction_duration_min=longest,fraction_total_min=float((fractions>.1).sum())*.1,
                controlled_duration_min=old.longest_true_run(any_c)*.1,
                controlled_total_min=float(any_c.sum())*.1,
                controlled_peak=float(c.max()),controlled_minimum=float(c.min()),
                q_peak=float(np.abs(r['actions'][start:]).max()),
                saturated_fraction=float(r['saturated'][start:].mean()),
                cost=float(-r['rewards'][start:].sum()),
                passes=bool(float(np.nanmax(v))<=1.10 and longest<5),
                terminal_violation_fraction=float(fractions[-1]),
                boundary_tolerance_1e4_duration_min=old.longest_true_run((((v<.9499)|(v>1.0501)).sum(axis=1)/finite.sum(axis=1))>.1)*.1)


def counterfactual(method,case):
    _,_,_,base,_=context(); r=base[method]; a=case['event']; b=case['stop']
    return dict(case=case,start=a,all_states=r['all_states'][a:b],actions=r['actions'][a:b],
                rewards=r['rewards'][a:b],saturated=r['saturated'][a:b])


def run(case,method,full_day=False,margin_match=False):
    runtime,data,p,base,protocol=context()
    variant='matched_margin' if margin_match else 'original'
    stage='full' if full_day else 'screen'
    path=data/'cache'/ROOT/stage/f"{case['id']}__{method}__{variant}.pkl.gz"
    key=dict(protocol=protocol,case=case,method=method,full_day=full_day,margin_match=margin_match)
    if path.exists():
        saved=load(path); assert saved['key']==key; return saved['summary']
    env,state=build_env(runtime)
    if margin_match: env.vmin,env.vmax=.955,1.045
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policies=old.load_policy_set(method,env,data,*runtime[2:5],runtime[5],device)
    a=0 if full_day else case['start']; b=14400 if full_day else case['stop']; n=b-a
    baseline=base[method]
    state=state if a==0 else baseline['states'][a].copy(); env.state=state.copy()
    last=np.zeros(5) if a==0 else baseline['actions'][a-1].copy()
    r0=env.network.line.r_ohm_per_km.to_numpy().copy(); x0=env.network.line.x_ohm_per_km.to_numpy().copy()
    feature=env.topology_init if a==0 else x0
    topology=torch.as_tensor(np.asarray(feature),dtype=torch.float32,device=device)[None]
    all_v=np.empty((n,56)); actions=np.empty((n,5)); rewards=np.empty(n); sat=np.zeros(n,dtype=bool)
    states=np.empty((n+1,5)); states[0]=state
    start_time=time.perf_counter()
    try:
        with torch.inference_mode():
            for j,k in enumerate(range(a,b)):
                apply_case(env,case,k,r0,x0)
                if margin_match and method=='Linear':
                    req=last-10*(np.maximum(state-1.045,0)-np.maximum(.955-state,0))
                else: req=old.controller_action(method,state,last,policies,topology,device)
                inactive=~env.network.sgen.loc[1:5,'in_service'].to_numpy(dtype=bool)
                req[inactive]=0.
                action=np.clip(req,-25,25)
                state,feature,reward,_=env.step_load(action.reshape(5,1),p['p'][k],p['q'][k],p['pv_p'][k])
                if not np.isfinite(state).all(): raise RuntimeError(f'controlled bus disconnected at {k}')
                topology=torch.as_tensor(np.asarray(feature),dtype=torch.float32,device=device)[None]
                states[j+1]=state; all_v[j]=env.network.res_bus.vm_pu.to_numpy()
                actions[j]=action; rewards[j]=reward; sat[j]=np.any(np.abs(req)>25)
                last=action.copy()
        result=dict(case=case,method=method,start=a,all_states=all_v,states=states,actions=actions,rewards=rewards,saturated=sat,
                    elapsed=time.perf_counter()-start_time)
        cutoff=case['event']-a
        window=dict(result,all_states=all_v[cutoff:cutoff+600],actions=actions[cutoff:cutoff+600],
                    rewards=rewards[cutoff:cutoff+600],saturated=sat[cutoff:cutoff+600],start=case['event'])
        summary=dict(case_id=case['id'],direction=case['direction'],method=method,variant=variant,status='ok',
                     elapsed=result['elapsed'],**metrics(window),counterfactual=metrics(counterfactual(method,case)))
        dump(path,dict(key=key,result=result,summary=summary))
    except Exception as exc:
        summary=dict(case_id=case['id'],direction=case['direction'],method=method,variant=variant,status='failed',
                     error=f'{type(exc).__name__}: {exc}',elapsed=time.perf_counter()-start_time)
        dump(path,dict(key=key,summary=summary))
    print(json.dumps(summary),flush=True)
    return summary


def verify_resume():
    runtime,_,p,baselines,_=context(); results=[]
    for method in old.CONTROL_METHODS:
        env,_=build_env(runtime); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policies=old.load_policy_set(method,env,Path(runtime[-1].data_path),*runtime[2:5],runtime[5],device)
        max_q=max_v=0.
        for k in (7238,7838,11679,12279):
            reference=baselines[method]; state=reference['states'][k].copy(); env.state=state.copy()
            topology=torch.as_tensor(env.network.line.x_ohm_per_km.to_numpy(),dtype=torch.float32,device=device)[None]
            with torch.inference_mode():
                q=old.controller_action(method,state,reference['actions'][k-1],policies,topology,device)
                v,_,_,_=env.step_load(q.reshape(5,1),p['p'][k],p['q'][k],p['pv_p'][k])
            max_q=max(max_q,float(np.abs(q-reference['actions'][k]).max()))
            max_v=max(max_v,float(np.abs(v-reference['states'][k+1]).max()))
        assert max_q<1e-4 and max_v<1e-8
        results.append(dict(method=method,action_error=max_q,voltage_error=max_v))
    return results


def aggregate(rows,out):
    successful=[r for r in rows if r['status']=='ok']
    scores=[]
    for case_id in sorted(set(r['case_id'] for r in successful)):
        group={r['method']:r for r in successful if r['case_id']==case_id and r['variant']=='original'}
        if set(group)!=set(old.CONTROL_METHODS): continue
        r=group['RLC-FT']; s=group['Safe-DDPG']; l=group['Linear']
        scores.append(dict(case_id=case_id,direction=r['direction'],rlc_pass=r['passes'],safe_pass=s['passes'],linear_pass=l['passes'],
                           rlc_duration=r['fraction_duration_min'],safe_duration=s['fraction_duration_min'],linear_duration=l['fraction_duration_min'],
                           rlc_peak=r['peak'],safe_peak=s['peak'],linear_peak=l['peak'],
                           rlc_cost_ratio_safe=r['cost']/s['cost'],
                           unique_pass=r['passes'] and not s['passes'] and not l['passes'],
                           induced_safe_failure=not s['passes'] and s['counterfactual']['passes'],
                           safe_added_duration=s['fraction_duration_min']-s['counterfactual']['fraction_duration_min']))
    out.mkdir(parents=True,exist_ok=True)
    (out/'summaries.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    pd.json_normalize(rows).to_csv(out/'summaries.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'case_scores.csv',index=False)
    return scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases-json',type=Path)
    parser.add_argument('--full-day',action='store_true')
    parser.add_argument('--matched-margin',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    _,data,_,_,protocol=context()
    root=data/'cache'/ROOT; root.mkdir(parents=True,exist_ok=True)
    verified=verify_resume(); (root/'resume_verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    if args.verify_only: print(verified); return
    chosen=json.loads(args.cases_json.read_text(encoding='utf-8')) if args.cases_json else cases()
    stage='full' if args.full_day else 'screen'
    tag=args.cases_json.stem if args.cases_json else 'initial'
    out=root/f'{stage}_{tag}'
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(dict(protocol=protocol,cases=chosen),indent=2),encoding='utf-8')
    print(f'START {len(chosen)} cases x 3 methods, {args.workers} workers, full_day={args.full_day}',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,case,method,args.full_day,args.matched_margin and method!='RLC-FT') for case in chosen for method in old.CONTROL_METHODS]
        for future in as_completed(futures):
            rows.append(future.result()); aggregate(rows,out)
            print(f'PROGRESS {len(rows)}/{len(futures)}',flush=True)
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__': main()
