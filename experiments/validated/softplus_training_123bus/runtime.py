"""Bounded 123-bus pilot. Shared original models and accepted speed helpers only."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,time,copy,hashlib,argparse,shutil,multiprocessing as mp,traceback
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
FEEDER=Path(__file__).resolve().parent
SPEED=ROOT/'experiments/validated/training_runtime'
sys.path[:0]=[str(FEEDER),str(SPEED)]
import numpy as np
import torch
import pandapower as pp
from environment123 import Feeder as OriginalFeeder,BUS,save_json
from model123 import Controller,Learner
from evaluation123 import load_reference,rollout,summary,act
from protocol123 import draw_graph,scene_for
from experiments.validated.training_runtime.support import initialize_weak_actor
from fast_environment import feeder_class
from fast_runtime import CachedReplay,sample_replay_fast,ActorMirror,configure_adam
Feeder=feeder_class(OriginalFeeder,BUS,'lightsim_recycle')
OUT=Path('D:/Code/Python/Flexible_Voltage_Control/experiments/123bus_pilot_2026-09-16')
PROTOCOL=OUT.parent/'topology_efficiency/seed2601_123_stream_v3/protocol.json'
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def append(p,row):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def gains(actor,scenes):
    xs=torch.tensor([s['features'] for s in scenes],dtype=torch.float32)
    with torch.no_grad():
        gs=torch.cat([p.topology_gain(xs) for p in actor.policies],dim=1).numpy()
        actions=actor(torch.full((len(xs),14),1.08),xs).numpy()
    cv=gs.std(axis=0)/np.maximum(gs.mean(axis=0),1e-12)
    return dict(gain_median=float(np.median(gs)),gain_p90=float(np.quantile(gs,.9)),gain_cv_median=float(np.median(cv)),
                gain_cv_per_agent=cv.tolist(),fixed_voltage_action_spread=float(np.mean(np.ptp(actions,axis=0))))

_worker_env=_worker_model=_worker_ref=None
def init_worker():
    global _worker_env,_worker_model,_worker_ref
    torch.set_num_threads(1);_worker_env=Feeder();_worker_model=Controller().eval();_worker_ref,_=load_reference(torch.device('cpu'))
def evaluate_job(payload):
    state,scenes,reference,keep=payload
    if not reference:_worker_model.load_state_dict(state)
    model=_worker_ref if reference else _worker_model;rows=[]
    for scene in scenes:
        row=rollout(model,scene,torch.device('cpu'),legacy=reference,env=_worker_env)
        v=np.asarray(row['voltage'])
        row['mean_voltage_violation']=float(np.maximum(np.abs(v-1)-.05,0).mean()) if len(v) else 1.
        row['terminal_voltage_violation']=float(np.maximum(np.abs(v[-1]-1)-.05,0).max()) if len(v) else 1.
        if not keep:row={k:v for k,v in row.items() if k not in ('voltage','action')}
        rows.append(row)
    return rows

def evaluate(pool,model,scenes,reference=False,keep=False):
    state=None if reference else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    chunks=[scenes[i:i+4] for i in range(0,len(scenes),4)]
    rows=[r for group in pool.map(evaluate_job,[(state,c,reference,keep) for c in chunks]) for r in group]
    return rows

def metrics(rows):
    return summary(rows)|dict(mean_voltage_violation=float(np.mean([r['mean_voltage_violation'] for r in rows])),
                             terminal_violation_median=float(np.median([r['terminal_voltage_violation'] for r in rows])))

def preflight():
    OUT.mkdir(parents=True,exist_ok=True);protocol=read(PROTOCOL);scenes=protocol['groups']['anchor'][:16]
    result=dict(protocol_sha256=sha(PROTOCOL),seed=2601,models={})
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for name,factor in [('reference',None),('original_init',1.),('weak_56_init',.05)]:
            torch.manual_seed(2601);actor=Controller().eval()
            if factor is not None and factor!=1:initialize_weak_actor(actor,factor)
            tick=time.perf_counter();rows=evaluate(pool,actor,scenes,reference=name=='reference',keep=True)
            delta=[];voltage=[]
            for row in rows:
                q=np.asarray(row['action']);v=np.asarray(row['voltage'])
                d=np.diff(np.vstack((np.zeros((1,14)),q)),axis=0)
                end=min(int(row['recovery']),len(d));delta.extend(d[:end].ravel().tolist());voltage.extend(np.maximum(np.abs(v[:end]-1)-.045,0).ravel().tolist())
            result['models'][name]=dict(metrics=metrics(rows),rows=rows,seconds=time.perf_counter()-tick,
                delta_abs_p50=float(np.median(np.abs(delta))),delta_abs_p90=float(np.quantile(np.abs(delta),.9)),
                delta_squared_mean=float(np.mean(np.square(delta))),band_violation_mean=float(np.mean(voltage)),
                topology=gains(actor,scenes) if factor is not None else None)
            save_json(OUT/'preflight.json',result);print(name,{k:v for k,v in result['models'][name].items() if k not in ('rows','topology')},flush=True)


def train(args):
    if not args.run_id.replace('_','').isalnum():raise ValueError('ASCII run ID required')
    run=OUT/args.run_id;started=time.perf_counter()
    protocol=read(PROTOCOL);protocol_hash=sha(PROTOCOL);excluded=set(protocol['excluded_graphs'])
    source_paths=[Path(__file__),SPEED/'fast_environment.py',SPEED/'fast_runtime.py',FEEDER/'environment123.py',FEEDER/'model123.py',FEEDER/'evaluation123.py',FEEDER/'protocol123.py',
                  FEEDER/'learner.py',SPEED/'support.py',ROOT/'src/softplus_controller.py',ROOT/'data/case_123.mat']
    config=dict(args={k:v for k,v in vars(args).items() if k not in ('resume','stop_at')},protocol_sha256=protocol_hash,
                sources={str(p):sha(p) for p in source_paths},from_scratch=True,output_scale=1.,
                runtime=dict(powerflow='lightsim_recycle',adam_fused=True,actor_inference='synchronized_cpu',replay='cached_severity',threads=1),
                reward=dict(band=1.,deviation=.001,delta=args.delta_weight,q=0.),critic_lr=1e-3,
                acceptance='engineering + learning trend; usable control separately relative to 123 reference',audit_performed=False)
    if args.resume:
        if read(run/'config.json')!=config:raise ValueError('Source/config changed; resume blocked')
    else:
        if run.exists():raise FileExistsError(run)
        run.mkdir(parents=True)
        save_json(run/'config.json',config)
        for src in source_paths:
            dest=run/'source'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    device=torch.device('cuda');learner=Learner(device,seed=args.seed)
    if args.initial_factor!=1:initialize_weak_actor(learner.actor,args.initial_factor)
    learner.target.load_state_dict(learner.actor.state_dict())
    for group in learner.actor_opt.param_groups:group['lr']=args.actor_lr;group['initial_lr']=args.actor_lr
    learner.actor_scheduler.base_lrs=[args.actor_lr];learner.actor_scheduler._last_lr=[args.actor_lr]
    configure_adam(learner,True)
    replay=CachedReplay(seed=args.seed+3000);rng=np.random.default_rng(args.seed+1000);noise_rng=np.random.default_rng(args.seed+2000)
    env=Feeder();used=set();records=[];episodes=[];solver_checks=[]
    counts=dict(interactions=0,successful_steps=0,failed_steps=0,graph_draws=0,reset_attempts=0,reset_failures=0)
    timing={k:0. for k in ('reset','action','powerflow','replay_reward','sample','update','actor_sync','evaluation','checkpoint','solver_check','prior_elapsed')}
    if args.resume:
        latest=read(run/'latest_checkpoint.json');assert sha(latest['path'])==latest['sha256']
        ck=torch.load(latest['path'],map_location='cpu',weights_only=False)
        learner.load(ck['learner']);configure_adam(learner,True)
        replay.rows=ck['replay'];replay.pos=ck['replay_pos'];replay.rng.bit_generator.state=ck['replay_rng'];replay.rebuild_severity()
        rng.bit_generator.state=ck['rng'];noise_rng.bit_generator.state=ck['noise_rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        counts=ck['counts'];timing=ck['timing'];timing['prior_elapsed']=ck['elapsed'];used=set(ck['used']);records=ck['records'];episodes=ck['episodes'];solver_checks=ck['solver_checks']
        (run/'training.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes),encoding='utf-8')
        save_json(run/'restore_check.json',dict(interactions=counts['interactions'],updates=learner.updates,replay_rows=len(replay.rows),cache_rebuilt=True,
                  actor_optimizer_devices=sorted({str(v.device) for s in learner.actor_opt.state.values() for v in s.values() if torch.is_tensor(v)})))
    mirror=ActorMirror(learner.actor)
    def elapsed():return timing['prior_elapsed']+time.perf_counter()-started
    def checkpoint():
        t=time.perf_counter();path=run/f'checkpoint_{counts["interactions"]:06d}.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(learner=learner.state(),replay=replay.rows,replay_pos=replay.pos,replay_rng=replay.rng.bit_generator.state,
                        rng=rng.bit_generator.state,noise_rng=noise_rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        counts=counts,timing=timing.copy(),used=sorted(used),records=records,episodes=episodes,solver_checks=solver_checks,elapsed=elapsed(),protocol_sha256=protocol_hash),temp)
        temp.replace(path);save_json(run/'latest_checkpoint.json',dict(path=str(path),sha256=sha(path)));timing['checkpoint']+=time.perf_counter()-t
    pre=read(OUT/'preflight.json');anchor=protocol['groups']['anchor'][:16]
    assert pre['protocol_sha256']==protocol_hash
    reference_anchor=pre['models']['reference']['rows']
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        def assess():
            t=time.perf_counter();index=len(records);rotation=protocol['groups']['rotation'][index*8:(index+1)*8];assert len(rotation)==8
            scene_batch=anchor+rotation
            rows=evaluate(pool,mirror.actor,scene_batch);refs=evaluate(pool,None,rotation,reference=True)
            a=metrics(rows[:16]);r=metrics(rows[16:]);refa=metrics(reference_anchor);refr=metrics(refs)
            passed=bool(a['failures']==0 and a['success_rate']>=.95 and .8*refa['recovery_median']<=a['recovery_median']<=1.2*refa['recovery_median'] and a['recovery_p90']<=max(30,refa['recovery_p90']*1.25))
            timing['evaluation']+=time.perf_counter()-t
            row=dict(interactions=counts['interactions'],num_topologies_seen=len(used),updates=learner.updates,anchor=a,rotation=r,reference_anchor=refa,reference_rotation=refr,
                     topology=gains(mirror.actor,anchor+protocol['groups']['rotation'][:16]),usable_development_pass=passed,
                     rows=rows,rotation_reference_rows=refs,timing=timing.copy(),elapsed=elapsed())
            records.append(row);save_json(run/'curve_data.json',records);checkpoint()
            print(json.dumps(dict(run=args.run_id,interactions=counts['interactions'],topologies=len(used),anchor=a,rotation=r,topology=row['topology'],usable_pass=passed,elapsed=elapsed())),flush=True)
        if not records:assess()
        reason='stage_budget'
        stop=min(args.max_interactions,args.stop_at)
        while counts['interactions']<stop:
            if elapsed()>args.wall_minutes*60:reason='wall_budget';break
            if len(records)>=3 and all(r['anchor']['failures']>=2 for r in records[-2:]):reason='evaluation_failure_stop';break
            t=time.perf_counter();topology=draw_graph(env,rng,used|excluded,counts);scene,state=scene_for(env,topology,rng,90000000+len(used),counts)
            timing['reset']+=time.perf_counter()-t;used.add(topology['id']);assert not used&excluded
            x=env.features.copy();previous=np.zeros(14,dtype=np.float32);reward_total=0.;losses={}
            budget=min(60,stop-counts['interactions']);actual_steps=0
            for step in range(budget):
                t=time.perf_counter()
                with torch.no_grad():delta=mirror.actor(torch.as_tensor(state).view(1,14),torch.as_tensor(x).view(1,113)).numpy().reshape(14)
                noise=noise_rng.normal(0,.25 if counts['interactions']<512 else .08,14)
                q=np.clip(previous-np.clip(delta+noise,-5,5),-50,50).astype(np.float32);timing['action']+=time.perf_counter()-t
                if not np.isfinite(q).all():raise ValueError('Nonfinite action')
                counts['interactions']+=1;actual_steps+=1;terminal=step+1==budget;t=time.perf_counter()
                try:
                    nxt=env.step(q)
                    if not np.isfinite(nxt).all() or nxt.min()<.75 or nxt.max()>1.25:raise ValueError('Hard voltage bound')
                    local=np.maximum(np.abs(nxt-1)-.045,0)+.001*np.abs(nxt-1)+args.delta_weight*(q-previous)**2
                    reward=-(.5*local+.5*local.mean());counts['successful_steps']+=1
                    terminal=terminal or bool(np.all((nxt>.955)&(nxt<1.045)))
                except (ValueError,pp.powerflow.LoadflowNotConverged):
                    counts['failed_steps']+=1;nxt=state.copy();reward=np.full(14,-100.,dtype=np.float32);terminal=True
                timing['powerflow']+=time.perf_counter()-t
                t=time.perf_counter();replay.push(state,x,previous,q,reward,nxt,x,[float(terminal)]);timing['replay_reward']+=time.perf_counter()-t
                if counts['interactions']>=512 and len(replay.rows)>=128:
                    t=time.perf_counter();batch=sample_replay_fast(replay,128,.5,.75);timing['sample']+=time.perf_counter()-t
                    t=time.perf_counter();losses=learner.update(batch);torch.cuda.synchronize();timing['update']+=time.perf_counter()-t
                    if any(v is not None and not np.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite loss')
                    if learner.updates%3==0:
                        if any(not bool(torch.isfinite(v).all()) for v in learner.actor.parameters()):raise ValueError('Nonfinite actor parameter')
                        t=time.perf_counter();mirror.sync(learner.actor);timing['actor_sync']+=time.perf_counter()-t
                reward_total+=float(reward.sum());state,previous=nxt,q
                if terminal:break
            if len(used) in (1,10,30,50) and counts['failed_steps']==0:
                t=time.perf_counter();baseline=OriginalFeeder();baseline.reset(scene);baseline.step(previous)
                error=float(np.nanmax(np.abs(baseline.voltage-env.voltage)))
                solver_checks.append(dict(interactions=counts['interactions'],topology=topology['id'],max_voltage_abs=error,passed=error<=1e-6));assert error<=1e-6
                timing['solver_check']+=time.perf_counter()-t
            episode=dict(interactions=counts['interactions'],num_topologies_seen=len(used),episode_steps=actual_steps,reward_per_step=reward_total/actual_steps,losses=losses,scene=scene,counts=counts.copy())
            episodes.append(episode);append(run/'training.jsonl',episode)
            save_json(run/'progress.json',dict(interactions=counts['interactions'],topologies=len(used),updates=learner.updates,counts=counts,timing=timing,elapsed=elapsed()))
            if counts['failed_steps']>=3:reason='training_failure_stop';break
            if counts['interactions']-records[-1]['interactions']>=1000:assess()
        if records[-1]['interactions']!=counts['interactions']:assess()
    save_json(run/'stage_result.json',dict(reason=reason,counts=counts,topologies=len(used),timing=timing,elapsed=elapsed(),solver_checks=solver_checks,
              train_evaluation_graph_overlap=len(used&excluded),latest=read(run/'latest_checkpoint.json'),final=records[-1],audit_performed=False,
              gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print(json.dumps(dict(stage_complete=args.run_id,interactions=counts['interactions'],reason=reason)),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','train'])
    parser.add_argument('--run-id',default='');parser.add_argument('--actor-lr',type=float,default=1e-4)
    parser.add_argument('--delta-weight',type=float,default=.2);parser.add_argument('--initial-factor',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=2601);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-interactions',type=int,default=3000);parser.add_argument('--stop-at',type=int,default=1000)
    parser.add_argument('--wall-minutes',type=float,default=45.);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='preflight':preflight()
    else:train(args)
if __name__=='__main__':main()
