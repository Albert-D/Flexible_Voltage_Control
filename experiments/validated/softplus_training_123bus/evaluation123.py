"""Reproducible controller pretraining experiment, not the efficiency benchmark."""
from __future__ import annotations

# Shared 56-bus dependencies now live in the validated experiment.
from pathlib import Path as _MigrationPath
import sys as _migration_sys
_migration_sys.path.insert(0, str(_MigrationPath(__file__).resolve().parents[3]))
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import pandapower as pp
from environment123 import Feeder, ROOT, BUS, save_json
from model123 import Controller, Reference, Learner, Replay, voltage_branch

BASE_ROOT=ROOT
ROOT=BASE_ROOT/'recovery_training_v4'

REF_ROOT=Path('D:/Code/Python/Flexible_Voltage_Control/check_points/policy_net/2025-08-19')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_reference(device):
    paths=[REF_ROOT/f'Step_950_Seed_2_a{i}.pth' for i in range(14)]
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    return Reference(states).to(device).eval(),[dict(path=str(p),sha256=sha(p)) for p in paths]

def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)

def act(model,state,features,previous,device,unbounded=False):
    with torch.no_grad():
        delta=model(tensor(state,device).view(1,14),tensor(features,device).view(1,113))
        q=(tensor(previous,device).view(1,14)-delta if unbounded else Learner.action(tensor(previous,device).view(1,14),delta))
    return q.cpu().numpy().reshape(14)

def legacy_cost(v,q):
    return float(np.linalg.norm(q)+100*np.linalg.norm(np.maximum(v-1,0))+100*np.linalg.norm(np.maximum(1-v,0)))

def first_and_sustained(values,horizon):
    first=next((i+1 for i,ok in enumerate(values) if ok),horizon)
    sustained=next((i+1 for i in range(max(0,len(values)-9)) if all(values[i:])),horizon)
    return first,sustained

def rollout(model,scene,device,legacy=False,env=None):
    env=Feeder() if env is None else env
    env.counts=dict(reset=0,step=0,event=0,failed=0)
    env.reset(scene)
    horizon=120 if scene.get('events') else 200
    events={e['step']:e['switches'] for e in scene.get('events',[])}
    q=np.zeros(14,dtype=np.float32); v=env.state.copy()
    flags=[]; all_flags=[]; costs=[]; before_costs=[]; vs=[]; qs=[]; phases=[]
    starts=[0]+sorted(events); failed=None; frozen=False
    for t in range(horizon):
        try:
            if t in events:
                v=env.event(events[t]);frozen=False
            old_v=v.copy()
            features=scene.get('legacy_features',env.legacy_features) if legacy else env.features
            if not frozen:
                previous_q=q.copy()
                q=act(model,v,features,q,device,unbounded=scene.get('historical',False))
                v=env.step(q)
                # Exact deterministic fixed point, not early success termination:
                # identical physical input implies identical output until an event.
                frozen=np.array_equal(q,previous_q) and np.array_equal(v,old_v)
            if not np.isfinite(v).all() or v.min()<.75 or v.max()>1.25:
                raise ValueError('voltage outside hard safety bound')
            flags.append(bool(np.all((v>.9499)&(v<1.0501))))
            energized=env.voltage[list(env.nodes)]
            all_flags.append(bool(np.all((energized>=.95)&(energized<=1.05))))
            costs.append(legacy_cost(v,q));before_costs.append(legacy_cost(old_v,q))
            vs.append(v.tolist());qs.append(q.tolist())
        except (ValueError,pp.powerflow.LoadflowNotConverged) as exc:
            failed=str(exc);break
    for start,end in zip(starts,starts[1:]+[horizon]):
        phase_flags=flags[start:end]
        first,sustain=first_and_sustained(phase_flags,end-start)
        # Historical notebook excludes terminal transition from transient objective.
        stop=start+max(first-1,0)
        success=(len(phase_flags)==end-start and sustain<end-start)
        phases.append(dict(start=start,end=end,recovery=first,sustained_recovery=sustain,success=success,
                           transient_objective=float(sum(before_costs[start:stop]))))
    # Failed cases receive the remaining horizon's explicit large cost, never a smaller cost.
    full_cost=sum(costs)+1000*(horizon-len(costs))
    arr=np.asarray(vs)
    return dict(seed=scene['seed'],topology_id=scene['topology_id'],events=bool(events),
                success=failed is None and all(p['success'] for p in phases),failure=failed,
                recovery=float(np.mean([p['recovery'] for p in phases])),
                sustained_recovery=float(np.mean([p['sustained_recovery'] for p in phases])),
                objective=float(full_cost),transient_objective=float(sum(p['transient_objective'] for p in phases)),
                max_violation=float(np.maximum(np.abs(arr-1)-.05,0).max()) if len(arr) else 1.,
                controlled_compliance=float(np.mean(flags)) if flags else 0.,
                all_energized_compliance=float(np.mean(all_flags)) if all_flags else 0.,
                phases=phases,voltage=vs,action=qs,counts=env.counts.copy())

def summary(rows):
    result=dict(n=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),
                failures=sum(r['failure'] is not None for r in rows))
    for key in ['objective','transient_objective','recovery','sustained_recovery','max_violation','all_energized_compliance']:
        result[key+'_median']=float(np.median([r[key] for r in rows]))
        result[key+'_p90']=float(np.quantile([r[key] for r in rows],.9))
    return result

from torch import nn
_env = _actor = _reference = None

def init_worker():
    global _env, _actor, _reference
    torch.set_num_threads(1)
    _env = Feeder()
    _actor = Controller().eval()
    _reference, _ = load_reference(torch.device('cpu'))
    _reference.eval()

class Override(nn.Module):
    def __init__(self, model, features):
        super().__init__()
        self.model=model
        self.register_buffer('features',torch.tensor(features,dtype=torch.float32).view(1,-1))
    def forward(self, voltage, features):
        return self.model(voltage,self.features.expand(len(voltage),-1))

def worker_job(payload):
    scenes, state, mode = payload
    start=time.perf_counter()
    if mode!='reference':
        _actor.load_state_dict(state)
    rows=[]
    for scene in scenes:
        model=_reference if mode=='reference' else _actor
        if mode=='wrong_x':model=Override(model,scene['wrong_features'])
        with torch.no_grad():
            row=rollout(model,scene,torch.device('cpu'),legacy=mode=='reference',env=_env)
        row={k:v for k,v in row.items() if k not in ('voltage','action')}
        row['initial_safe']=scene['initial_safe']
        rows.append(row)
    return dict(rows=rows,worker_seconds=time.perf_counter()-start)

def evaluate_pool(pool, scenes, state=None, mode='policy', workers=4):
    start=time.perf_counter()
    size=max(1,(len(scenes)+workers-1)//workers)
    jobs=[(scenes[i:i+size],state,mode) for i in range(0,len(scenes),size)]
    results=list(pool.map(worker_job,jobs))
    rows=[row for result in results for row in result['rows']]
    return dict(rows=rows,metrics=summary(rows),wall_seconds=time.perf_counter()-start,
                worker_seconds=sum(x['worker_seconds'] for x in results),
                pf_counts={key:sum(row['counts'][key] for row in rows) for key in ('reset','step','event','failed')})

def acceptance(metrics,reference):
    return bool(metrics['success_rate']>=.95 and metrics['failures']==0
                and .8*reference['recovery_median']<=metrics['recovery_median']<=1.2*reference['recovery_median']
                and metrics['recovery_p90']<=max(30.,reference['recovery_p90']*1.25))

def report_subsets(rows):
    violated=[x for x in rows if not x['initial_safe']]
    return dict(all=summary(rows),initial_violation=summary(violated) if violated else None,
                initially_safe_count=sum(x['initial_safe'] for x in rows))

def paired_report(correct,other):
    assert [(x['seed'],x['topology_id']) for x in correct]==[(x['seed'],x['topology_id']) for x in other]
    out={}
    for key in ('recovery','transient_objective','objective'):
        a=np.array([x[key] for x in correct]);b=np.array([x[key] for x in other])
        valid=a>1e-8
        out[key+'_other_minus_correct_median']=float(np.median(b-a))
        out[key+'_other_minus_correct_mean']=float(np.mean(b-a))
        out[key+'_other_over_correct_median']=float(np.median(b[valid]/a[valid])) if valid.any() else None
    out['recovery_changed_fraction']=float(np.mean([a['recovery']!=b['recovery'] for a,b in zip(correct,other)]))
    return out


# Adopted October 2026 learning experiment; historical rollout above stays available.
def _recovery15_delta(voltage, features, model):
    with torch.no_grad():
        return model(torch.as_tensor(voltage, dtype=torch.float32)[None],
                     torch.as_tensor(features, dtype=torch.float32)[None]).numpy().reshape(-1)


def recovery15_rollout(scene, model, env):
    system, method = "123", "softplus_main"
    voltage = env.reset(scene).astype(np.float32)
    q = np.zeros(len(voltage), dtype=np.float32)
    voltages = [voltage.tolist()]
    qs = [q.tolist()]
    deltas = []
    raw_deltas = []
    jv = jdelta = violation = 0.
    clips_step = clips_q = 0
    max_delta = max_q = max_voltage_violation = 0.
    initial_safe = bool(np.all((voltage >= .95) & (voltage <= 1.05)))
    failed = None
    recovered = initial_safe
    recovery = 0 if initial_safe else 200
    if not initial_safe:
        features = np.asarray(env.features, dtype=np.float32)
        limit_step, limit_q = (2.5, 25.) if system == "56" else (5., 50.)
        for step in range(1, 201):
            jv += 100. * float(np.square(voltage - 1.).sum())
            local_violation = np.maximum(np.abs(voltage - 1.) - .05, 0)
            violation += float(local_violation.sum())
            max_voltage_violation = max(max_voltage_violation, float(local_violation.max()))
            raw = np.asarray(_recovery15_delta(voltage, features, model), dtype=np.float32)
            if raw.shape != q.shape or not np.isfinite(raw).all():
                failed = "invalid_actor_output"
                break
            stepped = np.clip(raw, -limit_step, limit_step)
            new_q = np.clip(q - stepped, -limit_q, limit_q).astype(np.float32)
            clips_step += int(np.any(np.abs(raw) > limit_step + 1e-6))
            clips_q += int(np.any(np.abs(q - stepped) > limit_q + 1e-6))
            applied = q - new_q
            try:
                next_voltage = env.step(new_q)
                if not np.isfinite(next_voltage).all() or next_voltage.min() < .75 or next_voltage.max() > 1.25:
                    raise ValueError("hard_voltage_bound")
            except (ValueError, pp.powerflow.LoadflowNotConverged) as exc:
                failed = str(exc)
                break
            jdelta += float(np.square(applied).sum())
            max_delta = max(max_delta, float(np.abs(applied).max()))
            max_q = max(max_q, float(np.abs(new_q).max()))
            raw_deltas.append(raw.tolist())
            deltas.append(applied.tolist())
            voltage = np.asarray(next_voltage, dtype=np.float32)
            q = new_q
            voltages.append(voltage.tolist())
            qs.append(q.tolist())
            if bool(np.all((voltage >= .95) & (voltage <= 1.05))):
                recovered = True
                recovery = step
                break
    return dict(system=system, method=method, seed=int(scene["seed"]),
                topology_id=scene["topology_id"], x_id=scene["x_id"], initial_safe=initial_safe,
                success=bool(recovered and failed is None), failure=failed, recovery=recovery,
                jv=jv, jdelta=jdelta, voltage_violation=violation,
                max_voltage_violation=max_voltage_violation, max_delta=max_delta, max_abs_q=max_q,
                step_clip_events=clips_step, q_clip_events=clips_q,
                steps_executed=len(deltas), voltages=voltages, q=qs,
                raw_delta=raw_deltas, applied_delta=deltas)


def recovery15_summary(rows):
    succeeded = [row for row in rows if row["success"]]
    result = dict(n=len(rows), success=len(succeeded), hard_failures=sum(row["failure"] is not None for row in rows),
                  horizon_censored=sum(not row["success"] and row["failure"] is None for row in rows),
                  initial_safe=sum(row["initial_safe"] for row in rows),
                  step_clip_scenes=sum(row["step_clip_events"] > 0 for row in rows),
                  q_clip_scenes=sum(row["q_clip_events"] > 0 for row in rows),
                  powerflow_steps=sum(row["steps_executed"] for row in rows))
    for key in ("recovery", "jv", "jdelta", "voltage_violation", "max_voltage_violation", "max_delta", "max_abs_q"):
        # Include horizon-censored trajectories: they still incur cost through 200 steps.
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key + "_median"] = float(np.median(values)) if len(values) else None
        result[key + "_p90"] = float(np.quantile(values, .9)) if len(values) else None
    return result


def recovery15_compact_row(row):
    keys = ("topology_id", "x_id", "success", "failure", "initial_safe", "recovery",
            "jv", "jdelta", "voltage_violation", "max_voltage_violation",
            "max_delta", "max_abs_q", "step_clip_events", "q_clip_events",
            "steps_executed")
    return {key: row[key] for key in keys}
