"""One unique graph per scenario; no voltage-violation selection."""

# Shared 56-bus dependencies now live in the validated experiment.
from pathlib import Path as _MigrationPath
import sys as _migration_sys
_migration_sys.path.insert(0, str(_MigrationPath(__file__).resolve().parents[3]))
import hashlib
import json
import time
import numpy as np
import pandapower as pp
from environment123 import Feeder, BUS, SWITCH_LINES, sample_scene, save_json
from experiments.validated.training_runtime.support import safe_reset

def draw_graph(env,rng,excluded,counters):
    for _ in range(200000):
        counters['graph_draws']+=1
        key,nodes,mask=env.signature((rng.random(len(SWITCH_LINES))<.5).tolist())
        if key not in excluded and set(BUS).issubset(nodes):
            return dict(id=key,switches=[bool(mask[i]) for i in SWITCH_LINES])
    raise RuntimeError('Unique graph sampling exhausted')

def scene_for(env,topology,rng,seed,counters):
    for attempt in range(80):
        mult=rng.uniform(.5,1.5,len(env.x0)).tolist()
        counters['parameter_draws']=counters.get('parameter_draws',0)+1
        counters['reset_attempts']+=1
        scene=sample_scene(env,topology,seed+attempt*200000)
        scene.update(multipliers=mult,events=[])
        try:
            state=safe_reset(env,scene)
            scene.update(initial_v=state.tolist(),initial_safe=bool(np.all((state>.9499)&(state<1.0501))),features=env.features.tolist())
            return scene,state
        except (ValueError,pp.powerflow.LoadflowNotConverged):
            counters['reset_failures']+=1
    raise RuntimeError(f'Operating-point feasibility sampling exhausted: topology={topology}, seed={seed}, v={env.state.tolist()}')

def prepare(path,max_topologies,eval_every):
    start=time.perf_counter();rng=np.random.default_rng(20260916123);env=Feeder()
    excluded=set()
    counters=dict(graph_draws=0,reset_attempts=0,reset_failures=0)
    batches=(max_topologies+eval_every-1)//eval_every+1
    sizes=dict(anchor=32,rotation=32*batches,audit=1000)
    groups={};index=0
    for name,size in sizes.items():
        scenes=[]
        for _ in range(size):
            topo=draw_graph(env,rng,excluded,counters)
            scene,_=scene_for(env,topo,rng,80000000+index,counters)
            excluded.add(topo['id']);scenes.append(scene);index+=1
        groups[name]=scenes
        print(json.dumps(dict(prepared=name,scenes=len(scenes),seconds=time.perf_counter()-start)),flush=True)
    out=dict(schema='streaming_123_efficiency_v1',groups=groups,excluded_graphs=sorted(excluded),
             distribution=dict(switch_closed_probability=.5,line_admittance='independent uniform 0.5..1.5',injections='123bus PV ranges plus retained nominal loads scaled 0.8..1.2',reject_initial_voltage_violation=False),
             preparation_counters=counters,pf_counts=env.counts,wall_seconds=time.perf_counter()-start)
    save_json(path,out)
    return out
