"""Versioned 56-bus experiment environment; leaves historical Environment intact."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandapower as pp

BUS = np.array([17, 20, 29, 44, 52])
SWITCH_LINES = [7, 10, 12, 14, 22, 31, 33, 34, 35, 36, 37, 38, 41, 42, 46, 48, 50, 54]
ROOT = Path('D:/Code/Python/Flexible_Voltage_Control/experiments/controller_retraining_2026-09-11')

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

class Feeder:
    def __init__(self):
        self.net = pp.converter.from_mpc(str(Path(__file__).parent/'data/SCE_56bus.mat'), casename_mpc_file='case_mpc')
        self.net.sgen.loc[:, ['p_mw', 'q_mvar']] = 0.
        self.sgen_ids = [pp.create_sgen(self.net, int(b), p_mw=0., q_mvar=0.) for b in BUS]
        for line in SWITCH_LINES:
            pp.create_switch(self.net, int(self.net.line.at[line,'from_bus']), element=line, et='l', closed=True)
        self.r0 = self.net.line.r_ohm_per_km.to_numpy().copy()
        self.x0 = self.net.line.x_ohm_per_km.to_numpy().copy()
        self.p0 = self.net.load.p_mw.to_numpy().copy()
        self.q0 = self.net.load.q_mvar.to_numpy().copy()
        self.edges = self.net.line[['from_bus','to_bus']].to_numpy(dtype=int)
        self.slack = int(self.net.ext_grid.bus.iloc[0])
        self.counts = dict(reset=0, step=0, event=0, failed=0)

    def graph(self, switches):
        closed = np.ones(len(self.edges), dtype=bool)
        closed[SWITCH_LINES] = switches
        nodes = {self.slack}
        changed = True
        while changed:
            changed = False
            for active, (u,v) in zip(closed, self.edges):
                if active and ((u in nodes) != (v in nodes)):
                    nodes.update((int(u),int(v))); changed = True
        effective = closed & np.array([u in nodes and v in nodes for u,v in self.edges])
        return nodes, effective

    def signature(self, switches):
        nodes, mask = self.graph(switches)
        return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()[:20], nodes, mask

    def topology(self, switches):
        self.net.switch.loc[:, 'closed'] = np.array(switches, dtype=bool)
        self.nodes, self.mask = self.graph(switches)
        if not set(BUS).issubset(self.nodes):
            raise ValueError('disconnected controller')
        self.topology_id = self.signature(switches)[0]
        # Ratio of total series reactance to fixed baseline; line length cancels.
        self.features = (self.mask * self.x0 / self.net.line.x_ohm_per_km.to_numpy()).astype(np.float32)
        self.legacy_features = self.mask / self.net.line.x_ohm_per_km.to_numpy()

    def solve(self, category):
        self.counts[category] += 1
        try:
            pp.runpp(self.net, algorithm='bfsw', init='dc', numba=True)
            v = self.net.res_bus.vm_pu.to_numpy()
            if not np.isfinite(v[list(self.nodes)]).all():
                raise ValueError('nonfinite energized voltage')
            self.voltage = v.copy()
            self.state = v[BUS].astype(np.float32)
            return self.state.copy()
        except (pp.powerflow.LoadflowNotConverged, ValueError):
            self.counts['failed'] += 1
            raise

    def reset(self, scene):
        self.scene = scene
        mult = np.asarray(scene['multipliers'])
        self.net.line.loc[:, 'r_ohm_per_km'] = self.r0 / mult
        self.net.line.loc[:, 'x_ohm_per_km'] = self.x0 / mult
        self.net.load.loc[:, 'p_mw'] = np.asarray(scene['load_p'])
        self.net.load.loc[:, 'q_mvar'] = np.asarray(scene['load_q'])
        self.net.sgen.loc[:, ['p_mw','q_mvar']] = 0.
        self.net.sgen.loc[self.sgen_ids, 'p_mw'] = scene['pv_p']
        self.action = np.zeros(5, dtype=np.float32)
        self.topology(scene['switches'])
        return self.solve('reset')

    def event(self, switches):
        self.topology(switches)
        return self.solve('event')

    def step(self, q):
        self.action = np.asarray(q, dtype=np.float32).copy()
        self.net.sgen.loc[self.sgen_ids, 'q_mvar'] = self.action
        return self.solve('step')

    def snapshot(self):
        loadmask = self.net.load.bus.isin(self.nodes).to_numpy()
        return dict(topology_id=self.topology_id, energized_buses=sorted(self.nodes),
                    controlled_v=self.state.tolist(), voltage=[float(v) if np.isfinite(v) else None for v in self.voltage],
                    supplied_load_p=float(self.net.load.p_mw.to_numpy()[loadmask].sum()),
                    disconnected_load_p=float(self.net.load.p_mw.to_numpy()[~loadmask].sum()),
                    slack_p=float(self.net.res_ext_grid.p_mw.sum()),slack_q=float(self.net.res_ext_grid.q_mvar.sum()),
                    line_p=self.net.res_line.p_from_mw.fillna(0).tolist(),
                    line_q=self.net.res_line.q_from_mvar.fillna(0).tolist(),
                    line_i=self.net.res_line.i_ka.fillna(0).tolist())

def topology_pool(env, count=80):
    rng = np.random.default_rng(2026091101)
    pool = {}; attempts = 0
    while len(pool) < count and attempts < 50000:
        attempts += 1
        switches = (rng.random(len(SWITCH_LINES)) > .25).tolist()
        key, nodes, mask = env.signature(switches)
        if not set(BUS).issubset(nodes) or key in pool:
            continue
        # Canonicalize switches in disconnected subtrees; graph ID remains unchanged.
        switches = [bool(mask[line]) for line in SWITCH_LINES]
        pool[key] = dict(id=key, switches=switches)
    if len(pool) < count:
        raise RuntimeError(f'only {len(pool)} unique feasible graphs')
    return list(pool.values()), attempts

def sample_scene(env, topo, seed):
    rng = np.random.default_rng(seed)
    high = bool(seed % 2)
    load_scale = rng.uniform(1.0, 4.0)
    local = rng.uniform(.8, 1.2, size=len(env.p0))
    pv = (np.array([rng.uniform(1,5),rng.uniform(8,24),rng.uniform(1,5),rng.uniform(1,5),rng.uniform(1,5)])
          if high else rng.uniform(0,.5,size=5))
    return dict(seed=int(seed), topology_id=topo['id'], switches=topo['switches'],
                multipliers=rng.uniform(.8,1.2,size=len(env.x0)).tolist(),
                load_p=(env.p0*load_scale*local).tolist(),load_q=(env.q0*load_scale*local).tolist(),
                pv_p=pv.tolist(), regime='high' if high else 'low')

def make_scenes(env, pool, count, seed_start, events=False):
    scenes=[]; rejects=[]
    for i in range(count):
        topo=pool[i % len(pool)]
        for attempt in range(50):
            seed=seed_start+i+1000*attempt
            s=sample_scene(env,topo,seed)
            if events and i%2:
                s['events']=[dict(step=30,switches=pool[(i+1)%len(pool)]['switches']),dict(step=75,switches=topo['switches'])]
            else:
                s['events']=[]
            try:
                env.reset(s)
                voltages=[env.state.copy()]
                for event in s['events']:
                    voltages.append(env.event(event['switches']))
                if any(np.min(v)<.80 or np.max(v)>1.20 for v in voltages):
                    raise ValueError('initial voltage outside 0.80..1.20')
                scenes.append(s); break
            except (ValueError, pp.powerflow.LoadflowNotConverged) as exc:
                rejects.append(dict(seed=seed,reason=str(exc)))
        else:
            raise RuntimeError('scene sampling exhausted')
    return scenes,rejects

def prepare():
    env=Feeder()
    pool,attempts=topology_pool(env)
    splits=dict(train=pool[:16],validation=pool[16:28],test=pool[28:60])
    sets=[{t['id'] for t in p} for p in splits.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    scenes={}; rejected={}
    for name,count,seed in [('train',128,10000),('validation',24,20000),('test',128,30000)]:
        scenes[name],rejected[name]=make_scenes(env,splits[name],count,seed,events=True)
    save_json(ROOT/'protocol/scenarios.json',dict(splits=splits,scenes=scenes,rejected=rejected,topology_draws=attempts))
    rows=[]
    full=[True]*len(SWITCH_LINES)
    for i,topo in enumerate(pool[60:72]):
        for j in range(4):
            source=sample_scene(env,topo,40000+4*i+j)
            source['multipliers']=[1.]*len(env.x0)
            for load_mode in ['zero_load','retained_load']:
                s=copy.deepcopy(source); s['switches']=full
                if load_mode=='zero_load':
                    s['load_p']=[0.]*len(env.p0); s['load_q']=[0.]*len(env.q0)
                try:
                    env.reset(s); a=env.snapshot()
                    env.solve('reset'); repeat=env.state.copy()
                    env.event(topo['switches']); b=env.snapshot()
                    env.event(full); restored=env.state.copy()
                    rows.append(dict(case=i,operating_point=j,mode=load_mode,a=a,b=b,
                        dv_control=float(np.max(np.abs(np.array(a['controlled_v'])-b['controlled_v']))),
                        repeat_error=float(np.max(np.abs(np.array(a['controlled_v'])-repeat))),
                        restore_error=float(np.max(np.abs(np.array(a['controlled_v'])-restored)))))
                except (ValueError,pp.powerflow.LoadflowNotConverged) as exc:
                    rows.append(dict(case=i,operating_point=j,mode=load_mode,error=str(exc)))
    # Independent parameter-only control at a fixed graph, preserving R/X.
    admittance=[]
    for i in range(8):
        s=sample_scene(env,pool[i],50000+i); s['multipliers']=[1.]*len(env.x0)
        env.reset(s); a=env.snapshot()
        s['multipliers']=np.random.default_rng(60000+i).uniform(.8,1.2,len(env.x0)).tolist()
        env.reset(s); b=env.snapshot()
        admittance.append(dict(a=a,b=b,dv=float(np.max(np.abs(np.array(a['controlled_v'])-b['controlled_v'])))))
    summary={}
    for mode in ['zero_load','retained_load']:
        valid=[r for r in rows if r['mode']==mode and 'error' not in r]
        dv=[r['dv_control'] for r in valid]
        summary[mode]=dict(n=len(valid),errors=sum(r['mode']==mode and 'error' in r for r in rows),
                          dv_median=float(np.median(dv)),dv_max=float(max(dv)),
                          above_0p002=sum(v>=.002 for v in dv),
                          restore_error_max=max(r['restore_error'] for r in valid))
    save_json(ROOT/'powerflow_validation/results.json',dict(summary=summary,rows=rows,admittance=admittance,counts=env.counts))
    print(json.dumps(summary,indent=2),flush=True)
    for mode in ['zero_load','retained_load']:
        assert summary[mode]['restore_error_max']<1e-7
    assert summary['retained_load']['above_0p002']>0, 'no meaningful topology response'
    print('PREPARE_PASS',flush=True)

if __name__=='__main__':
    prepare()
