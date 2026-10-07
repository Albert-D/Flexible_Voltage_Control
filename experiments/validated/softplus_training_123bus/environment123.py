"""123-bus case environment retaining nominal loads; 114 modeled buses, 113 lines."""
from __future__ import annotations

# Shared 56-bus dependencies now live in the validated experiment.
from pathlib import Path as _MigrationPath
import sys as _migration_sys
_migration_sys.path.insert(0, str(_MigrationPath(__file__).resolve().parents[3]))
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandapower as pp

BUS = np.array([9,10,15,19,32,35,47,58,65,74,82,91,103,60])
PV_BUS = list(BUS)+[13,14,18]
SWITCH_LINES = [0,1,3,4,5,7,20,22,27,31,25,29,30,35,36,37,38,40,42,44,46,47,48,16,51,53,62,65,66,67,75,79,81,83,85,88,89,90,91,92,93,94,99,101,104]
ROOT = Path('D:/Code/Python/Flexible_Voltage_Control/experiments/controller_retraining_2026-09-11')

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

class Feeder:
    def __init__(self):
        self.net = pp.converter.from_mpc(str(Path(__file__).resolve().parents[3]/'data/case_123.mat'), casename_mpc_file='case_mpc')
        self.net.sgen.loc[:, ['p_mw', 'q_mvar']] = 0.
        self.sgen_ids = [pp.create_sgen(self.net, int(b), p_mw=0., q_mvar=0.) for b in PV_BUS]
        self.control_sgen_ids=self.sgen_ids[:len(BUS)]
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
        self.action = np.zeros(len(BUS), dtype=np.float32)
        self.topology(scene['switches'])
        return self.solve('reset')

    def event(self, switches):
        self.topology(switches)
        return self.solve('event')

    def step(self, q):
        self.action = np.asarray(q, dtype=np.float32).copy()
        self.net.sgen.loc[self.control_sgen_ids, 'q_mvar'] = self.action
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


def sample_scene(env,topo,seed):
    rng=np.random.default_rng(seed);high=bool(seed%2)
    if high:
        low=np.array([15,15,20,10,2,2,10,5,2,2,1,1,1,1,15,10,10])
        hi=np.array([60,50,60,34,20,80,80,50,30,30,40,30,30,24,25,50,20])
        scale=np.array([.8,.8,.8,.8,.8,.8,.8,.8,.7,.5,.4,.5,.5,.5,.5,.8,.8])
    else:
        low=np.array([15,10,10,10,1,2,2,1,1,1,1,1,1,2,10,10,10])
        hi=np.array([60,45,55,30,35,25,30,10,15,30,20,20,20,10,20,20,20])
        scale=-np.array([.8,.8,.8,.8,.6,.5,.8,.9,.7,.5,.3,.5,.4,.4,.4,.8,.8])
    pv=rng.uniform(low,hi)*scale
    load_scale=rng.uniform(.8,1.2);local=rng.uniform(.8,1.2,len(env.p0))
    return dict(seed=int(seed),topology_id=topo['id'],switches=topo['switches'],multipliers=rng.uniform(.5,1.5,len(env.x0)).tolist(),load_p=(env.p0*load_scale*local).tolist(),load_q=(env.q0*load_scale*local).tolist(),pv_p=pv.tolist(),regime='high' if high else 'low',events=[])
