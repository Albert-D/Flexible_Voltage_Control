"""Opt-in AC solver strategies. Reset/event always rebuild the network.

Caller must use reset/event for topology, line admittance, or load changes.
No linearized power flow, reduced tolerance, partial results or silent fallback.
"""
from __future__ import annotations
import numpy as np
import pandapower as pp

MODES = ('bfsw_dc', 'bfsw_warm', 'nr_warm', 'nr_recycle', 'lightsim_warm', 'lightsim_recycle')

def feeder_class(base, controlled_buses, mode='lightsim_recycle'):
    if mode not in MODES:
        raise ValueError(mode)
    class SolverFeeder(base):
        solver_mode = mode
        def solve(self, category):
            self.counts[category] += 1
            try:
                warm = category == 'step' and mode != 'bfsw_dc'
                kw = dict(algorithm='bfsw' if mode.startswith('bfsw') else 'nr',
                          init='results' if warm else 'dc', numba=True,
                          tolerance_mva=1e-8)
                if not mode.startswith('bfsw'):
                    # Disabling ZIP handling is valid only for constant-PQ loads.
                    for col in ('const_z_percent', 'const_i_percent'):
                        if col in self.net.load and self.net.load[col].abs().max() != 0:
                            raise ValueError('ZIP loads require a separately validated backend')
                    kw.update(voltage_depend_loads=False, lightsim2grid=mode.startswith('lightsim'),
                              max_iteration=100)
                    if warm and mode.endswith('recycle'):
                        kw['recycle'] = dict(bus_pq=True, trafo=False, gen=False)
                pp.runpp(self.net, **kw)
                if mode.startswith('lightsim') and not self.net['_options'].get('lightsim2grid'):
                    raise ValueError('lightsim requested but not active')
                v = self.net.res_bus.vm_pu.to_numpy()
                if not np.isfinite(v[list(self.nodes)]).all():
                    raise ValueError('nonfinite energized voltage')
                self.voltage = v.copy()
                self.state = v[controlled_buses].astype(np.float32)
                return self.state.copy()
            except (pp.powerflow.LoadflowNotConverged, ValueError):
                self.counts['failed'] += 1
                raise
    return SolverFeeder
