"""Actual load-branch maintenance, fixed controllers, and cache-only 2x3 review."""
from pathlib import Path
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as paths
import review_operational_direction_screen as vis
import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
import pandapower.topology as ppt
import matplotlib.pyplot as plt

OUT = paths.DATA / 'experiments/operational_path_shift/20260916_maintenance_restoration'
SOURCE = paths.DATA / 'cache/load_branch_restoration_2026-09-15/combined.pkl.gz'
CURRENT = None
ORIGINAL_BUILD = paths.base.build_env
METHODS = vis.METHODS


def cases():
    common = dict(direction='maintenance', lines=[0, 2, 18, 21, 23, 24, 30],
                  r_factor=2.5, x_factor=2., outage_start=5400, start=0, stop=14400,
                  interpretation='Existing load branch isolated for maintenance and restored; fixed high-impedance feeder variant')
    return [dict(common, id='maintenance54_midday', line=54, event=7838),
            dict(common, id='maintenance54_evening', line=54, event=12279),
            dict(common, id='maintenance48_midday', line=48, event=7838)]


def build_env(runtime):
    env, state = ORIGINAL_BUILD(runtime)
    if CURRENT is not None:
        env.network.line.loc[CURRENT['lines'], 'r_ohm_per_km'] *= CURRENT['r_factor']
        env.network.line.loc[CURRENT['lines'], 'x_ohm_per_km'] *= CURRENT['x_factor']
        env.topology_init = env.network.line.x_ohm_per_km.to_numpy().copy()
        env.topology = env.topology_init.copy()
        pp.runpp(env.network, algorithm='bfsw', init='dc')
        state = env.network.res_bus.vm_pu.iloc[vis.CONTROL].to_numpy().copy()
        env.state = state.copy()
    return env, state


def apply_case(env, case, k, r0=None, x0=None):
    mask = env.network.switch.et.eq('l') & env.network.switch.element.eq(case['line'])
    assert mask.sum() == 1
    env.network.switch.loc[mask, 'closed'] = not (case['outage_start'] <= k < case['event'])


def install(case):
    global CURRENT
    CURRENT = case
    paths.base.build_env = build_env
    paths.base.apply_case = apply_case
    paths.base.ROOT = str(OUT / 'cache')


def run_one(case, method):
    install(case)
    paths.context()[4]['maintenance_entry_hash'] = paths.base.digest(__file__)
    row = paths.base.run(case, method, full_day=True)
    path = OUT / 'cache/full' / f'{case["id"]}__{method}__original.pkl.gz'
    payload = vis.load(path)
    if 'result' in payload:
        row = dict(case_id=case['id'], method=method, status='ok',
                   **vis.metrics(payload['result']))
    return row


def topology_record(case):
    install(case)
    env, _ = build_env(paths.context()[0])
    net = env.network
    records = []
    for k in (0, case['outage_start'], case['event']):
        apply_case(env, case, k)
        g = ppt.create_nxgraph(net, respect_switches=True)
        online = sorted(int(n) for n in nx.node_connected_component(g, 0))
        assert set(vis.CONTROL).issubset(online) and nx.is_forest(nx.Graph(g))
        records.append(dict(step=k, online=online, closed=bool(net.switch.loc[
            net.switch.et.eq('l') & net.switch.element.eq(case['line']), 'closed'].iloc[0])))
    assert records[0]['online'] == records[2]['online'] and len(records[1]['online']) < 56
    offline = sorted(set(records[0]['online']) - set(records[1]['online']))
    load_rows = net.load.index[net.load.bus.isin(offline)].tolist()
    active_rows = sorted(set(load_rows) & {5, 8, 10, 12, 14, 19, 22, 33, 37, 38, 41})
    assert active_rows, 'Maintenance must disconnect an actual modeled load'
    p = paths.context()[2]
    return dict(case=case, branch_buses=(net.line.loc[case['line'], ['from_bus', 'to_bus']].astype(int)+1).tolist(),
                offline_bus_ids=[b+1 for b in offline], load_rows=active_rows,
                restoration_load_mw=float(.08*len(active_rows)*p['p'][case['event']]), snapshots=records)


def records():
    original = vis.load(SOURCE)
    rr = {r['method']: dict(r, all_states=r['all_states']) for r in original['results'] if r['event']}
    yield 'original_maintenance54_evening', dict(id='original_maintenance54_evening', line=54,
          outage_start=5400, event=12279, r_factor=1., x_factor=1.), rr
    for case in cases():
        rr = {}
        for method in ('No control', *METHODS):
            path = OUT / 'cache/full' / f'{case["id"]}__{method}__original.pkl.gz'
            if path.exists():
                p = vis.load(path)
                if 'result' in p:
                    rr[method] = p['result']
        if rr:
            yield case['id'], case, rr


def figure(name, case, rr, profiles):
    h = np.arange(14400)/600
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.5))
    fig.subplots_adjust(left=.08, right=.985, top=.84, bottom=.15, wspace=.53, hspace=.78)
    a, b, c, d, e, f = axes.flat
    for y, label, color in ((.88*profiles['p'], 'Load P', '#6F8FAC'),
                            (.88*profiles['q'], 'Load Q', '#C78F70'),
                            (1.75*profiles['pv_p'], 'PV P', '#6D8F5E')):
        a.plot(h, y, label=label, color=color, lw=1.)
    a.legend(ncol=3, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(-.1, 1.23),
             handlelength=1.1, columnspacing=.8)
    a.set(title='Demand and PV profiles', ylabel='Power (MW / Mvar)')
    extrema = []
    for ax, method, color in ((b, 'No control', '#777777'), (c, 'RLC-FT', vis.COLORS[2])):
        v = rr[method]['all_states']
        lo, hi = np.nanmin(v, axis=1), np.nanmax(v, axis=1)
        ax.fill_between(h, lo, hi, color=color, alpha=.15, lw=0)
        ax.plot(h, lo, color=color, lw=.85)
        ax.plot(h, hi, color=color, lw=.85)
        for limit in (.95, 1.05):
            ax.axhline(limit, color='#888888', ls=':', lw=.7)
        ax.set_title(method)
        extrema.extend([float(lo.min()), float(hi.max())])
    b.set_ylabel('Voltage envelope (p.u.)')
    for ax in (b, c):
        ax.set_ylim(min(.94, min(extrema)-.008), max(1.06, max(extrema)+.008))
    fig.text(.67, .94, 'Voltage envelope: all energized modeled buses', ha='center', fontsize=9)
    for ax in (a, b, c):
        ax.axvline(case['outage_start']/600, color='#888888', ls='--', lw=.8)
        ax.axvline(case['event']/600, color='#9B4E4A', ls=':', lw=1.)
        ax.set(xlim=(0,24), xticks=[0,6,12,18,24], xlabel='Time (h)')
    mm = {m: vis.metrics(rr[m]) for m in METHODS}
    for ax, key, title, ylabel, limit in ((d,'peak','Peak voltage','Voltage (p.u.)',1.10),
            (e,'duration','Longest voltage violation','Duration (min)',5.),
            (f,'cost','Total objective cost','Normalized cost',None)):
        vals = np.array([mm[m][key] for m in METHODS])
        if key == 'cost':
            vals /= vals[0]
        floor = .95 if key == 'peak' else min(.9, vals.min()-.05) if key == 'cost' else 0.
        ceiling = max(vals.max(), limit or vals.max()); span=max(ceiling-floor,.04)
        ax.set_ylim(floor-.035*span if key == 'duration' else floor, ceiling+.28*span)
        for j, value in enumerate(vals):
            ax.vlines(j, floor, value, color=vis.COLORS[j], lw=1.2)
            ax.scatter(j, value, s=48, facecolor=vis.FILLS[j], edgecolor=vis.COLORS[j], zorder=4)
            ax.annotate(f'{value:.1f}' if key=='duration' else f'{value:.3f}', (j,value),
                        xytext=(0,8), textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.6))
        ax.axhline(limit if limit is not None else 1., color='#A64F4A' if limit else '#999999',
                   ls='--' if limit else ':', lw=.8)
        ax.set(xlim=(-.45,2.45), title=title, ylabel=ylabel)
        ax.set_xticks(range(3), METHODS, rotation=15, ha='right')
    for letter, ax in zip('abcdef', axes.flat):
        ax.text(-.23, 1.08, letter, transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_axisbelow(True); ax.grid(axis='y',color='#e8e8e8',lw=.5)
    fig.text(.5,.062, f'Maintenance isolation: 09:00; restoration: {case["event"]/600:.2f} h | Metrics: full 24 hours',ha='center',fontsize=9)
    fig.text(.5,.029,'Duration: >10% of energized modeled buses outside 0.95-1.05 p.u.; references: 1.10 p.u. and 5 min.',ha='center',fontsize=8)
    folder=OUT/'figures'; folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(folder/f'{name}_2x3.{ext}',dpi=210,facecolor='white')
    plt.close(fig)


def review(make_notebook=False):
    vis.style(); profiles=vis.load(SOURCE)['profiles']; rows=[]
    for name, case, rr in records():
        for method,r in rr.items():
            rows.append(dict(case_id=name,method=method,**vis.metrics(r)))
        if set(('No control',*METHODS)).issubset(rr):
            figure(name,case,rr,profiles)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'full_metrics.csv',index=False)
    print(frame.to_string(index=False),flush=True)
    if make_notebook:
        import nbformat as nbf
        from nbclient import NotebookClient
        nb=nbf.v4.new_notebook()
        nb.cells=[nbf.v4.new_markdown_cell('# Operational maintenance restoration\n\nActual existing load-branch isolation and restoration, with all five controllers online. Fixed checkpoints and scales. Full-day metrics use energized modeled buses, not measured customer meters. This notebook reads saved results only.'),
                  nbf.v4.new_code_cell(f'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path({str(OUT)!r})\ndisplay(pd.read_csv(root / "full_metrics.csv"))')]
        for path in sorted((OUT/'figures').glob('*.png')):
            nb.cells.append(nbf.v4.new_markdown_cell('## '+path.stem.replace('_',' ')))
            nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename={str(path)!r}))'))
        nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
        NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(HERE)}}).execute()
        nbf.write(nb,HERE/'maintenance_review.ipynb')


def verify():
    checks=[]; runtime=paths.context()[0]; profiles=paths.context()[2]
    device=paths.torch.device('cuda' if paths.torch.cuda.is_available() else 'cpu')
    for case in cases():
        install(case)
        for method in ('No control',*METHODS):
            path=OUT/'cache/full'/f'{case["id"]}__{method}__original.pkl.gz'
            if not path.exists(): continue
            p=vis.load(path)
            if 'result' not in p: continue
            r=p['result']; env,_=build_env(runtime)
            policies=paths.old.load_policy_set(method,env,paths.DATA,*runtime[2:5],runtime[5],device)
            topology=paths.torch.as_tensor(env.topology_init,dtype=paths.torch.float32,device=device)[None]
            errors=np.zeros(3)
            for k in (0,5399,5400,case['event']-1,case['event'],case['event']+50,14399):
                apply_case(env,case,k); env.state=r['states'][k].copy()
                last=r['actions'][k-1] if k else np.zeros(5)
                with paths.torch.inference_mode():
                    q=np.clip(paths.old.controller_action(method,env.state,last,policies,topology,device),-25,25)
                    _,_,reward,_=env.step_load(q.reshape(5,1),profiles['p'][k],profiles['q'][k],profiles['pv_p'][k])
                v=env.network.res_bus.vm_pu.to_numpy()
                assert np.array_equal(np.isfinite(v),np.isfinite(r['all_states'][k]))
                errors=np.maximum(errors,[np.max(np.abs(q-r['actions'][k])),np.nanmax(np.abs(v-r['all_states'][k])),abs(reward-r['rewards'][k])])
            topo=topology_record(case); offline=[i-1 for i in topo['offline_bus_ids']]
            expected=np.ones_like(r['all_states'],dtype=bool)
            expected[case['outage_start']:case['event'],offline]=False
            assert np.array_equal(np.isfinite(r['all_states']),expected)
            assert max(errors)<1e-7
            checks.append(dict(case_id=case['id'],method=method,max_action_error=errors[0],max_voltage_error=errors[1],max_reward_error=errors[2],topology_verified=True))
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['prepare','run','review','verify'],default='prepare')
    parser.add_argument('--case',default='maintenance54_midday'); parser.add_argument('--methods',nargs='+',default=list(METHODS))
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--notebook',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':
        payload=dict(protocol=paths.context()[4],entry_hash=paths.base.digest(__file__),cases=[topology_record(c) for c in cases()])
        (OUT/'manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(json.dumps(payload,indent=2),flush=True); review(args.notebook)
    elif args.stage=='run':
        case=next(c for c in cases() if c['id']==args.case); rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_one,case,m) for m in args.methods]
            for f in as_completed(futures):
                rows.append(f.result()); print(json.dumps(rows[-1]),flush=True)
                (OUT/f'{case["id"]}_last_batch.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        review(args.notebook)
    elif args.stage=='review': review(args.notebook)
    else: verify()


if __name__=='__main__':
    main()
