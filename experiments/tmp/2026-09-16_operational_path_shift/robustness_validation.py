"""Predeclared operational validation; reuse original dynamics and fixed policies."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import joint_search as j

np, pd, m, paths, vis = j.np, j.pd, j.m, j.paths, j.vis
HERE = Path(__file__).resolve().parent
OUT = paths.DATA / 'experiments/operational_path_shift/20260916_robustness_validation'
OLD = paths.DATA / 'experiments/operational_path_shift/20260916_joint_search'
METHODS = vis.METHODS
CONTEXT = None
ACTIVE_LOAD_ROWS = {5, 8, 10, 12, 14, 19, 22, 33, 37, 38, 41}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def signature(c):
    return (c['line'], tuple(c['lines']), c['r_factor'], c['x_factor'],
            c['outage_start'], c['event'], c.get('load_scale', 1.), c.get('pv_scale', 1.))


def base_case():
    p = json.loads((OLD/'batches/load_timing/manifest.json').read_text())
    c = next(c for c in p['cases'] if c['id'] == 'm54_restore_14p5')
    return dict(c, load_scale=1., pv_scale=1.)


def context():
    assert CONTEXT is not None
    return CONTEXT


def install(c):
    global CONTEXT
    j.install(c)
    runtime, data, profiles, baselines, protocol = paths.context()
    scaled = {key: np.array(value, copy=True) * (c['pv_scale'] if key == 'pv_p' else c['load_scale'])
              for key, value in profiles.items()}
    protocol = copy.deepcopy(protocol)
    protocol['validation_entry_hash'] = paths.base.digest(__file__)
    protocol['profile_scaling'] = dict(load=c['load_scale'], pv=c['pv_scale'])
    protocol['initialization'] = 'full day, original zero-injection initialization, fixed scenario background'
    CONTEXT = runtime, data, scaled, baselines, protocol
    paths.base.context = context
    paths.base.ROOT = str(OUT/'cache')


def inventory():
    env, _ = j.BUILD(paths.context()[0])
    net = env.network
    g = m.nx.Graph(m.ppt.create_nxgraph(net, respect_switches=True))
    assert m.nx.is_tree(g)
    rows = []
    for idx in sorted(set(net.switch.loc[net.switch.et.eq('l'), 'element'].astype(int))):
        idx = int(idx)
        edge = net.line.loc[idx]
        a, b = int(edge.from_bus), int(edge.to_bus)
        h = g.copy()
        h.remove_edge(a, b)
        online = set(m.nx.node_connected_component(h, 0))
        off = sorted(int(n) for n in set(g) - online)
        loads = sorted(int(i) for i in set(net.load.index[net.load.bus.isin(off)]) & ACTIVE_LOAD_ROWS)
        eligible = set(vis.CONTROL).issubset(online) and bool(loads)
        rows.append(dict(line=idx, buses=[a+1, b+1], offline_buses=[v+1 for v in off],
                         active_load_rows=loads, controls_online=set(vis.CONTROL).issubset(online),
                         eligible=eligible, radial=m.nx.is_tree(h.subgraph(online))))
    eligible = [r for r in rows if r['eligible']]
    chosen = [next(r for r in eligible if r['line'] == 54)]
    # Farthest-point sampling on the original tree, without controller outcomes.
    while len(chosen) < min(3, len(eligible)):
        available = [r for r in eligible if r not in chosen]
        def distance(r):
            return min(m.nx.shortest_path_length(g, r['buses'][1]-1, s['buses'][1]-1) for s in chosen)
        chosen.append(max(available, key=lambda r: (distance(r), -r['line'])))
    return dict(all_switches=rows, selected=chosen,
                selection_rule='include line54 then maximize minimum original-tree distance; lower line ID breaks ties; no response screening')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        print('EXISTING MANIFEST: not changing the predeclared design', flush=True)
        return
    inv = inventory()
    center = base_case()
    design = {}
    def add(c, group):
        key = signature(c)
        if key in design:
            design[key]['groups'].append(group)
        else:
            design[key] = dict(c, id=f'local_{len(design):02d}', groups=[group], phase='local')
    for start in (4800, 5400, 6000):
        for event in (8100, 8700, 9300):
            add(dict(center, outage_start=start, event=event), 'timing')
    for load in (.9, 1., 1.1):
        for pv in (.9, 1., 1.1):
            add(dict(center, load_scale=load, pv_scale=pv), 'profile')
    for r, x in ((2.5, 2.), (2.25, 2.), (2.75, 2.), (2.5, 1.8), (2.5, 2.2)):
        add(dict(center, r_factor=r, x_factor=x), 'impedance')
    local = list(design.values())
    assert len(local) == 21
    old = {signature(c): c for c in j.all_cases() if c['direction'] == 'maintenance' and c.get('full')}
    for c in local:
        c['reuse_case_id'] = old.get(signature(c), {}).get('id')
    rng = np.random.default_rng(260916)
    independent = []
    for row in inv['selected']:
        for i in range(20):
            c = dict(center, id=f'validation_l{row["line"]}_{i:02d}', line=row['line'],
                     outage_start=int(rng.integers(4800, 6001)), event=int(rng.integers(8100, 9301)),
                     load_scale=float(rng.uniform(.9, 1.1)), pv_scale=float(rng.uniform(.9, 1.1)),
                     r_factor=float(rng.uniform(2.25, 2.75)), x_factor=float(rng.uniform(1.8, 2.2)),
                     groups=['independent'], phase='independent', reuse_case_id=None)
            assert signature(c) not in old and signature(c) not in design
            independent.append(c)
    code = [HERE/'joint_search.py', HERE/'maintenance.py', HERE/'run.py', Path(__file__),
            paths.REPO/'Environment.py', paths.REPO/'NN_Module.py', paths.REPO/'config.py',
            Path(paths.base.__file__), Path(paths.old.__file__), Path(vis.__file__)]
    protocol = copy.deepcopy(paths.context()[4])
    protocol['validation_source_hashes'] = {str(p): paths.base.digest(p) for p in code}
    write_json(OUT/'manifest.json', dict(seed=260916, local=local, independent=independent,
               inventory=inv, protocol=protocol, planned_method_days=3*(len(local)+len(independent)),
               note='Local sensitivity includes exploratory reuse; independent set frozen before validation. No new training.'))
    write_json(OUT/'topology_inventory.json', inv)
    print(json.dumps(dict(local=len(local), independent=len(independent), reused_local=sum(bool(c['reuse_case_id']) for c in local), selected=inv['selected'])), flush=True)


def manifest():
    p = json.loads((OUT/'manifest.json').read_text())
    for name, expected in p['protocol']['validation_source_hashes'].items():
        assert paths.base.digest(name) == expected, f'Changed source: {name}'
    return p


def cache_path(c, method):
    if c.get('reuse_case_id'):
        return OLD/'cache/full'/f'{c["reuse_case_id"]}__{method}__original.pkl.gz'
    return OUT/'cache/full'/f'{c["id"]}__{method}__original.pkl.gz'


def summarize(c, method, payload):
    row = dict(case_id=c['id'], method=method, phase=c['phase'], line=c['line'],
               groups=';'.join(c['groups']), load_scale=c['load_scale'], pv_scale=c['pv_scale'],
               r_factor=c['r_factor'], x_factor=c['x_factor'], outage_hour=c['outage_start']/600,
               restore_hour=c['event']/600, reused=bool(c.get('reuse_case_id')))
    if 'result' not in payload:
        return dict(row, status='failed', error=payload['summary']['error'])
    r = payload['result']
    assert r['all_states'].shape == (14400, 56)
    return dict(row, status='ok', elapsed=r['elapsed'], **vis.metrics(r))


def run_one(c, method):
    install(c)
    path = cache_path(c, method)
    if c.get('reuse_case_id'):
        payload = vis.load(path)
        assert signature(payload['key']['case']) == signature(c)
        for name, h in context()[4]['checkpoint_hashes'].items():
            assert payload['key']['protocol']['checkpoint_hashes'][name] == h
    else:
        paths.base.run(c, method, full_day=True)
        payload = vis.load(path)
    row = summarize(c, method, payload)
    write_json(OUT/'rows'/f'{c["id"]}__{method}.json', row)
    return row


def table():
    rows = [json.loads(p.read_text()) for p in sorted((OUT/'rows').glob('*.json'))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame.to_csv(OUT/'all_metrics.csv', index=False)
    summary = []
    for (phase, method), group in frame.groupby(['phase', 'method']):
        ok = group[group.status.eq('ok')]
        summary.append(dict(phase=phase, method=method, completed=len(group), valid=len(ok),
                       passes=int(ok.passes.sum()), peak_median=float(ok.peak.median()), peak_max=float(ok.peak.max()),
                       duration_median=float(ok.duration.median()), duration_max=float(ok.duration.max())))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
    return frame


def audit_one(c, method):
    install(c)
    p = vis.load(cache_path(c, method))
    if 'result' not in p:
        return dict(case_id=c['id'], method=method, status='not_completed')
    r = p['result']
    runtime, data, profiles, _, _ = context()
    env, _ = j.build(runtime)
    m.apply_case(env, c, c['outage_start'])
    graph = m.ppt.create_nxgraph(env.network, respect_switches=True)
    online = sorted(m.nx.node_connected_component(graph, 0))
    expected = np.ones((14400, 56), dtype=bool)
    offline = sorted(set(range(56))-set(online))
    expected[c['outage_start']:c['event'], offline] = False
    assert offline and set(vis.CONTROL).issubset(online)
    assert np.array_equal(np.isfinite(r['all_states']), expected)
    device = paths.torch.device('cuda' if paths.torch.cuda.is_available() else 'cpu')
    policies = paths.old.load_policy_set(method, env, data, *runtime[2:5], runtime[5], device)
    errors = np.zeros(3)
    points = (0, c['outage_start']-1, c['outage_start'], c['event']-1, c['event'], c['event']+50, 14399)
    for k in points:
        m.apply_case(env, c, k)
        env.state = r['states'][k].copy()
        topo = paths.torch.as_tensor(env.network.line.x_ohm_per_km.to_numpy(), dtype=paths.torch.float32, device=device)[None]
        with paths.torch.inference_mode():
            q = np.clip(paths.old.controller_action(method, env.state, r['actions'][k-1] if k else np.zeros(5), policies, topo, device), -25, 25)
            _, _, reward, _ = env.step_load(q.reshape(5, 1), profiles['p'][k], profiles['q'][k], profiles['pv_p'][k])
        v = env.network.res_bus.vm_pu.to_numpy()
        assert np.array_equal(np.isfinite(v), expected[k])
        errors = np.maximum(errors, [np.max(np.abs(q-r['actions'][k])), np.nanmax(np.abs(v-r['all_states'][k])), abs(reward-r['rewards'][k])])
    assert max(errors) < 1e-7, errors
    run = best = 0
    bad = ((r['all_states'] < .95) | (r['all_states'] > 1.05)).sum(axis=1)/expected.sum(axis=1) > .1
    for value in bad:
        run = run+1 if value else 0
        best = max(best, run)
    assert abs(.1*best-vis.metrics(r)['duration']) < 1e-9
    return dict(case_id=c['id'], method=method, status='verified', errors=errors.tolist(), offline_bus_ids=[v+1 for v in offline],
                full_online_mask=True, independent_duration=.1*best, cache_sha256=paths.base.digest(cache_path(c, method)))


def smoke():
    p = manifest()
    center = next(c for c in p['local'] if signature(c) == signature(base_case()))
    # Replay the accepted case under this wrapper before starting new scenarios.
    checks = [audit_one(center, method) for method in METHODS]
    scaled = next(c for c in p['local'] if c['load_scale'] == .9 and c['pv_scale'] == 1.1)
    install(scaled)
    raw = paths.context()[2]
    for key in ('p', 'q', 'pv_p'):
        assert np.array_equal(context()[2][key], raw[key]*(1.1 if key == 'pv_p' else .9))
    write_json(OUT/'smoke_verification.json', checks)
    print(json.dumps(checks), flush=True)


def run_phase(phase, workers, limit):
    p = manifest()
    chosen = p[phase]
    jobs = [(c, method) for c in chosen for method in METHODS if not (OUT/'rows'/f'{c["id"]}__{method}.json').exists()]
    if limit:
        jobs = jobs[:limit]
    print(f'START {phase}: {len(jobs)} jobs, {workers} workers', flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, c, method) for c, method in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            table()
            print('PROGRESS '+json.dumps(dict(done=i, total=len(jobs), wall=time.perf_counter()-start, **row)), flush=True)
    print('BATCH COMPLETE', flush=True)


def review():
    frame = table()
    vis.style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    (OUT/'figures').mkdir(exist_ok=True)
    for phase, data in frame.groupby('phase'):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), layout='constrained')
        rng = np.random.default_rng(19)
        for ax, key, limit, title, label in ((axes[0], 'peak', 1.1, 'Peak voltage', 'Voltage (p.u.)'),
                                           (axes[1], 'duration', 5., 'Longest voltage violation', 'Duration (min)')):
            for i, method in enumerate(METHODS):
                vals = data.loc[data.method.eq(method) & data.status.eq('ok'), key].to_numpy()
                ax.scatter(i+rng.uniform(-.13, .13, len(vals)), vals, s=17, color=vis.COLORS[i], alpha=.65, edgecolors='none')
                if len(vals):
                    q1, med, q3 = np.quantile(vals, [.25, .5, .75])
                    ax.vlines(i, q1, q3, color='#333333', lw=2)
                    ax.plot([i-.1, i+.1], [med, med], color='#111111', lw=2)
            ax.axhline(limit, color='#AD5149', ls='--', lw=1)
            ax.set(xticks=range(3), xticklabels=METHODS, ylabel=label, title=title, xlim=(-.45, 2.45))
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        axes[1].set_yscale('symlog', linthresh=1.)
        axes[1].set_yticks([0, 1, 5, 20, 100, 500])
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        axes[1].set_ylim(bottom=0)
        ax = axes[2]
        lines = sorted(data.line.unique())
        for i, method in enumerate(METHODS):
            for pos, line in enumerate(lines):
                g = data[data.method.eq(method) & data.line.eq(line)]
                count = int(g.loc[g.status.eq('ok'), 'passes'].sum())
                x = pos+(i-1)*.22
                ax.bar(x, count/len(g)*100, width=.19, color=vis.FILLS[i], edgecolor=vis.COLORS[i], linewidth=.7,
                       label=method if pos == 0 else None)
                ax.text(x, count/len(g)*100+2, f'{count}/{len(g)}', ha='center', fontsize=8)
        inv = json.loads((OUT/'topology_inventory.json').read_text())
        lookup = {r['line']: '-'.join(map(str, r['buses'])) for r in inv['all_switches']}
        ax.set(xticks=range(len(lines)), xticklabels=[lookup[x] for x in lines], xlabel='Maintenance branch (bus IDs)',
               ylabel='Scenarios passing both references (%)', ylim=(0, 124), title='Operational reference checks')
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc='upper right', fontsize=7, frameon=False)
        for letter, ax in zip('abc', axes):
            ax.text(-.12, 1.07, letter, transform=ax.transAxes, fontsize=12, weight='bold')
        fig.suptitle(('Local sensitivity (21 design points)' if phase == 'local' else 'Independent scenario validation')+' | Fixed trained controllers', fontsize=11)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT/'figures'/f'{phase}_robustness.{ext}', dpi=220)
        plt.close(fig)
    paired = []
    for (phase, cid), g in frame.groupby(['phase', 'case_id']):
        rows = {r['method']: r for r in g.to_dict('records') if r['status'] == 'ok'}
        if len(rows) != 3:
            continue
        for baseline in ('Linear', 'Safe-DDPG'):
            r, b = rows['RLC-FT'], rows[baseline]
            paired.append(dict(phase=phase, case_id=cid, line=r['line'], baseline=baseline,
                               peak_reduction=b['peak']-r['peak'], duration_reduction=b['duration']-r['duration'],
                               cost_ratio=r['cost']/b['cost']))
    pd.DataFrame(paired).to_csv(OUT/'paired_metrics.csv', index=False)
    import nbformat as nbf
    from nbclient import NotebookClient
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell('# Operational robustness validation\n\nFixed controllers, paired scenarios. Local sensitivity and independent validation are separate. Duration uses >10% energized modeled buses outside 0.95-1.05 p.u.; this is a modeled-node reference, not meter compliance. Duration axis uses a linear region near zero and a logarithmic scale above 1 min.'),
                nbf.v4.new_code_cell(f'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path({str(OUT)!r})\ndisplay(pd.read_csv(root/"summary.csv"))\ndisplay(pd.read_csv(root/"paired_metrics.csv").groupby(["phase","baseline"])[["peak_reduction","duration_reduction","cost_ratio"]].agg(["min","median","max"]))')]
    for f in sorted((OUT/'figures').glob('*_robustness.png')):
        nb.cells.append(nbf.v4.new_code_cell(f'display(Image(filename={str(f)!r}))'))
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
    nbf.write(nb, HERE/'robustness_validation_review.ipynb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'smoke', 'run', 'table', 'audit', 'review'])
    ap.add_argument('--phase', choices=['local', 'independent'], default='local')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'smoke': smoke()
    elif args.command == 'run': run_phase(args.phase, args.workers, args.limit)
    elif args.command == 'table':
        table()
        print((OUT/'summary.csv').read_text(), flush=True)
    elif args.command == 'review': review()
    elif args.command == 'audit':
        checks = []
        for c in manifest()[args.phase]:
            for method in METHODS:
                checks.append(audit_one(c, method))
        write_json(OUT/f'{args.phase}_audit.json', checks)
        print(json.dumps(dict(verified=sum(r['status']=='verified' for r in checks), total=len(checks))), flush=True)


if __name__ == '__main__':
    main()
