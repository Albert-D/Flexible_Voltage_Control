"""Bounded operational path-shift exploration using the existing AC runner."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'Environment.py').is_file())
ARCHIVE = REPO / 'tests/operational_candidates'
sys.path.insert(0, str(REPO))
import run_real_world_pv_outage_stress as old
sys.path.insert(0, str(ARCHIVE))
import run_operational_direction_screen as base
import networkx as nx
import numpy as np
import torch
from config import Config

DATA = Path(Config.data_path)
OUT = DATA / 'experiments/operational_path_shift/20260916'
STATE = None


def context():
    global STATE
    if STATE is not None:
        return STATE
    os.chdir(REPO)
    torch.set_num_threads(1)
    runtime = old.load_runtime(REPO)
    source = DATA / 'cache/load_branch_restoration_2026-09-15/combined.pkl.gz'
    reference = base.load(source)
    for name, expected in reference['settings']['scripts'].items():
        path = REPO / name
        if not path.is_file():
            path = ARCHIVE / name
        assert base.digest(path) == expected, str(path)
    for path, expected in reference['settings']['checkpoints'].items():
        assert base.digest(path) == expected, path
    assert base.digest(DATA / 'realworld_results.pkl.gz') == reference['settings']['source_sha256']
    assert reference['settings']['rlcft_scale'] == 1.
    old.RLCFT_SCALE = 1.
    baselines = {r['method']: r for r in reference['results'] if not r['event']}
    protocol = dict(source=str(source), source_hash=base.digest(source),
                    code_hashes={str(p): base.digest(p) for p in
                                 (Path(__file__), Path(base.__file__), Path(old.__file__))},
                    checkpoint_hashes=reference['settings']['checkpoints'],
                    rlcft_scale=1., safe_scale=10., linear_gain=10., q_cap_mvar=25.,
                    dt_seconds=6, feature='original 55-entry line reactance; original one-update timing',
                    initialization='method-specific cached daily state and previous reactive action',
                    criterion='modeled-node proxy; not meter compliance',
                    device='cuda' if torch.cuda.is_available() else 'cpu')
    STATE = runtime, DATA, reference['profiles'], baselines, protocol
    return STATE


def apply_case(env, case, k, r0, x0):
    active = k >= case['event']
    if case['direction'] == 'corridor_split':
        lines = case['lines']
        env.network.line.loc[lines, 'r_ohm_per_km'] = r0[lines] * (case['r_factor'] if active else 1.)
        env.network.line.loc[lines, 'x_ohm_per_km'] = x0[lines] * (case['x_factor'] if active else 1.)
    elif case['direction'] == 'branch_exchange':
        line = case['line']
        env.network.line.at[line, 'from_bus'] = case['new_parent'] if active else case['old_parent']
        env.network.line.at[line, 'r_ohm_per_km'] = case['tie_r'] if active else r0[line]
        env.network.line.at[line, 'x_ohm_per_km'] = case['tie_x'] if active else x0[line]
    else:
        raise ValueError(case['direction'])


base.context = context
base.apply_case = apply_case
base.ROOT = str(OUT / 'cache')


def graph(net):
    g = nx.Graph()
    g.add_nodes_from(net.bus.index)
    for idx, row in net.line.iterrows():
        if row.in_service:
            g.add_edge(int(row.from_bus), int(row.to_bus), line=int(idx))
    return g


def cases():
    env, _ = base.build_env(context()[0])
    net = env.network
    assert net.switch.closed.all() and net.sgen.in_service.all()
    g = graph(net)
    assert nx.is_tree(g)
    items = []
    for bus, event, label in ((32, 7838, 'midday'), (53, 12279, 'evening')):
        nodes = nx.shortest_path(g, 0, bus - 1)
        lines = [g[a][b]['line'] for a, b in zip(nodes, nodes[1:])]
        for rf, xf in ((2.5, 1.), (1., 2.5), (2., 2.5), (2.5, 2.), (2.5, 2.5), (2.75, 2.5)):
            items.append(dict(id=f'corridor{bus}_r{rf:g}_x{xf:g}_{label}',
                              direction='corridor_split', lines=lines, r_factor=rf, x_factor=xf,
                              event=event, start=event, stop=event + 600,
                              assumption='hypothetical equivalent path impedance; unchanged connectivity'))
    for line, new_parent in ((30, 20), (30, 29), (45, 44), (45, 37)):
        old_parent = int(net.line.at[line, 'from_bus'])
        child = int(net.line.at[line, 'to_bus'])
        path = nx.shortest_path(g, new_parent, child)
        links = [g[a][b]['line'] for a, b in zip(path, path[1:])]
        changed = g.copy()
        changed.remove_edge(old_parent, child)
        changed.add_edge(new_parent, child)
        assert nx.is_tree(changed) and set(changed) == set(g)
        assert not (net.switch.element.eq(line) & net.switch.et.eq('l')).any()
        for event, label in ((7838, 'midday'), (12279, 'evening')):
            items.append(dict(id=f'tie{new_parent+1}_to{child+1}_{label}', direction='branch_exchange',
                              line=line, old_parent=old_parent, new_parent=new_parent, child=child,
                              tie_r=float(net.line.loc[links, 'r_ohm_per_km'].sum()),
                              tie_x=float(net.line.loc[links, 'x_ohm_per_km'].sum()),
                              event=event, start=event, stop=event + 600, tree_verified=True,
                              assumption='synthetic normally-open tie; impedance equals original endpoint path sum; not verified SCE asset'))
    return items


def run(case, method, full=False):
    return base.run(case, method, full, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'screen', 'full'), default='prepare')
    parser.add_argument('--cases-json', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = json.loads(args.cases_json.read_text(encoding='utf-8')) if args.cases_json else cases()
    batch = args.cases_json.stem if args.cases_json else 'initial'
    dest = OUT / f'{args.stage}_{batch}'
    dest.mkdir(exist_ok=True)
    manifest = dict(protocol=context()[4], cases=chosen)
    path = dest / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Existing run protocol differs'
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.stage == 'prepare':
        verified = base.verify_resume()
        (dest / 'resume_verification.json').write_text(json.dumps(verified, indent=2))
        print(json.dumps(dict(cases=len(chosen), device=context()[4]['device'], replay=verified)), flush=True)
        return
    methods = ('No control', *old.CONTROL_METHODS) if args.stage == 'full' else old.CONTROL_METHODS
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, case, method, args.stage == 'full') for case in chosen for method in methods]
        for f in as_completed(futures):
            rows.append(f.result())
            base.aggregate([r for r in rows if r['method'] != 'No control'], dest)
            (dest / 'all_summaries.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(f'PROGRESS {len(rows)}/{len(futures)}', flush=True)
    print('COMPLETE ' + str(dest), flush=True)


if __name__ == '__main__':
    main()
