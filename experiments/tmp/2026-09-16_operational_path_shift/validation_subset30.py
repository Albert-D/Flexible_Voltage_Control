"""User-requested budget reduction; preserve the original frozen protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import robustness_validation as v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run', 'audit', 'report'])
    args = ap.parse_args()
    original = v.manifest()
    cases = [c for c in original['independent'] if int(c['id'].rsplit('_', 1)[1]) < 10]
    assert len(cases) == 30
    target = v.OUT/'subset30'
    target.mkdir(exist_ok=True)
    revision = target/'budget_revision.json'
    if not revision.exists():
        v.write_json(revision, dict(reason='User requested reducing runtime from 60 to 30 scenarios after partial execution.',
                     selection='First 10 original frozen draws per branch, independent of outcomes.',
                     original_manifest_sha256=v.paths.base.digest(v.OUT/'manifest.json'),
                     selected_case_ids=[c['id'] for c in cases],
                     completed_at_revision=[p.stem for p in sorted((v.OUT/'rows').glob('validation*.json'))],
                     note='All extra completed results and interrupted logs retained; this budget change was not preregistered.'))
    if args.command == 'prepare':
        missing = [(c['id'], m) for c in cases for m in v.METHODS if not (v.OUT/'rows'/f'{c["id"]}__{m}.json').exists()]
        print({'selected': len(cases), 'missing_method_days': len(missing), 'missing': missing}, flush=True)
    elif args.command == 'run':
        jobs = [(c, m) for c in cases for m in v.METHODS if not (v.OUT/'rows'/f'{c["id"]}__{m}.json').exists()]
        print(f'SUBSET30: {len(jobs)} missing jobs; 8 workers', flush=True)
        with ProcessPoolExecutor(max_workers=8) as pool:
            fs = [pool.submit(v.run_one, c, m) for c, m in jobs]
            for i, f in enumerate(as_completed(fs), 1):
                row = f.result()
                v.table()
                print(f'SUBSET PROGRESS {i}/{len(jobs)}: {row}', flush=True)
    elif args.command == 'audit':
        checks = [v.audit_one(c, m) for c in cases for m in v.METHODS]
        v.write_json(target/'audit.json', checks)
        print({'verified': sum(r['status'] == 'verified' for r in checks), 'total': len(checks)}, flush=True)
    elif args.command == 'report':
        frame = v.table()
        chosen = {c['id'] for c in cases}
        frame = frame[frame.phase.eq('local') | frame.case_id.isin(chosen)]
        assert len(frame) == 153
        frame.to_csv(target/'all_metrics.csv', index=False)
        summaries = []
        pairs = []
        for (phase, method), g in frame.groupby(['phase', 'method']):
            ok = g[g.status.eq('ok')]
            summaries.append(dict(phase=phase, method=method, completed=len(g), valid=len(ok), passes=int(ok.passes.sum()),
                                  peak_median=ok.peak.median(), peak_max=ok.peak.max(), duration_median=ok.duration.median(), duration_max=ok.duration.max()))
        for (phase, cid), g in frame.groupby(['phase', 'case_id']):
            rows = {r['method']: r for r in g.to_dict('records') if r['status'] == 'ok'}
            if len(rows) != 3:
                continue
            for b in ['Linear', 'Safe-DDPG']:
                r, baseline = rows['RLC-FT'], rows[b]
                pairs.append(dict(phase=phase, case_id=cid, line=r['line'], baseline=b,
                                  peak_reduction=baseline['peak']-r['peak'], duration_reduction=baseline['duration']-r['duration'], cost_ratio=r['cost']/baseline['cost']))
        v.pd.DataFrame(summaries).to_csv(target/'summary.csv', index=False)
        v.pd.DataFrame(pairs).to_csv(target/'paired_metrics.csv', index=False)
        v.write_json(target/'topology_inventory.json', original['inventory'])
        (target/'figures').mkdir(exist_ok=True)
        import plot_robustness_validation as plot
        plot.ROOT = target
        plot.INDEPENDENT_N = 30
        plot.NOTEBOOK_NAME = 'robustness_validation_subset30_review.ipynb'
        plot.main()
        print((target/'summary.csv').read_text(), flush=True)


if __name__ == '__main__':
    main()
