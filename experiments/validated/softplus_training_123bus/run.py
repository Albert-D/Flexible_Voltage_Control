"""Reproduce the adopted delta010 123-bus learning configuration."""
from runtime import *
import argparse, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def train(args):
    if not args.run_id.isascii() or not args.run_id.replace('_','').replace('-','').isalnum(): raise ValueError('ASCII run ID required')
    protocol_path=Path(args.protocol)
    protocol=load_recovery15_protocol(protocol_path)
    run=Path(args.output_root)/args.run_id; start=time.perf_counter()
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; select --device cpu')
    source_paths=[Path(__file__), FEEDER/'runtime.py', FEEDER/'evaluation123.py',
        FEEDER/'model123.py', FEEDER/'learner.py', FEEDER/'environment123.py', FEEDER/'protocol123.py',
        SPEED/'fast_environment.py', SPEED/'fast_runtime.py', SPEED/'support.py',
        ROOT/'src/softplus_controller.py', ROOT/'data/case_123.mat', RECOVERY15_MANIFEST]
    config=dict(args={k:v for k,v in vars(args).items() if k not in ('resume','stop_at')},
        protocol_sha256=sha(protocol_path),sources={str(p):sha(p) for p in source_paths},
        model_config=RECOVERY15_SPEC.as_config(),from_scratch=True,output_scale=1.,
        reward=dict(band=1.,deviation=.001,delta=args.delta_weight,q=0.),
        runtime=dict(device=str(device),solver='lightsim_recycle',fused_adam=device.type=='cuda',
                     actor='synchronized_cpu_mirror',replay='cached_severity',threads=1),
        terminal='physical recovery or hard failure; bootstrap collection and stage limits')
    if args.resume:
        if read(run/'config.json')!=config: raise ValueError('Config/source mismatch on resume')
    else:
        run.mkdir(parents=True,exist_ok=False); save_json(run/'config.json',config)
    l=recovery15_learner(args.seed,device,args.actor_lr,args.critic_lr)
    replay=CachedReplay(seed=args.seed+3000); rng=np.random.default_rng(args.seed+1000)
    noise_rng=np.random.default_rng(args.seed+2000); used=set(); records=[]; episodes=[]
    counts=dict(interactions=0,successful_steps=0,failed_steps=0,graph_draws=0,reset_attempts=0,reset_failures=0)
    timing={k:0. for k in ('reset','action','powerflow','replay_reward','sample','update','actor_sync','evaluation','checkpoint','solver_check')}
    prior_elapsed=0.
    if args.resume:
        ckmeta=read(run/'latest_checkpoint.json'); assert sha(Path(ckmeta['path']))==ckmeta['sha256']
        ck=torch.load(ckmeta['path'],map_location=device,weights_only=False); l.load(ck['learner'])
        replay.rows=ck['replay']; replay.pos=ck['replay_pos']; replay.rebuild_severity()
        replay.rng.bit_generator.state=ck['replay_rng']; rng.bit_generator.state=ck['rng']
        noise_rng.bit_generator.state=ck['noise_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
        counts=ck['counts']; timing=ck['timing']; used=set(ck['used']); records=ck['records']
        episodes=ck['episodes']; prior_elapsed=ck['elapsed']
        (run/'training.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes),encoding='utf-8')
    mirror=ActorMirror(l.actor); env=Feeder(); excluded=set(protocol['excluded_graphs'])
    def elapsed(): return prior_elapsed+time.perf_counter()-start
    def checkpoint():
        tick=time.perf_counter(); path=run/f'checkpoint_{counts["interactions"]:06d}.pt'
        temp=path.with_suffix('.tmp')
        torch.save(dict(learner=l.state(),model_config=l.actor.model_config(),replay=replay.rows,
            replay_pos=replay.pos,replay_rng=replay.rng.bit_generator.state,rng=rng.bit_generator.state,
            noise_rng=noise_rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            counts=counts,timing=timing.copy(),used=sorted(used),records=records,episodes=episodes,
            elapsed=elapsed(),protocol_sha256=config['protocol_sha256']),temp)
        temp.replace(path); save_json(run/'latest_checkpoint.json',dict(path=str(path),sha256=sha(path)))
        timing['checkpoint']+=time.perf_counter()-tick
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_recovery15_worker) as pool:
        def evaluate():
            tick=time.perf_counter(); ix=len(records)
            rotation=protocol['groups']['rotation'][ix*16:(ix+1)*16]
            assert len(rotation)==16
            rows=assess_recovery15(pool,mirror.actor,protocol['groups']['development']+rotation)
            timing['evaluation']+=time.perf_counter()-tick
            row=dict(interactions=counts['interactions'],num_topologies_seen=len(used),updates=l.updates,
                     development=recovery15_summary(rows[:64]),rotation=recovery15_summary(rows[64:]),rows=rows,
                     elapsed=elapsed(),timing=timing.copy())
            records.append(row); save_json(run/'curve_data.json',records); checkpoint()
            print(json.dumps(dict(run=args.run_id,interactions=counts['interactions'],topologies=len(used),
                       development=row['development'],rotation=row['rotation'],elapsed=elapsed())),flush=True)
        if not records: evaluate()
        stop=min(args.max_interactions,args.stop_at); reason='stage_budget'
        while counts['interactions']<stop:
            if (run/'STOP').exists(): reason='requested_stage_stop'; break
            if time.perf_counter()-start>args.wall_minutes*60: reason='wall_budget'; break
            if len(records)>=3 and all(r['development']['hard_failures']>2 for r in records[-2:]):
                reason='evaluation_failure_stop'; break
            tick=time.perf_counter(); topo=draw_graph(env,rng,used|excluded,counts)
            scene,state=scene_for(env,topo,rng,510000000+len(used),counts)
            timing['reset']+=time.perf_counter()-tick; used.add(topo['id']); assert not used&excluded
            x=env.features.copy(); previous=np.zeros(14,dtype='f4'); reward_total=0.; losses={}
            budget=min(60,stop-counts['interactions']); actual=0; terminal=False
            for step in range(budget):
                tick=time.perf_counter()
                with torch.no_grad(): raw=mirror.actor(torch.as_tensor(state)[None],torch.as_tensor(x)[None]).numpy().reshape(14)
                noise=noise_rng.normal(0,args.warm_noise if counts['interactions']<512 else args.noise,14)
                q=np.clip(previous-np.clip(raw+noise,-5,5),-50,50).astype('f4')
                timing['action']+=time.perf_counter()-tick
                counts['interactions']+=1; actual+=1; tick=time.perf_counter()
                try:
                    nxt=env.step(q)
                    if not np.isfinite(nxt).all() or nxt.min()<.75 or nxt.max()>1.25: raise ValueError('Hard voltage bound')
                    local=np.maximum(np.abs(nxt-1)-.045,0)+.001*np.abs(nxt-1)+args.delta_weight*(q-previous)**2
                    reward=-(.5*local+.5*local.mean()); counts['successful_steps']+=1
                    terminal=bool(np.all((nxt>.955)&(nxt<1.045)))
                except (ValueError,pp.powerflow.LoadflowNotConverged):
                    nxt=state.copy(); reward=np.full(14,-100.,dtype='f4'); terminal=True; counts['failed_steps']+=1
                timing['powerflow']+=time.perf_counter()-tick
                tick=time.perf_counter(); replay.push(state,x,previous,q,reward,nxt,x,[float(terminal)])
                timing['replay_reward']+=time.perf_counter()-tick
                if counts['interactions']>=512 and len(replay.rows)>=128:
                    tick=time.perf_counter(); batch=sample_replay_fast(replay,128,.5,.75); timing['sample']+=time.perf_counter()-tick
                    tick=time.perf_counter(); losses=l.update(batch)
                    if device.type=='cuda': torch.cuda.synchronize()
                    timing['update']+=time.perf_counter()-tick
                    if any(v is not None and not np.isfinite(v) for v in losses.values()): raise ValueError('Nonfinite loss')
                    if l.updates%3==0:
                        tick=time.perf_counter(); mirror.sync(l.actor); timing['actor_sync']+=time.perf_counter()-tick
                state,previous=nxt,q; reward_total+=float(reward.sum())
                if terminal: break
            if len(used) in (1,10,30) or len(used)%200==0:
                tick=time.perf_counter(); base=OriginalFeeder(); base.reset(scene); base.step(previous)
                error=float(np.nanmax(np.abs(base.voltage-env.voltage))); assert error<=1e-6
                timing['solver_check']+=time.perf_counter()-tick
            episode=dict(interactions=counts['interactions'],num_topologies_seen=len(used),episode_steps=actual,
                         terminal=terminal,truncated=actual==budget and not terminal,reward_per_step=reward_total/actual,
                         losses=losses,scene=scene,counts=counts.copy())
            episodes.append(episode); append(run/'training.jsonl',episode)
            save_json(run/'progress.json',dict(counts=counts,topologies=len(used),updates=l.updates,timing=timing,elapsed=elapsed()))
            if counts['failed_steps']>=5 and counts['failed_steps']/counts['interactions']>.01:
                reason='training_failure_stop'; break
            if counts['interactions']-records[-1]['interactions']>=args.eval_every: evaluate()
        if records[-1]['interactions']!=counts['interactions']: evaluate()
    import psutil
    save_json(run/'stage_result.json',dict(reason=reason,counts=counts,topologies=len(used),timing=timing,
        elapsed=elapsed(),latest=read(run/'latest_checkpoint.json'),final=records[-1],
        train_evaluation_graph_overlap=len(used&excluded),rss_mib=psutil.Process().memory_info().rss/2**20,
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None))
    print(json.dumps(dict(stage_complete=args.run_id,interactions=counts['interactions'],reason=reason)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True)
    p.add_argument('--delta-weight',type=float,default=.01); p.add_argument('--actor-lr',type=float,default=2.5e-5)
    p.add_argument('--critic-lr',type=float,default=1e-3); p.add_argument('--seed',type=int,default=2601)
    p.add_argument('--noise',type=float,default=.25); p.add_argument('--warm-noise',type=float,default=.75)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--max-interactions',type=int,default=30000); p.add_argument('--stop-at',type=int,default=30000)
    p.add_argument('--wall-minutes',type=float,default=300); p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p.add_argument('--protocol',default=str(Path(Config.data_path)/'experiments/123bus_recovery15_2026-10-02/protocol.json'))
    p.add_argument('--output-root',default=str(Path(Config.data_path)/'experiments/123bus_recovery15_reproduction'))
    mp.freeze_support(); train(p.parse_args())
